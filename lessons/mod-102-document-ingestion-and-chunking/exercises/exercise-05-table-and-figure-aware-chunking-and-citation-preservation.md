# Exercise 05 — Table- and figure-aware chunking with citation preservation

**Estimated effort:** ~3 hours.

## Objective

Extend the ingest pipeline to handle the three "hard content" types from chapter 8 — tables, formulas, and figures — without losing the source-cell / equation / figure reference the generator will need to cite. You will build the dual chunk shape (content + summary), preserve cell / equation / image anchors in metadata, and pass round-trip tests that prove a citation can be traced back to the exact source unit.

This exercise is the difference between a RAG system that says *"salary bands range from $120K to $200K"* (with no defensible source) and one that says *"the L30 mid-band is $165K"* citing a specific cell in a specific table on a specific page.

## Prerequisites

- Chapter 8 (*Tables, Formulas, and Figures*) read once.
- `exercise-01` router works.
- Preferred parser: **Docling** for tables and figures (chapter 2 explained why). `unstructured` with `strategy="hi_res", infer_table_structure=True` is an acceptable substitute.
- For formulas: **Nougat** on a GPU is ideal. If you do not have a GPU, download 2–3 short arXiv papers where Docling's formula handling is sufficient, or use MathPix's free-tier API for a handful of pages.
- Python deps (in addition to earlier exercises): `docling` (or ensure `unstructured[pdf]` is installed), optionally `nougat-ocr`.

## The corpus

Assemble `data/hard-content/`:

- **2 PDFs with substantial tables.** Financial reports, salary benchmarks, technical spec tables, comparison matrices. At least one table should have ≥ 20 rows so you exercise the row-group chunking case.
- **1 STEM PDF with equations.** Any recent arXiv paper is fine ([arxiv.org](https://arxiv.org/)); pick one with a mix of inline math and display equations, ideally with numbered equations.
- **1 PDF with figures.** A paper or report with at least 3 figures that have real captions. The arXiv paper above may double up.

## Part A — Tables (60 min)

Extend the ingest pipeline so a PDF with tables produces:

### A.1 — Content chunks (structural)

For each table detected:

- **Small tables (≤ 30 rows)** — one `Chunk` with `element_type="table"`. `text` is a Markdown rendering. `extras["table_id"]`, `extras["header_row"]`, `extras["cells"]` (list of lists of `{row, col, text}`), `extras["caption"]` if the parser found one.
- **Medium tables (30–200 rows)** — one `Chunk` per 20-row group. Every chunk repeats the header row at the top and includes a caption `"Table {n}, rows {a}-{b} of {N}"`. All chunks share the same `extras["table_id"]`. Row indices in `extras["row_range"]` per chunk.

For every content chunk: `bbox` populated when the parser gave it (Docling does; `unstructured` gives element bboxes via `metadata.coordinates`).

### A.2 — Summary chunks

For each *table* (not each table-chunk), produce one `Chunk` with `element_type="table_summary"`. Choose one:

- **Template summary.** `f"Table {n}: {caption}. Columns: {header_row}. {row_count} rows."` Deterministic, free.
- **LLM-generated summary.** One call to a small LLM (Claude Haiku, GPT-4o-mini, or a local model) asking "Describe this table in one sentence: {markdown}". Cache by table content hash.

The summary chunk's `extras["table_id"]` links back to the content chunk(s). The summary is what the retriever will hit; the content is what the assembler pulls in.

### A.3 — Test

- **`test_no_mid_row_splits`** — assert every content chunk's rows come in complete groups (no orphan row without its column values).
- **`test_header_repeated_in_row_group_chunks`** — for medium tables, assert every row-group chunk contains the header row at position 0.
- **`test_table_summary_linked_to_content`** — for every `table_summary` chunk, assert the `table_id` resolves to at least one `table` chunk.
- **`test_table_cell_round_trip`** — pick 5 random `(table_id, row, col)` triples, look up the cell, and assert the extracted text equals what `pdfplumber` or `Docling` re-extraction gives from the source.

## Part B — Formulas (45 min)

Handle the STEM PDF:

- **If using Nougat:** run `nougat path/to/paper.pdf -o out/`. Parse the `.mmd` output; segment into chunks at heading boundaries, preserve `$...$` and `$$...$$` markup verbatim.
- **If using Docling:** iterate `doc.iterate_items()` and treat formula items as their own chunks or as annotated inline elements per Docling's current spec ([Docling docs](https://ds4sd.github.io/docling/)).
- **If using MathPix:** call the API on a per-page basis for 2–3 pages containing equations; store the returned LaTeX.

Produce `Chunk`s where:

- **Display equations** get `element_type="formula"` with `extras["latex"]` and `extras["equation_number"]` (parsed from the paper's `(17)`-style numbering when present).
- **A formula is never split from its introductory sentence.** Chunk the sentence-and-formula together — either as `element_type="paragraph"` with the formula embedded in the text, or as two linked chunks (`paragraph` + `formula`) with a shared `extras["group_id"]`.

Test:

- **`test_latex_preserved`** — assert at least three chunks contain valid LaTeX (a `$...$` or `$$...$$` region parseable by a LaTeX-aware regex).
- **`test_display_equation_not_split`** — assert no `formula` chunk's text splits across a `$$` boundary.

## Part C — Figures (45 min)

For the PDF with figures:

- Extract each figure's image bytes (PyMuPDF's `page.get_images()` + `doc.extract_image(xref)` is convenient) to `data/hard-content/figures/{doc_id}/fig-{n}.png`.
- Produce a `Chunk` per figure with `element_type="figure_caption"`, `text` = the caption + surrounding paragraph, `extras["figure_id"]`, `extras["image_uri"]` pointing at the extracted file.
- If a figure has no caption, generate a one-sentence description with an LLM ("Describe this figure image in one sentence") and store as `element_type="figure_summary"` chunk, linked by `figure_id`.

Test:

- **`test_figure_bytes_written`** — assert every `figure_caption` chunk's `image_uri` resolves to a real file on disk.
- **`test_figure_round_trip`** — pick 3 figures, fetch `image_uri`, assert the file's `sha256` matches what direct extraction from the PDF produces.

## Part D — Citation contract (30 min)

Author `src/ingest/citation.py`:

```python
def cite(chunk: Chunk, *, cell: tuple[int, int] | None = None) -> str:
    """Return the canonical citation string for this chunk (chapter 8's table)."""
    ...
```

The function must handle every `element_type` your ingest produces (`paragraph`, `heading`, `list_item`, `table`, `table_summary`, `formula`, `figure_caption`, `figure_summary`, `code`) and return the `[cite: ...]` format from chapter 8's citation contract table. For a `table` chunk with a cell coordinate, the citation names the cell:

```
[cite: doc-42, table:3, row=17, col=2]
```

For a formula:

```
[cite: paper-42, eq.17]
```

Test:

- **`test_every_chunk_type_citable`** — for every chunk in the corpus, `cite(chunk)` returns a non-empty string and does not raise.
- **`test_citation_round_trip`** — for a hand-picked set of ~10 chunks across types, assert the citation string encodes enough to reconstruct which chunk it came from (parse the citation string and look up the chunk).

## Deliverables

- Updated `src/ingest/router.py` and `src/ingest/schema.py` handling all three hard content types.
- `src/ingest/citation.py` with the `cite(...)` function.
- Extracted figures under `data/hard-content/figures/`.
- Tests: all Part A–D tests present and passing.
- `docs/hard-content.md` with three short paragraphs: how your pipeline handles each of tables / formulas / figures, and one example citation per type from a chunk in your corpus.

## Acceptance criteria

- `pytest -q` passes.
- Every table chunk carries `extras["header_row"]` and `extras["cells"]`, and no chunk contains a mid-row split.
- Every table has both content chunk(s) *and* a summary chunk, linked by `table_id`.
- Every formula chunk contains real LaTeX (not `α ∼ (0, σ2)` line-noise).
- Every figure chunk's `image_uri` resolves; round-trip test passes.
- `cite(chunk)` returns a distinct, parseable citation string for every element type your ingest produces.

## Stretch goals

- Add a *multimodal* representation for figures: embed each figure image with `openai/clip-vit-base-patch32` (or `google/siglip-base-patch16-224`), store the vector alongside the text caption embedding, and support querying by both text-of-caption and image-of-figure. Chapter 8 named this as a mod-112 pattern; do a preview.
- Instead of one LLM call per table for the summary, batch: pass 5 tables in one prompt and parse a JSON array. Measure the wall-clock and token savings.
- End-to-end: build a small Streamlit / Gradio page that takes a query, retrieves the top-3 chunks including table/figure/formula chunks, and renders the citations as clickable links back to the source (opening the PDF at the correct page, or displaying the extracted figure image inline). This is the "we actually shipped citations" shape production RAG products aim at.
