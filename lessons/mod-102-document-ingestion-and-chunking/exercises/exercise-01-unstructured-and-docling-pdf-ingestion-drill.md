# Exercise 01 — unstructured and Docling PDF ingestion drill

**Estimated effort:** ~2 hours.

## Objective

Build a per-format router that ingests a small mixed-format corpus into the normalized `Chunk` shape from chapters 1 and 7. You will use both `unstructured` and `Docling` on the PDF path (and one of `MarkdownHeaderTextSplitter` or `unstructured.partition_html` for the non-PDF files), and end with a passing round-trip test: no content silently lost, every chunk carries the metadata the retriever will need.

The point is not "get any parser to run." The point is to have a working router that can grow into `exercise-02` (OCR fallback), `exercise-03` (chunker ablation), and `exercise-05` (tables and figures).

## Prerequisites

- Chapter 1 (*Ingestion Overview*), chapter 2 (*Native-Text Parser Toolchain*), chapter 7 (*Metadata as First-Class Fields*) — read once.
- The `mod-101-toolchain` project from mod-101 exercise 05 is green (`pytest -q` passes on `test_smoke.py`). You will grow it into `mod-102-ingest`.
- Add dependencies: `unstructured[pdf,html,md,docx]`, `docling`, `pypdf` (for a truthy-text-layer probe), `python-magic` (MIME detection).

## The corpus

You will build a tiny mixed corpus under `data/corpus/` with three files. Pick any files you can legally share; suggested set:

- `data/corpus/handbook.pdf` — a native-text PDF with clear H1/H2/H3 headings and at least one table. Any well-formatted PDF you have on hand (a book chapter, a technical spec, a policy handbook). If nothing else, use the [Kubernetes Documentation PDF export](https://kubernetes.io/docs/home/) or an [arXiv paper](https://arxiv.org/) with clean structure.
- `data/corpus/notes.md` — a Markdown document with at least three heading levels and one code block. Write your own; ~300 lines of anything.
- `data/corpus/article.html` — one saved-and-cleaned web article (right-click → save-as). Real HTML with sidebars, ads, and a `<article>` region is intentionally messy.

Commit the corpus (or `.gitignore` it and provide a `download.sh`; do not commit anything you cannot legally distribute).

## Part A — Router scaffold (30 min)

Create `src/ingest/router.py`. Implement:

```python
def ingest(path: pathlib.Path, *, permission_scope: str = "public") -> list[Chunk]:
    """Route a file to the right parser and return normalized Chunks."""
    ...
```

Requirements:

- MIME detection via `python-magic` (or file-extension fallback if magic is unavailable on your OS).
- Dispatch by MIME:
  - `application/pdf` → PDF path (Part B).
  - `text/html` → HTML path.
  - `text/markdown` → Markdown path.
- Every path returns `list[Chunk]` in the schema from chapter 7 (`chunk_id`, `doc_id`, `source_uri`, `ordinal`, `text`, `element_type`, `tokens`, `section_path`, `page`, `bbox`, `updated_at`, `permission_scope`, `parser`, `parser_version`, `chunker`, `chunker_version`, `content_hash`, `ingested_at`, at minimum).
- Ordinal is monotonic within a document; `chunk_id` is stable (a hash of `(doc_id, chunker_version, ordinal)`), not a UUID.

## Part B — Two-way PDF ingest: unstructured and Docling side-by-side (45 min)

Implement two PDF back-ends. Same input, two implementations, same `Chunk` output.

### B.1 — `unstructured`

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

elements = partition_pdf(filename=str(path), strategy="hi_res",
                         infer_table_structure=True, languages=["eng"])
title_chunks = chunk_by_title(elements, max_characters=2000, new_after_n_chars=1500)
```

Normalize each `title_chunk` into a `Chunk`. Preserve:

- `section_path` from the parent title chain.
- `page` from `element.metadata.page_number`.
- `element_type` mapped from `element.category` (`Title`, `NarrativeText`, `ListItem`, `Table`, etc.).
- Table elements' `metadata.text_as_html` in `extras["text_as_html"]`.

### B.2 — Docling

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert(str(path)).document
```

Iterate `doc.iterate_items()` and produce one `Chunk` per top-level structural item (heading, paragraph, table, figure caption). For headings, use the item to build a `section_path` you propagate into every subsequent chunk until a new heading of equal or higher level appears.

### B.3 — Compare

Write `scripts/compare_pdf_parsers.py` that runs both back-ends on `handbook.pdf` and prints a table:

```
parser        | n_chunks | avg_tokens | n_tables | n_headings | mean_section_path_depth
unstructured  |          |            |          |            |
docling       |          |            |          |            |
```

There is no "correct" answer; the point is to feel the difference.

## Part C — Markdown and HTML paths (30 min)

- **Markdown.** Use `MarkdownHeaderTextSplitter` (LangChain) or `MarkdownNodeParser` (LlamaIndex) to split at H1/H2/H3 and populate `section_path`. Recurse-split any section that exceeds 800 tokens with `RecursiveCharacterTextSplitter`. Preserve code blocks as their own `element_type="code"` chunks.
- **HTML.** Run `trafilatura.extract(...)` first to strip chrome. Feed the cleaned Markdown-ish output through the Markdown path *or* the cleaned HTML through `unstructured.partition_html`. Either is fine; document which you chose.

## Part D — Round-trip tests (30 min)

Create `tests/test_ingest.py`. Implement at least four tests:

1. **`test_no_content_silently_lost`** — for each file in `data/corpus/`, `sum(len(chunk.text) for chunk in chunks) >= 0.90 * len(raw_text_of_source)`. Whitespace / boilerplate stripping accounts for the 10%.
2. **`test_every_chunk_has_required_metadata`** — assert `doc_id`, `source_uri`, `ordinal`, `text`, `element_type`, `tokens`, `content_hash`, `parser`, `chunker_version`, `ingested_at` are non-null on every chunk. Assert `section_path` is non-empty on every chunk of `handbook.pdf` and `notes.md` (HTML may be flat).
3. **`test_stable_chunk_ids_across_runs`** — run `ingest(handbook.pdf)` twice, assert `chunk_id` lists are identical.
4. **`test_pdf_chunks_carry_page_numbers`** — assert every PDF chunk has a non-null `page` and it is `>= 1`.

`pytest -q` must pass.

## Deliverables

- `src/ingest/router.py` (~150–250 LoC).
- `src/ingest/schema.py` with the `Chunk` dataclass (or Pydantic model) from chapter 7.
- `scripts/compare_pdf_parsers.py` and its printed output committed as `docs/parser-comparison.txt`.
- `tests/test_ingest.py` — the four tests above.
- `data/corpus/` (or a `download.sh` if you cannot commit the files).
- A short `README.md` in `src/ingest/` documenting: which parser you defaulted to per format and why, one paragraph.

## Acceptance criteria

- `pytest -q tests/test_ingest.py` passes.
- Both `unstructured` and `Docling` back-ends produce well-formed `Chunk`s from `handbook.pdf` with `section_path`, `page`, and at least one `element_type="table"` chunk.
- `scripts/compare_pdf_parsers.py` runs and prints the comparison table.
- Every chunk has `content_hash` populated and `chunk_id` deterministic.
- Zero uses of `dict[str, Any]` metadata that bypass the `Chunk` schema.

## Stretch goals

- Add a fourth back-end: PyMuPDF (`fitz`). Extract text at the block level (`page.get_text("blocks")`) and populate `bbox` on every chunk. Include a PyMuPDF row in `parser-comparison.txt`.
- Add a `.docx` handler using `unstructured.partition.docx`. Confirm heading hierarchy propagates.
- Time each parser on `handbook.pdf` and add a `wall_clock_s` column to the comparison. Feel the "fast vs. hi_res" trade-off in seconds.
