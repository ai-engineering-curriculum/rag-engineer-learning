# Chapter 8 — Tables, Formulas, and Figures Without Losing the Citation

## Why this chapter exists

Naïve chunkers split a table across three chunks, drop the header row from two of them, mangle every formula into `α`-line-noise, and lose the caption of every figure. Then someone asks a question whose answer is a specific cell in that table, the retriever finds a nearby paragraph, and the LLM confidently makes up a number.

This chapter is about keeping the *hard* content types intact — tables, formulas, figures — through ingest and chunking so the retriever can find them, the LLM can use them, and the citation can point back to the exact row / equation / figure that grounds the answer. It builds on chapter 2 (parsers that emit structured tables and figures) and chapter 7 (metadata schema that carries the anchors).

## The unifying pattern: dual chunk shapes

For each hard content unit, index *two* representations and link them:

1. **The content chunk** — the table itself, the formula in LaTeX, the figure's bytes + caption. Preserves the exact source needed for citation.
2. **The summary chunk** — a short LLM-generated (or template-generated) natural-language sentence describing what the content shows. Retrievable by ordinary text queries.

The retriever hits the summary chunk on a text query; the assembler (mod-107) resolves to the content chunk before the LLM sees it; the generator can cite the content chunk by ID. This is the "table summary" pattern popularized by LlamaIndex's [`RecursiveRetriever` / index-per-table](https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/) example and equally applicable to formulas and figures.

## Tables

### Preserving structure at ingest

Chapter 2's parsers give you table structure natively:

- **Docling** — tables land as `DoclingDocument.tables` with cell coordinates, row/column indices, and header detection. Cells are addressable.
- **unstructured (`hi_res`)** — `Table` element with `metadata.text_as_html` containing the HTML representation, and `metadata.text_as_html` parseable back into a cell grid.
- **AWS Textract (`FeatureTypes=["TABLES"]`)** — returns `TABLE`, `CELL`, and `MERGED_CELL` blocks with row/column indices.
- **Azure Document Intelligence (`prebuilt-layout`)** — returns tables with `row_index`, `column_index`, `row_span`, `column_span`.
- **PyMuPDF `Page.find_tables()`** — extracts tables into a per-page table finder result with `.extract()` returning a list of rows.

Use whichever the parser gave you and keep the *structured* form. Do not stringify to `"col1 | col2 | col3\n..."` and lose the cell coordinates.

### Chunking a table

Three shapes based on size:

- **Small (≤ ~30 rows or ~1 embedding-context-worth of text)**. One chunk per table. `element_type="table"`, `text` = a Markdown or plain-text rendering (useful when the embedding model has not been trained on tables), plus a full `extras.table_cells` structured representation for later cell-level citation.
- **Medium (~30–200 rows)**. Split by row groups. Each chunk keeps `element_type="table"`, includes the *header row* at the top, and a caption `"Table 3, rows 41–80 of 200"` so a chunk retrieved in isolation still makes sense. Cross-reference every chunk to a `table_id` in metadata.
- **Large (200+ rows)**. Treat the table as tabular data, not text. Consider text-to-SQL (mod-112) instead of embedding every row.

Never split a table across a mid-row boundary. Never drop the header row on any chunk.

### The table-summary chunk

For every table, generate an additional short-text chunk that describes what the table shows:

```
Table 3, "Salary Bands by Level and Region", shows base compensation ranges
for engineering ladder levels 20–50 in USD, EUR, and GBP. Rows: 12 levels ×
3 regions = 36 rows. Header row: Level | Region | Min | Mid | Max.
```

- `element_type="table_summary"`.
- `parent_id` (or `table_id` in `extras`) points at the content chunk(s) of the actual table.
- The summary is what the retriever indexes; on hit, the assembler pulls in the table chunk itself so the LLM has both.

Two ways to produce the summary:

- **Template.** If tables have consistent structure (caption + header row), a template like "Table {n}, '{caption}', with columns {headers} and {N} rows" is deterministic, free, and often enough.
- **LLM-generated.** A one-shot call to `"describe this table in one sentence: {markdown_table}"`. Better on messy or ambiguous tables. Costs one LLM call per table at ingest. Cache aggressively.

Pin whichever you choose in the `Chunk.chunker_version` — a change in summary generation changes what enters the vector index.

### Preserving cell-level citation

The generator's job (mod-108) is to emit citations of the form `[cite: doc_id::table:3, cell(row=17, col=2)]` when it quotes a specific value. The chunker's job here is to make that possible by keeping the cell coordinate accessible:

```python
Chunk(
    chunk_id="doc-42::t3::rows-41-80",
    element_type="table",
    text=<markdown-rendered rows 41-80 with header>,
    extras={
        "table_id": "doc-42::t3",
        "row_range": (41, 80),
        "header_row": ["Level", "Region", "Min", "Mid", "Max"],
        "cells": [
            [{"row": 41, "col": 0, "text": "L30"}, ...],
            ...
        ],
        "caption": "Salary Bands by Level and Region",
    },
    ...
)
```

If the parser gave you cell bboxes on the PDF page, keep those too — a bbox-based citation deep-link (`?page=14#bbox=100,200,400,220`) is the gold standard.

## Formulas and LaTeX-aware chunking

### The problem

A general-purpose text parser (unstructured `fast`, PyMuPDF) reads `$\alpha \sim \mathcal{N}(0, \sigma^2)$` as a sequence of Unicode-ish glyphs (`α ∼ (0, σ2)` on a good day; complete garbage on a bad one). The embedding of that garbled string has no relationship to a query asking about the equation. Every academic-domain retrieval stack fails here first.

### The tools

- **Nougat** (chapter 2) — end-to-end transformer that OCRs academic PDFs into Markdown *with LaTeX* preserved (`$...$` and `$$...$$`). The right choice when equations are load-bearing.
- **Docling** — the layout model detects formulas; the output preserves them as formula items (see [Docling formulas](https://ds4sd.github.io/docling/) — check current docs for the latest formula-handling status).
- **MathPix** (commercial API) — high-accuracy OCR specifically for equations; overkill for a corpus with occasional formulas, essential for math-heavy content.

Whichever tool: the goal is `text` that contains `$\alpha \sim \mathcal{N}(0, \sigma^2)$`, not `α ∼ (0, σ2)`.

### Chunking around formulas

- **Do not split inside a formula.** A `$$...$$` display equation is one atomic unit, exactly like a table.
- **Keep the surrounding sentence.** A formula in isolation ("$E = mc^2$") is often useless without its introductory sentence ("Rest-mass energy is given by..."). Chunk the sentence-with-formula together.
- **Consider a "formula summary" chunk.** For display equations that name a phenomenon ("Bellman equation", "Bayes' rule"), a summary chunk `"Bellman equation for the state-value function of a Markov decision process."` gives the retriever a text handle on the equation.
- **Preserve the reference number.** Papers cite equations by number ("as in (17)"). If the parser gave you the equation number, keep it in `extras.equation_number` — the retriever might need to filter by "equation 17 in doc X" for a cross-reference lookup.

### Chunk metadata for a formula

```python
Chunk(
    chunk_id="paper-42::eq-17",
    element_type="formula",
    text="The Bellman equation for the state-value function is\n\n$$V^\\pi(s) = \\mathbb{E}_\\pi \\left[ R_{t+1} + \\gamma V^\\pi(S_{t+1}) \\mid S_t = s \\right]$$",
    extras={
        "latex": r"V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s \right]",
        "equation_number": "17",
        "display": True,
    },
    ...
)
```

## Figures

### The pairing rule

The unit of retrieval for a figure is **figure + caption + surrounding paragraph**, not the figure alone. A picture is worth a thousand words, but a picture's *embedding* is only useful if it lives near the text that describes it.

Three shapes:

- **Text-only pipeline** — index the caption + the surrounding paragraph as one chunk, with `element_type="figure_caption"` and a `extras.image_uri` pointing at the extracted image bytes (S3 key, local path). The LLM cites `[cite: doc-42::fig-3]`; the UI knows to render the image alongside.
- **Multimodal-embedding pipeline** — compute both a text embedding of the caption and a *visual* embedding of the figure (CLIP, SigLIP, ColPali) and store both. The retriever can then match text queries → captions AND image-like queries → figures. Multimodal retrieval is mod-112 territory; the chunker's job here is to keep both representations linked.
- **Figure-summary chunk** — for figures whose caption is terse ("Figure 3.") or missing, generate a one-sentence LLM description of the figure image and store it as a `figure_summary` chunk pointing back at the figure content.

### Extracting figure images at ingest

- **PyMuPDF** — `page.get_images()` and `doc.extract_image(xref)` extract raster images with their bytes. Vector illustrations may need `page.get_drawings()`.
- **unstructured (`hi_res`)** — `Image` and `FigureCaption` elements are emitted when the layout model detects them.
- **Docling** — figures land as first-class `PictureItem`s with their bounding box and (optionally) extracted image bytes.
- **Nougat** — figures come through as Markdown image references; the actual image bytes need to be extracted with a separate pass (PyMuPDF is convenient).

Store the extracted image alongside the corpus:

```
s3://corpus/doc-42/figures/fig-3.png
```

with the `extras.image_uri` pointing at it. Never inline binary bytes into `Chunk.text` — that path lies madness for the vector store and every JSON tool downstream.

## Citation contract across content types

The generator (mod-108) has to be able to say "this claim came from this specific thing." The chunker (this module) has to make the *specific thing* addressable. A citation contract sketch:

| Content type | Citation format | Fields required in `Chunk` |
|---|---|---|
| Paragraph | `[cite: doc_id, sec=Compensation, p.14]` | `doc_id`, `section_path`, `page` |
| Table cell | `[cite: doc_id, table:3, row=17, col=2]` | `doc_id`, `extras.table_id`, `extras.cells` |
| Formula | `[cite: doc_id, eq.17]` | `doc_id`, `extras.equation_number` |
| Figure | `[cite: doc_id, fig.3]` | `doc_id`, `extras.figure_id`, `extras.image_uri` |
| Code | `[cite: doc_id, path=/svc/router.py, L120-L160]` | `doc_id`, `extras.file_path`, `extras.line_range` |

Every chunk your chunker emits should be citable in one of these forms. If a chunk shape is not citable, either you are missing a field or the shape does not belong in the index.

## Round-trip test: "can I get back to the source?"

Two integration tests worth writing for every hard-content path:

1. **Table cell round-trip.** Given a random `(doc_id, table_id, row, col)` triple, look it up in the chunk store and assert the returned cell text matches the original PDF (`.extract()` from PyMuPDF as the ground truth).
2. **Figure round-trip.** Given a random `figure_id`, look up its `image_uri`, fetch the image, and assert it matches the image extracted directly from the source PDF page.

These tests catch the entire class of "we lost the pointer somewhere" bugs that hide until a compliance auditor asks "show me the exact cell you got this number from."

## What you should be able to do after this chapter

- Explain why hard-content units (tables, formulas, figures) need a *dual* chunk shape (content + summary) rather than a single embedded chunk.
- Preserve table structure through ingest using Docling, unstructured, Textract, Doc Intelligence, or PyMuPDF, and chunk small / medium / large tables correctly (never split mid-row, always repeat the header).
- Route STEM PDFs through Nougat (or an equivalent) so equations arrive as LaTeX, and chunk them so a formula is never split from its introductory sentence.
- Pair every figure with its caption and store the image separately with `extras.image_uri`, and know when a multimodal-embedding pipeline is justified.
- Design a citation contract that maps every chunk shape to a specific `[cite: ...]` format, and enforce the metadata each format requires at ingest.
- Write round-trip tests that verify a citation can be traced back to the exact source cell / equation / figure.

That closes out the ingestion + chunking side of the pipeline. You now have `Chunk`s worth embedding. Move to mod-103, where the *choice* of embedding model becomes the next decision — one you will defend with a domain-fit benchmark, not with an MTEB leaderboard row.

## Further reading

- LlamaIndex, [*Recursive Retriever + Node References*](https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/) — the canonical table-index-per-summary walk-through.
- LlamaIndex, [*Multi-Modal RAG*](https://docs.llamaindex.ai/en/stable/module_guides/models/multi_modal/) — the multimodal-pipeline framing referenced above.
- Docling, [`DoclingDocument` docs](https://ds4sd.github.io/docling/concepts/docling_document/) — the schema that models tables, figures, and formulas.
- Nougat: Blecher, L., et al. (2023). [arXiv:2308.13418](https://arxiv.org/abs/2308.13418).
- MathPix, [*API documentation*](https://docs.mathpix.com/) — commercial equation OCR (evaluate before adopting; check licensing).
- Anthropic engineering, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — a related pattern where each chunk is enriched with an LLM-generated context prefix before embedding.
