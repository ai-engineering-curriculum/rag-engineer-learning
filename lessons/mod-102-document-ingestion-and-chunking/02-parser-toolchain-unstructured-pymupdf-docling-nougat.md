# Chapter 2 — The Native-Text Parser Toolchain: unstructured, PyMuPDF, Docling, Nougat

## Why this chapter exists

Ninety percent of what makes production ingestion work well is boring: pick the right parser for the format in front of you and let it emit its native structured output. Ninety percent of what makes ingestion fail silently is also boring: pick one parser (usually the first one whose Getting-Started snippet ran), send everything through it, and lose tables / headings / reading order on the formats it was not designed for.

This chapter walks the four Python libraries you will reach for on the native-text branch of chapter 1's decision tree: `unstructured`, `PyMuPDF`, `Docling`, and `Nougat`. For each, the shape of the output, what it is genuinely good at, what it hides from you, and the concrete `import ... ; partition(...)`-shaped call. OCR fallbacks are chapter 3.

## The comparison the rest of the chapter unpacks

| Parser | Best at | Output shape | Structural fidelity | Speed on native text | License / origin |
|---|---|---|---|---|---|
| **unstructured** (`unstructured-io/unstructured`) | Broad format coverage (PDF, HTML, DOCX, PPTX, XLSX, EML, ...); one API for all | List of typed `Element` objects with `metadata` | Good headings, lists, tables (`hi_res` strategy). Reading order can wobble on multi-column PDF. | Fast strategy: ~10s of ms/page. `hi_res` uses a layout model: seconds/page. | Apache 2.0 |
| **PyMuPDF** (`fitz`) | Fast, accurate text + layout extraction from PDFs; page images; bbox-per-word | Per-page dicts with blocks → lines → spans, or raw text | Full bbox / span control; you build the higher-level structure yourself | Fast: milliseconds per page even on hundreds of pages | AGPL-3.0 (commercial license available) |
| **Docling** (IBM `DS4SD/docling`) | PDF-to-structured with layout, tables, formulas; unified `DoclingDocument` schema exportable to Markdown / JSON / HTML | `DoclingDocument` with hierarchical items, table cells, refs | Strong on tables and headings; explicit document model | Moderate; layout model on by default | MIT |
| **Nougat** (Meta `facebookresearch/nougat`) | Scientific / STEM PDFs — inline math, equations, references — end-to-end transformer OCR to Markdown+LaTeX | Markdown with embedded LaTeX | Best-in-class for equations and academic layout; not designed for enterprise forms | Slow: end-to-end transformer inference per page (GPU strongly recommended) | MIT (model weights CC-BY-NC-4.0 — check before commercial use) |

Nothing here says "one wins." The decision is per format per corpus. Chapter 3 adds the OCR fallback column; mod-105's Lucene chapters will add analyzer-side decisions that pair with these.

## unstructured — the one-API-for-all-formats default

`unstructured` ([docs](https://docs.unstructured.io/open-source/introduction/overview)) exposes a `partition(...)` function per format that returns a list of `Element` objects: `Title`, `Header`, `NarrativeText`, `ListItem`, `Table`, `Image`, `CodeSnippet`, ... Each element carries `.metadata` with `filename`, `page_number`, `parent_id` (for hierarchy), `text_as_html` for tables, and more.

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="handbook.pdf",
    strategy="hi_res",                 # "fast" | "hi_res" | "ocr_only" | "auto"
    infer_table_structure=True,        # table cells as HTML in element.metadata.text_as_html
    languages=["eng"],                 # improves OCR routing when it kicks in
)

for el in elements[:8]:
    print(el.category, "|", repr(el.text[:80]))
    md = el.metadata.to_dict()
    print("  page:", md.get("page_number"), "parent:", md.get("parent_id"))
```

Strategy is the knob that matters. `fast` uses PDFMiner-style text extraction and is millisecond-per-page but has no notion of layout (columns, tables, reading order). `hi_res` uses a layout-detection model (YOLOX-based) to segment pages into regions, then dispatches per region — much better for real reports, at seconds-per-page cost. `ocr_only` forces the OCR branch even when a text layer exists (useful when the text layer is corrupted). `auto` decides per document.

The one gotcha to know before you ship: `unstructured` does *not* stably return elements in visual reading order for complex multi-column PDFs. Sort by `metadata.page_number` first, then by the coordinate metadata (`metadata.coordinates.points`) top-to-bottom, left-to-right, before you serialize elements to a "reconstructed document." The [`chunk_by_title` helper](https://docs.unstructured.io/open-source/core-functionality/chunking) is often the shape you actually want to use for the chunker step (chapter 4), and it handles the ordering internally.

**Same-shape calls for other formats.** `partition_html`, `partition_docx`, `partition_pptx`, `partition_xlsx`, `partition_md`, `partition_email`, `partition_epub`. You will almost never learn a second per-format library because `unstructured` covers the common ones. The exceptions:

- **HTML from real-world web pages** — pre-process with `trafilatura` (main-content extraction) *first*, then hand the cleaned HTML to `partition_html`. Otherwise you index every navigation menu on the internet.
- **`.xlsx` with real spreadsheets, not tables** — a workbook of formulas and pivot tables is not "text with structure." Treat cells as data, not passages. `openpyxl` directly.
- **Code files** — use LlamaIndex `CodeSplitter` (tree-sitter) or LangChain `RecursiveCharacterTextSplitter.from_language`. `unstructured` treats code as plain text.

## PyMuPDF — the fast, low-level PDF layer

PyMuPDF ([docs](https://pymupdf.readthedocs.io/)), imported as `fitz`, is the fastest way in the Python ecosystem to get text + bbox out of a PDF that has a real text layer. It is also the library `unstructured`'s "fast" strategy and Docling's PDF backend both use under the hood in various paths, so knowing the primitives pays off.

```python
import fitz  # PyMuPDF

doc = fitz.open("handbook.pdf")
for page_index, page in enumerate(doc, start=1):
    # "blocks" | "dict" | "html" | "xhtml" | "xml" | "text"
    blocks = page.get_text("blocks")   # list[(x0, y0, x1, y1, text, block_no, block_type)]
    for x0, y0, x1, y1, text, *_ in blocks:
        yield Passage(page=page_index, bbox=(x0, y0, x1, y1), text=text, ...)
doc.close()
```

Two facts that decide when to pick it:

- **AGPL-3.0.** Commercial products that cannot ship AGPL code need a commercial license from Artifex, or must switch to an alternative like [`pdfplumber`](https://github.com/jsvine/pdfplumber) (MIT) or [`pypdfium2`](https://github.com/pypdfium2-team/pypdfium2) (Apache-2.0 / BSD-3-Clause dual). Check with your legal team *before* the corpus hits production. `pdfplumber` and `pypdfium2` are noticeably slower and expose fewer primitives, but the license is compatible with closed-source distribution.
- **You are on the hook for higher-level structure.** PyMuPDF hands you spans and blocks. Grouping them into headings / paragraphs / list items / tables is *your* code. That is fine when you want the control and speed; it is a footgun when you thought you were getting a full document model. If you want the model handed to you, reach for Docling or `unstructured`.

## Docling — one unified schema across formats

Docling ([repo](https://github.com/DS4SD/docling), IBM Research) is a newer entrant with the explicit goal of "one document model" (`DoclingDocument`) that you can round-trip to Markdown, HTML, or JSON without losing structure. It handles PDF, DOCX, PPTX, XLSX, HTML, and images, and includes a layout model and a table-structure model out of the box.

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("handbook.pdf")

doc = result.document
# Structural iteration
for item, level in doc.iterate_items():
    print("  " * level, item.label, "|", getattr(item, "text", "")[:80])

# Deterministic serialization
markdown = doc.export_to_markdown()
json_dict = doc.export_to_dict()
```

Why reach for it:

- **The `DoclingDocument` schema is your friend.** Tables land as first-class objects with cell coordinates and row/column indices, not as HTML strings. Headings carry hierarchy. Figures carry captions. This makes chapters 7 (metadata) and 8 (tables / figures) much less painful.
- **MIT-licensed.** No AGPL concerns.
- **First-class Markdown export.** Feeds cleanly into structural chunkers that were built for Markdown (LlamaIndex `MarkdownNodeParser`, LangChain `MarkdownHeaderTextSplitter`).

The trade-off: it does more work than PyMuPDF, so it is slower on plain-text-heavy documents. On a "just get me the text" corpus, PyMuPDF wins throughput; on a "keep the tables intact and let me cite the cell" corpus, Docling wins clarity.

## Nougat — the STEM / academic PDF specialist

Nougat ([Meta AI, 2023](https://arxiv.org/abs/2308.13418), [repo](https://github.com/facebookresearch/nougat)) is a transformer that OCRs a scientific PDF end-to-end into Markdown with embedded LaTeX. It exists because inline math (`$\alpha \sim \beta$`), display equations, cross-references (`[15]`), and multi-column academic layouts break every other PDF parser in specific, painful ways.

```bash
pip install nougat-ocr
nougat path/to/paper.pdf -o out/       # emits `paper.mmd` — Markdown + LaTeX
```

When to reach for it:

- Your corpus is scientific papers, textbooks with equations, or technical reports with heavy formula content, *and* you need those equations preserved as LaTeX (embeddable, renderable, citable) rather than as garbled OCR text.
- You have a GPU. Nougat's transformer inference is slow on CPU — expect single-digit seconds per page on a modern GPU, minutes per page on CPU.

When *not* to:

- Enterprise handbooks, HR forms, contracts, meeting notes — Docling or `unstructured` are cheaper and equally accurate on layouts without heavy math.
- Any product that ships closed-source with the model weights bundled: the pre-trained weights are **CC-BY-NC-4.0**, non-commercial. The code is MIT. Consult legal.

## The decision matrix, one line per shape

| You have… | Reach for | Why |
|---|---|---|
| A mixed corpus, one call to rule them all | `unstructured` with `strategy="hi_res"` (fall back to `"ocr_only"` per page — chapter 3) | Broad format coverage in one API. Ships with layout inference. |
| Native PDF, need speed and control over layout | `PyMuPDF` (AGPL-permitting) or `pdfplumber` | Fastest text + bbox extraction; you glue higher-level structure. |
| PDFs where tables and figures matter for citation | `Docling` | First-class table / figure objects with cell coordinates, MIT-licensed. |
| Scientific PDF with equations | `Nougat` on GPU | Only tool that preserves inline math and LaTeX faithfully. |
| Real-world HTML | `trafilatura` main-content extract → `unstructured.partition_html` | Kills the chrome; keeps the structure. |
| Markdown from a docs site | LlamaIndex `MarkdownNodeParser` or LangChain `MarkdownHeaderTextSplitter` | Respects heading hierarchy, which becomes the parent-child scaffold in chapter 6. |
| Office documents | `unstructured.partition_docx / _pptx / _xlsx` | One API; falls back to `python-docx` / `python-pptx` / `openpyxl` under the hood. |
| Source code | LlamaIndex `CodeSplitter` (tree-sitter) or LangChain `from_language` | Splits on function / class boundaries, not on characters. |

## Multi-parser pipelines are normal

Real corpora justify a **router**: probe each incoming file, dispatch to the right parser, normalize each parser's native output into the `Passage` shape from chapter 1. A minimal shape:

```python
def ingest(path: Path) -> list[Passage]:
    mime = detect_mime(path)  # e.g. python-magic
    if mime == "application/pdf":
        if is_scanned(path):        # heuristic from chapter 1; refined in chapter 3
            return with_ocr_fallback(path)
        if is_stem(path):           # e.g. arXiv / journal DOI in metadata
            return via_nougat(path)
        return via_docling(path)    # or via_pymupdf / via_unstructured per corpus policy
    if mime == "text/html":
        return via_unstructured_html(trafilatura_clean(path.read_text()))
    if mime == "text/markdown":
        return via_markdown(path)
    if mime in OFFICE_MIMES:
        return via_unstructured_office(path)
    if mime.startswith("text/x-"):
        return via_code(path)
    raise UnsupportedFormatError(mime)
```

The router is where the module's exercises will spend most of their code. Getting each per-format branch to emit *the same* `Passage` shape is the payoff.

## What you should be able to do after this chapter

- Pick a primary parser per format from the matrix above and explain why.
- Call `unstructured` `partition_pdf` with the right `strategy` for a given input.
- Extract text + bbox from a native-text PDF with PyMuPDF in ten lines.
- Export a `DoclingDocument` to Markdown and iterate its structural items.
- Identify when a PDF actually needs Nougat and when Docling / unstructured is sufficient.
- Name the AGPL-3.0 licensing consideration around PyMuPDF and the CC-BY-NC-4.0 consideration around Nougat's weights, and the closed-source-friendly alternatives to each.

Chapter 3 picks up the other branch of the decision tree — what happens when there is no honest text layer and OCR takes over.

## Further reading

- Unstructured, [*Overview*](https://docs.unstructured.io/open-source/introduction/overview), [*Chunking strategies*](https://docs.unstructured.io/open-source/core-functionality/chunking), [*Element metadata*](https://docs.unstructured.io/open-source/concepts/document-elements).
- PyMuPDF, [*Text extraction*](https://pymupdf.readthedocs.io/en/latest/textpage.html), [*Layout, blocks, and words*](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).
- Docling, [`DS4SD/docling` README](https://github.com/DS4SD/docling), [`DoclingDocument` spec](https://ds4sd.github.io/docling/concepts/docling_document/).
- Nougat: Blecher, L., et al. (2023). *Nougat: Neural Optical Understanding for Academic Documents.* [arXiv:2308.13418](https://arxiv.org/abs/2308.13418). [Repo](https://github.com/facebookresearch/nougat).
- Trafilatura, [*Overview*](https://trafilatura.readthedocs.io/en/latest/) — HTML main-content extraction.
- pdfplumber, [`jsvine/pdfplumber` README](https://github.com/jsvine/pdfplumber) — MIT-licensed alternative when AGPL is unacceptable.
