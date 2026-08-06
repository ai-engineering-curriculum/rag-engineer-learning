# Chapter 1 — Ingestion Overview: What This Stage Owns and What Comes Out of It

## Why this chapter exists

Ingestion is the first stage of the seven-stage pipeline you mapped in mod-101, and it is the stage `llm-application-developer` (level 20) is *allowed* to gloss over. Level-20 shape: `open(path).read()`, throw the string at a character-window splitter, embed. That shape ships a demo. It also silently caps every downstream metric — recall, faithfulness, citation quality — because the string it lands with has already lost the layout, the tables, the page numbers, and the permission scope the retriever needs to do its job.

This module builds the production-shaped ingestion layer. Chapter 1 sets the scope: what the stage owns, what the *output* of the stage actually looks like (spoiler: not a string), the format matrix you will handle, and the two-branch decision (native-text vs. OCR fallback) every parser call implicitly makes. Chapters 2 and 3 go deep on the parser and OCR toolchains; chapters 4–8 go deep on chunking.

## The output contract: passages, not text

The single most important shift from the level-20 shape is what leaves the stage. Level-20 ingestion returns a `str`. Level-30 ingestion returns a list of **structured passages**:

```python
@dataclass
class Passage:
    doc_id: str                 # stable identifier for the source document
    source_uri: str             # URL, S3 key, git blob SHA — the thing you can re-fetch
    text: str                   # normalized, encoding-fixed text of this passage
    element_type: str           # "title" | "heading" | "paragraph" | "list_item"
                                # | "table" | "table_summary" | "code" | "figure_caption" | ...
    section_path: list[str]     # ["Compensation Framework", "Salary Bands", "Level 30"]
    page: int | None            # 1-indexed page in the source, when the format has pages
    bbox: tuple[float, ...] | None  # (x0, y0, x1, y1) in the page coordinate system
    order: int                  # monotonic index in reading order within the doc
    updated_at: datetime | None # last-modified timestamp of the source
    author: str | None
    permission_scope: str | None    # tenant/group ACL tag; see mod-112
    parser: str                 # "unstructured@0.15.13", "pymupdf@1.24.10", "docling@2.4.0"
    ocr_used: bool              # True if any page fed through the OCR fallback
    extras: dict                # parser-specific structured metadata (table cells, LaTeX, ...)
```

Everything downstream needs at least one of those fields. The chunker (chapters 4–6) needs `element_type`, `section_path`, and `order` to keep boundaries meaningful. The retriever (mod-104, mod-106) needs `permission_scope`, `updated_at`, and `page` to filter and to feed a citation. The reranker (mod-107) can use `element_type` to down-weight boilerplate. The generator (mod-108) needs `doc_id` + `page` + `section_path` to emit a citation the user can click through.

The shape of `Passage` is the contract this module owns. Do not treat it as a schema you will get right the first time — you will grow it over the module — but do treat *every field it needs to carry* as a requirement the ingestion code owes the rest of the pipeline.

## The seven format families

Real corpora arrive as a mix. Pick the parser per format, not per corpus. This module's exercises will have you handle:

| Family | Concrete formats | Parser you will reach for first | Fallback |
|---|---|---|---|
| Native-text PDF | `application/pdf` with an embedded text layer | PyMuPDF (`fitz`) or Docling | unstructured "hi_res" |
| Scanned PDF | `application/pdf` with page images only | Docling + OCR / unstructured "ocr_only" | Tesseract, Textract, Document Intelligence, Document AI |
| Academic / STEM PDF | Papers with equations, references, figures | Nougat (Meta) or Docling | PyMuPDF for the plain text spine |
| HTML | `text/html`, cleaned or raw web | unstructured `partition_html`, `readability-lxml`, `trafilatura` | BeautifulSoup as a last resort |
| Markdown | `text/markdown` (Confluence exports, docs sites) | LlamaIndex `MarkdownNodeParser`, `mistune`, `markdown-it-py` | plain-text splitter, respecting `#` boundaries |
| Office | `.docx`, `.pptx`, `.xlsx` | unstructured `partition_docx` / `_pptx` / `_xlsx`; `python-docx`, `python-pptx`, `openpyxl` under the hood | Convert to PDF and re-parse (LibreOffice headless) |
| Code | source files (`.py`, `.ts`, `.go`, ...) | LlamaIndex `CodeSplitter` (tree-sitter) or LangChain `RecursiveCharacterTextSplitter.from_language` | tree-sitter directly |

Two calls this table hides:

- **`application/pdf` is not a format**, it is a container. The same MIME type covers a natively-authored LaTeX export (perfect text layer), a screenshot-of-a-fax "PDF" that is really a wrapper around JPEGs, and everything in between. Chapter 3 will show you how to detect which shape you have and route accordingly, because a parser that assumes "text layer present" will emit `""` for every page of a scanned brochure without raising.
- **HTML is adversarial by default.** A raw web page is 90% chrome, ads, and cookie banners around 10% of the content you actually want. Feed the whole DOM to your chunker and you will index navigation menus. Use `trafilatura` ([docs](https://trafilatura.readthedocs.io/)) or `readability-lxml` to extract the main content first, then hand *that* to a structural parser.

## The two branches every parser call implicitly picks

Every real ingestion pipeline is a decision tree with two branches at the top:

```
                    ┌── has-native-text?
                    │        │
                    │        ├── YES ──▶ native-text parser
                    │        │           (PyMuPDF, Docling native, unstructured "fast")
   incoming file ──▶│        │
                    │        └── NO  ──▶ OCR fallback
                    │                    (Tesseract local; Textract / Doc Intelligence / Doc AI hosted)
                    │
                    └── (fan-out by format for HTML, MD, Office, code)
```

- **Native-text branch.** Cheap (milliseconds per page, no per-page API cost), high fidelity when the text layer is honest. Fails silently on scanned-image PDFs: you get empty strings or line-noise ligatures that pass a truthy check but embed to gibberish. Chapter 2 walks the primary parsers.
- **OCR fallback branch.** Expensive (seconds per page, per-page API cost for hosted services), varying accuracy by service and by document layout. Chapter 3 walks the fallback options and the cost / latency / accuracy trade-off. The exercise you will do (`exercise-02`) forces you to measure that trade-off on your own sample rather than trust a vendor benchmark.

The "which branch?" decision is not the *file extension* — a scanned PDF has extension `.pdf` — it is a probe of the text layer. A common shape:

```python
def route(path: Path) -> Literal["native", "ocr"]:
    doc = fitz.open(path)
    total_chars = sum(len(page.get_text("text")) for page in doc)
    total_pages = doc.page_count
    doc.close()
    # Empirically: honest text layers hit hundreds of chars per page.
    # Below ~50 chars/page across the whole doc, treat as image-only.
    return "native" if (total_pages and total_chars / total_pages > 50) else "ocr"
```

That heuristic is fragile — a mixed doc (born-digital cover page + scanned body) needs *per-page* routing. Chapter 3 refines it.

## What is NOT in this module

To keep the boundaries clean:

- **Web crawling / connector authentication.** Getting bytes from Confluence / SharePoint / Google Drive / Salesforce is an integration problem the paired `applied-ai-engineer` track covers at the product surface. This module assumes bytes-on-disk (or bytes-in-memory) as the input.
- **The embedding call itself.** You will normalize `text` into `Passage.text` and stop. Choosing an embedding model and calling it is mod-103's job.
- **The chunk-*writer* to the vector DB.** You will produce chunk objects and stop. Landing them in pgvector / Elasticsearch is mod-104 and mod-105.
- **The scheduled ingestion DAG (Airflow / Dagster / Prefect).** You will write the pure functions; the scheduling and lag SLOs are mod-111.

Everything above the dotted lines is scope for this module. Everything below is scope for the module that owns it — resist the urge to reach across.

## The stage's success criterion

By the end of the module, you can point a `run_ingestion(corpus_dir) -> list[Chunk]` at a directory of mixed formats and get back chunks where:

1. **No content is silently lost.** Scanned PDFs went through OCR. Tables were preserved as tables. Code kept its indentation. Confluence-exported HTML lost the sidebar but kept the body.
2. **Every chunk carries the metadata retrieval needs.** `doc_id`, `section_path`, `page`, `updated_at`, `permission_scope`, and enough of `extras` to reconstruct a citation.
3. **The chunk boundary respects the document's structure.** Sections do not split mid-sentence; tables are not sliced across rows; code chunks do not cut a function in half.
4. **The output is reproducible.** Run twice, get identical output — because the parser version, the chunker version, and the corpus snapshot are pinned. This is the shape mod-111 needs to version indexes.

Chapter 2 starts on point 1 for the PDF case: the parser landscape and how to pick.

## What you should be able to do after this chapter

- Explain why the ingestion stage returns *passages with metadata*, not a `str`.
- Name the seven format families this module covers and pick a first-choice parser per family.
- State the two-branch decision (native-text vs. OCR fallback) that every PDF call implicitly makes and why file extension is not a reliable trigger.
- Draw the boundary between mod-102 and the modules on either side (level-20 minimal shape below; mod-103 embed / mod-104 index above; mod-111 orchestration around).

## Further reading

- Unstructured docs, [*Concepts: partitioning*](https://docs.unstructured.io/open-source/concepts/partitioning) — the reference framing for the "elements with metadata" output shape this chapter borrows.
- Docling repo, [`docling` GitHub](https://github.com/DS4SD/docling) — IBM's document-parsing library with a similar "one output schema per document" model.
- LlamaIndex, [*Loading Data*](https://docs.llamaindex.ai/en/stable/module_guides/loading/) — the framework's overview of readers, node parsers, and metadata as first-class fields.
- Trafilatura, [*Overview*](https://trafilatura.readthedocs.io/en/latest/) — HTML main-content extraction, used any time your source is real-world web pages.
