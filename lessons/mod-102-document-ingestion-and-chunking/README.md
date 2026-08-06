# mod-102 — Document Ingestion and Corpus Preparation

**Track:** RAG & Retrieval Engineer (role level 30, AI Engineering family)
**Estimated effort:** 14 hours
**Position in track:** second module. First stage-depth drill-down into the seven-stage pipeline mapped in mod-101.

## What this module does

This module builds the production-shaped ingestion and chunking layer that everything downstream in the track assumes. By the end you can (a) route a mixed-format corpus (native PDF, scanned PDF, HTML, Markdown, Office, code) through the right parser and OCR fallback so no content is silently lost, (b) design chunk boundaries that respect document structure — sections, tables, code, equations — rather than shredding them at arbitrary character offsets, (c) reason about the chunk-size / recall / precision trade-off *with numbers on your corpus*, (d) attach the metadata (`doc_id`, `section_path`, `page`, `updated_at`, `permission_scope`, ...) that the retriever will filter on and the generator will cite, and (e) handle the hard content types — tables, formulas, figures — without losing the source cell / equation / figure reference the citation contract requires.

## Boundary with adjacent modules

- **Below (level 20).** `llm-application-developer` mod-005 introduces text-only naïve character chunking as the day-one shape. This module replaces it — same *place in the pipeline*, production-shaped implementation.
- **Above (this track).** mod-103 picks up with the *choice* of embedding model for the chunks this module produces; mod-104 lands them in a vector index. Neither is in scope here.
- **Around.** mod-111 owns the scheduled ingestion DAG, incremental vs. full rebuilds, and the index-version key that includes `chunker_version` and `parser_version` — the fields chapter 7 introduces here.

## Learning objectives

- Author a document-ingestion pipeline that handles PDF (native + scanned), HTML, Markdown, Office documents, and code — using unstructured.io, PyMuPDF, Docling, Nougat, and OCR fallbacks (Tesseract, AWS Textract, Azure AI Document Intelligence, GCP Document AI).
- Design semantic, structural (section-aware, table-aware, code-aware), and metadata-enriched chunking strategies — moving beyond naïve character/token windows to embedding-similarity segmenters, hierarchical / parent-child chunks, and small-to-big retrieval.
- Reason about the chunk-size / recall / precision trade-off and its interaction with embedding-model context length, reranker cost, and generator context budget — drawing on the LlamaIndex and LangChain chunking guides.
- Extract and attach retrieval-critical metadata (document ID, section path, page number, publish date, author, permissions) as first-class fields the retriever can filter on.
- Handle table, formula, and figure content — LaTeX-aware chunking, table-summary augmentation, figure-caption pairing — without losing the source cell / equation / image reference needed for citation.

## Prerequisites

- The mod-101 toolchain exercise (`exercise-05-local-toolchain-bootstrap-and-smoke-test`) is green: pgvector reachable, Elasticsearch/OpenSearch reachable, `sentence-transformers` and `ir_measures` importable.
- Comfortable Python (dataclasses, typing, packaging) and Docker.
- Familiarity with the seven-stage RAG pipeline diagram from mod-101 chapter 1.

## Chapters

Read the chapters in order. Chapters 1–3 cover ingest (parser and OCR toolchain); chapters 4–8 cover chunking (strategies, size trade-off, hierarchical patterns, metadata, hard content types). Run the paired exercise before moving to the next block.

| # | Chapter | Focus | Paired exercise |
|---|---|---|---|
| 01 | [Ingestion Overview and Format Matrix](01-ingestion-overview-and-format-matrix.md) | The `Passage` output contract; seven format families; the native-text vs. OCR fork | — |
| 02 | [Native-Text Parser Toolchain: unstructured, PyMuPDF, Docling, Nougat](02-parser-toolchain-unstructured-pymupdf-docling-nougat.md) | Per-parser strengths, output shapes, and licensing | [exercise-01 — unstructured and Docling PDF ingestion drill](exercises/exercise-01-unstructured-and-docling-pdf-ingestion-drill.md) |
| 03 | [OCR Fallbacks: Tesseract, Textract, Document Intelligence, Document AI](03-ocr-fallbacks-tesseract-textract-doc-intelligence-doc-ai.md) | Per-page routing, hosted vs. local, escalation | [exercise-02 — OCR fallback: Tesseract vs. Textract cost / latency compare](exercises/exercise-02-ocr-fallback-with-textract-and-tesseract-cost-latency-compare.md) |
| 04 | [Chunking Strategies: Fixed, Structural, Semantic](04-chunking-strategies-fixed-structural-semantic.md) | The strategy families, and the LlamaIndex / LangChain map | (paired with chapter 5's exercise) |
| 05 | [Chunk-Size / Recall / Precision Trade-Off](05-chunk-size-recall-precision-tradeoff.md) | The four interacting constraints; the ablation harness | [exercise-03 — semantic vs. structural vs. fixed-window chunking ablation](exercises/exercise-03-semantic-vs-structural-vs-fixed-window-chunking-ablation.md) |
| 06 | [Hierarchical, Parent-Child, and Small-to-Big](06-hierarchical-parent-child-and-small-to-big.md) | Index-small-return-big patterns and their trade-offs | [exercise-04 — hierarchical parent-child and small-to-big chunk design](exercises/exercise-04-hierarchical-parent-child-and-small-to-big-chunk-design.md) |
| 07 | [Metadata as First-Class Retrieval Fields](07-metadata-as-first-class-retrieval-fields.md) | The `Chunk` schema and pre-filter vs. post-filter | (metadata discipline runs through exercises 01–05) |
| 08 | [Tables, Formulas, and Figures Without Losing the Citation](08-tables-formulas-figures-and-citation.md) | Dual chunk shape; citation contract | [exercise-05 — table- and figure-aware chunking with citation preservation](exercises/exercise-05-table-and-figure-aware-chunking-and-citation-preservation.md) |

## How to work through it

1. Read chapters 1–3 as one arc (the ingest side). Do exercise-01 after chapter 2, exercise-02 after chapter 3.
2. Read chapters 4–5 together (chunking strategies + size trade-off). Do exercise-03 after chapter 5.
3. Read chapter 6 and do exercise-04.
4. Read chapter 7 alongside exercise-04 — the metadata schema is what makes exercise-04's parent-child linkage sane.
5. Read chapter 8 and do exercise-05. This closes the module.

## Structure

- `01-…md` … `08-…md`: lecture chapters.
- `exercises/`: five hands-on prompts, matched to the chapters above.
- `labs/`: long-form hands-on labs (see `labs/README.md`).
- `quizzes/`: knowledge checks.
- `resources.md`: primary references and official documentation.

## Where to go next

Once exercise-05's round-trip tests pass, move to [`mod-103-embeddings-selection-and-benchmarking`](../mod-103-embeddings-selection-and-benchmarking/README.md). The chunks you produced here become the corpus mod-103's domain-fit benchmark evaluates embedding models against.
