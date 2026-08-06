# Resources — mod-102: Document Ingestion and Corpus Preparation

Primary sources for the chapters and exercises. Links go to official docs, the source repos, or the canonical paper — not to third-party summary blogs.

## Ingestion — parsers and toolchain

- Unstructured, [*Overview*](https://docs.unstructured.io/open-source/introduction/overview), [*Concepts: document elements*](https://docs.unstructured.io/open-source/concepts/document-elements), [*Chunking strategies*](https://docs.unstructured.io/open-source/core-functionality/chunking).
- PyMuPDF, [*Documentation home*](https://pymupdf.readthedocs.io/), [*Text extraction*](https://pymupdf.readthedocs.io/en/latest/textpage.html), [*Layout, blocks, and words*](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).
- Docling, [`DS4SD/docling` repo](https://github.com/DS4SD/docling), [`DoclingDocument` concept](https://ds4sd.github.io/docling/concepts/docling_document/).
- Nougat: Blecher, L., et al. (2023). *Nougat: Neural Optical Understanding for Academic Documents.* [arXiv:2308.13418](https://arxiv.org/abs/2308.13418). [Repo](https://github.com/facebookresearch/nougat). (Note: model weights are CC-BY-NC-4.0.)
- Trafilatura, [*Overview*](https://trafilatura.readthedocs.io/en/latest/) — HTML main-content extraction.
- pdfplumber, [`jsvine/pdfplumber` README](https://github.com/jsvine/pdfplumber) — MIT-licensed PDF alternative when PyMuPDF's AGPL is unacceptable.
- pypdfium2, [`pypdfium2-team/pypdfium2` README](https://github.com/pypdfium2-team/pypdfium2) — Apache-2.0 / BSD-3-Clause dual-licensed PDF alternative.

## OCR back-ends

- Tesseract, [user docs](https://tesseract-ocr.github.io/tessdoc/), [*Installation*](https://tesseract-ocr.github.io/tessdoc/Installation.html); Python: [`pytesseract`](https://github.com/madmaze/pytesseract). Reference on Page Segmentation Modes: [Tesseract manpage](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options).
- AWS Textract, [*Developer Guide*](https://docs.aws.amazon.com/textract/latest/dg/what-is.html), [*API reference*](https://docs.aws.amazon.com/textract/latest/dg/API_Operations.html), [pricing](https://aws.amazon.com/textract/pricing/).
- Azure AI Document Intelligence, [*Overview*](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview), [*Prebuilt models*](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/model-overview), [Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-documentintelligence-readme).
- GCP Document AI, [*Overview*](https://cloud.google.com/document-ai/docs/overview), [*Processor list*](https://cloud.google.com/document-ai/docs/processors-list), [Python client](https://cloud.google.com/python/docs/reference/documentai/latest).
- MathPix, [*API docs*](https://docs.mathpix.com/) — equation OCR (commercial; check licensing before adoption).

## Chunking strategies and orchestration frameworks

- LlamaIndex, [*Loading Data*](https://docs.llamaindex.ai/en/stable/module_guides/loading/), [*Node Parser Modules*](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/), [*Semantic Chunking*](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/), [*Chunk Size Optimization*](https://docs.llamaindex.ai/en/stable/examples/param_optimizer/param_optimizer/), [*Metadata Extraction*](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/).
- LlamaIndex, [*Auto Merging Retriever*](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/), [*Sentence Window Retrieval*](https://docs.llamaindex.ai/en/stable/examples/node_parsers/MetadataReplacementDemo/), [*Recursive Retriever for PDF tables*](https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/).
- LangChain, [*Text Splitters overview*](https://python.langchain.com/docs/how_to/#text-splitters), [*Split by tokens*](https://python.langchain.com/docs/how_to/split_by_token/), [*Semantic Chunker*](https://python.langchain.com/docs/how_to/semantic-chunker/), [*Parent Document Retriever*](https://python.langchain.com/docs/how_to/parent_document_retriever/), [*Multi-Vector Retriever*](https://python.langchain.com/docs/how_to/multi_vector/), [*Self-querying retriever*](https://python.langchain.com/docs/how_to/self_query/).
- Kamradt, G. [*5 Levels of Text Splitting*](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) — the practitioner walk-through of the strategy families, including the algorithm behind `SemanticChunker`.
- Anthropic engineering, [*Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — public engineering case study on chunk boundary + per-chunk context augmentation.

## Embedding-model context (for the size trade-off)

- OpenAI, [*Embeddings guide*](https://platform.openai.com/docs/guides/embeddings) — includes token limits for `text-embedding-3-*`.
- Cohere, [*Embed v3 docs*](https://docs.cohere.com/docs/embeddings) — token limits and length behavior.
- Voyage AI, [*Embeddings docs*](https://docs.voyageai.com/docs/embeddings) — token limits for `voyage-3` and long-context variants.
- Hugging Face model cards for `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `mixedbread-ai/mxbai-embed-large-v1` — check the current model card for the current token limit before designing chunk sizes around it.

## Metadata, filtering, and the pre-filter / post-filter trap

- pgvector, [`pgvector/pgvector` README](https://github.com/pgvector/pgvector), [*Filtering*](https://github.com/pgvector/pgvector#filtering).
- Elasticsearch, [*Bool query DSL*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html), [*kNN search*](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
- Weaviate, [*Filters*](https://weaviate.io/developers/weaviate/api/graphql/filters), [*Vector indexes: HNSW*](https://weaviate.io/developers/weaviate/concepts/vector-index).
- Qdrant, [*Filtering*](https://qdrant.tech/documentation/concepts/filtering/).

## Lost-in-the-middle (context-window motivation for small-chunk / small-to-big patterns)

- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).

## Evaluation utilities (used by exercise-03 and beyond)

- `ir_measures`, [docs](https://ir-measur.es/en/latest/), [*Getting started*](https://ir-measur.es/en/latest/getting-started.html) — installed with the mod-101 toolchain.
- `jiwer`, [`jitsi/jiwer` docs](https://github.com/jitsi/jiwer) — CER / WER for the OCR benchmark in exercise-02.
- TREC qrels format reference, [NIST TREC](https://trec.nist.gov/data.html).

## Track-internal cross-references

- [`../mod-101-retrieval-engineering-foundations/README.md`](../mod-101-retrieval-engineering-foundations/README.md) — the seven-stage pipeline this module drills into.
- [`../mod-103-embeddings-selection-and-benchmarking/README.md`](../mod-103-embeddings-selection-and-benchmarking/README.md) — where the chunks produced here get embedded.
- [`../mod-104-vector-databases-and-ann-tuning/README.md`](../mod-104-vector-databases-and-ann-tuning/README.md) — where the pre-filter / post-filter decision (chapter 7) gets fully worked out.
- [`../../CURRICULUM.md`](../../CURRICULUM.md) — track ownership rule and full module map.
- [`../../PREREQUISITES.md`](../../PREREQUISITES.md) — assumed entry skills.
