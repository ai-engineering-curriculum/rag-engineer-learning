# Chapter 1 — The End-to-End RAG Pipeline and the Decision at Each Stage

## Why this chapter exists

Before you can *engineer* retrieval you have to *see* the whole pipeline as a single system, name each stage, and know the engineering decision that stage forces you to make. Almost every module in this track will drill into one of those stages, so the picture in your head from this chapter is the map you will annotate for the rest of the curriculum.

The `llm-application-developer` track (level 20) introduces a five-stage minimal shape — **ingest → chunk → embed → retrieve → generate** — that is enough to ship a demo. This track works with a seven-stage production shape that surfaces two decisions the shorter shape hides: **rerank** (a second-pass scorer over the cheap first-pass candidates) and **assemble** (deduplication, ordering, and citation anchoring before the prompt is built).

## The seven stages

```
    ┌──────────┐   ┌────────┐   ┌───────┐   ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐
    │  Ingest  │──▶│ Chunk  │──▶│ Embed │──▶│  Retrieve  │──▶│ Rerank  │──▶│ Assemble │──▶│ Generate  │
    └──────────┘   └────────┘   └───────┘   └────────────┘   └─────────┘   └──────────┘   └───────────┘
     (offline)     (offline)    (offline)     (online)        (online)      (online)       (online)
```

Ingest, chunk, and embed run *offline* against the corpus and land bytes in an index. Retrieve, rerank, assemble, and generate run *online* on the hot path of each user query. Every one of them is a place a retrieval engineer will one day get paged.

## The decision at each stage

Each stage's engineering decision is a set of trade-offs you own. What follows is a one-paragraph orientation per stage; the depth lives in the module noted.

### 1. Ingest — how do bytes become documents?

You choose (a) what to parse (native text, HTML, PDF, Office, Markdown, code), (b) which parser to use per format (unstructured.io, PyMuPDF, Docling, Nougat, OCR fallbacks such as Tesseract / AWS Textract / Azure Document Intelligence / GCP Document AI), (c) what structural metadata to preserve (section path, page number, table cell coordinates), and (d) what governance metadata to attach (source URL, updated-at, author, permission scope). The output is not "text" — it is *structured passages with metadata*, because everything downstream will need that metadata for filtering, ordering, and citation. Depth: **mod-102**.

### 2. Chunk — how do documents become retrievable passages?

You choose the boundary policy (fixed character/token window with overlap, sentence-aware, section-aware, table-aware, code-aware, embedding-similarity semantic segmenter, hierarchical parent-child, small-to-big). Chunk size interacts with the embedding model's usable context, with the reranker's cost, and with what fraction of the generator's context window one passage will consume. Bad boundaries silently cap the ceiling of every downstream metric. Depth: **mod-102**.

### 3. Embed — how do passages become vectors?

You choose an embedding family (bi-encoder, cross-encoder, ColBERT-style late-interaction, SPLADE sparse-neural), a specific model (hosted APIs like OpenAI `text-embedding-3-*`, Cohere Embed v3, Voyage `voyage-3`, Google `text-embedding-005`; or self-hosted such as BGE, E5, Nomic, GTE, Snowflake Arctic Embed, Alibaba GTE-Qwen, Jina), and a dimensionality / quantization regime (int8, binary, Matryoshka MRL). The decision is defended with a **domain-fit benchmark on your corpus**, not with an MTEB leaderboard row. Depth: **mod-103**.

### 4. Retrieve — how do queries become candidate passages?

You choose the index (pgvector HNSW/IVFFlat, Pinecone, Weaviate, Qdrant, Milvus, FAISS as a library, Chroma; and/or a Lucene-family engine — Elasticsearch, OpenSearch, Solr, Vespa — for BM25 and dense fields), the ANN parameters (HNSW `M`, `efConstruction`, `efSearch`; IVF-PQ `nlist`, `nprobe`, `pq_m`, `pq_nbits`) tuned to a recall@k and p95 latency target, whether the retriever is dense-only, lexical-only (BM25), or hybrid (BM25 + dense, sometimes plus SPLADE), and how metadata filters compose with the ANN scan (pre-filter vs. post-filter — a well-known correctness trap). You also choose whether to rewrite the query (HyDE, decomposition, step-back, multi-query fanout) before it hits the index. Depth: **mod-104, mod-105, mod-106**.

### 5. Rerank — how do candidates become an ordered short-list?

Given the top *N* candidates, you rescore to a top *K* (typically N = 50–200, K = 5–20) with a slower, more accurate scorer: a cross-encoder like `bge-reranker` / `ms-marco-MiniLM-*`, a hosted reranker (Cohere Rerank v3, Voyage `rerank-2`, Jina Reranker), a classical LTR reranker (LightGBM / XGBoost pairwise trained on your judgments), or an LLM-as-reranker. You defend the funnel shape (why *N*, why *K*) against a latency and cost budget. Depth: **mod-107**.

### 6. Assemble — how does the short-list become a prompt?

Deduplicate by source and by near-identical content, order by score or by document structure, attach per-chunk citation anchors the generator can name (chunk ID, source URL + section, page number), reserve token budget for the answer, and — when the window is tight — apply contextual compression (LlamaIndex `ContextualCompressor`, LangChain `LLMChainExtractor`) that shrinks each chunk while keeping the anchor intact. "Just concatenate the top-K" is the shape that fails; the failure mode is the **lost-in-the-middle** effect documented by Liu et al. 2024 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)). Depth: **mod-107**.

### 7. Generate — how does the assembled context become a grounded answer?

You choose the prompt shape that forces citation (inline bracketed citations, structured JSON citations, answer-with-quote), the refusal shape when context is insufficient, the contradiction-handling policy when retrieved chunks disagree, and the post-generation citation-verification check that catches fabricated source IDs. Baseline prompt engineering is owned by `llm-application-developer` (level 20); this track adds only the *retrieval-specific* prompt slice. Depth: **mod-108**.

## The two cross-cutting decisions the diagram hides

The seven-stage picture is a snapshot of a query flowing through the system. Two more decisions run *around* the pipeline and belong on your mental map from day one:

- **Evaluation.** How do you know a change to any stage helped? You need a judgment set (query → relevant-doc-IDs), retrieval-only metrics (recall@k, precision@k, MRR, nDCG@k), and paired significance testing so a fluke does not become a launch. Depth: **mod-109**.
- **MLOps and orchestration.** Indexes are versioned artifacts keyed on `(embedding-model, chunker-version, schema-version, corpus-snapshot)`. Retriever CI runs your eval on every PR touching chunker / embedding / retriever / reranker code. Ingestion runs on a schedule (Airflow / Dagster / Prefect) with backfill and lag SLOs. Corpus / query / quality drift is monitored. Above all this, an orchestration layer (LlamaIndex, LangChain, LangGraph, Semantic Kernel) — or none — composes the pipeline, and an MCP server exposes the tuned retriever as a tool downstream agents can call. Depth: **mod-110, mod-111**.

## A concrete example, end-to-end

To make the shape concrete, walk through one query end-to-end against a mixed HR + engineering wiki corpus:

1. **Ingest.** A nightly job reads Confluence exports, sends PDF attachments through Docling with a Tesseract fallback for scanned pages, and lands one JSON row per section with `{doc_id, section_path, page, updated_at, permission_scope, text}`.
2. **Chunk.** Each row is split by heading; sections longer than 512 tokens are further split at paragraph boundaries with 64-token overlap. Tables are chunked whole with a one-sentence LLM-generated summary attached to preserve citation of the source cell.
3. **Embed.** Each chunk is embedded with a candidate embedding model chosen by the mod-103 domain-fit benchmark and stored in a Postgres table with a `pgvector` HNSW index.
4. **Retrieve.** The user asks *"how do I request parental leave?"* The query is rewritten by HyDE, run against both a BM25 index in Elasticsearch and the pgvector HNSW index, and the two candidate lists are fused with reciprocal-rank fusion (Cormack et al. 2009) to a top-100.
5. **Rerank.** A `bge-reranker-large` cross-encoder rescores the top-100 to a top-8.
6. **Assemble.** The top-8 is deduplicated by `doc_id`, ordered by score, each chunk carries `[cite: doc_id#section]` anchors, and a `LLMChainExtractor` compresses each chunk to at most 400 tokens while keeping the anchor.
7. **Generate.** A prompt instructs the model to answer only from the provided context, to cite every claim with `[cite: ...]`, and to refuse if the context does not contain the answer. A post-generation check verifies every emitted `[cite: ...]` matches a chunk that was actually in the assembled context.

Every arrow in that walk-through corresponds to a knob you now own.

## What you should be able to do after this chapter

- Draw the seven-stage pipeline from memory and label each stage as offline or online.
- Name at least one engineering decision made at each stage.
- Explain the difference between the level-20 "five-stage minimal shape" and the level-30 "seven-stage production shape."
- Point to the module in this track that owns the depth for each stage.

The next chapter refreshes the classical-IR mental model — the vocabulary of inverted indexes, tf-idf, BM25, and precision / recall / F1 — every one of these stage-level decisions ultimately hangs off.

## Further reading

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
