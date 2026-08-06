# Chapter 3 — Retrieval as a Feature vs. Retrieval as an Engineered System

## Why this chapter exists

The most common failure mode of an early-career RAG project is not a bug — it is a *shape* problem. A team wires an embedding API to a pgvector top-*k* to an LLM prompt, ships it under a launch date, and treats the retriever as a *feature* of the product. Six months later a support agent flags that answers cite the wrong page number, an SRE flags that p95 latency degrades every time the corpus grows, and legal flags that customers on one tenant can see chunks from another tenant. None of those are retrieval-*correctness* bugs in the naïve pipeline they wrote. They are the bugs that appear once retrieval is asked to behave like an **engineered system** rather than a feature.

This chapter names the shape difference. The rest of the track builds the engineered-system version.

## The two shapes, side by side

|  | **Retrieval as a feature** | **Retrieval as an engineered system** |
|---|---|---|
| Owner | An application developer building a product surface | A retrieval engineer building a retriever + reranker + index that other products consume |
| Corpus | Fixed, hand-picked, small | Corpus of record: mixed formats, evolving, permissioned, multi-tenant |
| Chunker | LangChain / LlamaIndex default (fixed window, one setting for all docs) | Format-aware (PDF vs. code vs. tables), hierarchical, benchmarked against the corpus |
| Embedding model | Whatever the hosted API recommends today | Selected by a domain-fit benchmark on the corpus, versioned as an artifact |
| Index | pgvector top-*k*, defaults | pgvector / Pinecone / Weaviate / Qdrant / Milvus / FAISS + Lucene, ANN parameters tuned to recall@k and p95 latency |
| Retrieval shape | Dense top-*k* | Hybrid (BM25 + dense, sometimes SPLADE), rank-fused, optionally query-rewritten |
| Reranking | None | Cross-encoder / hosted reranker / LTR reranker with an *N* → *K* funnel |
| Context assembly | "Concatenate the top-K" | Dedup, structural ordering, per-chunk citation anchors, contextual compression, reserved answer budget |
| Prompt shape | "Answer using the context" | Citation-forcing, refusal-when-no-context, contradiction-handling, post-generation citation verification |
| Evaluation | Vibes; a few hand-picked queries | Judgment set (query → relevant-doc-IDs), recall@k / MRR / nDCG, paired significance tests, per-axis ablation |
| MLOps | None; the index is rebuilt when someone remembers | Index versioning `(embedding, chunker, schema, corpus-snapshot)`, retriever CI on every PR, dual-index cutover with rollback, drift alerts |
| Orchestration | Framework call chain | Optionally a framework; retriever is *also* exposed as an MCP-compatible tool downstream agents can call |
| Enterprise concerns | Ignored until launch | ACL-aware retrieval, per-tenant isolation, text-to-SQL over structured data, Graph RAG, multimodal — all first-class |

The right column is not "more sophisticated for its own sake" — it is what any of the following requirements force onto the design:

- The corpus grows past a few thousand documents.
- Two tenants must not see each other's chunks.
- A change to the chunker or embedding model has to ship without a several-day-long backfill outage.
- The retriever has to be reused by more than one product surface.
- Someone has to *defend* a launch decision with numbers (recall@k, latency, cost).
- The product has to answer *"where did you get that from?"* with a URL and a page number.

Any two of those in the same project tips you into the right column.

## The five capabilities the right column adds

The engineered-system shape is not a bag of tricks — it is five capabilities the feature-shape does not have. Each is a module (or several) later in the track.

### 1. Chunker design as an engineering discipline (mod-102)

A production chunker is a system that knows the format of each document (PDF vs. HTML vs. code vs. tabular), preserves structural metadata (section path, page number, table cell coordinates), attaches governance metadata (permission scope, updated-at), and is *benchmarked* against the corpus rather than adopted by default. "One chunker for all formats" is a feature-shape assumption.

### 2. Embedding model selection as evidence-driven (mod-103)

An embedding model is a versioned artifact chosen by a **domain-fit benchmark** — a fixed eval set on your corpus, ≥5 candidate models evaluated under identical chunker + retriever conditions, ranked on recall@k / MRR. MTEB and BEIR are inputs to that decision, not substitutes for it. "We use OpenAI text-embedding-3-large" is not an answer; "we ran the domain-fit benchmark in October and it beat E5 and BGE by X points, at Y ms and Z dollars per million tokens" is.

### 3. Index tuning against a recall / latency frontier (mod-104, mod-105)

Every ANN index (HNSW, IVF, IVF-PQ, ScaNN) exposes tunables (`M`, `efConstruction`, `efSearch`; `nlist`, `nprobe`, `pq_m`, `pq_nbits`) that trade recall@k, memory, build time, and p95 latency. The ann-benchmarks project ([ann-benchmarks.com](https://ann-benchmarks.com/)) publishes the frontier per algorithm; the retrieval engineer's job is to place their own workload on that frontier and defend the placement. The same discipline applies to Lucene-family shard sizing, replica placement, and BM25 analyzer configuration.

### 4. Retrieval-quality evaluation as first-class engineering (mod-109)

A judgment set is a versioned artifact. `recall@k`, `precision@k`, `MRR`, `MAP`, and `nDCG@k` are computed with `ir_measures` / `trec_eval` / `pytrec_eval`. Paired significance testing (paired t-test, Wilcoxon signed-rank, bootstrap CI over per-query score deltas) — grounded in Sakai's *Metrics, Statistics, Tests* (2014) — decides whether a proposed change survives the noise floor. Ablation harnesses attribute score changes to *which* of `(chunker, embedding, retriever, reranker)` moved.

Application-layer RAG evaluation — RAGAS, TruLens, DeepEval, RAG-triad (faithfulness / answer relevance / context precision-recall) — is owned by the peer `ai-eval-engineer` track. This module owns *retrieval-only* eval.

### 5. Retrieval MLOps (mod-111)

Index versioning keyed on `(embedding-model, chunker-version, schema-version, corpus-snapshot)`. Retriever CI that runs the judgment-set eval on every PR touching chunker / embedding / retriever / reranker code and gates on a recall@k / nDCG regression threshold. Dual-index cutover with health checks and rollback. Ingestion DAGs (Airflow / Dagster / Prefect) with backfill and ingestion-lag SLOs. Cost + latency measurement (embedding-API cost per doc, vector-DB storage cost per M vectors, ANN-parameter-driven latency, cache tiers, quantization). Corpus / query / quality drift alerts. This is what makes the retriever a system you can *own on-call*, not a demo you shipped once.

## The shape decision, made explicitly

When a stakeholder asks "should we do RAG?" the useful re-framing is *"which shape of RAG?"* — feature or engineered system. The answer is decided by the requirements list above, not by ambition. Feature-shape RAG is the right answer for a hackathon demo, an internal single-user tool, or the earliest version of a product where the corpus is a hand-picked handful of documents. Engineered-system RAG is the right answer when any of those forcing requirements are on the table.

The rest of this track is the engineered-system shape.

## What you should be able to do after this chapter

- Given a project description, place it on the feature ↔ engineered-system spectrum with reasoning.
- Name the five capabilities the engineered-system shape adds and the module that owns each.
- Explain why "we use OpenAI text-embedding-3-large" is not a defensible embedding-selection answer for a level-30 role.
- Explain why "answer using the context" is not a defensible generation-prompt answer.

The exercise for this chapter (paired with chapter 4) asks you to write a one-page ownership contract for the retriever your team will build, explicitly delegating the feature-shape concerns *out* to the peers who own them.

## Further reading

- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) — read the "Advanced RAG" and "Modular RAG" sections; they map almost 1:1 onto the right column above.
- Anthropic (2024). *Introducing Contextual Retrieval.* [Anthropic engineering blog](https://www.anthropic.com/news/contextual-retrieval) — a public engineering case study of the exact shape-shift this chapter names.
