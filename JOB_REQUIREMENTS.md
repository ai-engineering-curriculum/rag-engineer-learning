# RAG & Retrieval Engineer — Job Requirements

<!-- Regenerated 2026-08-06 from .aicg/job-requirements.json. Do not hand-edit — re-run `aicg org research` and rewrite via the normalized JSON so both files stay in sync. -->

## Sample

- **Sampled at**: 2026-08-06
- **Search window**: 2026-05-08 → 2026-08-06 (90 days)
- **Titles searched**: RAG Engineer, Retrieval Engineer, Search & Retrieval Engineer, Knowledge Retrieval Engineer, RAG & Retrieval Engineer
- **Total postings retained**: 31
- **In-window (or currently live) postings**: 23
- **Older-than-window signal postings**: 8 (real listings from 2025 and early 2026, kept as background — NOT counted toward the 3-posting evidence threshold)
- **URL sample-verified via WebFetch**: 7 of 31 (LSEG, Experity, Vannevar, Capital One, Salesforce SE-RAG, CAIS, Instrumentl). One originally-sampled URL (Reddit "Senior Staff Software Engineer, Indexing & Retrieval Platform") was dropped after verification failed.

The role's title is emerging. Postings appear under a mix of `Retrieval Engineer`, `Search Engineer`, `Software Engineer, RAG`, `AI Engineer (RAG specialist)`, `Applied AI Engineer`, `Machine Learning Engineer, Search`, and — at senior/staff levels — `Principal AI (Search) Engineer` / `Staff Software Engineer, Search & Retrieval Infrastructure`. Postings under `AI Engineer` were included only when the JD explicitly centered retrieval-engineering scope (chunking / embeddings / vector-DB / hybrid-search / reranking / retrieval eval).

## Ownership model (peer-repo scan)

Retrieval-adjacent topics are already claimed by three peer curricula. This track owns retrieval-engineering DEPTH; peers own application-layer integration and RAG-eval-at-application-layer. Verified against authored CURRICULUM.md files on 2026-08-06:

| Peer role | Level | Retrieval scope owned | Deliberate delegation to `rag-engineer` |
|---|---|---|---|
| `llm-application-developer-learning` | 20 | mod-005 "Retrieval Basics for LLM Applications" — hosted embedding APIs, pgvector top-k, grounded prompts with citations, retrieval-vs-prompt triage | Chunking strategies, embedding-model benchmarking, vector-DB internals, retrieval-quality eval, hybrid search, reranking |
| `applied-ai-engineer-learning` | 30 | mod-103 "RAG Integration at the Product Surface" — consuming a retriever, freshness/coverage/permission contracts, RAG-triad instrumentation at product layer | Retriever / reranker / chunker / index tuning |
| `ai-eval-engineer-learning` | 30 | mod-105 "RAG Evaluation at the Application Layer" — RAGAS / TruLens / DeepEval, faithfulness / context precision-recall, distractor suites | Retriever tuning, reranker choice, embedding-model selection, chunker design, hybrid retrieval |

Therefore this track (`rag-engineer-learning`, level 30, peer) owns:

- Retrieval **engineering** (not consumption): chunking, embeddings, vector-DB internals, Lucene-family engines, hybrid search, ANN tuning, index versioning
- **Retrieval-specific** evaluation (recall@k, MRR, nDCG, hit-rate) — NOT the RAG-triad or application-layer eval
- **Retrieval-specific** observability (per-chunk provenance, retriever hit-rate telemetry) — NOT general LLM tracing
- **Retrieval-specific** prompt patterns (citation instruction, refusal-when-no-context, contradiction handling) — NOT general prompt engineering
- **Exposing** a retriever as a tool (MCP, function-calling) — NOT general agent orchestration

## Requirement inventory (grouped)

Each row lists the requirement, sample count (all / in-window), frequency (in-window used for delta threshold), the peer role that owns it, and coverage path in this repo.

### Group 1 — Core RAG engineering (owner: `rag-engineer`)

| # | Requirement | Postings (all / in-win) | Freq (in-win) | Owner | Coverage |
|---|---|---|---|---|---|
| 1 | End-to-end RAG pipeline design (ingest → chunk → embed → retrieve → rerank → generate) | 27 / 21 | 0.91 | rag-engineer | **partial** — [`lessons/mod-001-rag-foundations/`](lessons/mod-001-rag-foundations/README.md) introduces the four-stage pipeline conceptually and via a naïve lab. No hands-on coverage of the reranker stage, semantic chunking, or metadata enrichment. |
| 2 | Vector databases (pgvector, Pinecone, Weaviate, Qdrant, Milvus, FAISS, Chroma) | 26 / 20 | 0.87 | rag-engineer | **not covered** — planned module territory. |
| 3 | Lucene-family engines (Elasticsearch, OpenSearch, Solr, Vespa) for lexical + hybrid retrieval | 17 / 13 | 0.57 | rag-engineer | **not covered** — planned module territory. |
| 4 | LlamaIndex / LangChain / LangGraph / Semantic Kernel orchestration for retrieval | 17 / 14 | 0.61 | rag-engineer | **not covered** — planned module territory. |
| 5 | Hybrid retrieval (BM25 + dense) with reciprocal-rank fusion | 15 / 12 | 0.52 | rag-engineer | **not covered** — planned module territory. |
| 6 | Embedding model selection, tuning, evaluation (bi-encoders, cross-encoders) | 14 / 11 | 0.48 | rag-engineer | **not covered** — planned module territory. |
| 7 | Chunking strategy (semantic, structural, metadata-enriched) | 13 / 11 | 0.48 | rag-engineer | **not covered** — planned module territory. |
| 8 | Reranking (cross-encoders, Cohere/Voyage rerankers, LTR) | 12 / 9 | 0.39 | rag-engineer | **not covered** — planned module territory. |
| 9 | MLOps for retrieval (model versioning, index versioning, CI/CD for RAG, drift monitoring) | 12 / 9 | 0.39 | rag-engineer | **not covered** — planned module territory. |
| 10 | Cost / latency optimization of retrieval (caching, tiering, quantization) | 10 / 7 | 0.30 | rag-engineer | **not covered** — planned module territory. |
| 11 | Citations / grounding / attribution and hallucination mitigation | 9 / 7 | 0.30 | rag-engineer | **partial** — motivated in [`lessons/mod-001-rag-foundations/lectures/01-what-is-rag.md`](lessons/mod-001-rag-foundations/lectures/01-what-is-rag.md); no hands-on citation mechanics. |
| 12 | Retrieval evaluation metrics (recall@k, MRR, nDCG, precision@k, hit-rate) | 8 / 6 | 0.26 | rag-engineer | **not covered** — planned module territory. Below the 0.30 threshold in-window but tightly coupled to req #1. |
| 13 | Structured extraction from PDFs / OCR / document intelligence for corpora | 8 / 6 | 0.26 | rag-engineer | **not covered** — planned module territory. |
| 14 | Context-window management / context assembly | 7 / 6 | 0.26 | rag-engineer | **not covered** — planned module territory. |
| 15 | Large-scale ANN tuning / index scaling (HNSW, IVF-PQ, Lucene shard sizing) | 9 / 6 | 0.26 | rag-engineer | **not covered** — planned module territory. |
| 16 | Query rewriting / decomposition / semantic routing | 6 / 5 | 0.22 | rag-engineer | **not covered** — planned module territory. |
| 17 | Permission / ACL-aware retrieval for enterprise corpora | 6 / 4 | 0.17 | rag-engineer | **not covered** — planned module territory. Below threshold. |
| 18 | Learning-to-rank / classical IR (LambdaMART, LightGBM ranking) | 6 / 4 | 0.17 | rag-engineer | **not covered** — planned module territory. Below threshold. |
| 19 | Graph RAG / knowledge graphs (Neo4j, ontologies) | 7 / 5 | 0.22 | rag-engineer | **not covered** — emerging. Below threshold. |
| 20 | SQL / text-to-SQL retrieval over structured data | 4 / 3 | 0.13 | rag-engineer | **not covered** — niche. |
| 21 | Multimodal retrieval (image / audio / video embeddings) | 3 / 2 | 0.09 | rag-engineer | **not covered** — niche. |

### Group 2 — Delegated to peer curricula

| # | Requirement | Postings (all / in-win) | Freq (in-win) | Owner | Handoff |
|---|---|---|---|---|---|
| 22 | Production Python services (FastAPI, asyncio) for retrieval APIs | 22 / 17 | 0.74 | llm-application-developer | Assumed as prereq. See [`PREREQUISITES.md`](PREREQUISITES.md). |
| 23 | Cloud deployment (AWS/GCP/Azure) + containerization (Docker/K8s) | 21 / 15 | 0.65 | ml-platform / cross-cutting | Assumed as prereq. |
| 24 | Prompt engineering for RAG (general prompt-engineering baseline) | 16 / 12 | 0.52 | llm-application-developer | This track only covers RAG-**specific** patterns (citation instruction, refusal-when-no-context) — not baseline prompt engineering. |
| 25 | Agentic retrieval / tool-use retrievers / MCP servers | 14 / 11 | 0.48 | agentic-ai-engineer | This track covers **exposing** a retriever as an MCP-compatible tool; agent orchestration is delegated. |
| 26 | Guardrails / PII redaction / responsible-AI on retrieval | 8 / 6 | 0.26 | ai-eval-engineer | This track covers ACL-aware retrieval mechanics (req #17); content-safety methodology is delegated. |
| 27 | Observability / tracing for RAG (LangSmith, Langfuse, Arize, Phoenix) | 8 / 6 | 0.26 | ai-eval-engineer | This track covers retrieval-specific traces (per-chunk provenance); general LLM tracing is delegated. |
| 28 | RAG evaluation frameworks (RAGAS, TruLens, DeepEval, OpenAI Evals) | 7 / 4 | 0.17 | ai-eval-engineer | Explicitly owned by ai-eval-engineer's mod-105. This track covers retrieval-**specific** eval (recall@k, MRR, nDCG). |
| 29 | Fine-tuning embeddings / domain adaptation (SFT/LoRA/PEFT) | 6 / 4 | 0.17 | fine-tuning-engineer | Cross-linked but not owned here. |
| 30 | Streaming / real-time indexing (Kafka / Flink / Spark) | 8 / 4 | 0.17 | staff+ / data-platform | Below threshold in-window; concentrated in staff+ web-scale postings. Not required for a level-30 IC track. |

## External resources for out-of-scope / niche items

For requirements deliberately not owned or currently below threshold, linked resources learners can use directly:

- **RAG evaluation frameworks** — [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval](https://docs.confident-ai.com/). Peer-owned in [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) mod-105.
- **MCP servers** — [Model Context Protocol spec](https://modelcontextprotocol.io/) and Anthropic's [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). Peer-owned in agentic tracks.
- **Fine-tuning embeddings** — [sentence-transformers training guide](https://www.sbert.net/docs/training/overview.html). Peer-owned in `fine-tuning-engineer-learning`.
- **Graph RAG** — [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) and [LlamaIndex Property Graph Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/). Below threshold; watch for delta candidate next cycle if signal grows.
- **Streaming indexing** — Elastic's [reindexing / ingest pipelines docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/ingest.html) and Vespa's [real-time indexing docs](https://docs.vespa.ai/en/reads-and-writes.html). Out of scope for a level-30 IC track.

## Emerging-signal watchlist (below-threshold this cycle)

Track these for the next research cycle. They cleared 3-posting evidence but not the 0.30 frequency threshold — a rebound above 0.30 would justify a delta:

- **Graph RAG / knowledge graphs** (0.22 in-window). Only 4 employers name it directly (CAIS, Octopus, CoreStory, Rayda). Growing but not yet mainstream.
- **Query rewriting / decomposition** (0.22 in-window). Language showing up under "advanced RAG techniques" in senior/staff postings only.
- **Permission-aware / ACL-aware retrieval** (0.17 in-window). Consistent-but-quiet enterprise signal.
- **MCP-server construction of retrievers** (a subset of the 0.48 agentic-retrieval bucket). Only Vertex Inc. and LSEG name MCP servers as required rather than nice-to-have. Worth re-checking; MCP adoption is moving fast.

## Delta rationale (this cycle)

See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json). This cycle proposes **zero new modules, exercises, or projects**. Reasoning:

1. **The curriculum is a scaffold** — only [`lessons/mod-001-rag-foundations/`](lessons/mod-001-rag-foundations/README.md) is authored; [`CURRICULUM.md`](CURRICULUM.md) marks the rest `TBD`. Almost every core-owned requirement above (rows 2–21) is legitimately uncovered.
2. **The 20% additions cap** would auto-flag any material seeding of the roadmap for human review. A three-line delta cannot responsibly define the whole track.
3. **The market has not "materially shifted" in the last 90 days** — the top-frequency requirements (vector DBs, hybrid retrieval, chunking, reranking, embedding evaluation) are the enduring shape of the role, not a recent shift. The one thing that IS a recent shift (MCP-server construction of retrievers) does not yet clear the 3-posting-as-required-skill bar.
4. **Correct routing** for build-out is the primary content-generation loop (per-module prompts under `.aicg/prompts/` as seen in sibling authored repos), not this research packet.

The purpose of this packet is to **inform** the roadmap and lock in the ownership boundary with peer curricula, not to seed the module tree.
