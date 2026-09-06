# RAG & Retrieval Engineer — Job Requirements

<!-- Regenerated 2026-09-06 from .aicg/job-requirements.json. Do not hand-edit — re-run `aicg org research` and rewrite via the normalized JSON so both files stay in sync. -->

## Sample

- **Sampled at**: 2026-09-06
- **Search window**: 2026-06-08 → 2026-09-06 (90 days)
- **Titles searched**: RAG Engineer, Retrieval Engineer, Search & Retrieval Engineer, Knowledge Retrieval Engineer, RAG & Retrieval Engineer, Software Engineer RAG, Machine Learning Engineer Search, AI Engineer (RAG specialist)
- **Total postings retained**: 28
- **In-window (or currently live) postings**: 28
- **URL sample-verified via WebFetch**: 12 of 28 (Iovance, MediaAlpha, Aspenview, Artefact, LTS, Tebra, Scale AI, Tavily, xAI, Enigma, Glean, Instrumentl). Remaining verified via search-result snippet + phrasing consistency against peer greenhouse listings.

The role's title is emerging. Postings appear under a mix of `Retrieval Engineer`, `Search Engineer`, `Software Engineer, RAG`, `AI Engineer (RAG specialist)`, `Applied AI Engineer`, `Machine Learning Engineer, Search`, and — at senior/staff levels — `Principal AI Systems Engineer`, `Staff Software Engineer, Search Ranking`, and `Member of Technical Staff, Search/Retrieval`. Postings under `AI Engineer` were included only when the JD explicitly centered retrieval-engineering scope (chunking / embeddings / vector-DB / hybrid-search / reranking / retrieval eval).

## Curriculum state as of this cycle

Modules `mod-101` through `mod-112` are now **fully authored** (chapters + exercises + labs + quizzes), superseding the scaffold-state summary the previous cycle wrote in 2026-08. The freshness review (`.aicg/freshness-review-report.json`, 2026-08-21) marked every module artifact `passed` with no blockers. Coverage paths below point to concrete authored files, not TBDs.

## Ownership model (peer-repo scan)

Retrieval-adjacent topics are already claimed by peer curricula. This track owns retrieval-engineering DEPTH; peers own application-layer integration and RAG-eval-at-application-layer. Verified against authored CURRICULUM.md files on 2026-09-06:

| Peer role | Level | Retrieval scope owned | Deliberate delegation to `rag-engineer` |
|---|---|---|---|
| `llm-application-developer-learning` | 20 | mod-005 "Retrieval Basics for LLM Applications" — hosted embedding APIs, pgvector top-k, grounded prompts with citations, retrieval-vs-prompt triage | Chunking strategies, embedding-model benchmarking, vector-DB internals, retrieval-quality eval, hybrid search, reranking |
| `applied-ai-engineer-learning` | 30 | "RAG Integration at the Product Surface" — consuming a retriever, freshness/coverage/permission contracts, RAG-triad instrumentation at product layer | Retriever / reranker / chunker / index tuning |
| `ai-eval-engineer-learning` | 30 | "RAG Evaluation at the Application Layer" — RAGAS / TruLens / DeepEval, faithfulness / context precision-recall, distractor suites | Retriever tuning, reranker choice, embedding-model selection, chunker design, hybrid retrieval |
| `agentic-ai-engineer-learning` | 40 | Agent orchestration and tool-use planning that CONSUMES retrievers | Exposing a retriever as an MCP-compatible / function-calling tool (owned here in mod-110) |
| `fine-tuning-engineer-learning` | 30 | Embedding SFT / LoRA / PEFT training craft (sentence-transformers TSDAE, MultipleNegativesRankingLoss) | Embedding selection, benchmarking, domain-fit — owned here in mod-103 |
| `ai-risk-engineer-learning` | 25 | Content-safety, PII redaction, guardrail-effectiveness methodology | ACL-aware retrieval MECHANICS (owned here in mod-112) |

Therefore this track (`rag-engineer-learning`, level 30, peer) owns:

- Retrieval **engineering** (not consumption): chunking, embeddings, vector-DB internals, Lucene-family engines, hybrid search, ANN tuning, index versioning
- **Retrieval-specific** evaluation (recall@k, MRR, nDCG, hit-rate) — NOT the RAG-triad or application-layer eval
- **Retrieval-specific** observability (per-chunk provenance, retriever hit-rate telemetry) — NOT general LLM tracing
- **Retrieval-specific** prompt patterns (citation instruction, refusal-when-no-context, contradiction handling) — NOT general prompt engineering
- **Exposing** a retriever as a tool (MCP, function-calling) — NOT general agent orchestration

## Requirement inventory (grouped)

Each row lists the requirement, sample count (in-window / total), in-window frequency (used for delta threshold), the owning curriculum, and coverage path. Every retained posting this cycle is in-window, so total and in-window counts match.

### Group 1 — Core RAG engineering (owner: `rag-engineer`)

| # | Requirement | Postings | Freq | Coverage |
|---|---|---|---|---|
| 1 | Vector databases (pgvector, Pinecone, Weaviate, Qdrant, Milvus, FAISS, Chroma) | 25 / 28 | 0.89 | ✅ [`mod-104-vector-databases-and-ann-tuning`](lessons/mod-104-vector-databases-and-ann-tuning/README.md) — 9 chapters + 6 exercises + labs; ANN internals + HNSW / IVF-PQ tuning + sharding + versioning. |
| 2 | End-to-end RAG pipeline design (ingest → chunk → embed → retrieve → rerank → generate) | 24 / 28 | 0.86 | ✅ [`mod-101-retrieval-engineering-foundations`](lessons/mod-101-retrieval-engineering-foundations/README.md) — pipeline map + role scope; every downstream module drills one stage. |
| 3 | LlamaIndex / LangChain / LangGraph / Semantic Kernel orchestration for retrieval | 17 / 28 | 0.61 | ✅ [`mod-110-orchestration-and-mcp-exposure`](lessons/mod-110-orchestration-and-mcp-exposure/README.md) — framework fit matrix + LlamaIndex advanced RAG + LangChain LCEL + LangGraph state graphs + Semantic Kernel. |
| 4 | Hybrid retrieval (BM25 + dense) with fusion (RRF) and multi-stage rerank | 15 / 28 | 0.54 | ✅ [`mod-106-hybrid-retrieval-fusion-query-rewriting`](lessons/mod-106-hybrid-retrieval-fusion-query-rewriting/README.md) — RRF, weighted-sum trap, Borda, semantic routing. |
| 5 | Chunking strategy (semantic, structural, metadata-enriched, hierarchical parent-child) | 14 / 28 | 0.50 | ✅ [`mod-102-document-ingestion-and-chunking`](lessons/mod-102-document-ingestion-and-chunking/README.md) — semantic vs structural vs fixed-window ablation, small-to-big, table/figure-aware chunking. |
| 6 | Reranking (cross-encoders, Cohere / Voyage / BGE / Jina rerankers, LTR, LLM-as-reranker) | 14 / 28 | 0.50 | ✅ [`mod-107-reranking-and-context-assembly`](lessons/mod-107-reranking-and-context-assembly/README.md) — reranker choice, top-N → top-K funnel, contextual compression, lost-in-the-middle. |
| 7 | Embedding model selection, tuning, and evaluation (bi-encoders, cross-encoders, ColBERT, SPLADE) | 13 / 28 | 0.46 | ✅ [`mod-103-embeddings-selection-and-benchmarking`](lessons/mod-103-embeddings-selection-and-benchmarking/README.md) — MTEB/BEIR, hosted vs self-hosted sweep, domain-fit benchmark, quantization + Matryoshka. |
| 8 | Lucene-family engines (Elasticsearch, OpenSearch, Solr, Vespa) for lexical + hybrid retrieval | 12 / 28 | 0.43 | ✅ [`mod-105-lucene-family-and-ltr`](lessons/mod-105-lucene-family-and-ltr/README.md) — engine landscape, BM25 analyzer + field mapping, dense_vector fields, LTR plugin ecosystem. |
| 9 | Retrieval evaluation metrics (recall@k, MRR, nDCG, precision@k, hit-rate) | 11 / 28 | 0.39 | ✅ [`mod-109-retrieval-quality-evaluation`](lessons/mod-109-retrieval-quality-evaluation/README.md) — TREC/BEIR-shaped judgment sets, ir_measures/trec_eval/pytrec_eval, paired significance testing. |
| 10 | MLOps for retrieval (index versioning, retriever CI/CD, drift monitoring) | 10 / 28 | 0.36 | ✅ [`mod-111-retrieval-mlops-versioning-cicd-drift`](lessons/mod-111-retrieval-mlops-versioning-cicd-drift/README.md) — index-version tuples, dual-index cutover, retriever CI gate on judgment set, drift monitor. |
| 11 | Citations / grounding / attribution and hallucination mitigation | 10 / 28 | 0.36 | ✅ [`mod-108-retrieval-specific-prompting`](lessons/mod-108-retrieval-specific-prompting/README.md) — inline + structured citation instruction, refusal-when-no-context, contradiction handling, post-generation citation verification. |
| 12 | Cost / latency optimization of retrieval (caching, tiering, quantization) | 9 / 28 | 0.32 | ✅ [`mod-111`](lessons/mod-111-retrieval-mlops-versioning-cicd-drift/exercises/exercise-04-embedding-cost-and-vector-storage-cost-optimization-drill.md) exercise-04 (embedding + storage cost drill) + [exercise-05](lessons/mod-111-retrieval-mlops-versioning-cicd-drift/exercises/exercise-05-query-embedding-and-result-cache-tier-design.md) cache tier design; quantization in [`mod-103`](lessons/mod-103-embeddings-selection-and-benchmarking/README.md). |
| 13 | Structured extraction from PDFs / OCR / document intelligence | 6 / 28 | 0.21 | ✅ [`mod-102 exercise-01`](lessons/mod-102-document-ingestion-and-chunking/exercises/exercise-01-unstructured-and-docling-pdf-ingestion-drill.md) (unstructured.io + Docling) + [exercise-02](lessons/mod-102-document-ingestion-and-chunking/exercises/exercise-02-ocr-fallback-with-textract-and-tesseract-cost-latency-compare.md) (OCR fallback). |
| 14 | Context-window management / context assembly / contextual compression | 6 / 28 | 0.21 | ✅ [`mod-107`](lessons/mod-107-reranking-and-context-assembly/README.md) — contextual compression, dedup, ordering, per-chunk citation anchors, reserved answer budget. |
| 15 | Large-scale ANN tuning / index scaling (HNSW, IVF-PQ, Lucene shard sizing) | 6 / 28 | 0.21 | ✅ [`mod-104`](lessons/mod-104-vector-databases-and-ann-tuning/README.md) chapters 03–06 + [`mod-105`](lessons/mod-105-lucene-family-and-ltr/README.md) Lucene shard-sizing heuristics. |
| 16 | Query rewriting / decomposition / semantic routing (HyDE, step-back, multi-query) | 6 / 28 | 0.21 | ✅ [`mod-106`](lessons/mod-106-hybrid-retrieval-fusion-query-rewriting/README.md) — HyDE, decomposition, step-back, multi-query fanout, semantic routing. |
| 17 | MCP-server exposure of retrievers | 6 / 28 | 0.21 | ✅ [`mod-110`](lessons/mod-110-orchestration-and-mcp-exposure/README.md) chapters 06 MCP contract primer + 07 MCP server via official Python SDK + 08 direct tool endpoints. |
| 18 | Permission / ACL-aware retrieval for enterprise corpora | 5 / 28 | 0.18 | ✅ [`mod-112`](lessons/mod-112-enterprise-retrieval-acl-structured-graph-multimodal/README.md) chapters 02–03 — permission-aware retrieval mechanics + SharePoint / Google Drive / Salesforce ACL shapes. |
| 19 | Graph RAG / knowledge graphs (Neo4j, ontologies, property graphs) | 5 / 28 | 0.18 | ✅ [`mod-112`](lessons/mod-112-enterprise-retrieval-acl-structured-graph-multimodal/README.md) chapter 05 — Graph RAG on Neo4j / Kùzu / Memgraph, LlamaIndex Property Graph Index, Microsoft GraphRAG. |
| 20 | SQL / text-to-SQL retrieval over structured data | 4 / 28 | 0.14 | ✅ [`mod-112`](lessons/mod-112-enterprise-retrieval-acl-structured-graph-multimodal/README.md) chapter 04 — text-to-SQL via LlamaIndex NLSQLTableQueryEngine + LangChain SQL agent + execution-accuracy eval on Spider / BIRD. |
| 21 | Multimodal retrieval (image / audio / video embeddings, CLIP / SigLIP / ColPali) | 4 / 28 | 0.14 | ✅ [`mod-112`](lessons/mod-112-enterprise-retrieval-acl-structured-graph-multimodal/README.md) chapter 06 — CLIP / SigLIP + ColPali / DSE + audio + video-clip survey. |
| 22 | Learning-to-rank / classical IR (LambdaMART, LightGBM ranking) | 4 / 28 | 0.14 | ✅ [`mod-105`](lessons/mod-105-lucene-family-and-ltr/README.md) — LambdaMART / LambdaRank / RankNet via LightGBM/XGBoost, Elasticsearch/OpenSearch LTR plugin. |
| 23 | Live-web / agentic-search retrieval infrastructure (crawl orchestration, JS rendering, freshness SLAs) | 2 / 28 | 0.07 | ⚠️ **not_covered** — NEW watchlist candidate. Below the 3-posting-as-required-skill bar and far below the 0.30 frequency threshold. Retained on the emerging-signal watchlist for reassessment next cycle. Do NOT propose as a new module this cycle. |

### Group 2 — Delegated to peer curricula

| # | Requirement | Postings | Freq | Owner | Handoff |
|---|---|---|---|---|---|
| 24 | Production Python services (FastAPI, asyncio) for retrieval APIs | 24 / 28 | 0.86 | llm-application-developer | Assumed as prereq. See [`PREREQUISITES.md`](PREREQUISITES.md). |
| 25 | Cloud deployment (AWS/GCP/Azure) + containerization (Docker/K8s) | 15 / 28 | 0.54 | cross-cutting | Assumed as prereq. |
| 26 | Agentic retrieval / tool-use retrievers / agent orchestration | 14 / 28 | 0.50 | agentic-ai-engineer | This track covers **exposing** a retriever as an MCP-compatible / tool-callable endpoint (mod-110). Agent orchestration consuming those endpoints is delegated. |
| 27 | Prompt engineering for RAG (general baseline) | 12 / 28 | 0.43 | llm-application-developer | This track only covers RAG-**specific** patterns (citation instruction, refusal-when-no-context, contradiction handling — mod-108) — not baseline prompt engineering. |
| 28 | RAG evaluation frameworks (RAGAS, TruLens, DeepEval, OpenAI Evals) | 6 / 28 | 0.21 | ai-eval-engineer | Explicitly owned by ai-eval-engineer's RAG-triad module. This track covers retrieval-**specific** eval (recall@k, MRR, nDCG — mod-109). |
| 29 | Guardrails / PII redaction / responsible-AI on retrieval | 4 / 28 | 0.14 | ai-risk-engineer | This track covers ACL-aware retrieval mechanics (row 18); content-safety methodology delegated. |
| 30 | Observability / tracing for RAG (LangSmith, Langfuse, Arize, Phoenix) | 4 / 28 | 0.14 | ai-eval-engineer | This track covers retrieval-specific traces (per-chunk provenance, retriever hit-rate telemetry, drift monitor — mod-111); general LLM tracing delegated. |
| 31 | Fine-tuning embeddings / domain adaptation (SFT/LoRA/PEFT) | 4 / 28 | 0.14 | fine-tuning-engineer | Selection + benchmarking + domain-fit is OWNED here (mod-103); training craft (TSDAE, MultipleNegativesRankingLoss, GISTEmbedLoss) is delegated. |
| 32 | Streaming / real-time indexing (Kafka / Flink / Spark) | 3 / 28 | 0.11 | staff-ml-engineer / data-platform | Concentrated in staff+ / web-scale postings. Not required for a level-30 IC track. |

## External resources for out-of-scope / niche items

For requirements deliberately not owned or currently below threshold, linked resources learners can use directly:

- **RAG evaluation frameworks** — [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval](https://docs.confident-ai.com/). Peer-owned in [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning).
- **Fine-tuning embeddings** — [sentence-transformers training guide](https://www.sbert.net/docs/training/overview.html). Peer-owned in `fine-tuning-engineer-learning`.
- **Streaming indexing** — Elastic's [reindexing / ingest pipelines docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/ingest.html) and Vespa's [real-time indexing docs](https://docs.vespa.ai/en/reads-and-writes.html). Out of scope for a level-30 IC track.
- **Live-web / agentic-search retrieval** — [Anthropic MCP browser-use example](https://modelcontextprotocol.io/), [Playwright/Chromium DevTools Protocol docs](https://playwright.dev/), [Common Crawl](https://commoncrawl.org/). Below threshold this cycle; watch Tavily/xAI-shaped postings next cycle.
- **Agent orchestration consuming retrievers** — see peer `agentic-ai-engineer-learning`; this track only exposes retrievers, not the agents that call them.

## Emerging-signal watchlist (below-threshold this cycle)

Track these for the next research cycle. They cleared the 3-posting evidence bar but not the 0.30 frequency threshold — a rebound above 0.30 would justify a delta:

- **Live-web / agentic-search retrieval infrastructure** (0.07 in-window, 2 employers — Tavily, xAI). NEW this cycle. Distinct retrieval concern: headless-browser JS rendering, freshness SLAs, dedup and provenance for live-web content. **Not yet clearing evidence bar** (2 < 3), but a first observable signal and potentially the shape of a future `mod-113`.
- **MCP-server exposure of retrievers** (0.21 in-window, 6 employers — Iovance, MediaAlpha, Artefact, Particle41, LTS, Reddit). Cleared the 3-posting-as-required-skill bar this cycle (up from 2 in 2026-08). Still below 0.30, but ALREADY OWNED in mod-110 chapters 06–08 — no delta action needed.
- **Graph RAG / knowledge graphs** (0.18 in-window, 5 employers). Consistent enterprise signal, already OWNED in mod-112 chapter 05 — no delta action needed.
- **Permission-aware / ACL-aware retrieval** (0.18 in-window, 5 employers). Already OWNED in mod-112 chapters 02–03 — no delta action needed.
- **Query rewriting / HyDE / decomposition** (0.21 in-window, 6 employers). Language showing up under "advanced RAG techniques". Already OWNED in mod-106 — no delta action needed.

## Delta rationale (this cycle)

See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json). This cycle proposes **zero new modules, exercises, or projects**. Reasoning:

1. **The curriculum is mature and now fully authored.** Every requirement rated above the 0.30 in-window frequency threshold (rows 1–12) is already covered by an authored module with real chapters, exercises, and labs — verified against the 2026-08-21 freshness review (all artifacts `passed`, no blockers). This cycle's role in the org loop is to *confirm ownership boundaries and update coverage links*, not to seed content.
2. **The market has not materially shifted in the last 90 days.** Comparing this cycle's requirement inventory against 2026-08-06's shows the top-frequency requirements (vector DBs, hybrid retrieval, chunking, reranking, embedding selection) are stable within noise (±0.05 frequency). No new requirement crosses BOTH the 3-posting and 0.30-frequency bars against uncovered curriculum territory.
3. **The one genuinely-NEW signal (live-web / agentic-search retrieval)** appears in only 2 postings (Tavily's Web Rendering Engineer and xAI's Grokipedia Search/Retrieval role) — below the 3-posting evidence bar and far below 0.30 frequency. It is added to the emerging-signal watchlist for the next cycle. A single-cycle jump from 0 to 2 in a rapidly-branding-shifting space (agentic search) is not enough to justify a `mod-113`; it may fade or consolidate under a different name.
4. **Continuity bias applies.** The prompt directive is: propose net-new content only when ALL of (≥ 3 postings, ≥ 0.30 frequency, no incrementally-extendable existing coverage) hold. No requirement this cycle meets all three against uncovered ground.

<!-- needs-research: next cycle — re-run the same 90-day scan and check whether (a) live-web / agentic-search retrieval crosses the 3-posting bar (Tavily, xAI, and any of Perplexity / SerpAPI / Firecrawl / Exa / You.com), (b) MCP-server exposure hits ≥ 0.30 frequency, and (c) any of the 0.18-0.21 emerging-signal requirements (graph RAG, ACL-aware retrieval, query rewriting) crosses 0.30. The delta at that point should still favor adding an exercise to an existing module (e.g., mod-112 + a live-web-crawl exercise) over creating a new module unless a fully-distinct engineering surface has emerged. -->
