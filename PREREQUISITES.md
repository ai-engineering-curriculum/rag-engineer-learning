# Prerequisites — RAG & Retrieval Engineer track

**Role level:** 30 (retrieval-engineering specialist, AI Engineering family)
**Track:** `rag-engineer-learning`

This track is a hands-on retrieval-engineering specialty. It assumes the learner arrives with retrieval basics fluent, production Python service craft muscle-memoried, and classical ML fundamentals in place. The modules step directly onto chunker design, embedding benchmarking, vector-DB internals, and retrieval evaluation — fundamentals are **not** re-taught here.

## Assumed lower-level curriculum

Complete or be practically fluent in the equivalent of these lower-level tracks before starting:

- **[`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) — level 10** — engineering-craft prerequisites (Linux, Git, Python packaging, HTTP APIs, unit testing, logging, Docker basics). If you cannot ship a small Python service and instrument it, start here.
- **[`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) — level 20** — classical ML with scikit-learn, deep learning with PyTorch and Hugging Face Transformers, standard sklearn / HF metrics, packaging with FastAPI + Docker, experiment tracking with MLflow / Weights & Biases. This is where the tf-idf / BM25 primer and the sentence-transformers introduction live; mod-101 refreshes rather than re-teaches.
- **[`llm-application-developer-learning`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) — level 20** — production Python service craft (FastAPI, asyncio, streaming), general prompt engineering, hosted LLM provider APIs (OpenAI, Anthropic, Google, Bedrock), model selection + cost + prompt caching basics, and — critically — mod-005 "Retrieval Basics for LLM Applications" (hosted embedding APIs, pgvector top-k, grounded prompts with citations, retrieval-vs-prompt triage). This track *builds on top of* that mod-005 shape; it does not re-teach naïve top-k or hosted-API embedding calls.

## Assumed working knowledge

Even without holding the equivalent role formally, you should be practically fluent in:

- **Python** — comfortable with packaging (pyproject.toml, uv / poetry, virtualenvs), unit testing (pytest, hypothesis), typed code (mypy / pyright), async (asyncio), and CI (GitHub Actions).
- **FastAPI + async I/O** — comfortable shipping a small retrieval API with streaming responses, request tracing, and reasonable timeout / retry / circuit-breaker handling. Baseline covered in `llm-application-developer` (level 20).
- **SQL and Postgres** — comfortable authoring schemas, indexes, and query plans; comfortable enough with EXPLAIN to reason about pgvector query plans. Table-scan-vs-index-scan intuition matters when tuning pgvector HNSW / IVFFlat.
- **Docker + Docker Compose** — comfortable standing up a multi-container local stack (Postgres + Elasticsearch + a model server) for reproducible dev.
- **PyTorch + Hugging Face Transformers + sentence-transformers** — enough to load a Hugging Face model, run inference, use sentence-transformers to embed a batch. Fine-tuning depth is *not* required (it's owned by `fine-tuning-engineer`, peer, level 30).
- **Statistics & probability** — reading precision / recall / F1, ROC / PR curves, bootstrap confidence intervals; comfortable enough with hypothesis testing that a paired t-test / Wilcoxon signed-rank discussion is not new. Deep statistical methodology depth (Bayesian priors, sequential testing, judge-vs-human calibration) is linked out to `model-evaluation-engineer` (peer, level 30).
- **Classical IR intuition** — inverted indexes, tf-idf, BM25, precision-vs-recall trade-off. Mod-101 refreshes this; if it's brand new, work through the relevant chapters of Manning-Raghavan-Schütze *Introduction to Information Retrieval* before mod-102.
- **LLM provider APIs** — comfortable calling OpenAI / Anthropic / Google / Cohere / open-source (vLLM / TGI) endpoints from Python, including embedding endpoints, and reading trace output from Arize Phoenix, Langfuse, or LangSmith at the app-developer level.
- **Basic security literacy** — HTTPS / auth basics, secrets handling, OWASP Top 10 for web at the general awareness level. Deep application security is not required — this track adds retrieval-specific ACL mechanics on top of that baseline.
- **Reading the primary literature** — you should be able to open a paper on arXiv (e.g., Lewis et al. RAG 2020, Malkov & Yashunin HNSW 2016, Cormack et al. RRF 2009, Khattab & Zaharia ColBERT 2020, Gao et al. HyDE 2022, Liu et al. Lost-in-the-Middle 2023) and translate the method into runnable code within a working day.

## Not assumed, taught here

You do **not** need prior experience with:

- Any specific chunking library — LlamaIndex node parsers, LangChain text splitters, unstructured.io, Docling, PyMuPDF. Mod-102 covers the working suite.
- Any specific embedding model catalog or benchmark — MTEB leaderboard, BEIR, OpenAI text-embedding-3, Cohere Embed v3, Voyage voyage-3, BGE, E5, Nomic, GTE, Snowflake Arctic Embed, ColBERT, SPLADE. Mod-103 covers the working set with a domain-fit benchmark methodology.
- Any specific vector database beyond a passing familiarity with pgvector — pgvector internals, Pinecone, Weaviate, Qdrant, Milvus, FAISS as a library, Chroma. Mod-104 covers the working suite plus HNSW / IVF-PQ / ScaNN tuning.
- Any specific Lucene-family engine — Elasticsearch, OpenSearch, Solr, Vespa; BM25 analyzer configuration; the Elasticsearch / OpenSearch LTR plugins. Mod-105 covers the working suite.
- Any specific fusion or query-rewriting technique — reciprocal-rank fusion, HyDE, query decomposition, step-back prompting, semantic routing. Mod-106 covers the working suite.
- Any specific reranker — bge-reranker / ms-marco MiniLM cross-encoders, Cohere Rerank v3, Voyage rerank-2, Jina Reranker, LTR rerankers. Mod-107 covers the working suite plus contextual compression.
- Any specific retrieval evaluation toolchain — ir_measures, trec_eval, pytrec_eval; how to author a judgment set. Mod-109 covers the working suite plus paired significance testing.
- Any specific retrieval orchestration framework — LlamaIndex, LangChain, LangGraph, Semantic Kernel — beyond the light introduction in `llm-application-developer` mod-005. Mod-110 covers the working suite plus the MCP-server exposure pattern.
- Any specific text-to-SQL, Graph RAG, or multimodal-retrieval stack — LlamaIndex NLSQLTableQueryEngine, LangChain SQL agent, Microsoft GraphRAG, Neo4j / Kùzu / Memgraph, CLIP / SigLIP / ColPali. Mod-112 covers the working suite.

## Recommended reading before starting

- Lewis, P. et al. ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) — Meta, 2020. The canonical RAG paper; read once end-to-end.
- Gao, Y. et al. ["Retrieval-Augmented Generation for Large Language Models: A Survey"](https://arxiv.org/abs/2312.10997) — 2024. The map of the landscape as of the survey date; read the section headings and pick two subsections you want to know first.
- Manning, Raghavan, Schütze [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) — Cambridge, 2008. Skim chapters 1–8 (Boolean retrieval, index construction, dictionaries, tolerant retrieval, positional indexes, scoring / vector-space model, evaluation) before mod-101 if the classical-IR ideas are new.
- Malkov & Yashunin ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/abs/1603.09320) — 2016. The HNSW paper; every vector-DB tunable in mod-104 traces back to this.
- [Anthropic Model Context Protocol specification](https://modelcontextprotocol.io/). Skim the tools + resources shape so mod-110 lands with a mental model.
- One recent public system card that includes retrieval (e.g., a Claude tool-use disclosure, an OpenAI file-search launch note) — read it as an engineer would, tracing each capability claim to an engineering decision.

## Not required, useful to have

- Prior work with an eval-tooling platform (Arize Phoenix, Langfuse, W&B Weave, Braintrust, Promptfoo, DeepEval, RAGAS, LangSmith) — the peer `ai-eval-engineer-learning` (level 30) owns this stack; a working feel is useful before mod-109 (retrieval eval) and mod-111 (retriever CI + drift).
- Prior work with an experiment-tracking platform (MLflow, W&B) — assumed at the `ml-engineer` level; useful when logging benchmark sweeps in mod-103 / project-102.
- Familiarity with a workflow orchestrator (Airflow, Dagster, Prefect) — useful before mod-111's ingestion DAG work; mod-111 walks the shape without assuming prior mastery.
- Familiarity with an observability platform (Grafana, Datadog, OpenTelemetry) — useful when standing up the cost / latency / drift dashboards in project-103.

## Not in scope for this track (linked out)

- **RAG *integration* at the product surface** — freshness / coverage / permission contracts consumed from a retriever, RAG-triad instrumentation at product layer, product-facing prompt shape, feature-flagged rollout of a RAG feature — owned by [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) (peer, level 30, AI Engineering family).
- **Application-layer RAG evaluation** — RAGAS, TruLens, DeepEval, RAG triad (faithfulness / answer relevance / context precision-recall), general LLM / agent tracing (LangSmith, Langfuse, Arize, Phoenix) — owned by [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) (peer, level 30, AI Engineering family, mod-105).
- **Content-safety and responsible-AI evaluation methodology** — guardrail effectiveness measurement, PII detection / redaction methodology, jailbreak / prompt-injection evaluation — owned by [`ai-risk-engineer-learning`](https://github.com/ai-governance-curriculum/ai-risk-engineer-learning) (peer, level 25, Governance family). This track covers ACL-*mechanics* only.
- **Embedding fine-tuning depth** — sentence-transformers SFT / LoRA / PEFT, MultipleNegativesRankingLoss, GISTEmbedLoss, TSDAE, domain-adaptive contrastive pretraining — owned by [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) (peer, level 30, ML Engineering family). This track covers embedding *selection* + benchmarking (mod-103) and hands off the fine-tuning brief.
- **Benchmark / statistical / judge-vs-human calibration methodology depth** — owned by [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) (peer, level 30, ML Engineering family).
- **General MLOps automation** — model registry, generic CI/CD-for-ML — owned by [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) (peer, level 25, Infra family). This track covers *retrieval-specific* MLOps (mod-111) — index versioning, retriever CI, drift monitoring keyed on retrieval metrics.
- **Agent orchestration and tool-use planning** consuming the MCP retrievers this track exposes — owned by [`agentic-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/agentic-ai-engineer-learning) (level 40, AI Engineering family).
- **Retrieval infra at platform / web scale** — GPU-accelerated ANN, sharded cross-DC replication, HA plans, streaming-ingest backed corpora (Kafka / Flink / Spark) — owned by [`ai-infra-senior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-learning) (level 40) and [`staff-ml-engineer-learning`](https://github.com/ml-engineering-curriculum/staff-ml-engineer-learning) (level 50).
- **Legal opinion** — retention / data-residency / copyright / privacy-law obligations attached to the corpus — out of scope; routed to counsel.
