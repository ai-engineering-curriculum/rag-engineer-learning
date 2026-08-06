# Chapter 4 — Where a Retrieval Engineer Sits: Peers, Levels, and Ownership Contract

## Why this chapter exists

Chapter 3 named the shape difference between feature-RAG and engineered-system-RAG. This chapter names *who does what* in an organization that has both — because a retrieval engineer whose scope is unclear either gets pulled into product-facing prompt tuning at the expense of the retriever, or reinvents an eval framework the eval team already owns. Both patterns show up in real teams and both are avoidable by writing the ownership contract down.

The picture below is the AI Engineering family view; the retrieval role you are training for sits at level 30, has peers at level 30, and is bounded above by roles at level 40+.

## The level ladder around this role

```
                                            ↑
                                      level 60  ai-infra-principal-engineer     — cross-org retrieval-platform architecture
                                      level 50  staff-ml-engineer               — retrieval under web-scale streaming ingest
                                      level 40  agentic-ai-engineer             — agent orchestration consuming MCP retrievers
                                      level 40  ai-infra-senior-engineer        — retrieval-serving infra at platform scale
                                      ────────────────────────────────────────
                                *YOU → level 30  rag-engineer                    — retrieval-engineering ownership
                                      level 30  applied-ai-engineer  (peer)     — product-surface RAG integration
                                      level 30  ai-eval-engineer     (peer)     — application-layer RAG evaluation
                                      level 30  fine-tuning-engineer (peer, ML) — embedding SFT / LoRA / PEFT
                                      level 30  model-evaluation-engineer (peer, ML) — benchmark & statistical depth
                                      level 25  ai-risk-engineer     (peer, Gov)— content-safety, PII, jailbreak eval
                                      level 25  ai-infra-mlops       (peer, Inf)— general MLOps automation
                                      ────────────────────────────────────────
                                      level 20  llm-application-developer       — retrieval basics, prompt eng., FastAPI
                                      level 20  ml-engineer                     — classical ML, tf-idf/BM25 primer, HF/PyTorch
                                      level 10  ai-infra-junior-engineer        — Linux, Git, Python, HTTP APIs, Docker
                                            ↓
```

You inherit engineering craft and retrieval basics from the level-10 and level-20 tracks. You defer *sideways* to your level-30 peers for anything outside retrieval engineering. You defer *up* to level-40+ when scope crosses into agents or platform infra.

## What this role owns

The `rag-engineer` role owns **the engineering craft of retrieval end-to-end**, as an artifact other product surfaces consume:

- Document ingestion and chunker design (mod-102).
- Embedding-model selection and benchmarking on the corpus (mod-103).
- Vector-database internals and ANN tuning (mod-104).
- Lucene-family engines and classical LTR (mod-105).
- Hybrid retrieval, fusion, and query rewriting (mod-106).
- Reranking and context assembly (mod-107).
- Retrieval-specific prompting — citation, refusal, contradiction (mod-108).
- **Retrieval-only** evaluation — recall@k / MRR / nDCG on judgment sets, paired significance testing, ablation attribution (mod-109).
- Retrieval orchestration and MCP-exposure of retrievers as tools (mod-110).
- Retrieval MLOps — index versioning, retriever CI, ingestion DAGs, cost / latency, drift monitoring (mod-111).
- Enterprise retrieval — ACL-aware, structured (text-to-SQL), Graph RAG, multimodal (mod-112).

The output artifact this role ships is not "a demo answer" — it is **a tuned retriever with a stable interface (typically MCP), a versioned index, a judgment-set-backed quality claim, a cost + latency claim, and an on-call runbook**.

## What this role defers *down*

The following capabilities are prerequisites, not scope; the track expects you to arrive with them:

- **Engineering craft** (Linux, Git, Python, packaging, HTTP APIs, Docker) — `ai-infra-junior-engineer-learning` (level 10).
- **Classical ML fundamentals and the tf-idf / BM25 primer** — `ml-engineer-learning` (level 20). Chapter 2 refreshes; it does not re-teach.
- **Retrieval basics** — hosted embedding APIs, pgvector top-*k*, grounded prompts with citations, retrieval-vs-prompt triage — `llm-application-developer-learning` (level 20, mod-005). This track builds on top of that shape; the seven-stage production pipeline in chapter 1 is a superset of that mod-005 five-stage minimal shape.
- **Production Python service craft** — FastAPI, asyncio, streaming, retries, circuit-breakers — `llm-application-developer-learning` (level 20).
- **General prompt engineering** — same track.

## What this role defers *sideways* — the four contracts to write down

Every one of these boundaries has been a real inter-team conflict. Naming them up front avoids it.

### 1. → `applied-ai-engineer-learning` (peer, level 30, AI Engineering family)

They own **RAG *integration* at the product surface**. That includes:

- The freshness / coverage / permission contract the product surface *consumes* from your retriever.
- RAG-triad instrumentation at the *product* layer.
- Product-facing prompt shape, including how citations render in the UI, refusal copy, and the "insufficient information" fallback message.
- Feature-flagged rollout of a RAG feature to a user cohort.

You give them a retriever with a stable interface, a documented quality/latency/cost claim, and a support contract. They own the product decision of when to call it and how the result becomes an answer the user reads.

### 2. → `ai-eval-engineer-learning` (peer, level 30, AI Engineering family)

They own **application-layer LLM and RAG evaluation**. That includes:

- RAGAS, TruLens, DeepEval — the tooling for application-layer evaluation.
- The RAG-triad: **faithfulness, answer relevance, context precision-recall** at the *application* level.
- General LLM / agent tracing platforms — LangSmith, Langfuse, Arize, Phoenix.
- LLM-judge calibration for application eval.

You own **retrieval-only eval**: `recall@k`, `precision@k`, `MRR`, `MAP`, `nDCG@k`, and paired significance testing on judgment sets. The two evaluations are *both* required in a production RAG stack, and the boundary is: **retrieval-only eval attributes to your artifact; application eval attributes to theirs**.

### 3. → `fine-tuning-engineer-learning` (peer, level 30, ML Engineering family)

They own **embedding fine-tuning depth**:

- sentence-transformers SFT, LoRA, PEFT.
- Loss functions such as `MultipleNegativesRankingLoss`, `GISTEmbedLoss`, `TSDAE`.
- Domain-adaptive contrastive pretraining.

You *select* and *evaluate* embedding models on the corpus (mod-103). When the selection benchmark says "the best off-the-shelf option is still X points short of what we need," you hand off a brief — corpus, current best model, target metric — to a fine-tuning engineer. You do not train the LoRA adapter yourself as part of the retrieval-engineering role.

### 4. → `ai-risk-engineer-learning` (peer, level 25, Governance family) and `ai-infra-mlops-learning` (peer, level 25, Infra family)

- `ai-risk-engineer` owns **content-safety and responsible-AI methodology** — guardrail effectiveness measurement, PII detection / redaction methodology, jailbreak / prompt-injection evaluation. You cover ACL *mechanics* (mod-112); they cover *methodology*.
- `ai-infra-mlops` owns **general MLOps automation** — model registry, generic CI/CD-for-ML. You cover *retrieval-specific* MLOps (mod-111) — index versioning, retriever CI, drift monitoring keyed on retrieval metrics.

## What this role defers *up*

- **`agentic-ai-engineer-learning` (level 40)** — agent orchestration and tool-use planning *consuming* the MCP retrievers this track builds. When the design question is "how does the agent decide when to call the retriever and how many times," it is theirs.
- **`ai-infra-senior-engineer-learning` (level 40)** — retrieval-serving infra at platform scale (GPU-accelerated ANN, sharded cross-DC replication, HA plans, GPU / accelerator budgeting for embeddings).
- **`staff-ml-engineer-learning` (level 50)** — retrieval under web-scale streaming ingest (Kafka / Flink / Spark backing corpora), retrieval as one component of a larger ML platform.
- **`ai-infra-principal-engineer-learning` (level 60)** — cross-org retrieval-platform architecture, standardization across many product areas.

## The ownership contract — a template

The exercise for this chapter asks you to write this out for a specific team. The template:

```markdown
## Retriever ownership contract — v1

Owned by: retrieval-engineering (this role)
Consumers: <team A>, <team B>

The retriever this team ships:
  - Interface: <MCP tool schema | HTTP endpoint spec>
  - Quality claim: recall@10 = X on judgment set v<N>, paired-significance p < 0.05 vs. previous version
  - Latency claim: p50 = X ms, p95 = Y ms measured under <workload>
  - Cost claim: $Z per 1M queries at current cache hit ratio W%
  - Freshness claim: corpus lag ≤ T hours; SLO on ingestion pipeline
  - Permission claim: per-tenant isolation via <mechanism>; ACL model matches <source>

Not this team's scope; delegated to:
  - Product-surface prompt / UX / rollout → applied-ai-engineer
  - Application-layer RAG-triad / tracing → ai-eval-engineer
  - Embedding fine-tuning (SFT / LoRA / PEFT) → fine-tuning-engineer
  - Content-safety / PII methodology → ai-risk-engineer
  - Agent orchestration / when-to-call → agentic-ai-engineer
  - Platform-scale serving infra / HA → ai-infra-senior-engineer

Escalation triggers (we own, but pull the delegated party in):
  - Retrieval quality regression discovered in production → ai-eval-engineer
  - Embedding candidate-set fully exhausted, gap remains → fine-tuning-engineer
  - New corpus source with a fundamentally different ACL model → ai-risk-engineer
```

## What you should be able to do after this chapter

- Place your role on the level ladder relative to each peer and each higher role.
- State, without looking, which two peers own the two adjacent capabilities most often confused with retrieval engineering (product-surface integration and application-layer eval).
- Write a one-page ownership contract for a hypothetical team, listing what you own, what you defer, and the escalation triggers.

## Further reading

- The track root [`CURRICULUM.md`](../../CURRICULUM.md), section *Ownership rule*, is the canonical statement of the deferral graph above.
- The track root [`PREREQUISITES.md`](../../PREREQUISITES.md), section *Not in scope for this track (linked out)*, is the shorter version of the delegation contract for a learner deciding whether this is the right track.
