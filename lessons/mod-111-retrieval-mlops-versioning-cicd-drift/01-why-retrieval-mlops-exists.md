# Chapter 1 — Why Retrieval MLOps Exists

## Why this chapter exists

Every prior module in this track produced an artifact whose *quality is measurable* — a chunker, an embedding pass, an ANN index, a fusion recipe, a reranker window, a judgment set + metric harness. The gap this module closes is what happens *between* those artifacts and production traffic: how they get built, versioned, promoted, rolled back, monitored, and paid for.

The concrete symptom of skipping this module: a retrieval stack that looked good on the day it shipped and then, six months later, no one can answer any of these:

- Which `(embedding-model, chunker, schema, corpus-snapshot, ANN-config)` tuple built the index that is currently serving traffic?
- If a PR bumps the chunker's overlap from 40 tokens to 80, does anything block the merge if recall@50 falls by 10%?
- If the corpus grew by 12% this week, when did the last full rebuild happen and when does the next one need to run?
- What does an average query cost in embedding-API tokens, vector-DB storage, and reranker calls, and which of those three dominates?
- Has the quality regressed since last month? If yes, is it the corpus that changed, the queries that changed, or the retriever?

Every one of those questions is a **retrieval-specific MLOps** question. The general-purpose MLOps track (`ai-infra-mlops`, level 25) owns the general shape — job orchestrators, artifact registries, feature stores, model registries, deployment strategies. This module owns the *retrieval-shaped* instances of that shape: the index tuple as the versioned artifact, the judgment-set eval as the CI gate, the ingestion-lag SLO as the freshness contract, and the three-axis drift monitor (corpus × query × quality) as the production health signal.

## Where this module sits between two peers

Two peer tracks at level 30 sit next to this module, and drawing their boundaries early is what keeps the module scoped. Chapter 8 goes into detail; the short version:

- **`ai-infra-mlops` (foundational, level 25)** owns *general* MLOps automation: how to run Airflow / Dagster / Prefect, how a model registry works, what a CI pipeline is, how to structure a Terraform module for a training job. This module *uses* those primitives; it does not re-teach them. When you learn how to author a Dagster asset, that is `ai-infra-mlops`'s scope; when you learn what asset shape a retrieval index build has, that is this module's scope.
- **`ai-eval-engineer` (peer, level 30)** owns *general* LLM / agent telemetry — LangSmith, Langfuse, Arize AI, Arize Phoenix — the trace / evaluation systems that observe an application-level LLM call end to end. This module owns the *retrieval-specific* telemetry that hooks into those systems: per-chunk provenance on every retrieval, retriever hit-rate as a first-class metric, per-query recall against a held-out judgment set as a rolling gauge. The eval peer owns the tracing platform; this module owns which retrieval fields it emits.

```
                                   ai-infra-mlops (foundational, level 25)
                                   ─────────────────────────────────────────
                                   - Airflow / Dagster / Prefect
                                   - MLflow / W&B registry
                                   - Terraform for infrastructure
                                   - GitHub Actions / GitLab CI plumbing
                                   - blue/green + canary as deployment shapes
                                          │
                                          │  provides the primitives
                                          ▼
                                   rag-engineer / mod-111 (this module)
                                   ─────────────────────────────────────────
                                   - index tuple as the versioned artifact
                                   - dual-index cutover automation (mod-104 pattern, now scripted)
                                   - retriever CI with judgment-set eval gate (mod-109 harness)
                                   - incremental vs. full rebuild triggers
                                   - retrieval cost model + cache tier design
                                   - corpus / query / quality drift monitor
                                          │
                                          │  emits retrieval telemetry
                                          ▼
                                   ai-eval-engineer (peer, level 30)
                                   ─────────────────────────────────────────
                                   - LangSmith / Langfuse / Arize Phoenix
                                   - trace correlation across the full LLM call
                                   - application-layer eval dashboards
                                   - alerting on RAG-triad regressions
```

Getting the boundaries wrong is the most common way to bloat this module. If you find yourself writing "how to author an Airflow DAG from scratch," you have crossed into `ai-infra-mlops`. If you find yourself writing "how LangSmith's span tree renders," you have crossed into `ai-eval-engineer`. Chapter 8 is the contract with both peers; chapters 2–7 stay inside the retrieval slice.

## What retrieval-shaped MLOps actually is — five surfaces

Chapters 2–7 each pick one of these surfaces and go deep on the retrieval instance of it.

1. **Index versioning + cutover automation** — the `(embedding-model, chunker-version, schema-version, corpus-snapshot, ANN-config)` tuple as the versioned unit; dual-index build + verify + flip + cool-down as the deployment shape; health checks and rollback as first-class states. mod-104 chapter 8 defined the *convention*; chapter 2 of this module turns it into scripted automation.
2. **Retriever CI** — every PR that changes the chunker, embedding config, retriever, or reranker triggers the judgment-set eval from mod-109, and gates on a recall@k / nDCG@k regression threshold with paired significance. Chapter 3 builds it.
3. **Index-build automation** — the batch job that turns corpus deltas into new index chunks. Incremental vs. full rebuild triggers, backfill runs, ingestion-lag SLOs, orchestrator choice (Airflow / Dagster / Prefect / cloud-native). Chapter 4 covers it.
4. **Cost + latency optimization** — embedding-API cost per doc, vector-DB storage cost per million vectors, ANN parameters as latency levers, cache tiers (query-embedding cache, top-k result cache, generation cache), and quantization as a cost-vs-recall trade. Chapters 5 and 6.
5. **Drift monitoring** — three orthogonal drifts (corpus, query, quality) with independent detectors and alert thresholds. Chapter 7.

Each of the five is a real production incident category — teams that skip them find each one the expensive way.

## Two failures that motivate the discipline

### Failure 1: "the index built fine and then quality got worse and no one can say why"

An unlogged embedding-model bump on the hosted side, a chunker PR that silently reverted an overlap change, a nightly rebuild that dropped a subset of chunks because a parser exception was swallowed — all of these produce the same symptom: **quality regressed, but no artifact, no CI run, no log line tells you which build did it**.

Chapters 2 and 3 stop this. The index tuple + manifest gives you the "which build" answer. The retriever CI with the judgment-set gate refuses to let a regressing PR merge in the first place. Chapter 4's build-automation logs every rebuild against its inputs so an incident timeline is a `grep` away.

### Failure 2: "the cost of retrieval got 4× and we're not sure which line item"

Corpus grew, embedding pass re-ran nightly instead of incrementally, the reranker call fired even for cache-hittable repeats, the vector DB was over-provisioned for peak-of-peak because storage was hot rather than warm. Each contributes a fraction; the aggregate is a bill that surprises. This module's answer is a **cost model with a per-query breakdown** (chapter 5) and a **three-tier cache** (chapter 6) that closes each shortfall.

## What this module does *not* teach

- **Writing an Airflow DAG from scratch, running an EKS cluster, structuring a Terraform module.** All `ai-infra-mlops` (level 25) — this module assumes them.
- **LangSmith / Langfuse / Phoenix trace UIs, dashboard authoring, general LLM span correlation.** All `ai-eval-engineer` (peer, level 30). This module emits the retrieval fields; the peer runs the platform.
- **Application-layer RAG eval (RAGAS faithfulness, TruLens RAG triad).** `ai-eval-engineer` again. This module's CI gate uses the retrieval-only metrics from mod-109.
- **Deep MLOps deployment strategies at platform scale** (cross-region replication of the index, per-tenant sharding at scale, hot-standby failover across DCs). `ai-infra-senior-engineer` at level 40; this module covers the single-DC dual-index case.
- **Kubernetes autoscaling of retriever pods.** Level 40 platform work; touched only at the boundary in chapter 8.

## The rest of the module

- **Chapter 2** — the index-tuple convention as *scripted* automation: dual-index cutover with health checks and rollback as a state machine, alias-flip scripts per engine, cool-down policy, and the `INDEX_ID` config contract.
- **Chapter 3** — the retriever CI pipeline: what a PR touches that triggers it, how the judgment-set eval from mod-109 runs against a rebuilt index, the recall@k / nDCG@k regression threshold, paired-significance gating (Smucker et al. 2007), and how the CI's artifact caches keep it under a wall-clock budget.
- **Chapter 4** — index-build automation with Airflow / Dagster / Prefect (and the cloud-native equivalents): the DAG / asset shape, incremental vs. full rebuild triggers, backfill and reprocessing, and an ingestion-lag SLO.
- **Chapter 5** — the retrieval cost model: embedding-API cost per doc, vector-DB storage cost per million vectors, ANN-parameter-driven latency, and quantization as a cost-vs-recall trade.
- **Chapter 6** — the three cache tiers: query-embedding cache, top-k result cache, generation cache. TTL, invalidation on index cutover, and how the tiers interact.
- **Chapter 7** — retrieval drift monitoring: corpus drift, query drift, and quality drift as three independent detectors with independent alert thresholds and a shared incident-response runbook.
- **Chapter 8** — the boundaries: what this module hands to `ai-infra-mlops` (foundational, level 25) and to `ai-eval-engineer` (peer, level 30), and what each peer hands back.

## What you should be able to do after this chapter

- Name the five retrieval-MLOps surfaces (index versioning, retriever CI, index-build automation, cost+latency, drift monitoring) and the chapter of this module that owns each.
- Draw the boundary to `ai-infra-mlops` (general MLOps primitives) and to `ai-eval-engineer` (general LLM tracing) and name at least one artifact this module hands to each.
- State the two motivating failures (un-diagnosable quality regression, un-attributed cost blowup) and which chapters are the defense against each.
- Explain why the index tuple from mod-104 chapter 8 is a *pre-requisite* for the CI machinery in chapter 3 rather than an alternative to it.

## Further reading

- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. [PDF](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — the canonical taxonomy of the failures MLOps disciplines defend against; every failure category has a retrieval-shaped instance.
- Google Cloud Architecture Center: [*MLOps: Continuous delivery and automation pipelines in machine learning*](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — the general MLOps reference this module's retrieval-specific patterns specialize.
- Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly. Chapters 8 (data distribution shift) and 10 (infrastructure) are the general treatment; chapters 5 and 7 of this module are the retrieval-specific instances.
- The `ai-infra-mlops` track curriculum — the foundational MLOps peer at level 25 whose primitives this module composes.
- The `ai-eval-engineer` track curriculum — the observability / eval peer at level 30 whose platforms this module emits into.
