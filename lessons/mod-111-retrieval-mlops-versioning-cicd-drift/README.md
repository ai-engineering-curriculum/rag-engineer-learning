# mod-111 · Retrieval MLOps — Index Versioning, CI/CD, Cost/Latency, and Drift

Eleventh module in the RAG & Retrieval Engineer track. Takes the artifacts every earlier module produced — a chunker, an embedding pass, an ANN index, a fusion recipe, a reranker window, a judgment set + metric harness — and puts a *production discipline* around them: how those artifacts are versioned, promoted, rolled back, monitored, and paid for. This is the module where a "good retriever" becomes a system that stays good six months later.

**Estimated effort:** 14 hours.

## Learning objectives

By the end of this module a learner can:

- Establish an index-versioning convention keyed on `(embedding-model, chunker-version, schema-version, corpus-snapshot, ANN-config)` and automate dual-index cutover with health checks and rollback.
- Build a retriever CI pipeline — every PR that touches chunker, embedding config, retriever, or reranker code triggers the mod-109 judgment-set eval and gates on a recall@k / nDCG@k regression threshold with paired significance.
- Set up index-build automation — incremental vs. full rebuild triggers, backfill runs, an ingestion-lag SLO — on Airflow / Dagster / Prefect or the cloud-native equivalent.
- Measure and optimize retrieval cost and latency across the three cost line items (embedding, vector-DB storage + compute, reranker) and the three latency line items (query-embed, ANN, rerank + generate). Use quantization and ANN parameters as the primary levers.
- Design the three cache tiers — query-embedding cache, top-k result cache, generation cache — with correct keys, TTLs, invalidation hooks, and stampede protection.
- Monitor for retrieval drift across three orthogonal channels — corpus drift, query drift, and quality drift — with independent alert thresholds and a cross-drift triage runbook.
- Cite the boundary to `ai-infra-mlops` (foundational, level 25) which owns *general* MLOps primitives (orchestrators, CI systems, artifact registries, monitoring stacks); this module owns the *retrieval-specific* instances.
- Cite the boundary to `ai-eval-engineer` (peer, level 30) which owns *general* LLM / agent tracing (LangSmith, Langfuse, Arize, Phoenix) and application-layer RAG eval (RAGAS faithfulness, TruLens RAG triad); this module owns *retrieval-specific* telemetry (per-chunk provenance, retriever hit-rate, per-query recall on a pinned judgment set).

## Prerequisites

- All prior modules in this track. In particular:
  - `mod-104-vector-databases-and-ann-tuning` — chapter 8's index tuple and manifest schema are the versioning primitive this module automates; chapters 4 and 5 are the ANN-parameter latency levers chapter 5 of this module reuses.
  - `mod-107-reranking-and-context-assembly` — chapter 5's funnel curve is the reranker-window latency lever chapter 5 of this module reuses; chapter 4's `AssembledContext` shape is the hand-off to `ai-eval-engineer`.
  - `mod-109-retrieval-quality-evaluation` — the judgment set, the metric triple, and chapter 4's paired-significance test are the engine both the CI gate (chapter 3) and the drift alert (chapter 7) run on.
- Comfort with a CI system (GitHub Actions is used in the examples; GitLab CI or Buildkite work with parallel wording).
- Familiarity with one workflow orchestrator (Airflow / Dagster / Prefect / Step Functions / Vertex AI Pipelines). Authoring the DAG shape is `ai-infra-mlops`'s scope; this module assumes you can.
- A local vector DB you can script (pgvector / Qdrant / Weaviate / Chroma) for the cutover and cache exercises.

## Structure

| Path | What it is |
|---|---|
| [`01-why-retrieval-mlops-exists.md`](01-why-retrieval-mlops-exists.md) | Five retrieval-MLOps surfaces, two motivating failures, and the boundaries to the two peer tracks. |
| [`02-scripted-dual-index-cutover-with-health-checks-and-rollback.md`](02-scripted-dual-index-cutover-with-health-checks-and-rollback.md) | The five-state cutover as scripted automation; health-check families; the `INDEX_ID` request-time contract; rollback vs. discard. |
| [`03-retriever-ci-with-a-judgment-set-eval-gate.md`](03-retriever-ci-with-a-judgment-set-eval-gate.md) | Path-filtered CI, tuple-keyed index cache, mod-109 eval, paired-significance gate, PR-comment shape. |
| [`04-index-build-automation-with-airflow-dagster-or-prefect.md`](04-index-build-automation-with-airflow-dagster-or-prefect.md) | The retrieval asset graph; incremental-vs-full trigger matrix; embedding-pass shape; ingestion-lag SLO. |
| [`05-retrieval-cost-and-latency-optimization.md`](05-retrieval-cost-and-latency-optimization.md) | Per-query cost model; the six line items; quantization tiers; ANN parameters as latency knobs; the three-way trade. |
| [`06-cache-tiers-query-embedding-top-k-and-generation.md`](06-cache-tiers-query-embedding-top-k-and-generation.md) | Three cache tiers with keys, TTLs, invalidation on cutover, single-flight, sizing. |
| [`07-monitoring-corpus-query-and-quality-drift.md`](07-monitoring-corpus-query-and-quality-drift.md) | Corpus, query, and quality drift as three independent detectors; rolling paired-significance alert; per-chunk provenance. |
| [`08-boundaries-to-ai-infra-mlops-and-ai-eval-engineer.md`](08-boundaries-to-ai-infra-mlops-and-ai-eval-engineer.md) | What this module owns vs. what the two peers own; hand-off artifacts in both directions. |
| [`exercises/`](exercises/) | Six hands-on prompts — one per major objective. |
| [`labs/`](labs/) | Long-form hands-on labs (scaffolded separately). |
| [`quizzes/`](quizzes/) | Knowledge check (scaffolded separately). |
| [`resources.md`](resources.md) | Primary-source references for the topics covered. |

## How to work through it

1. Read chapter 01 first. It states the five surfaces the rest of the module covers and draws the two peer boundaries; every later chapter refers back to that frame.
2. Read chapters 02 and 03 back-to-back, then do `exercises/exercise-01-…` and `exercises/exercise-02-…`. Chapter 02's cutover state machine is what chapter 03's CI ultimately feeds a promoted tuple into; running both drills gives you a working retriever-promotion pipeline end to end.
3. Read chapter 04 and do `exercises/exercise-03-…`. This is where the "how does a rebuild actually happen" plumbing lives; the asset graph you build here is what chapter 07's drift monitor observes.
4. Read chapters 05 and 06 together, then do `exercises/exercise-04-…` and `exercises/exercise-05-…`. Cost and cache design are two sides of the same lever set; a shipped optimization touches both.
5. Read chapter 07 and do `exercises/exercise-06-…`. The drift monitor is the always-on defense that catches what the CI gate and cutover verify can't — the unplanned regressions.
6. Read chapter 08 last. With direct experience of the retriever-side artifacts, the boundary contract to the foundational MLOps peer and the eval peer reads as a checklist rather than as theory.

## Ownership boundary

This module owns the *retrieval-shaped* MLOps discipline: the index tuple as the versioned artifact, the cutover state machine, the judgment-set-eval CI gate, the retrieval-index asset graph, the per-query cost model, the three-tier cache design, and the three-channel drift monitor.

It does **not** own the general primitives that host all of this. The **foundational MLOps peer** ([`ai-infra-mlops`](https://github.com/ai-engineering-curriculum) at level 25) owns orchestrators, CI systems, artifact registries, infrastructure-as-code, and the general monitoring stack. It also does **not** own application-layer eval or general LLM tracing. The **eval peer** ([`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum) at level 30) owns LangSmith / Langfuse / Arize / Phoenix, RAGAS / TruLens / DeepEval, and the cross-layer dashboards that combine retrieval and generation metrics. Chapter 08 draws both lines explicitly with hand-off artifact shapes.

## Primary sources

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
- Sato, D., Wider, A., & Windheuser, C. (2019). *Continuous Delivery for Machine Learning.* martinfowler.com. <https://martinfowler.com/articles/cd4ml.html>
- Fowler, M. (2010). *BlueGreenDeployment.* <https://martinfowler.com/bliki/BlueGreenDeployment.html>
- Google. *Site Reliability Engineering* — chapters on SLOs, Monitoring, and Release Engineering. <https://sre.google/sre-book/table-of-contents/>
- Google Cloud Architecture Center. *MLOps: Continuous delivery and automation pipelines in machine learning.* <https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM. <https://dl.acm.org/doi/10.1145/1321440.1321528>
- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS. <https://arxiv.org/abs/1810.11953>

Further reading — chapter-by-chapter and grouped by topic — is in [`resources.md`](resources.md).
