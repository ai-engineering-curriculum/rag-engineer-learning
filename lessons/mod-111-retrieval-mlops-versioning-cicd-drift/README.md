# mod-111-retrieval-mlops-versioning-cicd-drift: Retrieval MLOps: Index Versioning, CI/CD, Cost/Latency, and Drift

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Establish an index-versioning convention keyed on (embedding-model, chunker-version, schema-version, corpus-snapshot); automate dual-index cutover with health checks and rollback
- Build a retriever CI pipeline — every PR that touches chunker, embedding-model, retriever, or reranker code triggers the judgment-set eval from mod-109 and gates on a recall@k / nDCG regression threshold
- Set up index-build automation — incremental vs. full rebuild triggers, backfill runs, ingestion-lag SLOs — using Airflow / Dagster / Prefect or the cloud-native equivalent
- Measure and optimize retrieval cost + latency — embedding-API cost per doc, vector-DB storage cost per million vectors, ANN-parameter-driven latency, cache tiers (query-embedding cache, top-k result cache, generation cache), and quantization
- Monitor for retrieval drift — corpus drift (new documents, deleted documents, changed distributions), query drift (new intents, distribution shift), and quality drift (falling recall@k on a held-out judgment set)
- Cite the boundary to `ai-infra-mlops` (level 25) which owns general MLOps automation; this module owns retrieval-specific concerns
- Cite the boundary to `ai-eval-engineer` (peer, level 30) which owns general LLM/agent tracing (LangSmith, Langfuse, Arize, Phoenix); this module owns retrieval-specific telemetry (per-chunk provenance, retriever hit-rate, per-query recall)

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
