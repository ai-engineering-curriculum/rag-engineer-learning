# Chapter 8 — The Boundaries to `ai-infra-mlops` and `ai-eval-engineer`

## Why this chapter exists

Chapter 1 named the two peer tracks. Chapters 2–7 stayed inside the retrieval-specific slice. This chapter is the contract when the retrieval-specific slice exchanges artifacts with the two peers, and the two failure modes that misdrawing the boundary produces.

1. **`ai-infra-mlops` (foundational, level 25)** owns *general* MLOps automation — orchestrators, artifact registries, CI systems, deployment strategies. This module *uses* those primitives.
2. **`ai-eval-engineer` (peer, level 30)** owns *general* LLM / agent tracing and application-layer evaluation — LangSmith, Langfuse, Arize AI, Arize Phoenix, the RAG-triad metrics (faithfulness, answer relevance, context precision-recall). This module *emits into* those platforms; it does not run them.

Getting the boundary wrong produces two symmetric failures: the rag-engineer either **over-reaches** (re-implements an orchestrator, builds a trace-viewer UI) or **under-delivers** (hands off telemetry without provenance fields, publishes CI numbers without the tuple manifest). This chapter is the artifact-shaped contract for both directions of both boundaries.

## Where this module sits between the two peers

```
                       ai-infra-mlops (foundational, level 25)
                       ──────────────────────────────────────────
                       - Airflow / Dagster / Prefect / Step Fns
                       - GitHub Actions / GitLab CI / Buildkite
                       - MLflow / Weights & Biases / model registry
                       - Terraform / Kubernetes deployment
                       - blue/green / canary as deployment shapes
                                        │
                                        │  primitives this module uses
                                        ▼
                       rag-engineer / mod-111 (this module)
                       ──────────────────────────────────────────
                       - the index tuple (mod-104 chapter 8)
                       - dual-index cutover automation (ch. 2)
                       - retriever CI + judgment-set gate (ch. 3)
                       - index-build asset graph (ch. 4)
                       - retrieval cost model (ch. 5)
                       - three-tier cache design (ch. 6)
                       - three-channel drift monitoring (ch. 7)
                                        │
                                        │  retrieval telemetry this module emits
                                        ▼
                       ai-eval-engineer (peer, level 30)
                       ──────────────────────────────────────────
                       - LangSmith / Langfuse / Arize / Phoenix
                       - end-to-end LLM span / trace correlation
                       - application-layer RAG eval (RAGAS, TruLens,
                         DeepEval — faithfulness, answer relevance,
                         context precision-recall)
                       - dashboards spanning retrieval + generation
```

The rag-engineer produces the retrieval-specific slice. The foundational MLOps track provides the primitives; the peer eval track consumes the retrieval-side telemetry (plus its own generator-side telemetry) into the app-layer dashboards.

## Boundary 1: to `ai-infra-mlops` (foundational, level 25)

### What this module owns

Everything in chapters 2–7 that is *specifically retrieval-shaped*:

- The index tuple as the versioned artifact identity.
- The dual-index cutover state machine and health checks (the *pattern*, not the general blue/green primitive).
- The retriever CI's judgment-set gate (the *retrieval-specific gate*, not the general CI system).
- The retrieval-index asset graph (`cleaned_docs → chunked_docs → embedded_chunks → payload → built_index → cutover`).
- The retrieval cost model line items.
- The retrieval telemetry field set on every request.
- The three retrieval-drift channels.

### What the foundational MLOps peer owns

The general primitives every ML-shaped project uses:

- **Orchestrators** — Airflow / Dagster / Prefect / Step Functions / Vertex AI Pipelines / etc. How to author a DAG, how to schedule it, how retries work, how backfills work in general.
- **Artifact registries and model registries** — MLflow, Weights & Biases, DVC, the general "artifact + manifest + stage" pattern. mod-104 chapter 8's manifest is an instance of this pattern.
- **CI/CD systems** — GitHub Actions, GitLab CI, Buildkite, Argo Workflows. How to author a workflow, run steps, cache artifacts, gate merges.
- **Infrastructure-as-code** — Terraform, Pulumi, CloudFormation, kubectl / Helm. How to provision the queue, the compute, the storage.
- **Deployment strategies** — blue/green, canary, shadow, feature flags at the general level. Chapter 2's dual-index cutover is a retrieval-shaped instance of blue/green.
- **Monitoring stack** — Prometheus, Grafana, OpenTelemetry, Datadog. How to author a dashboard, how to set an alert, how to route a page.

### Where the boundary lives in code

- **This module produces:** `configs/tuple.yaml`, `configs/ci-thresholds.yaml`, `.ci/build_slice_index.py`, `.ci/gate.py`, `assets/embedding_pipeline.py`, `assets/retriever_ci.py`, `dashboards/retrieval-drift.json`.
- **Foundational peer produces:** the Terraform for the Airflow cluster, the Helm chart for the retriever service, the GitHub Actions runner setup, the Prometheus scrape config, the MLflow deployment.

If you find yourself writing `terraform` or `helm` or CI-runner setup, you have crossed into the foundational MLOps peer's scope. If you find yourself writing "which fields to put in the tuple" or "which metric to gate on," you are in this module's scope.

### Confusable topics that live on this side of the boundary

Two topics look like MLOps generalities but are retrieval-specific:

- **The index tuple.** MLOps' general "versioned artifact" is a model; the retrieval instance is an index. Same shape, different fields. This module owns the retrieval fields.
- **The CI eval gate.** MLOps' general "test-gated CI" is a test suite; the retrieval instance is a judgment-set eval with paired significance. Same shape, different metric. This module owns the retrieval-specific gate.

The rule: when a topic is a specialization of a general MLOps pattern *on retrieval-shaped inputs and outputs*, this module owns it. When it's the general pattern itself, the peer owns it.

### Hand-off shape

The foundational MLOps peer provides:

- Working orchestrator (Airflow / Dagster / Prefect / cloud-native).
- Working CI system (GitHub Actions / GitLab CI / equivalent).
- Working artifact registry (MLflow / W&B / S3+manifest).
- Working monitoring stack (Prometheus / Grafana / Datadog).

This module provides:

- The retrieval-specific *asset definitions* to run on the orchestrator.
- The retrieval-specific *workflow definitions* to run in CI.
- The retrieval-specific *manifest schema* to publish to the registry.
- The retrieval-specific *dashboards and alert rules* to install in the monitoring stack.

### The failure this boundary defends against

**Over-reach.** The rag-engineer decides the existing Airflow install is "not quite right for retrieval" and starts building a home-grown orchestrator. Three months later there's an under-tested piece of infrastructure that only the rag-engineer knows and the foundational peer's general improvements to Airflow don't reach it.

**Defense.** If the orchestrator has a rough edge for retrieval, file the improvement upstream with the foundational peer. Do not fork the primitive.

## Boundary 2: to `ai-eval-engineer` (peer, level 30)

### What this module owns

- The retrieval-specific telemetry fields on every retrieval request (chapter 7): query, `INDEX_ID`, top-k doc-IDs, retrieval scores, cache-hit flags, tenant, filters, per-chunk provenance.
- The retrieval-side judgment set (mod-109) and the retriever CI gate (chapter 3) that runs on it.
- The three retrieval-drift dashboards (chapter 7).
- The rolling retrieval quality-drift alert on the judgment set.

### What the eval peer owns

The application-layer eval + tracing platform:

- **The tracing platform.** LangSmith, Langfuse, Arize AI, Arize Phoenix — the systems that ingest per-request events across the whole LLM app (embedding call, retrieval call, reranker call, generator call, downstream tool calls) and correlate them into a single trace.
- **The trace-viewer UI.** Span trees, per-request timelines, per-user history — the human interfaces to those platforms.
- **Application-layer RAG eval metrics.** RAGAS faithfulness, RAGAS answer-relevance, RAGAS context-precision-recall (which is *not* mod-109's precision@k / recall@k — see mod-109 chapter 7's confusable-metrics section), TruLens RAG triad, DeepEval faithfulness. Anything that requires the *generator's output* to compute.
- **The application-layer dashboards.** The single-pane dashboards that combine retrieval-side metrics and generator-side metrics into one view; the alerts that fire on cross-layer regressions ("recall went up but faithfulness went down").
- **General LLM span/trace correlation.** The infrastructure that ties an embedding call to a retrieval call to a generator call, correlated by request ID, across process boundaries.

### Where the boundary lives in code

- **This module produces:** `src/retriever/telemetry.py` (emits the retrieval-side fields with per-chunk provenance), `dashboards/retrieval-drift.json`, `eval/qrels.tsv`, `eval/queries.jsonl`, `.ci/gate.py` (the retriever CI gate).
- **Peer produces:** `src/observability/langsmith_client.py` (or Langfuse / Phoenix — whichever the org uses), the app-layer eval jobs that consume the peer's platform, the dashboards that combine retrieval-side and generator-side metrics.

### Confusable metrics that live on the peer's side

Two RAGAS metrics have names that look like retrieval-side ones but are application-layer:

- **RAGAS `context_precision`** is *not* this module's `precision@k`. RAGAS calls an LLM to judge whether each retrieved chunk was "useful" to the answer; mod-109's `precision@k` checks whether the chunk is in the qrels' relevant set. Different owner.
- **RAGAS `context_recall`** is *not* this module's `recall@k`. RAGAS atomizes the ground-truth answer with an LLM and checks whether each atom has a supporting chunk; mod-109's `recall@k` checks whether the qrels' relevant docs are in the top-k. Different owner.

mod-109 chapter 7 covered this in more detail; when a stakeholder says "context precision dropped," always clarify which metric they mean.

### Hand-off shape

The eval peer needs, from this module:

1. **The retrieval telemetry fields on every request** — chapter 7's list, emitted in whatever event shape the platform ingests (LangSmith event, Langfuse event, OpenTelemetry span, etc.). Per-chunk provenance is not optional.
2. **The judgment set (`eval/qrels.tsv`)** and its version — the peer's application-layer metrics that need retrieval-side truth (RAGAS `context_recall`) can consume this.
3. **The retriever's per-query top-k output for pinned eval queries** — so the peer can call the generator on the same ranked list.
4. **The `AssembledContext` shape (mod-107 chapter 4)** — the deduplicated, cite-anchored bundle the generator sees. The peer's faithfulness / answer-relevance metrics need this exact shape.

The peer hands back:

1. **Application-layer eval reports** — RAGAS faithfulness, answer-relevance, context-precision-recall over the same queries.
2. **App-layer trace dashboards** — the one-request timeline that shows retrieval + reranker + generator side-by-side. This module's dashboards link to those.
3. **Cross-layer alerts** — "retrieval nDCG@10 improved but faithfulness dropped" is a peer-owned alert; when it fires, both owners investigate.

### The failure this boundary defends against

**Over-reach.** The rag-engineer decides to install RAGAS in the retriever CI and gate on faithfulness. Three effects:

- The CI wall-clock triples (RAGAS needs generator calls).
- When faithfulness moves, no one is sure whether it was a retriever regression or a generator instability.
- When RAGAS changes its metric definition between releases (they have, several times), the retriever CI silently changes behavior on a peer-owned dependency.

**Defense.** Publish the retrieval-only metrics from the retriever CI. Link to the peer's application-layer report (which runs in a separate pipeline). If a stakeholder wants both in one dashboard, that dashboard is the peer's — this module's contribution is the retrieval panels within it.

**Under-delivery.** The retriever emits `top_k_returned` as a list of doc-IDs and no more. The peer's citation-verification metric can't map back to source URIs, section paths, or timestamps. The application-layer eval can't diagnose which chunk caused which bad answer.

**Defense.** Per-chunk provenance is a hard requirement (chapter 7). Emit it always; make missing provenance a CI failure via a schema check on the telemetry event.

## Two failures the boundaries together defend against

### Failure A: the "everything-in-one dashboard" reflex

The rag-engineer builds a single retrieval-and-generation dashboard, installs it in Grafana, and updates it manually. When retrieval telemetry moves through five iterations of the eval peer's LangSmith setup, the rag-engineer's dashboard doesn't follow. When the peer's app-layer metrics change definitions, no one updates the rag-engineer's panels. The dashboard becomes stale, everyone stops trusting it, and no one owns the fix.

**Defense.** This module's dashboards show *retrieval-only* metrics — pure recall@k, nDCG@10, MRR@10 on a pinned judgment set, plus the three drift channels. The peer's platform hosts the cross-layer dashboards; the rag-engineer's dashboards link into those.

### Failure B: "we ship without the eval peer's sign-off"

The retriever CI passes (chapter 3). The rag-engineer flips the alias. Two weeks later the eval peer notices RAGAS faithfulness on the same tuple dropped by 8 percentage points — the retriever surfaced longer, more topically-diverse chunks that hurt the generator's grounding.

**Defense.** For non-trivial retriever changes (embedding model bump, chunker rewrite, reranker swap), require the peer's app-layer eval to run *in parallel* with the retriever CI and to publish before the flip. The dual-index cutover's cool-down window (chapter 2) is the natural home for this cross-check.

## Concrete: a boundary-aware handoff artifact

When the rag-engineer promotes a new tuple, the artifact bundle includes:

```
    build-2026-08-01/
    ├── manifest.json                  # mod-104 chapter 8's manifest
    ├── retriever-ci-report.md         # chapter 3's PR-comment shape, expanded
    ├── per-query-scores.csv           # from mod-109 chapter 3
    ├── cost-report.md                 # chapter 5's cost model, updated for this tuple
    ├── drift-baseline.json            # chapter 7's re-baselined detectors
    └── handoff-to-eval-peer.md        # what the eval peer needs to know
```

The `handoff-to-eval-peer.md` is short:

```markdown
# Handoff: build-2026-08-01

## What changed
- chunker version 3 → 4 (paragraph-boundary policy changed)
- rebuilt index; INDEX_ID = hr_wiki__bge-large-en-1.5__chunker-4__schema-2__snap-2026-08-01__hnsw-M32-ef200

## What we verified (retriever CI, chapter 3)
- recall@50: baseline 0.803, new 0.811 (+0.008; Wilcoxon p = 0.128; not significant)
- nDCG@10:   baseline 0.487, new 0.481 (−0.006; Wilcoxon p = 0.253; not significant)
- MRR@10:    baseline 0.612, new 0.618 (+0.006; Wilcoxon p = 0.407; not significant)

## What we did NOT verify (deferred to the eval peer)
- RAGAS faithfulness on this tuple.
- RAGAS answer-relevance on this tuple.
- Full RAG-triad report.

## Cool-down
- Alias flip planned for 2026-08-02 09:00 UTC.
- 48-hour cool-down; drop v1 at 2026-08-04 09:00 UTC.
- Peer's app-layer numbers requested by 2026-08-03 close-of-business so we can decide on rollback within the cool-down window.
```

Every clause is on the correct side of the boundary. The peer reads it, runs the app-layer eval on the *same* tuple, and either signs off or asks for a rollback within the cool-down window. Neither side re-derives the other's work.

## What you should be able to do after this chapter

- Name what this module owns (retrieval-shaped MLOps: tuple, cutover, CI gate, asset graph, cost model, cache tiers, drift monitor) and what it does not (general orchestrator authoring, general trace-viewer UIs, application-layer RAG metrics).
- Distinguish retrieval-only `precision@k` / `recall@k` from RAGAS's application-layer `context_precision` / `context_recall`.
- Name at least three artifacts this module hands to the foundational MLOps peer (asset definitions, workflow definitions, dashboard configs) and three it hands to the eval peer (per-request telemetry with per-chunk provenance, judgment set + version, `AssembledContext` shape).
- State when a decision is inside this module's scope (retriever-only metric moved, tuple field changed) and when it belongs to a peer (orchestrator rough edge → foundational; faithfulness dropped → eval peer).
- Author a handoff artifact for a tuple promotion that includes the retriever CI numbers, the cost delta, the drift baseline, and the ask for the peer's app-layer eval.

## Further reading

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — canonical reference for the app-layer eval metrics on the peer's side of the boundary.
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) — another app-layer eval framework.
- LangSmith documentation. [`docs.smith.langchain.com`](https://docs.smith.langchain.com/) — the eval peer's canonical tracing platform.
- Langfuse documentation. [`langfuse.com/docs`](https://langfuse.com/docs) — open-source LLM tracing / eval platform in the peer's stack.
- Arize AI [`Phoenix` documentation](https://docs.arize.com/phoenix) — the eval peer's open-source tracing/eval framework.
- TruLens documentation. [`trulens.org`](https://www.trulens.org/) — canonical RAG-triad framework, peer-owned.
- DeepEval documentation. [`github.com/confident-ai/deepeval`](https://github.com/confident-ai/deepeval) — application-layer eval harness the peer runs.
- MLflow: [*Model Registry*](https://mlflow.org/docs/latest/model-registry.html) — the general artifact + manifest + stage pattern this module's index-manifest schema instantiates; the foundational peer owns the registry.
- Apache Airflow, Dagster, Prefect — the foundational peer's orchestration tools; this module authors the retrieval-specific assets that run on them.
- Google SRE Book, Ch. 8 (*Release Engineering*). [Full text](https://sre.google/sre-book/release-engineering/) — the general release-engineering discipline this module's cutover automation specializes.
- mod-109 chapter 7 — the paired-track boundary from the retrieval-eval side; a sister chapter to this one.
- The [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum) curriculum — the eval peer's track.
- The [`ai-infra-mlops`](https://github.com/ai-engineering-curriculum) curriculum — the foundational MLOps track.
