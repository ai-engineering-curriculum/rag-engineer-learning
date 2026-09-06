# Exercise 03 — Index-Build Automation with Airflow or Dagster

**Estimated effort:** ~3 hours. Python, an orchestrator you can run locally (Dagster recommended for the shortest setup; Airflow, Prefect, or a cloud-native pipeline definition work with parallel wording), the corpus + judgment set from prior exercises, and the build/verify/flip scripts from exercise 01.

## Objective

Author the retrieval index-build pipeline from chapter 4 as an orchestrator-native asset graph. The graph must run in *two shapes* — incremental and full — with the correct trigger for each, produce an ingestion-lag gauge that maps to a stated SLO, and hand its output to exercise-01's cutover automation.

By the end you have:

- An asset graph (Dagster) / DAG (Airflow) / flow (Prefect) covering `raw_corpus → cleaned_docs → chunked_docs → embedded_chunks → payload_records → built_index → cutover_verified → cutover_flipped`.
- An **incremental** run that reads a corpus delta and upserts / deletes only the affected chunks in the live index.
- A **full** run that snapshots the corpus, rebuilds every asset, and hands the result to the cutover state machine.
- An **ingestion-lag** metric emitted on every run and a stated SLO (`L`) with an alert when the gauge exceeds it.
- A short write-up covering the trigger matrix (which changes fire incremental vs. full), the embedding-pass's five must-haves, and the SLO rationale.

## Prerequisites

- Chapter 4 read; chapters 2 and 7 helpful.
- Exercise 01 done (build/verify/flip scripts exist and are callable).
- An orchestrator you can run locally. Recommended:
  - **Dagster** — the asset abstraction maps 1:1 to chapter 4's graph. Install with `pip install dagster dagster-webserver`.
  - **Airflow** — install locally via `pip install apache-airflow` and run in standalone mode, or use the Astronomer local runtime.
  - **Prefect** — `pip install prefect` and run against the local server.
- A small corpus (500–5,000 chunks) that you can label with real `updated_at` timestamps. A directory of Markdown files with `mtime`-based deltas is enough.
- Python 3.10+, plus the client for your vector DB from exercise 01.

## Steps

**Step 1 — Model the asset graph (25 minutes).**

Author `assets/index_pipeline.py` (Dagster) or the equivalent `dags/index_pipeline.py` (Airflow) / `flows/index_pipeline.py` (Prefect). One node per box in chapter 4's asset graph:

- `raw_corpus` — reads the source (a directory listing is fine for this exercise).
- `cleaned_docs` — normalizes text (Unicode NFKC, whitespace collapse, HTML strip if applicable).
- `chunked_docs` — depends on `cleaned_docs` and `configs/chunker.yaml`.
- `embedded_chunks` — depends on `chunked_docs` and `configs/embedding.yaml`.
- `payload_records` — depends on `embedded_chunks` and `configs/schema.yaml`.
- `built_index` — depends on `payload_records` and `configs/ann.yaml`. Emits an `INDEX_ID` and writes the manifest.
- `cutover_verified` — depends on `built_index`; calls exercise-01's `verify.py`.
- `cutover_flipped` — depends on `cutover_verified`; calls exercise-01's `flip.py`.

Draw the graph in the orchestrator UI and screenshot it for your write-up.

**Step 2 — Wire the full-rebuild run (30 minutes).**

Materialize every asset once, top to bottom, on a snapshotted corpus (freeze the `mtime` cutoff at the start of the run). Confirm:

- Each asset's output is persisted (Dagster IO manager / Airflow XCom + object store / Prefect result storage) so a failure downstream does not re-run upstream.
- The `built_index` step writes a manifest per exercise-01's schema.
- The `cutover_verified` step calls exercise-01's `verify.py`; a verify failure short-circuits the flip.
- The `cutover_flipped` step calls exercise-01's `flip.py`; on success, the retriever picks up the new `INDEX_ID` on the next request.

Record the wall-clock per asset in the orchestrator's run log.

**Step 3 — Wire the incremental-rebuild run (35 minutes).**

Author a second entry-point that reads only a delta:

- Input: a "since" timestamp (from the last successful run's completion time).
- `raw_corpus_delta` — files whose `mtime` is greater than "since", plus a list of deleted file paths (compare current listing to a previously-persisted manifest).
- Downstream assets process only the delta; the `built_index` step performs `upsert(new_chunks)` followed by `delete(removed_chunks)` on the *live* collection (do NOT create a new collection for incrementals — the tuple's corpus-snapshot field bumps its date, but the collection name stays put).
- Skip the `cutover_verified` and `cutover_flipped` steps for incrementals; live-index mutation does not go through the state machine.
- Emit a delta manifest to `audit.log` capturing which chunk-IDs were upserted and deleted.

Confirm: after the incremental, the retriever returns new documents on a query that hits them.

**Step 4 — Implement the incremental-vs-full trigger matrix (20 minutes).**

Author `triggers/decide.py`. Given a diff over `configs/` and the corpus delta size, decide which shape to fire:

| Change | Fires |
|---|---|
| Corpus-only delta ≤ 5% of `n_chunks` | incremental |
| Corpus-only delta > 5% of `n_chunks` | incremental (but consider full for next scheduled cadence) |
| Any change to `configs/{chunker,embedding,schema,ann}.yaml` | full |
| Any change to `src/chunkers/**` (behavior-affecting) | full |
| Weekly cadence | full (safety rebuild) |

Wire the decider into the orchestrator's scheduler / sensor so a config commit triggers the full pipeline and a file drop triggers the incremental. Log the decision on every fire.

**Step 5 — Embedding pass hardening (25 minutes).**

The `embedded_chunks` asset is the dominant cost and the most-common failure. Add all five of chapter 4's must-haves:

1. **Batching.** Send chunks in batches sized to the vendor limit; if you are self-hosting, size to the model's efficient batch width. Do NOT send one chunk per request.
2. **Concurrency.** Cap concurrent batches (start at 4–8); back off on HTTP 429 or vendor-specific rate-limit signals.
3. **Retry with exponential backoff.** Use `tenacity` or the orchestrator's built-in retry primitive. Idempotency comes from the fact that the same text + model produces the same vector.
4. **Checkpointing.** Persist a `(chunk_id, vector)` write-ahead after every batch so a mid-run failure resumes from the last checkpoint, not from scratch. Dagster IO managers or Airflow's task-instance state are fine backends.
5. **Cost accounting.** Emit a `tokens_embedded` counter per batch to metrics. Sum to a run-total; log at end.

Simulate a rate-limit failure (mock the embed client to raise on the 3rd batch, twice, then succeed). Confirm the retry backs off and the checkpoint prevents duplicate re-embedding.

**Step 6 — Ingestion-lag SLO and gauge (20 minutes).**

State an SLO for your pipeline (choose based on your corpus). Reasonable defaults from chapter 4:

- `L = 24h` for a slow-moving corpus.
- `L = 1h` for support-ticket / business-hour corpora.
- `L = 5min` for chat / streaming (would need event-driven ingestion; out of scope).

Emit `ingestion_lag_seconds` on every retriever request: `now - max(payload.updated_at over all served chunks)`. Also emit it as a periodic pipeline gauge (`ingestion_lag_p95` over the last hour).

Wire a simple alert: log a WARN if the gauge exceeds `L`; log an ERROR if it exceeds `2 × L`. In production this maps to a Prometheus alertmanager rule; here, a log line is enough.

Confirm: pause the pipeline for a period longer than `L` while a new file lands; observe the alert fire; resume; observe it clear.

**Step 7 — A simulated backfill (15 minutes).**

Change the embedding config (e.g., from a small model to another small model) and let the trigger matrix fire the full pipeline. Confirm:

- The backfill runs end-to-end.
- The manifest for the new index records the changed tuple field.
- The cost accounting from step 5 gives a total-tokens-embedded number for the backfill.
- The cutover state machine flips the alias after verify passes.

Write down the backfill's total cost using vendor pricing (or your estimated self-hosted GPU cost). This is the number a stakeholder needs to sign off on before a real backfill.

**Step 8 — Write up (10 minutes).**

In `pipelines/README.md`, publish:

- The asset graph diagram (screenshot from the orchestrator UI).
- The trigger matrix as a table, with your project-specific thresholds.
- The stated SLO `L` and one paragraph on why you chose it for this corpus.
- The five embedding-pass properties and where each one is implemented.
- A short table of asset-level wall-clocks from step 2 and the simulated-backfill cost from step 7.

## Acceptance criteria

- The asset graph loads in the orchestrator UI and every arrow in chapter 4's diagram is present.
- A single command runs the full pipeline end-to-end on a snapshot; a second command runs the incremental on a delta.
- Trigger decisions in step 4 fire the correct shape on each simulated change; the decision is logged.
- The embedding pass survives an injected rate-limit failure via backoff + retry and does not re-embed previously-embedded chunks after resume.
- `ingestion_lag_seconds` is emitted on every retriever request and periodically as a pipeline gauge; the alert fires when the SLO is violated.
- A simulated backfill runs the full pipeline, publishes a new manifest, and hands to exercise-01's cutover automation.
- `pipelines/README.md` documents the graph, trigger matrix, SLO, and backfill cost estimate.

## Stretch goals

- **Dagster freshness policies.** Use Dagster's `FreshnessPolicy(maximum_lag_minutes=…)` on `embedded_chunks` and `built_index`; observe the UI's freshness indicator drive auto-materialization.
- **Airflow datasets.** Model the same graph with Airflow's `Dataset` scheduling (2.4+); a change to `raw_corpus` fires the downstream DAG automatically.
- **Reprocessing.** Change the *cleaner* (add an HTML-parser fix); mark `cleaned_docs` stale; watch the orchestrator recompute the downstream chain end-to-end.
- **Per-source-system sensor.** Replace the mtime-diff sensor with a real change-feed reader (a Notion webhook, a Confluence `since` API poll, a git `--since` diff); measure how much sooner the pipeline picks up the change.
- **Cloud-native port.** Re-implement the pipeline on Step Functions + Batch or Vertex AI Pipelines and compare LOC and observability out of the box.
- **Cost + retry telemetry to a dashboard.** Push `tokens_embedded` and `retry_count` to Prometheus; author a Grafana panel; watch the embedding-cost budget in real time.

## Notes

- **The orchestrator is not the point.** The asset graph and the trigger discipline transfer between orchestrators — this exercise is Dagster-friendly because assets map cleanly, not because Dagster is the right choice for your team. Use whichever your org uses.
- **Do not skip the incremental case.** The full-only shape is easy; the incremental (delete-then-upsert against a live collection, delta manifest, live-corpus consistency) is where production bugs actually happen.
- **Do not simulate the embedding-pass failure by short-circuiting.** Wire the actual retry / backoff / checkpoint path so a real vendor 429 (which *will* happen at 3 AM Saturday) is handled correctly.
- **The SLO is not the number your dashboard reports.** It is the promise the pipeline makes to callers. Publish it; alert on it; violate it only after conversation with the callers who depend on it.
- **Backfill sign-off is a real gate.** Before you run a real backfill, someone with the cost authority signs off on the estimate. Reinforce that habit here even though the exercise cost is trivial.
