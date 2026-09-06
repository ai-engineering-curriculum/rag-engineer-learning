# Exercise 06 — Corpus / Query / Quality Drift Monitor with Alert Thresholds

**Estimated effort:** ~3 hours. Python, the mod-109 judgment set, the retriever pipeline from exercises 04 and 05, the ingest pipeline from exercise 03, and a metrics backend you can chart against (Prometheus + Grafana locally, or plain SQLite + a Jupyter notebook — the shape matters more than the tool).

## Objective

Build the three drift detectors from chapter 7 — corpus, query, and quality — each with its own gauges, its own alert threshold calibrated against a measured noise floor rather than an intuited constant, and a cross-drift triage runbook. Emit the retrieval-side telemetry field set (including per-chunk provenance) so the eval peer's tracing platform could ingest it. Simulate each of the three drift conditions and confirm the detector fires with the correct root-cause attribution.

By the end you have:

- `drift/corpus.py`, `drift/query.py`, `drift/quality.py` — the three detectors as callable modules.
- `drift/telemetry.py` — the retrieval-side field-set emitter, including per-chunk provenance.
- `drift/thresholds.yaml` — the alert thresholds with a comment showing the noise measurement that set each.
- `drift/dashboard.md` (or a Grafana JSON export) — the three-panel dashboard shape.
- `drift/runbook.md` — the cross-drift triage runbook.
- Three recorded simulations (corpus, query, quality) with the detector firing and the runbook diagnosis.

## Prerequisites

- Chapter 7 read; chapters 2, 3, and 4 helpful.
- Exercise 03 done (ingest pipeline producing an audit log of adds / updates / deletes).
- Exercise 04 done (retrieval-side telemetry with `INDEX_ID`, top-k, latency).
- Exercise 05 done (cache-hit flags in the telemetry field set).
- mod-109 chapter 4 read (paired-significance testing).
- Python 3.10+, `pip install scipy numpy scikit-learn` (for KS / MMD; `alibi-detect` is optional but recommended).

## Steps

**Step 1 — Extend the telemetry field set (20 minutes).**

Update the retriever from exercise 04 to emit chapter 7's complete field list on every request:

- `query_text` (hashed if you treat query text as sensitive)
- `query_normalized`
- `query_embedding` (or an ID pointing to it if the payload is large)
- `INDEX_ID`
- `top_k_returned` — list of `{chunk_id, doc_id, source_uri, section_path, updated_at, retrieval_score, rerank_score}` — **per-chunk provenance is not optional.**
- `retriever_latency_ms`, `reranker_latency_ms`
- `cache_hit_query_embed`, `cache_hit_top_k`, `cache_hit_generation`
- `tenant_id`
- `filters_applied`
- `request_id`

Assert with a schema check (`pydantic` or `jsonschema`) that every telemetry event contains every field. Refuse to emit an event that fails the schema; a failing schema check is a CI failure — this is chapter 8's defense against the "under-delivery to the eval peer" boundary failure.

**Step 2 — Corpus-drift detector (30 minutes).**

`drift/corpus.py` reads the ingest audit log (exercise 03) and the current index-manifest and emits daily gauges:

- **Volume gauges.** `n_documents`, `n_chunks`, `avg_chunks_per_doc`, `avg_tokens_per_chunk`.
- **Rate gauges.** `documents_added_today`, `documents_updated_today`, `documents_deleted_today`.
- **Content-change gauge.** Fraction of documents whose content hash changed today.
- **Length-distribution shift.** KS statistic between today's chunk-length distribution and a rolling 30-day baseline (`scipy.stats.ks_2samp`).
- **Embedding-space centroid shift.** Cosine distance between today's newly-added-doc mean embedding and the reference mean.

Calibrate the thresholds by measuring the noise floor:

1. For a period of stable operation (or by resampling the historical ingest log), compute each gauge's daily value.
2. Compute the standard deviation of each gauge across the stable window.
3. Set the alert threshold at ~3σ above the rolling baseline (or the equivalent p-value for the distributional tests).
4. Write each threshold into `drift/thresholds.yaml` with a comment showing the measured noise floor and the date it was measured.

**Step 3 — Query-drift detector (30 minutes).**

`drift/query.py` reads the retriever-side telemetry and emits:

- **Shape gauges.** `query_length_p50 / p95`, `distinct_normalized_queries_per_day`.
- **Zero-result rate.** Fraction of queries whose `top_k_returned` was empty.
- **Reformulation rate** (if you have session tracking; otherwise skip and note in the write-up).
- **New-intent rate.** Cluster the query embeddings with KMeans (or reuse a saved sklearn model); track the fraction of queries falling outside the known clusters (distance to nearest centroid above a threshold).
- **Distributional shift on query-embedding space.** KS or MMD test between today's query embeddings and the 30-day baseline.

Calibrate as in step 2. Give the zero-result rate the tightest threshold — chapter 7 flags it as the highest-signal alert; a 2× rise from baseline should page.

**Step 4 — Quality-drift detector (25 minutes).**

`drift/quality.py` runs nightly and computes:

- The mod-109 metric triple against the pinned judgment set at the current `INDEX_ID`.
- Per-query values (not just aggregates) so the significance test can read them.
- The rolling paired-Wilcoxon `p` between the last 7-day window's per-query values and the previous 7-day window's (chapter 7's "gate on the trend, not on a single-day drop").

Wire the alert:

- If Wilcoxon `p < 0.05` and the mean delta is negative → page.
- If `p ≥ 0.05` → no alert.
- If any single-day metric drops by more than a hard threshold (`ε_hard` from exercise 02) → page immediately (independent of the trend test).

Persist per-run per-query values to disk / your metrics backend so historical windows are recomputable.

**Step 5 — Build the dashboard (15 minutes).**

`drift/dashboard.md` (or a Grafana JSON export). One page with three panels:

- **Corpus drift panel.** Volume, rates, content-change, distributional gauges. Include the 30-day baseline and the alert-threshold overlay.
- **Query drift panel.** Query shape, zero-result rate, new-intent rate, embedding-space shift.
- **Quality drift panel.** `recall@50`, `nDCG@10`, `MRR@10` on the nightly rolling window; the paired-Wilcoxon `p` value; per-slice tabs if you have a stratified judgment set.

Every panel links to `drift/runbook.md`.

**Step 6 — Author the cross-drift runbook (20 minutes).**

`drift/runbook.md` is the operational artifact. Chapter 7's cross-drift correlation section is the template:

- **Corpus drift up + quality drift down** → the new documents retrieve worse; investigate the ingest step; check for parser regressions (chapter 4 Failure A).
- **Query drift up + quality drift down** → users are asking new things the retriever handles poorly; the judgment set is stale; expand it (mod-109 chapter 2 protocol B).
- **Quality drift down + corpus flat + query flat** → something silent moved; diff the tuple manifests (chapter 2); check for a hosted-model revision bump, a schema drift, a downstream cache misconfig.
- **Corpus volume spike** → check whether the ANN parameters are still right at the new scale (chapter 7 Failure B).

For each row, include: what to check, the artifact to consult, and the escalation path.

**Step 7 — Simulate the three drifts (30 minutes).**

Run each simulation independently and record the detector's response.

- **Simulation A — corpus drift.** Bulk-ingest 20% more documents in one day (repeat existing docs with a new hash; append random text; whatever produces the shape). Confirm the corpus-drift alert fires on the volume + content-change gauges. Follow the runbook; diagnose to "large content-change fraction; recommend full rebuild."
- **Simulation B — query drift.** Replay a set of queries whose embeddings sit outside your baseline clusters (translate to a different language, use jargon-heavy queries, generate paraphrases in a new register). Confirm the new-intent-rate alert fires and the zero-result rate rises if the corpus doesn't cover the new intents.
- **Simulation C — quality drift.** Downshift the embedding model (or lower `efSearch` aggressively) so recall drops by 0.02–0.05 across the judgment set. Run the nightly eval for several simulated windows so the rolling Wilcoxon has data. Confirm the alert fires when the drop becomes significant.

For each simulation, record:

- Which detector fired.
- The runbook diagnosis.
- The action taken (in real life; you don't need to execute it here — noting "would trigger full rebuild" is enough).

**Step 8 — Emit into the eval peer's platform (10 minutes).**

Chapter 8's hand-off: the retrieval telemetry lands in `ai-eval-engineer`'s tracing platform. Pick one platform:

- LangSmith: `pip install langsmith`; wrap the retriever call in a `@traceable` decorator; attach the telemetry field set as span attributes.
- Langfuse: `pip install langfuse`; use the `@observe` decorator; attach fields.
- Arize Phoenix: `pip install arize-phoenix`; use the OpenTelemetry integration.

Fire 10 requests. Screenshot the platform showing the retrieval span with the per-chunk provenance attached. This is the artifact the eval peer will use — verify it's ingestible.

**Step 9 — Write up (10 minutes).**

In `drift/README.md`:

- The three detectors with their gauges and thresholds.
- The noise-floor measurement for each threshold (or a `<!-- needs-research: measure the noise floor for the corpus centroid-shift gauge on a stable corpus window -->` note if you couldn't run it).
- The dashboard structure.
- The runbook cross-drift matrix.
- The three simulation results.
- The eval-peer hand-off (which platform, which field the provenance lives on).

## Acceptance criteria

- The retriever emits every chapter-7 telemetry field on every request, verified by a schema check.
- The three drift detectors run and populate their gauges from real data.
- `drift/thresholds.yaml` documents each threshold with the noise-floor measurement behind it (or a `needs-research` note).
- The quality-drift alert uses the rolling paired-Wilcoxon test, not a raw-drop-only heuristic.
- The three-panel dashboard exists and each panel links to the runbook.
- The runbook covers at least the four cross-drift scenarios in chapter 7.
- All three simulations fire the correct detector; the runbook diagnosis matches the root cause.
- One eval-peer platform ingests the telemetry and shows the per-chunk provenance.

## Stretch goals

- **Alibi Detect integration.** Replace the hand-rolled KS test with Alibi Detect's `KSDrift`, `MMDDrift`, or `TabularDrift` classes for a more mature detector implementation.
- **Per-slice quality drift.** If the judgment set is stratified (mod-109 chapter 2's stretch), report per-slice paired-Wilcoxon. Catch the "aggregate is flat but one slice regressed" case chapter 3 flags.
- **Automated corpus-drift → ANN-config nudge.** When corpus volume climbs 3×, log a recommendation to re-tune HNSW `efSearch` (chapter 7 Failure B).
- **Judgment-set refresh trigger.** If the new-intent rate stays above threshold for N days, open an issue in the eval-repo asking for a judgment-set expansion (mod-109 chapter 2 protocol B).
- **Alert history + MTTR panel.** Track alert firings and the time to resolution; publish a monthly MTTR gauge. Not the drift signal itself, but the operational quality of the drift response.
- **Cross-layer alert.** Combine this module's quality-drift with the eval peer's app-layer faithfulness gauge into a single alert rule ("retrieval nDCG improved but faithfulness dropped" — chapter 8's cross-layer failure). The alert belongs on the peer's side; wire the retrieval feed from here.

## Notes

- **Per-chunk provenance is a hard requirement.** Chapter 8's under-delivery failure is real — an incident review that starts with "this answer was wrong, which chunk produced it?" cannot end well if the retriever didn't emit the chunk's `source_uri` and `updated_at`. Make provenance a schema failure, not a nice-to-have.
- **Thresholds without noise measurement are numbers you made up.** Chapter 7 is explicit: calibrate. If you can't measure, write a `needs-research` note and disable the alert until you can; a muted alert is safer than a false-positive-training one.
- **The alert firing is not the goal.** The runbook diagnosis is. If a corpus-drift alert fires and no one can decide whether to full-rebuild or wait, the runbook is missing a row.
- **Do not compute app-layer metrics here.** RAGAS faithfulness, TruLens RAG triad — those go through the eval peer's pipeline (chapter 8). This drill's quality signal is retrieval-only, from mod-109.
- **The Wilcoxon-on-rolling-windows shape is worth the extra machinery.** Chapter 7 Failure A ("slowly boiling frog": recall drifted from 0.803 to 0.71 over six months, no single-week drop was alertable") is the reason a raw-drop-only quality alert is inadequate.
- **Emit to the eval peer's platform even in the drill.** The hand-off is easier to design when the receiving surface is real. Screenshots into the write-up make the boundary tangible.
