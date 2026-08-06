# Chapter 7 — Monitoring Corpus, Query, and Quality Drift

## Why this chapter exists

Chapters 2, 3, and 4 are the discipline for **planned** changes — a PR ships, an index rebuilds, a cutover flips. This chapter is the discipline for **unplanned** ones — the corpus you index is not the corpus you had last month; the queries users send today are not the queries they sent last quarter; the retrieval quality has quietly drifted below the threshold that made the eval pass on the day you shipped.

Retrieval drift decomposes cleanly into three orthogonal channels:

1. **Corpus drift** — the documents changed. New docs, deleted docs, changed content, shifted length or topic distribution.
2. **Query drift** — the queries changed. New intents, new vocabulary, distribution shift in what users ask.
3. **Quality drift** — the metric numbers changed on a held-out judgment set, even though corpus and queries and code are all the same.

Each channel needs its own detector, its own alert threshold, and its own runbook. This chapter builds all three.

Boundary reminder: the general LLM-application drift and tracing lives in `ai-eval-engineer` (LangSmith, Langfuse, Arize AI, Arize Phoenix). This chapter is the *retrieval-specific* fields that get emitted into whichever platform the org uses. Chapter 8 draws that boundary explicitly.

## The three drifts as three graphs

Each drift is a time series on a specific quantity. Alerts fire when the time series moves.

```
  corpus drift                query drift                      quality drift
  ───────────                 ───────────                      ─────────────
  n_docs                      distinct_query_terms             recall@50 (rolling eval)
  n_chunks                    query_length_p50                 nDCG@10  (rolling eval)
  add_rate / day              embedding_distribution_shift     MRR@10   (rolling eval)
  delete_rate / day           unseen_intent_rate               hit-rate@10
  content_hash_change_rate    zero-result-rate
  topic_dist_shift            reformulation_rate
```

The graphs are cheap to produce (an ingestion pipeline already knows adds and deletes; a retriever service already knows what it returned) and expensive to skip.

## Corpus drift — the "the input to the index moved" channel

### What to measure

- **Volume gauges.** `n_documents`, `n_chunks`, `avg_chunks_per_doc`, `avg_tokens_per_chunk`. All computed after each ingest run and stored as time series.
- **Rate gauges.** `documents_added_per_day`, `documents_updated_per_day`, `documents_deleted_per_day`. Chapter 4's incremental-build audit log is the source.
- **Content-change gauges.** Per document: hash of the content. Per day: fraction of documents whose hash changed. Detects "the whole corpus was rewritten and no one told us."
- **Distributional gauges** (more work; higher signal):
  - **Length distribution shift.** KS statistic between today's chunk-length distribution and the reference. Or a simpler percentile-drift check (p50 length ±X%).
  - **Topic distribution shift.** Cluster the embedded documents (KMeans on a sample, or reuse HDBSCAN clustering); track the population of each cluster over time. A cluster's share jumping from 5% to 25% in a week is an intent shift.
  - **Embedding-space centroid shift.** Compute the mean embedding of newly-added docs; compare to the reference set's mean via cosine distance. Reference the [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/mmddrift.html) MMD/KS detector patterns for the general shape.

### Alert thresholds

Alert thresholds are ranges to sanity-check on your own corpus, not universal constants. Common shapes:

- **Volume:** absolute-change thresholds relative to a rolling 7-day baseline; e.g. `|n_documents_today − n_documents_7d_avg| > 10%` — investigate.
- **Rate:** the incremental-build audit log delta is the numerator; the source system's expected delta is the denominator. A 3× spike or a 3× dip both warrant a look.
- **Content-change:** > 5% of docs changing in a day, when the source usually sees < 1%, is unusual.
- **Distributional:** shift beyond the noise level you measure by re-running the detector on stable-corpus windows.

Calibrate every threshold before enabling the alert. An alert that fires weekly gets muted; an alert that fires once a quarter gets attention.

### What to do when it fires

- **Investigate the source.** Which upstream system emitted the delta? Confluence, Notion, S3, the file share — every source has an audit trail.
- **Check the ingest pipeline for silent failures.** Chapter 4's structural sanity check; count the docs that entered vs. what should have entered.
- **If the drift is real, decide on a full rebuild trigger.** A large content-change fraction usually means the incremental has enough deltas that a full rebuild is faster than repeated incrementals.
- **Update the reference baseline.** Once the drift is understood as a real change in the corpus (not a bug), re-baseline the detectors so they don't keep firing.

## Query drift — the "the input to the retriever moved" channel

### What to measure

- **Query-shape gauges.** `query_length_p50 / p95`, `distinct_query_terms_per_day`, `distinct_normalized_queries_per_day`. Sourced from the retrieval service's request logs.
- **Zero-result rate.** Fraction of queries where `top_k` came back empty. A sudden climb is either a corpus problem (documents disappeared) or a query problem (users are asking about something the corpus doesn't cover). Either way, page.
- **Reformulation rate.** Fraction of queries followed within a session by a *similar* query. High reformulation is a UX proxy for "the first query didn't get the right answer." Not a retrieval metric per se, but useful as a canary.
- **New-intent rate.** Cluster queries in embedding space; track the fraction that don't fit any known cluster. Rising fraction = new intents entering the query distribution.
- **Distributional shift on the query-embedding space.** Same MMD / KS / centroid-shift shapes as corpus drift, applied to embedded queries. [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/latest/) is one of several libraries with off-the-shelf detectors.
- **Query-vocabulary shift.** Top-N term frequency changes over time. New terms rising fast = a new topic entering user vocabulary.

### Alert thresholds

Same discipline as corpus drift: measure the noise floor before setting the threshold. Query drift alerts are usually noisier than corpus drift (users are unpredictable), so err toward wider bands.

- **Zero-result rate spike** is the highest-signal alert; err small (a 2× rise from the rolling baseline should page).
- **Distributional shift** should alert on a p-value threshold from the detector, not on the raw distance — the detector's null distribution has the noise floor built in.

### What to do when it fires

- **Check the retriever's fresh releases.** A new retriever config that changed normalization may have broken the vocabulary — a query-drift alert is sometimes a code drift.
- **Sample the new-intent queries** for human review. Users may be asking about a subject the corpus should cover but doesn't (a gap-analysis opportunity), or the queries may be noise / attack traffic.
- **Update the query-drift baseline** once the change is understood.
- **Consider expanding the judgment set** (mod-109 chapter 2) to cover the new intents. A judgment set that's a year old on a corpus that has new intents will silently under-cover regressions on those intents.

## Quality drift — the "the metrics moved" channel

### What to measure

The mod-109 metric triple, on a fixed judgment set, on a schedule.

- **`recall@N`** — the retriever's job-doing metric.
- **`nDCG@10`** — the ranked-order metric.
- **`MRR@10`** — the top-1 metric.
- **`hit_rate@10`** — a smooth canary metric.
- Per-slice versions of each — if the judgment set is stratified (mod-109 chapter 2 stretch goal), publish per-slice too.

### The scheduling shape

- Run the mod-109 eval nightly, on `main`, against the current `INDEX_ID`.
- Store per-run per-query values, not just aggregates. The rolling significance test (below) reads per-query.
- Post the aggregate to the dashboard; alert when the aggregate moves.

### Alert threshold — the rolling paired-significance test

The right alert isn't "recall@50 dropped by more than X." That gates on noise. The right alert is a **rolling paired significance test**: compare the last window's per-query values to the previous window's. If the drop is statistically significant (paired Wilcoxon at `p < 0.05` on per-query deltas, mod-109 chapter 4), page. Otherwise, no alert.

This is the same test the retriever CI uses in chapter 3; the difference is the *cadence* (nightly vs. per-PR) and the *baseline* (the previous rolling window vs. `main`).

### What to do when it fires

- **Diff the tuple.** Did anything about the index change since the last-known-good window? Chapter 2's manifest is where you look.
- **Check for corpus or query drift on the same time window.** Often the corpus / query drift alert fired first and this is the downstream effect; wire the alerts to cross-reference.
- **If nothing else moved, the corpus or query distribution changed in a way the eval set didn't cover.** Expand the judgment set (see below).
- **If the drop is real and unrecoverable at the current tuple, cut a rebuild PR.** Chapter 3's CI runs and confirms whether the change fixes it.

### When the judgment set itself has drifted

Every judgment set decays. New queries appear that the set doesn't include; new relevant docs appear that the set's labels don't reflect. **Refresh the eval set on a cadence** (quarterly is a reasonable default). mod-109 chapter 2's protocol B (LLM-assisted labeling with a human spot-check) is the cheap way to expand the set as the corpus and queries evolve.

Every time the eval set is expanded, run a **transition window** — publish the new-set's metrics alongside the old-set's for one cycle so the delta is a set change, not a drift signal.

## Cross-drift correlation — three graphs together

The three drift graphs are most useful *together*.

- Corpus drift up + quality drift down → the new documents are lower-quality retrievals; investigate the ingest step.
- Query drift up + quality drift down → users are asking new things the retriever handles poorly; the judgment set is stale.
- Quality drift down + corpus drift flat + query drift flat → something silent moved (a hosted embedding-model revision bump; a downstream cache misconfig; a schema drift). Chapter 2's manifest diff catches this class.

Publish all three drift dashboards on one page. Cross-linked alerts (a corpus-drift alert links to the quality-drift dashboard for the same window) make the triage fast.

## The retriever telemetry — what fields go on every request

For any drift detector to work, the retrieval service must emit specific fields on every request. The retrieval-side contract (this module's responsibility):

- `query_text` (hashed if PII-sensitive; the drift detectors work on the hash + embedding for privacy)
- `query_normalized`
- `INDEX_ID`
- `top_k_returned` (the list of doc-IDs)
- `top_k_scores`
- `retriever_latency_ms`
- `reranker_latency_ms` (if invoked)
- `cache_hit_query_embed` (bool)
- `cache_hit_top_k` (bool)
- `cache_hit_generation` (bool)
- `tenant_id` (for tenant-scoped drift)
- `filters_applied` (predicate names, for selective-predicate canary tracking)
- `request_id` (for trace correlation)

The `ai-eval-engineer` platform's job is to correlate these with the LLM-side trace (prompt, response, cost). This module's job is to emit them in a form the platform can ingest — Langsmith / Langfuse / Phoenix / Arize all accept a general "trace event" shape.

## The per-chunk provenance field — a retrieval-specific must-emit

On every retrieval, emit the *provenance* for each returned chunk:

- `chunk_id`
- `doc_id`
- `source_uri`
- `section_path`
- `updated_at` (the source document's timestamp)
- `retrieval_score`
- `rerank_score` (if applicable)

Provenance is what makes an incident review possible ("this bad answer came from chunk X, from document Y, which was last updated on Z, and was retrieved with score S"). It is also what `ai-eval-engineer`'s citation-verification metrics consume.

The rule: provenance is not optional and is not batched separately from the retrieval log. It ships alongside every top-k result, always.

## Concrete example — the drift dashboard shape

Illustrative Grafana dashboard structure:

```
┌────────────────────── Retrieval drift ──────────────────────┐
│                                                                │
│  ┌──── Corpus drift ────┐  ┌──── Query drift ────┐             │
│  │ n_docs (7d)          │  │ query_len_p50 (7d)  │             │
│  │ n_chunks (7d)        │  │ zero_result_rate    │             │
│  │ add/update/delete/d  │  │ new_intent_rate     │             │
│  │ topic_dist_shift     │  │ embed_dist_shift    │             │
│  └──────────────────────┘  └─────────────────────┘             │
│                                                                │
│  ┌────────── Quality drift ──────────┐                          │
│  │ recall@50 (nightly, 30d)          │                          │
│  │ nDCG@10   (nightly, 30d)          │                          │
│  │ MRR@10    (nightly, 30d)          │                          │
│  │ per-slice tabs                     │                          │
│  │ paired-Wilcoxon p on rolling 14d   │                          │
│  └────────────────────────────────────┘                         │
│                                                                │
│  ┌────────── Alert history ──────────┐                          │
│  │ time × alert × runbook link        │                          │
│  └────────────────────────────────────┘                         │
└────────────────────────────────────────────────────────────────┘
```

Every panel links to the incident runbook. Every alert page includes a link to the panel that fired it. The dashboard is the operational center of the retrieval system.

## Two failures the drift monitor defends against

### Failure A: the "slowly boiling frog"

Recall@50 was 0.803 at ship. Six months later it is 0.71. No single incident is visible in the history. Each week the corpus grew a bit, each week a small class of queries got a bit harder, and the rebuild cadence didn't keep up. Because no alert ever fired ("no single week dropped enough"), no investigation happened.

**Defense.** The rolling paired-significance test doesn't gate on single-week drops; it gates on the accumulated trend over a rolling window. A 0.005-per-week drift that adds up to 0.09 over 18 weeks fires the alert on week 4 or 5, not on week 18.

### Failure B: the corpus grew 10× and no one adjusted the ANN parameters

A new department onboarded and dumped their docs into the corpus. `n_chunks` went from 500k to 5M. The `HNSW efSearch` that was optimal at 500k is under-tuned at 5M — recall drops by a few points, latency climbs because the graph is bigger. No one connects the corpus-drift alert (which fired) to the retrieval-config that should have changed.

**Defense.** Cross-alert correlation: a corpus-drift volume spike triggers a note in the runbook to check ANN config alongside quality metrics. Cross-references between drift dashboards make the connection obvious.

## What you should be able to do after this chapter

- Name the three drift channels (corpus, query, quality), what each measures, and at least two gauges per channel.
- Set an alert threshold for each channel using measured noise floors rather than intuited constants.
- Wire the rolling paired-significance test (mod-109 chapter 4) as the quality-drift alert; explain why gating on the raw metric drop is inferior.
- Enumerate the retrieval-side telemetry fields this module owns emitting (query, top-k, cache-hit flags, tenant, filter predicates, per-chunk provenance).
- Distinguish this module's retrieval-specific telemetry from `ai-eval-engineer`'s LLM-application-level tracing; state what each hands the other.
- Author a cross-drift triage runbook — a corpus-drift alert on Tuesday + a quality-drift alert on Thursday tells you what to check.

## Further reading

- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS. [arXiv:1810.11953](https://arxiv.org/abs/1810.11953) — the reference on distribution-shift detection methods.
- Alibi Detect: [*Drift Detection Methods*](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods.html) — an off-the-shelf library that implements many of the tests this chapter cites (KS, MMD, chi-square).
- Evidently AI: [*Data drift docs*](https://docs.evidentlyai.com/user-guide/tests/data-drift) — an alternative library with similar coverage.
- WhyLabs [whylogs](https://whylogs.readthedocs.io/) — data-profile-based drift detection, useful for the corpus-side gauges.
- Google SRE Book, Ch. 6 (*Monitoring Distributed Systems*) and Ch. 4 (*Service Level Objectives*). [Full text](https://sre.google/sre-book/monitoring-distributed-systems/) — the alert-thresholding discipline this chapter borrows.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys. [DOI:10.1145/2523813](https://doi.org/10.1145/2523813) — a foundational survey; the concepts translate directly to corpus and query drift.
- Chip Huyen (2022). *Designing Machine Learning Systems*, Ch. 8 (Data Distribution Shifts and Monitoring) — the general framing this chapter's retrieval-specific instances specialize.
- LangSmith [Tracing docs](https://docs.smith.langchain.com/observability), Langfuse [docs](https://langfuse.com/docs), Arize Phoenix [docs](https://docs.arize.com/phoenix) — the peer-owned platforms this module's retrieval telemetry lands in; chapter 8 draws the boundary.
- mod-109 chapter 4 — the paired-significance-test primitive the rolling quality-drift alert uses.
