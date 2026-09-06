# Exercise 05 — Filter-Heavy Hybrid Composition Drill

**Estimated effort:** ~2.5 hours. Python, a Lucene-family engine (Elasticsearch 8.10+ or OpenSearch 2.10+) or a dedicated vector database that supports filtered ANN (Qdrant, Weaviate, or pgvector), and a small corpus with rich metadata.

## Objective

Reproduce, measure, and fix the four filter-heavy failure modes named in chapter 6 — post-filter ANN recall collapse, dense-arm-becomes-decorative on tight filters, weighted-sum warp after filtering, and asymmetric filters across arms — on a real hybrid stack. You will end the exercise with (a) a per-filter-tightness recall report, (b) a compose-with-metadata query that mirrors filters across both arms with proper `num_candidates`, (c) a short design memo defending your pre-filter-vs-boost-not-filter choices per query type, and (d) an adversarial ACL test that proves permission filters never leak. The point is to build the reflexes that keep a hybrid retriever from silently collapsing to a single arm when filters get tight — not to author novel fusion.

## Prerequisites

- Chapter 6 (filter-heavy hybrid composition) read. Chapter 2 (RRF) and chapter 3 (weighted-sum) as background.
- A running retrieval stack with both a lexical arm and a dense arm on the same corpus, each supporting a filter clause at query time. Options in decreasing setup cost:
    - **Full:** Elasticsearch 8.10+ or OpenSearch 2.10+ with a `dense_vector` / `knn_vector` field on the same index as your BM25 fields. Use the RRF retriever (ES) or hybrid query search-pipeline (OS).
    - **Split-engine:** A Lucene-family engine for BM25 + a dedicated vector database (Qdrant, Weaviate, or pgvector) for dense; do the fusion in application code using your `rrf.py` from exercise 01.
    - **Minimum:** in-memory BM25 (`rank_bm25`) + FAISS with a Python-side filter that mirrors the same predicate on both arms. Loses the engine-native filtered-ANN experience, so you will not observe failure mode 1 in its native form — note this in the writeup.
- A corpus of **≥ 20,000 docs** with **at least three structured filter fields** worth filtering on. Suggested corpora:
    - Your own team's corpus with real metadata (product, language, updated_at, ACL / tenant).
    - A BEIR dataset augmented with synthetic metadata (attach `topic`, `year`, `source`, `acl_group` fields to each doc — random-but-stable values are fine for the exercise).
    - Amazon reviews or MS MARCO passage subsets partitioned into synthetic tenants / categories.
- A small labeled eval set — **50–150 (query, filter, relevant-doc-ids)** triples. For each query, record:
    - `query`: the natural-language query text.
    - `filter`: the structured predicate (e.g., `{"product": "Acme 4402", "language": "en"}`).
    - `filter_tightness`: the fraction of the corpus that matches the filter (compute once, cache it).
    - `relevant_docs`: the qrels list.
- Your `rrf.py` from exercise 01 (or the engine-native RRF).

## The four filter-tightness buckets

Bucket your eval set by `filter_tightness` before doing anything else. The failure modes surface differently per bucket, and aggregate-only metrics hide them.

| Bucket | `filter_tightness` (docs matching / total) | Example |
|---|---|---|
| **Loose** | > 20% of corpus | `language = en` on a mostly-English corpus. |
| **Moderate** | 1% – 20% | `product = Acme 4402 AND language = en`. |
| **Tight** | 0.1% – 1% | `product = Acme 4402 AND language = en AND updated_at > 2024-06-01`. |
| **Very tight** | < 0.1% | `product = Acme 4402 AND tenant = customer-77 AND status = archived`. |

Aim for **≥ 10 queries per bucket** (40+ total). If the corpus has no naturally-tight filters, synthesize by tightening the predicate on some queries (e.g., add `updated_at > <recent>` to force tightness) and re-labeling the relevant docs.

## Steps

**Step 1 — Baseline: unfiltered hybrid recall (20 min).** Ignore filters entirely. For each eval query, run:

- BM25 arm, top-100 unfiltered.
- Dense arm, top-100 unfiltered.
- RRF (`k=60`, `top_n=100`) over the two.

Report recall@100 per bucket and overall. This is your "filter-free ceiling" — the number you'd get if filters were free and never damaged retrieval. You will compare every filtered variant back to this.

**Step 2 — Reproduce failure mode 1: post-filter ANN recall collapse (25 min).** For each query, use the *post-filter* approach on the dense arm:

- Retrieve unfiltered top-100 from the dense arm.
- Drop any doc that does not match the filter.
- Measure `recall_kept = |returned ∩ relevant| / |relevant|` and `count_kept = |returned after filter|`.

Report per bucket:

- **Mean `count_kept`.** On very-tight filters this collapses to a handful of docs (often < 10), which is the failure.
- **Mean `recall_kept`.** Recall crashes because most of the top-100 unfiltered candidates were out of scope.

Compare to the filter-free ceiling from step 1. On loose filters post-filtering barely hurts; on very-tight filters recall drops by 30–70 percentage points. Write two sentences naming which buckets are the danger zone on your corpus.

**Step 3 — Fix failure mode 1: pre-filtered ANN with `num_candidates` tuning (30 min).** Rewire the dense arm to pass the filter into the retrieval call (Elasticsearch `knn.filter`, OpenSearch `knn.filter`, Qdrant `filter`, Weaviate `where`, pgvector `WHERE`).

Sweep `num_candidates ∈ {100, 500, 1000, 5000}` on the same eval set, per bucket, and report:

- Dense arm's recall@100 per (bucket, `num_candidates`).
- Median query latency per (bucket, `num_candidates`) — you need the latency because bumping `num_candidates` is not free.

Two observations you should be able to make in the writeup:

- Recall on tight and very-tight buckets rises with `num_candidates`, sometimes dramatically (0.20 → 0.75 is a realistic jump on the very-tight bucket).
- Latency scales roughly linearly with `num_candidates` at the HNSW level (engine-specific; measure yours). Pick a defensible operating point per bucket, or a single `num_candidates` that clears the recall bar across all buckets.

**Step 4 — Reproduce failure mode 2: dense-arm-becomes-decorative on tight filters (25 min).** With pre-filtered ANN set to the `num_candidates` you picked in step 3, run RRF over the two arms. Compare, per bucket:

- BM25-alone recall@100.
- Dense-alone recall@100 (with pre-filtered ANN).
- RRF recall@100.

Compute `RRF_lift = RRF_recall - max(BM25_recall, Dense_recall)`. On the loose and moderate buckets this should be positive (the standard fusion lift). On the tight and very-tight buckets, if the dense arm has degraded, `RRF_lift` may be near-zero or negative — the dense arm is contributing noise, and its rank votes drag the fusion.

If you see the decorative-dense-arm failure on your corpus, write down which bucket and by how many percentage points RRF underperforms BM25-alone. If you do not see it, note the reason (your `num_candidates` was high enough, your corpus is not tight-filter-hostile, the eval set is too small).

**Step 5 — Reproduce failure mode 3: weighted-sum warp after filtering (20 min).** Repeat step 4 but swap RRF for weighted-sum fusion with per-query min-max normalization (from your exercise 02 implementation, or a fresh one). Use equal weights (`w_lex = w_dense = 0.5`).

Report per bucket:

- Weighted-sum recall@100.
- RRF recall@100 (same numbers from step 4).

The delta `RRF - weighted_sum` should be largest on the tight and very-tight buckets — RRF's rank-based scoring is not affected by per-query score distributions warping when the candidate pool shrinks; weighted-sum with min-max is. Write one sentence stating whether your corpus shows this pattern.

**Step 6 — Reproduce failure mode 4: asymmetric filters across arms (10 min).** Intentionally break the fusion by filtering the dense arm and *not* the BM25 arm (or vice versa). Rerun on the very-tight bucket only. Report:

- Recall@100 of the asymmetric-filter RRF vs. the mirrored-filter RRF.
- Qualitative: sample 3 queries and list the top-10 doc IDs before and after. You should see the asymmetric variant leaking out-of-scope docs at the top of the ranking, dominated by whichever arm was filtered.

This is a five-minute demo of why the compose-with-metadata pattern insists on mirroring filter clauses across both arms.

**Step 7 — Compose-with-metadata worked query (20 min).** Author one production-shape query on your engine that puts the chapter-6 pattern together, for a real query from your eval set:

- Both arms carry the same filter clauses.
- `num_candidates` set from step 3.
- Optionally: a soft recency preference (Gaussian decay via `function_score` on `updated_at`) on the BM25 arm — chapter 6 explains why not to put it on the dense arm.
- Optionally: an ACL predicate (`terms` on `acl_group`) mirrored across both arms.

Save this query as `compose_with_metadata.json` alongside the writeup.

**Step 8 — Design memo: hard filter vs. soft boost per query type (15 min).** Take five representative queries from your eval set. For each, decide:

- Which filter fields are **hard** (pre-filter, non-negotiable — permissions, tenant, product, language)?
- Which are **soft** (boost-not-filter — recency preference, popularity boost, editorial pin)?
- Which are **filter-as-fusion-arm** candidates (chapter 6's table row three — rare, but call it out if you have one)?

Justify each choice in one sentence. Aim for the mistake chapter 6 warns about — treating everything as hard — to be visible in at least one of your five if you started naively.

**Step 9 — Adversarial ACL leak test (15 min).** Optional but recommended if your corpus has any real permission analog. Author 5–10 adversarial queries whose *natural* top hits are docs the querying user should not see (e.g., queries that would surface `acl_group = admin_only` docs to an `acl_group = customer` user). Run them through your compose-with-metadata retrieval with the ACL as a pre-filter mirrored across both arms.

Report:

- Number of not-allowed docs returned in the top-100. This should be **zero** for every query.
- If any test returns a not-allowed doc, treat it as a bug and diagnose (missing filter on one arm, wrong ACL field name, engine-side post-filter default).

**Step 10 — Write it up (~20 min).** Author `writeup.md` in your working directory containing:

- The filter-free ceiling table from step 1.
- The post-filter collapse table from step 2, per bucket.
- The `num_candidates` sweep from step 3 with your chosen operating point and one paragraph defending it.
- The decorative-dense-arm result from step 4 (with or without evidence of the failure on your corpus).
- The RRF-vs-weighted-sum delta table from step 5.
- The asymmetric-filter demo from step 6.
- The `compose_with_metadata.json` inlined (or linked).
- The design memo from step 8.
- The ACL leak test results from step 9 (if run).
- One paragraph naming which of the four failure modes you found in your stack and which you did not, with a reason for each.

## Acceptance criteria

- Eval set is bucketed by filter tightness (loose / moderate / tight / very tight) with ≥ 10 queries per bucket, or the writeup explains which buckets were unobtainable and why.
- Step 2 shows a measurable recall collapse on at least one bucket (tight or very-tight) when the dense arm is post-filtered, or the writeup defends why post-filtering did not collapse (e.g., the corpus was too small for the effect to appear).
- Step 3 shows a `num_candidates` sweep across ≥ 4 values per bucket with both recall and latency numbers, and a defended operating-point pick.
- Step 4 reports `RRF_lift = RRF_recall - max(BM25_recall, Dense_recall)` per bucket. On tight/very-tight buckets the writeup either shows the decorative-dense-arm failure or defends its absence.
- Step 5 reports RRF vs. weighted-sum-with-normalization per bucket and calls out the pattern (RRF wins more on tight buckets).
- Step 6 demonstrates the asymmetric-filter failure with three sample queries showing out-of-scope docs in the top-10.
- `compose_with_metadata.json` contains one query that mirrors filters across both arms, sets `num_candidates` from step 3, and matches an engine-native RRF / hybrid pipeline. It should validate with the engine (a live search returns results without error).
- Design memo covers ≥ 5 queries with a hard-vs-soft-vs-filter-as-arm decision per query.
- If ACL leak tests were run, they return zero not-allowed docs, or the writeup treats any leak as a bug and shows the fix.
- Writeup names which failure modes were present and which were not on your corpus, with reasons.

## Stretch goals

- **Filter-aware ANN algorithm comparison.** If your engine supports multiple filter strategies (Weaviate's flat / HNSW / IVF fallback, pgvector's iterative HNSW scans, Qdrant's payload-index-driven filter estimator), rerun step 3 with each and compare recall / latency on the very-tight bucket. Cite the [Filtered-DiskANN](https://arxiv.org/abs/2401.01715) paper if you want the algorithms context.
- **Soft-recency preference as a scoring modifier.** Add a Gaussian decay on `updated_at` to the BM25 arm's `function_score` (Elasticsearch) and measure the impact on nDCG@10 on a recency-sensitive subset of your eval set. Try the same as a post-fusion rerank instead. Compare and defend the choice.
- **Tight-filter routing (chapter 5 crossover).** Route very-tight-filter queries to BM25-only, moderate-filter queries to full RRF. Compare cost-adjusted quality on your eval set to running full RRF on everything.
- **Per-tenant sharding sanity check.** If your `acl_group` is a tenant identifier and your very-tight bucket is dominated by per-tenant queries, sketch (do not implement) an alternative index shape where each tenant is its own index or its own routing key. When would that beat a single mixed index with a `tenant` pre-filter? Mod-112 (enterprise retrieval) is the peer chapter.
- **ACL as a fusion arm (compare to pre-filter).** Chapter 6 argues ACL should be a hard pre-filter. Try implementing it as an RRF arm (score 1 for matching, 0 otherwise) and measure. You should observe two failures: (a) rare but real leakage in the top-100 (RRF gives 0-scored docs some rank, and low-signal queries let out-of-scope docs float up), and (b) the extra arm dilutes the useful fusion signal. Document both.

## Notes

- Do not tune `num_candidates` on the same queries you measure recall on. Hold out 30% of the eval set as a test set that you measure the final config on, once.
- The four failure modes are stated as if they always fire. In practice, on small corpora with loose filters, none of them may fire and the naïve implementation looks fine. That is a feature of your setup, not a general refutation of the chapter — note it and move on.
- The `count_kept` metric in step 2 is often more visceral than `recall_kept`. Seeing "our top-100 dense query returned 4 in-scope docs" is what convinces a team to switch to pre-filtered ANN.
- Engine-native RRF (Elasticsearch's `rrf` retriever, OpenSearch's hybrid pipeline) already carries filters correctly per sub-retriever when you author the query as chapter 6 shows. This exercise is about proving to yourself that "correctly" is not the default of every framework or client library — verify by construction.
- Mod-104 (vector databases and ANN tuning) is the deep dive on filtered ANN internals. Mod-112 (enterprise retrieval) is the deep dive on ACL-aware retrieval at scale. This exercise sits at the intersection and is deliberately shallow on both.
