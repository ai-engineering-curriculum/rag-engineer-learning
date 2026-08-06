# Exercise 02 — Retrieve-Top-N → Rerank-to-K Funnel Tuning

**Estimated effort:** ~2.5 hours. Python, an eval set with per-query graded (or binary) relevance labels, a first-stage retriever that can return top-500, one cross-encoder reranker, and — for the K sweep — a way to measure end-to-end answer quality (a running LLM or the RAG-triad harness).

## Objective

Run the N-K joint tuning recipe from chapter 5 end-to-end on your own data. Produce two calibration curves — recall@N over N, and end-to-end answer quality over K — and defend the operating point (N*, K*) you pick against the four constraints (quality, latency, cost, context budget). This is the exercise that makes chapter 5 concrete.

## Prerequisites

- Chapter 5 read. Chapter 2 read (for the reranker). Chapter 8 read (for why K peaks rather than climbing monotonically).
- A first-stage retriever configurable to return top-500 candidates per query (mod-106 stack, or a simple BM25 + dense fallback).
- One reranker from chapters 2 or 3 (cross-encoder open-weight, or hosted API).
- An eval set with ~50–200 (query, relevant-doc-ids) pairs; ideally with graded relevance for nDCG.
- Python 3.10+ and either an LLM API (OpenAI, Anthropic, or a hosted open model) or an eval harness ([RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval](https://docs.confident-ai.com/)) for the K sweep. If you cannot run end-to-end LLM eval, do a chunk-level nDCG@K sweep and note the substitution in your writeup.

## Steps

**Step 1 — Baseline the retriever's recall curve (30 minutes).** For each query, retrieve top-500 candidates. Compute:

```
    N       recall@N over the eval set
    ---     -------------------------
    20      ?
    50      ?
    100     ?
    200     ?
    500     ?
```

Plot recall@N vs. N. Confirm it is concave (rises fast, then plateaus). Identify the elbow — the N past which the recall curve is essentially flat.

If your curve is *not* concave — recall@200 < recall@100, or recall@500 = recall@50 — something is broken in your retriever. Diagnose before proceeding: (a) is the retriever deterministic across N, or is it sampling? (b) is the eval set too small to see the curve above noise?

**Step 2 — Sweep N with the reranker (30 minutes).** For each N ∈ {20, 50, 100, 200, 500} from step 1, rerank the top-N with your reranker and take the top-K=10. Compute nDCG@10 for each N. Plot.

Observations to make:

- Where is the elbow of the *end-to-end* (post-rerank) quality curve? Usually it is at a smaller N than the raw recall curve — a well-tuned reranker often extracts most of the gain from N=100, so N=500 is wasteful.
- What does p95 latency look like at each N? Add a second axis or a second column to your table.

Pick N* — the smallest N inside your latency budget where nDCG@10 is within 1 percentage point of its plateau value.

**Step 3 — Fix N=N*, sweep K (60 minutes).** For each K ∈ {1, 3, 5, 10, 15, 20}, run the pipeline (retrieve → rerank at N* → assemble top-K → LLM answer), then compute end-to-end answer quality.

The metric options for end-to-end:

- **RAG-triad via RAGAS / TruLens / DeepEval.** Faithfulness + answer relevance + context precision. Preferred; matches mod-109's methodology.
- **A task-specific metric.** If your eval queries have gold-standard answers, use exact-match, BLEU, or a semantic-similarity metric between the generated answer and the gold.
- **Fallback: chunk-level nDCG@K.** Not end-to-end but a reasonable substitute if you can't run an LLM leg. Note the substitution explicitly.

Plot answer quality vs. K. Confirm the shape from chapter 5: rises to a peak, then flat or declines. Identify K*.

**Step 4 — Verify the N-K interaction (30 minutes).** Re-run step 2's sweep at K=K* (not K=10). Do you still pick the same N*? Sometimes the best N depends on the K you eventually take; verify.

If N* changes, that is a real insight — the N and K knobs interact and one-at-a-time optimization can find a false optimum. Note it in your writeup.

**Step 5 — The budget worksheet (20 minutes).** Fill in chapter 5's latency budget worksheet with real numbers from your stack:

```
    Total user-visible latency (p95)            = your product target
    − Generator latency at your output length   = ?
    − Network + app overhead                    = ?
    ─────────────────────────────────────
    Retrieve + rerank + assemble budget         = ?

    Retrieval (BM25 + dense + RRF)              = ?
    Reranker at N*, top_K=K*                    = ?
    Compression (optional; ex. 03 covers)       = ?
    Assembly                                    = ~10 ms
    Slack                                        = ?
    ─────────────────────────────────────
    Sum                                          = should match the budget above
```

If the sum exceeds the budget, apply the four levers from chapter 5:
1. Shrink N.
2. Switch to a smaller reranker.
3. Skip compression.
4. Cache.

Report which lever you would pull first and why.

**Step 6 — Write it up (20 minutes).** In a `funnel-tuning.md`:

- The two calibration curves (recall@N vs. N; answer quality vs. K).
- The chosen (N*, K*).
- The budget worksheet, filled in.
- One paragraph on the N-K interaction: did N* change when you re-ran the sweep at the peak K? If so, what did you learn?
- One paragraph on which chapter 5 anti-pattern (default N, coupled N-K, K-at-window-max) your prior stack was closest to before this exercise.

## Acceptance criteria

- Recall@N curve with at least 5 N values (20, 50, 100, 200, 500).
- nDCG@10 sweep over the same N values with the same reranker.
- Answer-quality sweep over at least 5 K values (1, 3, 5, 10, 15, or 20).
- Chosen (N*, K*) with a written justification against the four constraints.
- Latency budget worksheet with real p95 numbers, not estimates.
- The N-K interaction re-check is done and reported (whether N* changed or not).

## Stretch goals

- **Per-query-class funnel tuning.** Split the eval set by query type (short/factoid vs. long/multi-hop vs. filter-heavy) and run the sweep per bucket. Do different query classes want different (N, K)? This is a per-task-K argument (chapter 5) with numbers.
- **Reranker family sensitivity.** Rerun the sweep with two different rerankers (small MiniLM + large BGE, say). Does the N* elbow shift with the reranker? A stronger reranker often extracts value from smaller N, letting you spend less compute.
- **Latency-quality Pareto plot.** Instead of picking a single (N*, K*), plot the Pareto frontier: for each (N, K) pair, plot (p95 latency, nDCG@10) as a point. Which (N, K) points are on the frontier? Which are dominated?
- **Cost curve alongside quality curve.** For each N, compute cost per 1000 queries (per exercise 01's methodology). At what N does the marginal cost per marginal nDCG@10 point become uneconomic?
- **Compare to chapter 5's "real-world numbers for calibration" section.** Where does your corpus sit relative to the four archetypes (enterprise search, very heterogeneous, narrow product, multi-hop research)? Do your numbers match the archetype you expected?

## Notes

- Hold out 30% of eval queries as a test set. Sweep N and K on the training 70%, then measure the chosen (N*, K*) once on the held-out set. This is the standard hyperparameter-selection discipline; not doing it invites overfitting the funnel to your eval set.
- End-to-end answer quality is expensive to measure (an LLM call per query per K value = several thousand LLM calls). Cache results per (query, chunks_signature) so re-running is cheap.
- Do not report a single "best" (N, K). Report the trade-off: "N=100, K=5 is Pareto-optimal at our latency budget; N=200, K=10 wins on quality but doesn't fit p95." Curriculum-authoring shorthand: **the picks are always joint**.
- If your first-stage retriever's recall@500 is already low (say < 0.6), no reranker or funnel-tuning fixes that. Go back to mod-106 first.
- If the K sweep shows quality still rising at K=20, either your corpus is very synthesis-heavy (long-form summarisation) or your reranker is ranking poorly (the top-10 doesn't contain what a K=20 pass would surface). Diagnose which case.
