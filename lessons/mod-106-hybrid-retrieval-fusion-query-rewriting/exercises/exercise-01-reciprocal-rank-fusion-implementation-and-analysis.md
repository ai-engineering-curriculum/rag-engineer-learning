# Exercise 01 — Reciprocal Rank Fusion Implementation and Analysis

**Estimated effort:** ~2 hours. Python, a small labeled dataset, and Elasticsearch or OpenSearch (or a simple in-memory retriever pair).

## Objective

Implement Reciprocal Rank Fusion (Cormack et al. 2009) from scratch in ~15 lines of Python, wire it to a two-arm retrieval stack (BM25 + dense), run it against a labeled eval set, and analyze its behavior across four dimensions: (1) recall@k lift vs. either single arm, (2) sensitivity to the `k` parameter, (3) sensitivity to the per-arm `top_n` window, and (4) how the fused ranking degrades when one arm is deliberately broken. The point is not the code (RRF is 15 lines) — the point is the *behavior* you observe when you turn the knobs.

## Prerequisites

- Chapter 1 (motivation, score-scale mismatch) and Chapter 2 (RRF) read.
- Python 3.10+.
- A retrieval stack with a BM25 arm and a dense arm. Options in decreasing order of setup cost:
    - Full: a local Elasticsearch 8.10+ or OpenSearch 2.10+ ([Elastic quick start](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html), [OpenSearch quick start](https://opensearch.org/downloads.html)), a BM25 mapping (per mod-105 chapter 3), a dense field with a sentence-transformer embedding, and a small corpus indexed.
    - Medium: Python with `rank_bm25` for the lexical arm and `sentence-transformers` + FAISS or Chroma for the dense arm.
    - Minimum: two pre-computed ranked lists per query on a public dataset (see below), no live retrieval.
- A small labeled eval set of ~50–200 (query, relevant-doc-ids) pairs. Suggested options:
    - A BEIR subset: [SciFact](https://github.com/beir-cellar/beir) (300 test queries, ~5K docs, small enough to run locally); FiQA (~650 test queries, ~57K docs).
    - Your own team's corpus with hand-labeled qrels (~20 queries, top-5 relevant per query, is enough for directional signal but not for statistical significance).
- Familiarity with recall@k as the metric (mod-101 refresher, mod-109 rigorous).

## Steps

**Step 1 — Implement RRF (15 minutes).** Write the reference implementation from chapter 2 in a file `rrf.py`:

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    ...
```

Write a small unit test that reproduces the chapter 2 worked example (doc_A wins with score 0.03252) and confirms the sort order.

**Step 2 — Wire the two-arm retriever (30 minutes).** Author two functions that each return a top-100 ranked list of doc IDs for a query:

- `bm25_retrieve(query: str) -> list[str]` — the lexical arm.
- `dense_retrieve(query: str) -> list[str]` — the dense arm.

If you're using Elasticsearch or OpenSearch, wrap the `_search` call. If you're using in-memory retrievers (`rank_bm25` + sentence-transformers), just call the library. The key requirement: both return top-100 as a list of doc IDs.

**Step 3 — Compute per-arm and RRF baseline (20 minutes).** Loop over your eval-set queries and compute recall@100 for:

- BM25 alone.
- Dense alone.
- RRF over both, with `k = 60`, `top_n = 100`.

Report the three numbers as a table. You should see RRF beat both single arms by at least 2–5 percentage points on any heterogeneous corpus. If RRF loses to one arm, diagnose: either (a) the losing arm is producing degenerate rankings (all-zeros, wrong analyzer, model mismatch), or (b) the eval set is very small (< 30 queries) and you're seeing noise. Fix (a) before proceeding; note (b) explicitly.

**Step 4 — Sweep the `k` parameter (20 minutes).** For `k ∈ {5, 20, 40, 60, 100, 200}`, compute recall@100 for RRF over the same two arms and the same eval set. Plot the sweep (a simple line plot or a text table is fine).

Two questions the plot should answer:

- Where is the peak, and how far from `k = 60`?
- How flat is the curve? A flat curve (~±0.5 percentage points across the range) means RRF is not sensitive to `k` on your corpus; a peaked curve means the default 60 is worth revisiting.

Write one paragraph explaining what you see.

**Step 5 — Sweep the `top_n` per-arm window (15 minutes).** For `top_n ∈ {10, 50, 100, 200, 500}`, compute recall@100 for RRF with `k = 60`. Report the curve. Two observations you should be able to make:

- Recall usually rises with `top_n` (more candidates per arm → more union recall) and plateaus somewhere between 100 and 500.
- The compute cost per query rises linearly with `top_n` on the retriever side (BM25 is close to free at 500; dense ANN cost grows). Pick a defensible operating point.

**Step 6 — Break one arm, observe RRF's degradation (25 minutes).** Deliberately damage one arm and re-run RRF vs. either single arm:

- **Damage 1: shuffle the top-100 of the dense arm.** Randomly permute the ranking. Rerun RRF. Compare to (a) BM25 alone, (b) the pristine two-arm RRF. Report by how many percentage points RRF drops. It should stay meaningfully above BM25-alone if the shuffle is over top-100 — the shuffled arm's *set* is still meaningful, just not ordered.
- **Damage 2: replace the dense arm with random doc IDs.** Return 100 random IDs from the corpus, unrelated to the query. Rerun RRF. Compare to BM25-alone. RRF should be *worse* than BM25-alone here (the random arm is worse than absent).

Write one paragraph on what this teaches you about RRF's robustness: it tolerates a *noisier* arm but not a *lying* arm. If an arm is broken enough to return junk, it drags down the fusion.

**Step 7 — Write it up (~15 minutes).** Author a short report `writeup.md` in your working directory containing:

- The results table from step 3 (per-arm and RRF baseline).
- The `k` sweep curve from step 4 with one paragraph of interpretation.
- The `top_n` sweep curve from step 5 with a defended operating-point pick.
- The two damage-arm results from step 6.
- One paragraph naming which of chapter 2's three caveats (no per-arm weighting, agreement signal hidden, clone-arm double-counting) you saw evidence of in this exercise.

## Acceptance criteria

- `rrf.py` implements RRF, passes a unit test that reproduces the chapter 2 worked example.
- Report table shows BM25 recall@100, dense recall@100, and RRF recall@100 with `k=60`.
- `k` sweep across ≥ 5 values with a plot or table.
- `top_n` sweep across ≥ 4 values with a plot or table.
- Two damage experiments (shuffled dense, random dense) with numeric results.
- Writeup includes at least one non-obvious observation tied to a chapter 2 concept.
- If RRF does not beat both single arms on your corpus, the writeup explains why (small eval set, degenerate arm, mismatch) rather than pretending it did.

## Stretch goals

- **Add a third arm.** Bring in a SPLADE arm ([naver/splade](https://github.com/naver/splade)) or a second dense arm with a different embedding model (e.g., MiniLM + BGE). Report three-arm RRF vs. two-arm RRF. Note any diminishing-returns effect — chapter 1 named this.
- **Bootstrap confidence intervals.** Compute a 95% bootstrap confidence interval for each recall@100 number by resampling queries. Determine whether the RRF vs. best-single-arm delta is statistically significant. (Mod-109 covers paired significance testing; the bootstrap is a reasonable first pass.)
- **Per-query-class analysis.** Split your eval set into buckets: short queries (< 5 tokens) vs. long queries; single-entity queries vs. multi-entity queries. Report per-bucket recall for each arm and RRF. Look for buckets where RRF does *not* beat both arms — this signals the query class where routing (chapter 5) or a different fusion (chapter 3) might help.
- **Weighted-RRF.** Add a per-arm weight to your RRF implementation: `Σ_r w_r / (k + rank_r(d))`. Sweep `w_dense ∈ {0.1, 0.3, 0.5, 0.7, 1.0}` with `w_lex = 1 - w_dense`. Compare to unweighted RRF. Does per-arm weighting help on your corpus? (Usually it helps when the arms are very unequal in quality; it usually doesn't when they are near-equal.)
- **Compare against the engine's native RRF.** If you have Elasticsearch 8.8+ or OpenSearch 2.19+, run the same eval through the native `rrf` retriever / hybrid pipeline. Confirm your Python implementation produces the same ranking (or explain the delta, e.g., different `rank_window_size`).

## Notes

- Do not tune `k` on the same queries you measure recall on. That is eval-on-training-set and will overfit. Hold out 30% of the queries as a test set that you measure the final config on, once.
- Recall@100 is the right metric for this exercise (mod-106 optimizes recall for downstream rerankers). If you also want MRR@10 or nDCG@10, run them alongside — but do not swap them in as the primary metric here.
- The two-page Cormack paper is worth a genuine read this week. It is short, clear, and the intuition it builds is the foundation for chapter 3's alternatives.
