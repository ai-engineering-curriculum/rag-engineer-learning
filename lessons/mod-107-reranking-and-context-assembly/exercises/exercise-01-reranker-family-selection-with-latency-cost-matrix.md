# Exercise 01 — Reranker Family Selection With a Latency + Cost Matrix

**Estimated effort:** ~3 hours. Python, an eval set with ~50–200 (query, relevant-doc-ids) pairs, a first-stage retriever (BM25 + dense) producing top-100 candidates, at least one GPU-accessible open-weight cross-encoder, and at least one hosted rerank API key. Optional: a LightGBM feature pipeline; an LLM API for an LLM-as-reranker leg.

## Objective

Benchmark four reranker families on **the same eval set, the same first-stage candidate pool, and the same top-K** — and produce a single decision-worthy matrix of (quality, latency, cost, ops overhead) that you can defend to your team. This exercise operationalises chapters 2, 3, and 4: you build the numbers that make the family choice a decision, not a guess.

The deliverable is not the "best" reranker on a benchmark — public benchmarks are already saturated. The deliverable is a defensible pick for *your* data, *your* latency budget, and *your* cost budget, with a written trade-off for the choice you did not make.

## Prerequisites

- Chapters 1, 2, 3, and 4 read.
- A first-stage retriever from mod-106 (or a simpler BM25 + dense fallback) that returns top-100 candidates for each eval query.
- A labeled eval set (~50–200 queries). Options:
    - A BEIR subset ([beir-cellar/beir](https://github.com/beir-cellar/beir)) — SciFact (300 queries, ~5K docs), FiQA (~650 queries), or NFCorpus.
    - Your own team's corpus with hand-labeled qrels (~30 queries, top-5 relevant per query, is enough for directional signal).
- Python 3.10+.
- One or more of:
    - A GPU (T4 / L4 / A10 / equivalent) for open-weight cross-encoder inference. CPU is workable for the smallest MiniLM but painful past batch size 16.
    - A [Cohere Rerank](https://docs.cohere.com/docs/rerank-overview) API key (free tier is sufficient for this exercise).
    - A [Voyage AI](https://docs.voyageai.com/) API key.
    - A [Jina Reranker](https://jina.ai/reranker/) API key.

Bring at least three of the following four families to the matrix: cross-encoder open-weight, hosted API, LTR (if you have labels), LLM-as-reranker (if you have an LLM API key).

## Steps

**Step 1 — Fix the eval set and the first-stage output (30 minutes).** Cache the top-100 first-stage candidates per eval query as a JSON file:

```json
{
  "query_id": "q001",
  "query": "How do I cancel my subscription?",
  "candidates": [
    {"doc_id": "d123", "chunk_id": "d123#c04", "text": "...", "first_stage_score": 0.87},
    ...
  ]
}
```

Every reranker in this exercise runs against these exact 100 candidates. This is the single most important control — a benchmark on different candidate pools is not a benchmark.

**Step 2 — Baseline: no reranker (10 minutes).** Compute recall@10 and nDCG@10 for the first-stage top-10 (candidates[0..9] from step 1 unchanged). This is the number every reranker must beat to justify its cost.

**Step 3 — Cross-encoder open-weight leg (45 minutes).** Pick two open-weight cross-encoders that span size / cost:

- One small: `cross-encoder/ms-marco-MiniLM-L-6-v2` or `jina-reranker-v1-turbo-en`.
- One large: `BAAI/bge-reranker-v2-m3` or `BAAI/bge-reranker-large`.

For each: load the model, run at fp16, batch all 100 candidates per query, compute rerank scores, take top-10. Measure:

- nDCG@10 and recall@10 over the eval set.
- p50 / p95 / p99 latency per query (rerank step only, not first-stage).
- Cost per 1000 queries (compute GPU-hours × your $/hour rate; if you're using Colab or a shared machine, use a defensible substitute).

**Step 4 — Hosted API leg (30 minutes).** Pick at least one of Cohere Rerank, Voyage `rerank-2`, or Jina Reranker. Send the same 100 candidates per query. Measure the same three metrics:

- nDCG@10, recall@10.
- p50 / p95 / p99 latency (this is *end-to-end* including network hop — call it out explicitly in your matrix, since the network hop is a real production cost).
- Cost per 1000 queries — look up the vendor's current per-call or per-token price and compute. Do not omit this; the cost column is why hosted vs. self-hosted is a decision.

**Step 5 (optional if you have labels) — LTR leg (60 minutes).** Only meaningful if your eval set has enough labels (~200+ examples with per-query relevance grades). Build a feature vector per (query, candidate):

- First-stage score (from step 1).
- Cross-encoder score (from step 3, small model — reuse the fast pass).
- Query length, chunk length.
- Any behavioural or metadata features your corpus supports.

Train a LightGBM LambdaMART on 70% of the queries; measure nDCG@10 on the held-out 30%. Latency: how long does a LightGBM `predict` on 100 rows × ~10 features take (should be well under 1 ms).

If you skip this leg, note it explicitly in your writeup — this is a common defensible skip for early-stage projects.

**Step 6 (optional if you have an LLM API budget) — LLM-as-reranker leg (45 minutes).** Use a strong LLM (GPT-4, Claude 3.5, or an open-weight equivalent) as a listwise reranker. Feed the top-20 candidates from the first stage (not all 100 — cost) and prompt for a listwise ranking (chapter 4's RankGPT-style prompt). Take top-10 from the output. Measure the same three metrics.

Bound the cost — if you can't afford to run 100 queries at this leg, run 20 and note the small sample size.

**Step 7 — Assemble the matrix (30 minutes).** Produce a single table:

| Reranker | nDCG@10 | recall@10 | p95 latency | Cost / 1000 q | Ops overhead |
|---|---|---|---|---|---|
| Baseline (no rerank) | X.XX | X.XX | ~0 ms | $0 | none |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | | | | | GPU maintenance |
| BAAI/bge-reranker-v2-m3 | | | | | GPU + more memory |
| Cohere rerank-v3.5 | | | | | vendor + region |
| Voyage rerank-2 | | | | | vendor + region |
| LightGBM LTR | | | | | feature pipeline |
| GPT-4 listwise | | | | | LLM API + rate limits |

The "ops overhead" column is text, not a number — but it is part of the decision.

**Step 8 — Write the pick (20 minutes).** In a `pick.md` in your working directory:

- Name the family and specific model you would deploy.
- Justify it against the four constraints: quality, latency, cost, ops.
- Explicitly name one runner-up and the constraint under which you would switch to it.
- Note the strongest single-metric leader you did NOT pick and why (e.g., "GPT-4 listwise wins nDCG@10 by 3 points but costs 100× more and adds 800 ms; not worth it for our budget").

## Acceptance criteria

- Same first-stage top-100 per query across all rerankers — verified by running two rerankers over the same input file.
- Matrix with at least 3 reranker families (cross-encoder + hosted API + one of LTR / LLM) plus a baseline.
- Latency reported as p50 / p95 / p99, not just an average. Measure with real per-query timing, not a rough estimate.
- Cost reported per 1000 queries with the arithmetic shown (queries × per-query cost, or GPU $/hour × time). Vendor prices verified from the current pricing page, not from a training-data snapshot.
- Pick includes an explicit runner-up and the constraint switch condition.
- Writeup names at least one insight that is corpus-specific (e.g., "hosted APIs win on latency at our 5 QPS load; if we scale to 50 QPS the self-hosted line takes over").

## Stretch goals

- **A/B the hosted vendors head-to-head.** Add Cohere + Voyage + Jina all three to the matrix.
- **Cache-cost analysis.** Rerank scores are cacheable on `(query_hash, chunk_hash)`. Measure your eval set's cache hit rate assuming a rolling 24-hour cache. What fraction of hosted API cost would this save?
- **Failure mode sampling.** Pick 5 queries where the "best" reranker (by nDCG@10) *loses* to a cheaper one. Read them. Is there a query class where the loser dominates? This is the routing insight from mod-106 chapter 5 in a reranker context — some query classes benefit from a cheap fast reranker, some from an expensive slow one.
- **Reproducibility check.** Rerun the whole matrix a week later. Which cell drifted? Hosted APIs occasionally silently swap the underlying model; this is the diagnostic.
- **Add the "cross-encoder + LTR" stacked stack** (chapter 4). Rerank 100 → 20 with the cross-encoder; rerank 20 → 10 with LightGBM using the cross-encoder's score as one feature. Does the stack beat either single-stage leg on your data?

## Notes

- Do not tune your reranker choice on the same queries you measure recall on. Hold out 30% as a test set that you touch only for the final numbers.
- If you can't run *all* legs, run three and note the omissions explicitly. Missing legs are honest; fabricating numbers is not.
- Public benchmark numbers (BEIR, MS MARCO leaderboards) belong in your writeup only as *calibration* — "our number on FiQA is close to the published one, so our harness is likely correct." Do not report benchmark leaderboard rank as if it were an eval on your corpus.
- The dollars column matters *especially* for hosted APIs. A 2-point nDCG@10 improvement at 10× the cost is often a bad trade; the writeup should surface that trade explicitly.
- Chapter 3 named the network-hop latency floor. If your app runs in a different region from the vendor, the hosted-API p95 in this exercise is inflated; your production p95 will be different. Note the region alignment in your writeup.
