# Exercise 04 — Embedding Cost and Vector-Storage Cost Optimization Drill

**Estimated effort:** ~3 hours. Python, the mod-109 judgment set, a vector DB that exposes at least one quantization mode (Qdrant / Weaviate / pgvector's `halfvec` + IVFFlat / Milvus / Pinecone with `int8` — Qdrant is recommended for the cleanest quantization API), and the build scripts from exercise 01.

## Objective

Author the per-query cost model from chapter 5, instrument the retrieval service to populate it, and run a **cost + latency + quality sweep** across a small set of well-defined optimization moves (cache-on, `efSearch` lowering, scalar quantization, embedding-model downshift, reranker-window shrink). Produce a Pareto table that a stakeholder can read to pick the tuple they want to ship, and confirm each move through the mod-109 judgment-set eval so no move ships silent-regression risk.

By the end you have:

- `cost/model.py` — a per-query cost function that reads the vendor-published rates and instrumentation output.
- `cost/dashboard.md` — a per-day summary of embedding cost, vector-DB cost amortized, reranker cost, and cache hit rate for your project.
- `sweep/results.md` — a table of at least five optimization moves, each with `(recall@k, nDCG@10, MRR@10, p95 latency, cost/month)` deltas plus paired-Wilcoxon `p` values.
- `sweep/decision.md` — a one-paragraph recommendation for the tuple to ship and the tradeoffs the stakeholder should be aware of.

## Prerequisites

- Chapter 5 read; chapters 3 and 6 helpful.
- Exercise 01 done (build scripts) and exercise 02 done (CI eval harness).
- mod-104 chapters 4 and 5 read (ANN parameters as latency levers) and mod-107 chapter 5 read (reranker funnel).
- A vector DB that supports at least one quantization mode. Qdrant's [quantization guide](https://qdrant.tech/documentation/guides/quantization/) is the most-approachable local option; Weaviate's [compression docs](https://weaviate.io/developers/weaviate/configuration/compression) work as well.
- Access to two embedding models at different price/quality points (e.g., a `small` and a `large` variant from the same vendor, or a local `bge-small` + `bge-large` pair).
- Python 3.10+, `pip install ir_measures scipy`.

## Steps

**Step 1 — Instrument the retrieval service (25 minutes).**

Emit chapter 5's per-request field set on every retrieval call from exercise 01's retriever:

- `embedding_tokens` (query embedding, or `0` on cache hit)
- `retriever_latency_ms`
- `retriever_top_n`
- `reranker_input_pairs` and `reranker_latency_ms` (0 if reranker was skipped)
- `cache_hit_query_embed`, `cache_hit_top_k`, `cache_hit_generation` (all `false` for now — you'll wire caches in exercise 05)
- `INDEX_ID`
- `tenant_id` (if you have one; otherwise `"default"`)

Log them to a JSONL file or Prometheus counters — whatever your local setup supports. Run 200 queries from the judgment set through the service and confirm the log populates.

**Step 2 — Author the per-query cost model (30 minutes).**

`cost/model.py` implements chapter 5's formula:

```python
def cost_per_request(fields, rates):
    embed = fields["embedding_tokens"] * rates["embed_per_token"]
    db    = rates["db_amortized_per_query"]
    rerank = fields["reranker_input_pairs"] * rates["rerank_per_pair"]
    return embed + db + rerank
```

`rates` reads:

- `embed_per_token` from the vendor pricing page (OpenAI, Voyage, Cohere) or your self-hosted GPU-hour amortization.
- `db_amortized_per_query` = total DB monthly cost / total monthly queries. Use a stand-in for the exercise (e.g., $50 / month at 1M queries = $5e-5 per query).
- `rerank_per_pair` from the reranker vendor pricing or your self-hosted GPU cost.

Also implement `cost_per_stored_vector(dim, quantization, storage_rate)` for the storage side. `4 × dim` bytes for float32, `1 × dim` for int8, `dim / 8` bytes for binary; sum across `n_vectors`; multiply by the storage rate.

Publish the current numbers into `cost/dashboard.md` as a starting-point snapshot.

**Step 3 — Baseline the sweep (15 minutes).**

Establish the reference:

- **Tuple:** whichever you built in exercise 01 (e.g., `bge-large` + HNSW `M32 efSearch=200` + no reranker).
- **Judgment set:** the mod-109 set.
- **Metrics:** `recall@50, nDCG@10, MRR@10` (aggregate + per-query arrays).
- **Cost:** run 200 queries through the instrumented service, sum the per-request costs, and extrapolate to a month at your assumed query volume.
- **Latency:** `p50` and `p95` on a warm-cache-free canary set (20 queries, cold).

Record the baseline row in `sweep/results.md`.

**Step 4 — Run the sweep (60 minutes).**

Each move is one variation from the baseline; run each independently (do not compound moves in this exercise — the aim is to attribute deltas cleanly).

| # | Move | What changes |
|---|---|---|
| 1 | Add query-embedding cache | `cache_hit_query_embed` fires on repeats; cost of embed goes to 0 on hit |
| 2 | Lower `efSearch` 200 → 100 | ANN parameter; expect small recall drop, meaningful latency drop |
| 3 | Enable scalar (int8) quantization | Storage ~4×; small recall drop; small latency change |
| 4 | Lower reranker window 50 → 20 (if you have a reranker; skip if not) | Reranker cost / latency drop proportional; quality drop on tail queries |
| 5 | Downshift the embedding model to the small variant | Cost drops significantly; quality drops noticeably; latency drops modestly |

For each row:

- Rebuild whatever the change requires (a full rebuild for the quantization or model change; live for the ANN param).
- Rerun the mod-109 eval and record per-query values.
- Compute paired-Wilcoxon `p` on the delta vs. baseline (mod-109 chapter 4 + `scipy.stats.wilcoxon`).
- Compute the cost model on 200 fresh queries and extrapolate.
- Measure latency on the canary set.

Populate `sweep/results.md`:

```markdown
| Move | recall@50 | nDCG@10 | p (Wilcoxon) | p95 latency | cost/month | storage GB |
|---|---|---|---|---|---|---|
| baseline | 0.803 | 0.487 | — | 82 ms | $1240 | 4.10 |
| +query-embed cache (40% hit) | 0.803 | 0.487 | — | 51 ms | $850 | 4.10 |
| efSearch 200 → 100 | 0.788 | 0.480 | 0.041 | 63 ms | $1240 | 4.10 |
| int8 quantization | 0.795 | 0.481 | 0.312 | 78 ms | $1240 | 1.03 |
| rerank 50 → 20 | 0.803 | 0.472 | 0.006 | 68 ms | $920 | 4.10 |
| downshift embed model | 0.751 | 0.443 | <0.001 | 64 ms | $410 | 2.05 |
```

Fill with your own numbers.

**Step 5 — Storage sizing math (10 minutes).**

Independent of the sweep, apply chapter 5's storage formula. For your corpus's `n_chunks`:

- Compute float32 storage in GB.
- Compute int8 (~4×), PQ-16 (~32×), binary (~32×) storage.
- Look up your vector-DB vendor's storage rate ($/GB/month). Pinecone, Weaviate Cloud, and Qdrant Cloud publish these; self-hosted is your amortized disk cost.
- Report per-quantization annual storage cost in `cost/dashboard.md`.

**Step 6 — Interpret and recommend (15 minutes).**

Write `sweep/decision.md`:

- **Which tuple you would ship**, and why. Show the recall / cost / latency tradeoff explicitly ("we accept X ms latency for Y% cost savings; recall drop is not significant").
- **Which moves are safe to combine**, and which conflict (e.g., quantization + `efSearch` reduction may compound recall drop non-additively — flag it and note that a combined-move eval is a follow-up).
- **Which moves you rejected**, and why (e.g., the model downshift dropped recall significantly on jargon queries; not worth the cost saving for this product).

**Step 7 — Alert-on-cost-drift (5 minutes).**

Add a single alert rule (log line is fine) to `cost/model.py`:

- Cache-hit-rate drops by ≥ 20% relative over a rolling 7-day window → WARN.
- Per-request cost climbs by ≥ 20% over rolling 7-day → WARN.

Chapter 5's Failure A ("cache-hit rate fell from 40% to 3% overnight; embedding bill 4×'d") is what this alert defends against. Mention the failure and the defense in `cost/dashboard.md`.

## Acceptance criteria

- The retrieval service emits all six per-request fields on every call.
- `cost/model.py` computes per-request cost and per-vector storage cost from vendor-published rates.
- `sweep/results.md` contains at least five optimization moves each with recall, nDCG, MRR, latency, and cost columns and a paired-Wilcoxon `p` value.
- The three-way trade (cost vs. latency vs. quality) is explicit in the write-up; each move states which axes it moves.
- `sweep/decision.md` names the tuple you would ship and the tradeoffs.
- Storage sizing math is documented at each quantization level for your `n_chunks`.
- A cost-drift alert rule is implemented and referenced in the dashboard.

## Stretch goals

- **PQ or binary quantization.** Add a PQ-16 row to the sweep. Confirm the recall drop is larger and more workload-dependent than scalar int8. If your DB supports it, add a two-stage rerank-with-original for PQ / binary rows and observe how much recall comes back.
- **Concurrent moves.** Pick the two most-attractive moves from the sweep, apply both, and re-eval. Report whether the combined delta equals the sum of individual deltas or shows non-additive behavior.
- **Self-hosted vs. hosted crossover.** Given your query volume, compute the volume at which switching from hosted embedding to self-hosted (on a specific GPU class) breaks even. Cite your amortized GPU-hour cost and the vendor's per-token rate.
- **Per-slice cost report.** If the judgment set is stratified, report per-slice cost and quality so cost savings that concentrate a regression on one slice are visible.
- **Budget-aware CI.** Extend exercise-02's CI gate to also compute the cost delta and comment it alongside the quality delta.
- **Reranker off entirely.** Add a "no reranker" row to the sweep. For some workloads the retriever + `top_k=50` alone is within the quality budget; skipping the reranker is the largest single cost win available.

## Notes

- **Measure before you optimize.** The cost model built in step 2 is what tells you which line item dominates. Guessing wastes weeks; a per-day dashboard is the antidote.
- **Every sweep row is a full mod-109 eval.** Skipping the eval on the "obviously safe" quantization row is chapter 5's Failure B — don't ship it.
- **Rate cards drift.** OpenAI, Voyage, Cohere, and every vector DB vendor change their pricing multiple times per year. Bake the rate lookup into a config file so a rate update is a one-line change, not a scavenger hunt.
- **The stakeholder decision belongs in `sweep/decision.md`, not in a Slack thread.** The write-up is the artifact a future maintainer reads to answer "why did we ship this tuple and not the cheaper one" — invest in it.
- **Do not confuse latency (ANN parameter) with cost (token rate) in the write-up.** Chapter 5's three-way trade separates them; keep the columns distinct.
