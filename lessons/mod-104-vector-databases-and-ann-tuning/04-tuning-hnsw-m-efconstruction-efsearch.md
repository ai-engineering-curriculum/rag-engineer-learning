# Chapter 4 — Tuning HNSW: `M`, `efConstruction`, `efSearch` Against a Recall and Latency Budget

## Why this chapter exists

HNSW's defaults get you 80% of the way there — and that is the trap. The other 20% is the difference between "we have an ANN index" and "we have an ANN index that hits a recall@10 of 0.95 at a p95 of 40 ms on our corpus." That 20% is three knobs: **`M`**, **`efConstruction`**, and **`efSearch`**. This chapter builds the intuition for each, gives you a repeatable tuning procedure, and lists the traps that trip up first-timers.

Chapter 3 explained *what* HNSW does. This chapter is about tuning it to a target — because the interviewer at level 30 will not ask "what is HNSW"; they will ask "how did you pick `M = 32` on this corpus?"

## The three knobs and what each moves

| Knob | When it takes effect | Higher value → | Cost of a higher value |
|---|---|---|---|
| **`M`** | Index build (per-node degree) | Better recall ceiling, more graph edges, more memory | Longer build; larger RAM footprint; slower insert. Cannot be changed after build. |
| **`efConstruction`** | Index build (search width during insertion) | Better graph quality, higher recall ceiling | Longer build. Cannot be changed after build. |
| **`efSearch`** | Query time (search width during traversal) | Higher recall, higher latency | Slower queries. Can be changed per-query, per-request. |

The critical distinction: **`M` and `efConstruction` are build-time**, so you commit to them when you build the index. **`efSearch` is per-query**, so you can (and should) tune it against a live recall/latency target without rebuilding.

## `M` — the target degree

Every node in the HNSW graph has up to `M` outbound edges at layers > 0 and up to `2M` at layer 0. `M` sets the graph's density.

- **Too low.** The graph becomes disconnected in dense regions of the vector space; the greedy walk gets stuck in local minima; recall ceiling drops. The Malkov & Yashunin 2016 paper shows `M = 5`–`8` failing to reach high recall on higher-dim datasets even with generous `efSearch`.
- **Too high.** Memory grows linearly with `M`, and per-node work at query time grows too. Diminishing returns kick in quickly past `M ≈ 32` on typical text-embedding dimensions.

**Rule of thumb.** `M = 16` for lower-dim (≤ 256) or when memory is tight; `M = 32` for typical text-embedding dimensions (384 – 1024); `M = 48` – `64` only if you need every last point of recall and are willing to pay the memory cost. The Malkov paper's figure 3 sweeps `M ∈ {4, 8, 16, 32, 48, 96}` on multiple datasets and shows the curve flattens around 32 for most.

**Memory cost.** Rough estimate for `float32` vectors of dimension `d` on `N` vectors:

- **Raw vectors:** `N · d · 4` bytes.
- **HNSW graph:** ~`N · M · 4 · 2` bytes (edges are 4-byte IDs; roughly `2M` edges at layer 0 dominate). For `M = 32`, `N = 10⁷`: ~2.5 GB just for the graph.

For a `d = 1024`, `N = 10⁷`, `M = 32` corpus: ~40 GB raw + ~2.5 GB graph ≈ 42.5 GB in RAM. Doubling `M` to 64 adds another 2.5 GB, and adds a proportional per-query traversal cost.

## `efConstruction` — how carefully we insert

`efConstruction` is the beam width HNSW uses to *find candidate neighbors during insertion*. If it is set too low, each new node ends up connected to a nearest-but-not-quite-nearest neighbor set — the graph is technically valid but suboptimal, and every subsequent query pays.

- **Too low.** Recall ceiling drops even at high `efSearch`; the graph itself is impoverished.
- **Too high.** Build time grows linearly (and beyond a point, no measurable recall improvement).

**Rule of thumb.** `efConstruction = 200` is the widely-used default across pgvector, Weaviate, Qdrant, and hnswlib for good reason: it is a solid recall/build-time trade-off on typical corpora. Go to `400`–`500` for maximum-recall builds, or drop to `64`–`100` for rapid prototype builds where you plan to rebuild soon.

**Diagnostic.** If you sweep `efSearch` at query time and your recall curve saturates below your target, `efConstruction` was too low at build time — you need to rebuild. That is a costly loop, so lean toward the higher-quality default for the production build.

## `efSearch` — the query-time knob

Everything you actually operate on at runtime hangs off `efSearch`. It is the beam width the search uses on the bottom layer.

- **`efSearch < k`** — the algorithm still returns `k` neighbors but can't explore properly; recall collapses. Set `efSearch ≥ k` as a floor.
- **`efSearch ≈ k`** — fastest queries, lowest recall.
- **`efSearch >> k`** — highest recall, longest queries. Beyond a point (usually `efSearch ≥ 4k` – `8k`), returns diminish.

**How to use it.** Sweep `efSearch ∈ {10, 20, 40, 80, 160, 320, 640}` against your recall@k target on a held-out judgment set. The curve is monotonic: recall increases with `efSearch` (until it saturates), latency also increases. Pick the smallest `efSearch` that hits your recall floor.

**Per-request tuning.** Because `efSearch` is per-request, you can serve different users different values — e.g. an admin dashboard that tolerates 500 ms can use `efSearch = 640` for extra recall while a customer-facing search uses `efSearch = 80`. This is the shape most production stacks converge on.

**Availability by system.**

- pgvector: `SET hnsw.ef_search = 100;` per session or per transaction.
- FAISS: `index.hnsw.efSearch = 100`.
- Qdrant: `hnsw_ef` in `SearchParams`.
- Weaviate: `ef` on the collection or overridable per query since v1.19.
- Chroma: exposed as a configuration option on the collection.

## The tuning procedure — a repeatable playbook

The procedure below assumes you have (a) an embedding model chosen (mod-103), (b) a corpus embedded and stored, (c) a small labeled judgment set (queries → true relevant document IDs — mod-109 covers building this), and (d) a way to measure recall@k and p95 latency end-to-end.

**Step 0 — Pin the ground truth.**

You need a "true nearest neighbor" set to compute recall against. If your corpus is small (`N ≤ 10⁶`), run an exact brute-force nearest-neighbor pass over your judgment queries once and cache the ground-truth top-100 per query. That cache is your recall reference — every ANN configuration's recall is computed as `|ANN_top_k ∩ exact_top_k| / k`.

If your corpus is large, run the exact pass on a stratified sample of the corpus (per Aumüller et al. 2020, ann-benchmarks methodology, this is acceptable practice).

**Step 1 — Pick a build config.**

Start with `M = 32`, `efConstruction = 200`. Build. Measure build time and memory. If either is unacceptable, tune down `M` (to 16) or `efConstruction` (to 100).

**Step 2 — Sweep `efSearch`.**

For each `efSearch ∈ {10, 20, 40, 80, 160, 320, 640}`, measure:

- `recall@k` (for the `k` your application uses; commonly `k = 10` or `k = 20`).
- `p50, p95, p99 latency` end-to-end (query encode + ANN scan + result assembly — measure the shape your API serves, not just the ANN call).

Plot `recall` (y) vs `p95 latency` (x). This is your recall–latency Pareto curve for this build.

**Step 3 — Pick the operating point.**

Pick the smallest `efSearch` that hits your recall floor at a p95 within budget. Almost every mature retrieval stack has this operating point written down as `recall@10 ≥ 0.95 @ p95 ≤ 50 ms on hardware X`.

**Step 4 — Verify with a rebuild.**

If your recall curve saturates below the target — even at `efSearch = 640` — your build is limited by `M` or `efConstruction`. Rebuild with `M = 48` and/or `efConstruction = 400`. Repeat steps 2–3.

**Step 5 — Freeze and record.**

Record the tuple `(M, efConstruction, efSearch, embedding-model, chunker-version, corpus snapshot, hardware)` alongside the resulting `recall@k` and `p95 latency`. That tuple is what you defend the choice with — it is the artifact chapter 8's index-versioning convention is built around.

## What the ann-benchmarks methodology adds

[ann-benchmarks](https://ann-benchmarks.com/) (Aumüller, Bernhardsson & Faithfull 2020; [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)) is the reference benchmark for ANN algorithm comparisons. It runs each configuration of each algorithm on standardized public datasets (GloVe, SIFT, Deep1B, GIST, NYTimes, etc.) and plots the recall–QPS Pareto front.

The methodology is important beyond the specific datasets:

1. **Report the Pareto front, not one point.** A single "recall = 0.9 at 20 ms" is meaningless without knowing what other points that algorithm could have hit on the same hardware. A full sweep of `efSearch` (or `nprobe`) gives you the curve to compare against.
2. **Compare like-with-like hardware.** ann-benchmarks pins hardware; your internal comparison should too. A different CPU changes numbers.
3. **Distance measure matters.** Same corpus, cosine vs. dot product vs. L2 gives different recall curves.

You will not typically run ann-benchmarks *itself* against your production corpus — the tool is set up for public datasets — but you will apply its methodology: sweep, plot Pareto, pick a point. Chapter 5's exercise puts this into your hands on FAISS.

## The five traps that trip up first-timers

1. **Post-filter under a selective predicate.** Your recall is 0.95 measured with no filter, then in production you add `WHERE tenant_id = 42` and recall craters. That is not an HNSW-tuning problem; it is a pre-filter-vs-post-filter problem — chapter 7.
2. **Rebuild is not free.** Old pgvector versions with heavy incremental inserts drift below a clean-build's recall on the same `(M, efConstruction)`. Periodic `REINDEX CONCURRENTLY` is part of the operational plan.
3. **`efSearch < k` returns garbage.** Silently. There is no warning; recall just drops. Set `efSearch ≥ k` as a floor everywhere.
4. **Distance measure mismatch.** If your embedding model was trained for cosine similarity and you index with L2, your recall against ground truth is off — sometimes badly. Normalize your vectors and use inner product (equivalent to cosine on L2-normalized vectors), or use the engine's cosine operator, but do not mix them.
5. **Measuring recall against ground truth from a different `k`.** Recall@10 with `k = 10` requested vs. recall@10 with `k = 100` requested (truncated to top-10) are different numbers, both of which can be legitimately reported. Fix `k` and the ground-truth cardinality at the start of the sweep.

## A concrete tuning script — FAISS

The shape below is what an exercise-02 script looks like against a single index. Real sweeps loop the outer `M` and `efConstruction` too.

```python
import numpy as np
import faiss
import time

d = 384
N = 100_000
np.random.seed(0)
xb = np.random.randn(N, d).astype("float32")

# Small judgment set: 200 held-out queries with known top-10 ground truth
xq = np.random.randn(200, d).astype("float32")

# Ground truth via brute-force
flat = faiss.IndexFlatIP(d)
faiss.normalize_L2(xb); faiss.normalize_L2(xq)
flat.add(xb)
_, gt_top10 = flat.search(xq, 10)

# Build once
index = faiss.IndexHNSWFlat(d, 32)          # M = 32
index.hnsw.efConstruction = 200
index.add(xb)                                # (times & memory logged elsewhere)

# Sweep efSearch
for ef in [10, 20, 40, 80, 160, 320, 640]:
    index.hnsw.efSearch = ef
    t0 = time.perf_counter()
    _, ann_top10 = index.search(xq, 10)
    dt_ms = (time.perf_counter() - t0) * 1000 / len(xq)

    recall = np.mean([
        len(set(ann_top10[i]) & set(gt_top10[i])) / 10
        for i in range(len(xq))
    ])
    print(f"efSearch={ef:>4}  recall@10={recall:.3f}  mean latency={dt_ms:.2f} ms")
```

A production version measures `p50 / p95 / p99` (not just mean), runs on the actual serving hardware, and uses real judgment queries with real relevance labels. Exercise 02 walks you through that end-to-end using ann-benchmarks on a standard dataset.

## What is NOT in this chapter (and where it goes)

- **IVF-PQ tuning** — chapter 5. The two chapters mirror each other: this one tunes the graph, the next one tunes the partitioning.
- **Filter interaction with HNSW** — chapter 7. Filter selectivity changes what `efSearch` means.
- **Sizing and shard/replica strategy** — chapter 6. This chapter tunes a single-node index; the platform pieces come next.
- **Managed-service tuning** (Pinecone, OpenSearch Serverless) — you tune shard/replica count and pod tier rather than `M` / `efSearch` directly. The reasoning below still applies; the knobs are one level higher.

## What you should be able to do after this chapter

- State what each of `M`, `efConstruction`, `efSearch` moves and what it costs.
- Describe the tuning procedure from ground-truth cache through `efSearch` sweep to Pareto plot.
- Name the five traps and how to detect each in an existing deployment.
- Read someone else's HNSW config and predict, roughly, where it sits on the recall–latency curve.

The next chapter runs the same shape for IVF-PQ (`nlist`, `nprobe`, `pq_m`, `pq_nbits`) and introduces ann-benchmarks as the reference methodology.

## Further reading

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320). Figures 3–5 show the `M` / `efConstruction` / `efSearch` sweeps you should recognize.
- hnswlib documentation, [`github.com/nmslib/hnswlib#hnsw-algorithm-parameters`](https://github.com/nmslib/hnswlib#hnsw-algorithm-parameters) — Malkov's own parameter-guidance write-up.
- FAISS wiki, [*HNSW*](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#hnsw) — implementation notes and defaults.
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems 87. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614). Also the [live leaderboard](https://ann-benchmarks.com/).
- pgvector README, [*Index Options*](https://github.com/pgvector/pgvector#hnsw) — how the same knobs appear in Postgres syntax.
