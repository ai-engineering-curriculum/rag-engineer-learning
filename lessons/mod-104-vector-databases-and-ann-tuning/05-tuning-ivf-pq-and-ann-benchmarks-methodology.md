# Chapter 5 — Tuning IVF-PQ (`nlist`, `nprobe`, `pq_m`, `pq_nbits`) and the ann-benchmarks Methodology

## Why this chapter exists

Chapter 4 tuned a graph. This chapter tunes a partitioning with compressed codes. The mental model changes: with HNSW you are choosing *how much of the graph to walk*; with IVF-PQ you are choosing *how many cells to visit* and *how aggressively to compress the vectors themselves*. That second knob — compression — is the reason IVF-PQ exists, and it is the reason recall degrades in a way HNSW's does not: quantization introduces a distance-approximation error that a plain `efSearch` bump cannot recover.

The chapter also introduces **ann-benchmarks** as the reference methodology — the discipline every level-30 tuning writeup borrows from — because IVF-PQ has four knobs to sweep and unstructured sweeps go nowhere.

## The four knobs and what each moves

| Knob | Effect | Build cost | Query cost | Memory cost | Runtime-tunable? |
|---|---|---|---|---|---|
| **`nlist`** | Number of IVF cells / posting lists | k-means training grows with `nlist` | Slight — more centroids to compare `q` against | Small — centroid table only | No — set at build |
| **`nprobe`** | Number of cells to inspect at query time | — | Linear in `nprobe` | — | Yes — per query |
| **`pq_m`** | PQ code length (subvectors per vector) | PQ training per subspace | Linear in `pq_m` (distance table lookups) | Linear in `pq_m` (bytes per code) | No — set at build |
| **`pq_nbits`** | Bits per PQ code (almost always 8) | Trains `2^pq_nbits` centroids per subspace | Fixed per lookup | Multiplies code size by `pq_nbits / 8` | No — set at build |

The critical distinction: **only `nprobe` is runtime-tunable.** The other three are baked into the index. That has two consequences: (1) you sweep the build parameters offline and pick a build; (2) at runtime, `nprobe` is your recall/latency dial the way `efSearch` was for HNSW.

## `nlist` — how finely you partition the corpus

`nlist` is the number of k-means centroids. Each corpus vector is assigned to its nearest centroid; that assignment defines the posting list ("cell") the vector lives in.

- **Too low.** Each cell is huge, so each probe scans many vectors — high recall but bad latency.
- **Too high.** Each cell is tiny, so you need more probes to cover the true neighborhood — worse recall at the same `nprobe`, and k-means training gets expensive.

**Rule of thumb** (from the FAISS wiki [*Guidelines to choose an index*](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)):

| Corpus size `N` | Recommended `nlist` |
|---|---|
| < 10⁶ | `4·√N` … `16·√N` (e.g. 128 – 1024 for `N = 10⁵`) |
| 10⁶ – 10⁷ | `√N` … `4·√N` (a few thousand) |
| 10⁷ – 10⁸ | `4·√N` … `16·√N` (~30k – 100k) |
| > 10⁸ | Consider IVF composed with a coarser index (`IVF65536_HNSW32,PQ64`-style factory strings) |

Training requires `≥ ~30·nlist` training vectors per FAISS guidance — feed enough of your corpus into `index.train(...)` or you will get warnings and poorly-centred cells.

## `nprobe` — the runtime recall/latency knob

At query time, IVF computes distances from `q` to each of the `nlist` centroids, picks the `nprobe` closest cells, and exhaustively scores the vectors in those cells (with PQ, that scoring is the compact-code dot product).

- **`nprobe = 1`** — fastest, lowest recall.
- **`nprobe = nlist`** — equivalent to exact within the cells (no partition pruning); slowest.
- **`nprobe = 5%–15% of nlist`** — the typical operating range for high-recall use cases.

Sweep `nprobe ∈ {1, 2, 4, 8, 16, 32, 64, 128}` (or whatever powers of two land inside your `nlist`) and plot recall vs. latency, just like `efSearch`.

## `pq_m` — how many bytes per vector

`pq_m` is the number of subvectors the vector is split into. Each subvector becomes one PQ code (one byte at the default `pq_nbits = 8`). Higher `pq_m` = larger code = better recall + more memory.

Constraints and trade-offs:

- **`pq_m` must divide `d`.** For `d = 1024`, valid `pq_m` include 8, 16, 32, 64, 128, 256. For `d = 768`, valid `pq_m` include 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384.
- **Compression ratio** = `d · sizeof(float) / pq_m` (at `pq_nbits = 8`). For `d = 1024`, `pq_m = 64`: 4096 / 64 = 64× compression.
- **Recall / compression frontier** is monotonic: more bytes per vector = higher recall, all else equal.

**Rule of thumb** (again from the FAISS wiki):

- `pq_m ∈ {d/8, d/4}` is the usual range — big enough to preserve recall, small enough to matter for memory.
- Pushing to `pq_m = d/2` or `pq_m = d` gives you a strong recall at ~4×–8× compression and is a reasonable default when memory is not the binding constraint.
- Aggressive compression (`pq_m = d/16` or lower) needs an OPQ pre-rotation to hold recall (see below), and even then a re-rank pass on the raw vectors for the top candidates.

## `pq_nbits` — bits per code

Almost always 8. Some FAISS index variants support 4 bits (`IndexIVFPQFastScan`, `IndexPQFastScan`) which halves memory at some recall cost and enables SIMD-friendly scoring on modern CPUs. If you ever consider 12 or 16 bits, you are usually better off increasing `pq_m` instead.

## OPQ — the free-recall pre-rotation

Ge et al. 2013 (*Optimized Product Quantization*, [DOI:10.1109/CVPR.2013.379](https://ieeexplore.ieee.org/document/6619297)) showed that PQ's error depends on how variance is distributed across subspaces. If you pre-rotate the vectors so that each subspace carries roughly equal variance, PQ quantizes each subvector more accurately — at query time you pay one extra `d × d` matrix multiply, essentially free compared to the scoring loop.

**When to use it.** Almost always, if your database exposes it. FAISS: `faiss.index_factory(d, "OPQ64,IVF4096,PQ64")` builds the composed index. It is a "free" recall lift at aggressive compression settings.

## Re-rank — the cheap recall recovery

At small `pq_m` (aggressive compression), the top-`k` from IVF-PQ is often noisy. The fix that works well in practice is:

1. Retrieve top-`R` from IVF-PQ, with `R > k` (e.g. `R = 100` and `k = 10`).
2. Fetch the *raw* `float32` vectors for those `R` candidates (from wherever you stored them — a side database, a mmap'd file).
3. Rescore exactly and return the top-`k`.

FAISS ships this as `IndexRefine` and `IndexRefineFlat`. The re-rank cost is `O(R · d)` per query — a few hundred µs when `R = 100` and `d = 1024` — and buys back a meaningful chunk of recall lost to aggressive PQ. This is why IVF-PQ + refine sits above pure IVF-PQ on the frontier chart in chapter 3.

## The tuning procedure — a repeatable playbook

Same shape as chapter 4, adjusted for the four knobs.

**Step 0 — Pin the ground truth.**

Exact brute-force top-100 per judgment query, cached. (Same as chapter 4.)

**Step 1 — Pick a build config.**

A reasonable starting point for `d = 1024`, `N = 10⁷`:

- `nlist = 4096` (≈ 4·√N ≈ 12,000; scaled down to a friendly power of two — either works).
- `pq_m = 64` (16× compression at `pq_nbits = 8`).
- `pq_nbits = 8`.
- OPQ pre-rotation on.

Train on ~1M vectors from the corpus. Measure build time and memory.

**Step 2 — Sweep `nprobe`.**

For each `nprobe ∈ {1, 4, 8, 16, 32, 64, 128}`, measure `recall@k` and `p50 / p95 / p99` latency on your judgment set. Plot recall vs. p95 latency.

**Step 3 — Decide whether to rebuild.**

If the recall curve saturates below your target — even at `nprobe = 128` — the ceiling is set by `pq_m` (compression too aggressive) or `nlist` (partitioning misaligned with the corpus distribution). Rebuild with a larger `pq_m`, or add `IndexRefine` on top, and repeat.

**Step 4 — Compare against IVF-PQ + refine.**

Sweep `R ∈ {50, 100, 200, 400}` for the refine step. Plot the additional latency cost against the recall lift. Often you can push `pq_m` *down* (more compression, less memory) and use refine to buy the recall back — a strictly better memory point at the same recall.

**Step 5 — Freeze and record.**

Record `(nlist, nprobe, pq_m, pq_nbits, OPQ, refine_R, embedding-model, chunker-version, corpus snapshot, hardware) → (recall@k, p95 ms, RAM GB)`. That tuple is the defended choice.

## The ann-benchmarks methodology — the discipline you borrow

Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems 87. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614). Website: [ann-benchmarks.com](https://ann-benchmarks.com/). Repo: [`github.com/erikbern/ann-benchmarks`](https://github.com/erikbern/ann-benchmarks).

**What the tool does.** ann-benchmarks pins a set of standard datasets (SIFT1M, GloVe, GIST, NYTimes, Deep1B, and a few more), a distance function, a `k`, and a hardware target. It then runs each configuration of each ANN algorithm on that fixed setup, sweeping the algorithm's runtime knob (`efSearch` for HNSW, `nprobe` for IVF variants), and produces a plot of **recall vs. queries-per-second**.

**What the methodology teaches you** — five discipline points a tuning writeup should absorb whether or not it uses the tool:

1. **Always plot a curve, never a point.** A single "recall = 0.9 at 20 ms" is a snapshot along a curve; the curve is the artifact.
2. **Fix the hardware.** Same CPU, same threading model, same query batch size. Otherwise numbers do not compose across runs.
3. **Fix `k` and the ground-truth cardinality.** Recall@10 with ground-truth-10 and recall@10 with ground-truth-100 are different numbers.
4. **Cite the distance function.** Cosine, inner product, and L2 give different curves on the same corpus.
5. **Compare like-with-like.** IVF-PQ at 64× compression vs. HNSW at raw `float32` is not a fair Pareto comparison — HNSW's memory cost is 64× higher. Report memory in your comparison, not just recall / latency.

**When to use ann-benchmarks itself.** For algorithm-comparison sanity checks (does my HNSW build reach the recall the paper claims on GloVe?) and for reproducing published numbers. For *your corpus*, you replicate the methodology on your own data (that is exercise 02).

## The three traps specific to IVF-PQ

1. **Under-trained centroids.** k-means with too few training vectors produces skewed cells that hurt recall at every `nprobe`. Feed at least ~30·`nlist` training vectors, ideally more, per FAISS guidance.
2. **Distribution drift.** IVF-PQ trained on today's corpus starts drifting once the corpus mutates — new document topics land in cells that were assigned before those topics existed. Monitor recall on a stable judgment set and re-train PQ + IVF when it drops. Chapter 8's versioning + dual-index cutover is the operational recipe.
3. **PQ code collisions cap top-1 recall.** Compact codes mean many corpus vectors share the same code — `2^(pq_m · pq_nbits)` unique codes, versus `N` corpus vectors. Top-1 recall on a large corpus has a *combinatorial* ceiling below 1.0 at aggressive compression. Refine restores it.

## Composing IVF-PQ with other indexes — FAISS `index_factory` strings

FAISS's `index_factory` composes these building blocks into named configurations. The wiki page ([*The index factory*](https://github.com/facebookresearch/faiss/wiki/The-index-factory)) is the reference; a few common shapes:

- `"IVF4096,Flat"` — plain IVF, no quantization. Baseline.
- `"IVF4096,PQ64"` — IVF-PQ, 64-byte codes.
- `"OPQ64,IVF4096,PQ64"` — OPQ pre-rotation + IVF-PQ.
- `"IVF4096,PQ64,RFlat"` — IVF-PQ with exact-`float32` refine of the top candidates.
- `"IVF65536_HNSW32,PQ64"` — IVF with a HNSW-based coarse quantizer (faster centroid lookup at very large `nlist`).
- `"HNSW32,Flat"` — plain HNSW.
- `"HNSW32,PQ64"` — HNSW with PQ storage inside each node.

Notice the last one: HNSW + PQ is a common shape in the specialist databases (Qdrant, Weaviate) — the graph is HNSW, but the vectors it stores are PQ-compressed. The tuning applies both chapters' knobs at once.

## A concrete tuning script — FAISS IVF-PQ

```python
import numpy as np, faiss, time

d, N = 128, 200_000
np.random.seed(0)
xb = np.random.randn(N, d).astype("float32")
xq = np.random.randn(200, d).astype("float32")
faiss.normalize_L2(xb); faiss.normalize_L2(xq)

# Ground truth
flat = faiss.IndexFlatIP(d); flat.add(xb)
_, gt10 = flat.search(xq, 10)

# IVF-PQ build sweep
for nlist in [256, 1024]:
    for pq_m in [8, 16, 32]:
        assert d % pq_m == 0
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.train(xb); index.add(xb)

        for nprobe in [1, 4, 16, 64]:
            index.nprobe = nprobe
            t0 = time.perf_counter()
            _, top10 = index.search(xq, 10)
            dt_ms = (time.perf_counter() - t0) * 1000 / len(xq)
            recall = np.mean([
                len(set(top10[i]) & set(gt10[i])) / 10
                for i in range(len(xq))
            ])
            bytes_per_vec = pq_m
            print(
                f"nlist={nlist:<4} pq_m={pq_m:<2} nprobe={nprobe:<3} "
                f"recall@10={recall:.3f} lat={dt_ms:.2f}ms bytes/vec={bytes_per_vec}"
            )
```

Exercise 02 runs a real sweep on a standard ann-benchmarks dataset (SIFT1M is a good starter) and produces the Pareto plot as the deliverable.

## What is NOT in this chapter (and where it goes)

- **Sizing and shard/replica strategy for the deployed IVF-PQ index** — chapter 6.
- **How this composes with metadata filtering** — chapter 7.
- **DiskANN's disk-tiered graph shape** — mentioned in chapter 3; deeper coverage is out of scope for this module.
- **GPU-accelerated IVF-PQ (FAISS-GPU, cuVS/RAFT)** — level-40 platform concern; chapter 9's boundary.

## What you should be able to do after this chapter

- Name the four IVF-PQ knobs and what each moves.
- Describe the tuning procedure: pin ground truth → build → sweep `nprobe` → decide re-build vs. refine → freeze.
- Cite the ann-benchmarks methodology as the reason your comparison plots have five properties (curve, hardware pinned, `k` fixed, distance function named, memory reported).
- Read a FAISS `index_factory` string and predict its recall–latency–memory profile.

The next chapter zooms back out to operational concerns: memory footprint per million vectors, shard/replica placement, and the update cadence for the tuned index.

## Further reading

- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). [PDF](https://inria.hal.science/inria-00514462v2/document).
- Ge, T., et al. (2013). *Optimized Product Quantization.* CVPR 2013. [DOI:10.1109/CVPR.2013.379](https://ieeexplore.ieee.org/document/6619297).
- Guo, R., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396) — the ScaNN paper; the anisotropic loss idea generalizes to any product quantizer.
- Sun, P., et al. (2024). *SOAR: Improved Indexing for Approximate Nearest Neighbor Search.* [arXiv:2404.00774](https://arxiv.org/abs/2404.00774).
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks.* [arXiv:1807.05614](https://arxiv.org/abs/1807.05614) · [ann-benchmarks.com](https://ann-benchmarks.com/) · [`github.com/erikbern/ann-benchmarks`](https://github.com/erikbern/ann-benchmarks).
- FAISS wiki: [*Guidelines to choose an index*](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), [*The index factory*](https://github.com/facebookresearch/faiss/wiki/The-index-factory), [*Faster search*](https://github.com/facebookresearch/faiss/wiki/Faster-search).
