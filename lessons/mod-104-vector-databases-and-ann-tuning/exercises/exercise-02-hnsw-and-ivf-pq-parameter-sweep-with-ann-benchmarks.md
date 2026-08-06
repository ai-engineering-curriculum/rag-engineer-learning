# Exercise 02 — HNSW and IVF-PQ Parameter Sweep with `ann-benchmarks` Methodology

**Estimated effort:** ~3 hours. All Python + FAISS. No new infrastructure.

## Objective

Reproduce the discipline of the ann-benchmarks methodology (Aumüller, Bernhardsson & Faithfull 2020, [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)) on a standard dataset with a small parameter sweep for HNSW and IVF-PQ. The deliverable is a **recall vs. queries-per-second Pareto plot** with both algorithms overlaid, plus a written interpretation. The point is not to beat the leaderboard — it is to internalize the shape of the tuning loop so that when you tune a real corpus in exercise 03 or in production, you already know the moves.

## Prerequisites

- Chapters 3, 4, 5 read.
- Python ≥ 3.10, `faiss-cpu`, `numpy`, `matplotlib`. Install with `pip install faiss-cpu numpy matplotlib`.
- ~2 GB of free disk for the dataset download.

## The dataset

Use **SIFT1M** — a 128-dim, 1 M-vector dataset with 10 k query vectors and pre-computed ground-truth top-100. It is small enough to sweep on a laptop and standard enough that your numbers can be sanity-checked against ann-benchmarks or FAISS's own examples.

Download from [`corpus-texmex.irisa.fr`](http://corpus-texmex.irisa.fr/) (choose "ANN_SIFT1M") or use FAISS's demo script that fetches it. If SIFT1M is inaccessible, use **GloVe-100** (100-dim, 1.18 M vectors, cosine distance) from the ann-benchmarks datasets list — the shape of the exercise is identical.

Note down the dataset version and the URL you fetched it from in the manifest cell of your writeup (chapter 8).

## Steps

**Step 1 — Load the data and ground truth.**

Load the 1 M base vectors, 10 k query vectors, and the ground-truth top-100 per query. Confirm shapes and dtype. Compute a small stat sheet: dimensionality, vector count, `mean(||v||)`, `std(||v||)`. If you plan to use cosine, L2-normalize; if L2, do not.

**Step 2 — Baseline: exact top-k.**

Build a `faiss.IndexFlatL2` (or `IndexFlatIP` for cosine). Compute the top-10 for the 10 k queries. Confirm exact top-10 recall against the provided ground truth equals 1.0 (small floating-point differences are fine). Record wall-time — that is your latency ceiling for "no ANN."

**Step 3 — HNSW sweep.**

Build **three** HNSW indexes:

- `(M=8, efConstruction=100)` — small.
- `(M=16, efConstruction=200)` — medium.
- `(M=32, efConstruction=200)` — large.

Record build time and post-build RAM (via `psutil.Process().memory_info().rss` before and after). For each built index, sweep `efSearch ∈ {10, 20, 40, 80, 160, 320, 640}` and measure:

- `recall@10` against ground-truth top-10.
- Wall-time per query (repeat 3 runs, use the min of the three medians).

**Step 4 — IVF-PQ sweep.**

Pick two `(nlist, pq_m)` combinations that fit SIFT1M's `d = 128`:

- `(nlist=256, pq_m=8)` — aggressive compression (8 bytes/vec).
- `(nlist=1024, pq_m=16)` — moderate compression (16 bytes/vec).

Train each on the base vectors, add all 1 M. Record build time and post-build RAM. For each built index, sweep `nprobe ∈ {1, 4, 8, 16, 32, 64, 128}` and measure the same two metrics as HNSW.

**Step 5 — IVF-PQ + refine.**

For the aggressive-compression config `(nlist=256, pq_m=8)`, wrap it with `faiss.IndexRefineFlat` and add the same base vectors. Re-run the `nprobe` sweep with an additional inner-loop knob `R ∈ {50, 100, 200}` (the number of PQ candidates the refine step rescores exactly). Record the same two metrics.

**Step 6 — Plot the Pareto front.**

Produce a single plot with:

- **x-axis:** queries per second (`1 / median_latency_seconds`).
- **y-axis:** `recall@10`.
- **Series:** one per `(algorithm, build-config)`, connected by the swept knob.

Label the axes with units. Overlay all HNSW builds, all IVF-PQ builds, and IVF-PQ+refine on the same axes. Log-scale the x-axis; that is the ann-benchmarks convention.

**Step 7 — Write it up.**

In ≤ 500 words, cover:

- **What the plot shows.** Which algorithm / config sits on the frontier at high recall vs. high QPS?
- **Where the trade-offs bite.** At `recall@10 ≥ 0.95`, what is your fastest config? At `p95 latency ≤ 5 ms`, what is your highest-recall config?
- **What did refine buy for IVF-PQ?** Quantify the recall lift and the latency cost.
- **What is the memory story?** For each config, state RAM footprint per million vectors (measured, not estimated).
- **A one-paragraph "if I had to pick one config for prod"** answer, defended by the plot.

## Acceptance criteria

- All three HNSW builds, both IVF-PQ builds, and IVF-PQ+refine measured.
- Recall @ 10 measured against real ground truth; do not fabricate.
- Pareto plot rendered with legend, axis labels, units.
- Written interpretation covers each of the five sub-points from Step 7.
- Every reported latency includes a run count and either median or the minimum-of-medians rule; do not report a single-run number.

## Stretch goals

- **Deep1B-10M slice.** Move to a 10 M-vector subset of Deep1B (or SIFT10M if you have the disk). Repeat the exercise. Notice which config's frontier moves and which stays roughly where it was.
- **Add OPQ pre-rotation.** For the aggressive IVF-PQ config, add `OPQ` via `faiss.index_factory(d, "OPQ8,IVF256,PQ8")`. Plot the OPQ variant on the same axes as vanilla IVF-PQ. Quantify the recall lift.
- **Compare with `hnswlib`.** Repeat the HNSW build/sweep using `hnswlib` (Malkov's reference implementation). Do the numbers match FAISS's HNSW? If not, find the parameter or implementation detail that explains it (`allow_replace_deleted`? `num_threads` at build? Different `efConstruction` default?).
- **Sensitivity check.** Fix `(M=16, efConstruction=200, efSearch=80)` and re-run three times on different machine states (cold cache, warm cache, with a background CPU-heavy process). Report how much wall-time varies. This is why ann-benchmarks pins hardware.
- **Publish-worthy plot.** Take one of the plots and clean it up to publication quality — axis labels, log-scale annotations, an operating-point marker. Compare with a figure from the ann-benchmarks website. Is yours legible?

## Handoff to exercise 03

The winning HNSW and IVF-PQ configs from this exercise feed exercise 03, where you build the same shapes in pgvector / Qdrant / Milvus and see whether the vendor's implementation reaches the FAISS reference's numbers.
