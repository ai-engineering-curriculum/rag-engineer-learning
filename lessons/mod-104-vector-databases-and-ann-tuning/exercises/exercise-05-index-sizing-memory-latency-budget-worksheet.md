# Exercise 05 — Index Sizing: Memory / Latency Budget Worksheet

**Estimated effort:** ~2 hours. Spreadsheet + Markdown. No code strictly required; a short Python script for the back-of-envelope math is optional.

## Objective

Produce three sized index plans — one at **10 M vectors**, one at **100 M vectors**, one at **1 B vectors** — using the chapter 6 sizing formula and the vendor rubric from chapters 1 and 2. Each plan ends with (a) a per-vector RAM cost, (b) a total RAM including headroom, (c) a node count, (d) a replica count sized to a target QPS, (e) a storage budget that includes the cutover-doubling, and (f) a one-paragraph vendor / algorithm / quantization recommendation defended against the numbers.

The point of the exercise is not the answer — it is the *worksheet shape*. Every real sizing conversation on your team starts from these seven inputs and seven outputs; once the shape is in your fingers, the first-pass plan for a new project takes 20 minutes instead of a week of Slack threads.

## Prerequisites

- Chapter 6 read (the "memory per million vectors" formula and the shard / replica table).
- Chapters 1 and 2 skimmed again — you will cite specific vendor features in the recommendation.
- Exercise 02 completed — the recall / QPS numbers from the Pareto plot inform the per-replica QPS estimates.
- Exercise 03 completed (optional but useful) — the measured RAM footprints give you a sanity check against the formula.

## The seven inputs you must specify per scenario

For each of the three scenarios, write down:

1. **`N`** — vector count (fixed per scenario below; stretch goal: use your own project's 18-month estimate).
2. **`d`** — embedding dimensionality (pick from one of three canonical widths: 384, 1024, or 1536).
3. **`dtype`** — one of `float32`, `float16`, `int8`, `pq_m=64/nbits=8`, or `pq_m=32/nbits=8`.
4. **Payload size per vector** — bytes of metadata (`doc_id`, `tenant_id`, `permission_scope`, timestamps). Reasonable defaults: 200 B minimum, 500 B realistic, 2 KB rich.
5. **Target `recall@10`** — pick 0.90, 0.95, or 0.99.
6. **Target `p95` end-to-end latency** — pick 20 ms, 50 ms, or 200 ms.
7. **Target sustained QPS** — pick 10, 100, or 1 000.
8. **Refresh cadence** — pick hourly, nightly, or weekly; state the acceptable staleness SLO.

(That is eight bullets because item 7 implicitly needs item 8 for the build-hardware line.)

## The three scenarios

### Scenario A — 10 M vectors, "the departmental RAG"

- `N = 10_000_000`.
- Context: internal HR / docs wiki, ~12 M chunks after mod-102 chunking.
- Choose `d`, dtype, payload, recall target, latency, QPS, cadence given the context — but set `recall@10 ≥ 0.95` and `p95 ≤ 50 ms` as hard floors.

### Scenario B — 100 M vectors, "the product-catalog retrieval"

- `N = 100_000_000`.
- Context: multi-tenant product search; 300 tenants, each with 100 k – 1 M chunks.
- `recall@10 ≥ 0.90` is acceptable; `p95 ≤ 100 ms` is the SLO; sustained 500 QPS peak; nightly rebuild acceptable.

### Scenario C — 1 B vectors, "the archive-scale retrieval"

- `N = 1_000_000_000`.
- Context: large-corpus retrieval over web / archive text; single-tenant.
- `recall@10 ≥ 0.80` is acceptable (operate on the Pareto frontier where quantization bites); `p95 ≤ 200 ms`; 50 QPS; weekly rebuild.
- At this scale, pure `float32` HNSW on one node is impossible; you *must* reach for IVF-PQ or DiskANN or multi-node sharding, and defending that choice is the exercise.

## The seven outputs per scenario

For each scenario, compute:

1. **Per-vector RAM cost (bytes)** — raw vector + graph overhead + payload. Show the arithmetic.
2. **Total RAM (GB)** — `N × per-vector × headroom`. Use a 1.5× headroom factor for steady state, 2× if the refresh cadence is hourly.
3. **Node count** — ceil(total RAM / per-node RAM). Pick a per-node size (16, 32, 64, 128, 256 GB). State which cloud instance you are modelling (e.g. `r6i.4xlarge` at 128 GB) and link to its spec sheet.
4. **Replica count** — ceil(target QPS / per-replica QPS). Use the exercise-02 Pareto number for per-replica QPS at your chosen operating point; if you did not do exercise 02, cite the ann-benchmarks SIFT1M number for the same `(algorithm, config)`.
5. **Storage during cutover (GB)** — `2 × total index size` plus payload / WAL overhead. This is the number ops will push back on.
6. **Build hardware and expected build time** — target: build time ≤ ½ the refresh interval. Cite the exercise-02 build-time number scaled roughly linearly in `N` for HNSW, or cite the FAISS tutorial numbers for IVF-PQ at your chosen `(nlist, pq_m)`.
7. **One-paragraph vendor / algorithm / quantization recommendation** — name the engine, the algorithm, the quantization, and the shard / replica layout. Cite the matrix cell(s) from exercise 01 that justify the pick.

## Steps

**Step 1 — Open the worksheet.** Create `sizing_worksheet.md` with three H2 sections: one per scenario. Pre-populate each with the eight-input and seven-output skeleton.

**Step 2 — Fill in the inputs.** Decide each input value per scenario; defend non-obvious choices in one sentence (e.g. "picking `int8` scalar quantization over `float16` because chapter 6 says recall drop is ≤ 1 point and the 4× memory saving is load-bearing at `N = 100 M`").

**Step 3 — Compute the outputs.** Use the chapter 6 formula and the table it links. For the arithmetic, either show the formula inline or attach a `sizing.py` with ~30 lines of pure-numeric math. Do not estimate — compute.

**Step 4 — Pick the vendor.** For each scenario, read your exercise-01 matrix for the shortlist (chapter 1's rubric already filtered out the systems that cannot play at that scale). Pick one, name a runner-up, and write a 100-word defense that cites specific matrix cells.

**Step 5 — Sanity check against reality.** For scenario A at 10 M vectors, your plan should fit on one `r6i.4xlarge`-class node with headroom; if your arithmetic says you need five, re-check the dtype choice. For scenario C, your plan should *not* fit on one node of any size at `float32`; if it does, re-check `d` and the headroom factor.

**Step 6 — Author the one-page summary.** On the last page of the worksheet, produce a 3-row table with the three scenarios across columns: `(scenario, N, dtype, nodes, replicas, storage, vendor, algo)`. This is the artifact your project lead reads before the funding conversation.

## Acceptance criteria

- All three scenarios have all eight inputs and all seven outputs filled with real numbers (no "TBD", no "~").
- Arithmetic is visible — either shown inline or in an attached script. The grader can rerun the numbers.
- Scenario A fits on one node; scenario B uses ≤ 10 nodes with ≥ 1 replica for QPS; scenario C uses either multi-node sharding or a disk-tiered / PQ-compressed index on 1 – 3 fat nodes. If your plan deviates, defend it in prose.
- Each scenario's vendor recommendation cites at least one specific cell of the exercise-01 matrix.
- Cutover-doubling storage is explicitly line-itemed in every scenario (chapter 6's rule #2 of the three sizing mistakes).
- Headroom factor is 1.5× or 2× — stated and defended against the refresh cadence. Not 1.0×.
- Nowhere in the worksheet does the phrase "millions / billions of vectors" appear without the exact `N` next to it.

## Stretch goals

- **Carry a real project through.** Replace one scenario with your actual project's 18-month estimate. Walk the same seven outputs. The artifact doubles as a sizing memo you can send to your team.
- **Sensitivity table.** For scenario B, vary one input at a time (dim 1024 → 1536, dtype `float32` → `int8`, QPS 500 → 2 000) and show how each output changes. A one-page sensitivity table is a very effective tool during vendor-conversation triage.
- **Cloud cost column.** Add a ninth output: estimated monthly cost at list price on your target cloud (AWS, GCP, Azure, or the vendor's managed tier for Pinecone / Qdrant Cloud / Milvus Cloud / Weaviate Cloud). Cite the published price page. Order-of-magnitude is enough; the point is that cost is a first-class input to sizing.
- **Shard-vs-single-fat-node sweep.** For scenario C, produce two plans: one with 8 × 128 GB nodes and HNSW `float32`; one with 1 × 256 GB node and IVF-PQ. Compare recall (from exercise 02's Pareto), latency, and cost. Note which the ann-benchmarks frontier favors at your target point.
- **Build-time scaling check.** Instead of citing a scaled number, actually run a 10 M-vector build on your laptop and measure. Compare to your scenario-A build-time estimate. If they diverge by > 3×, you have found a wrong assumption in the formula and should document it.
- **Draw the state diagram.** For scenario A's refresh cadence, draw the four-state cutover diagram (steady → build → verify → flip → cool-down) from chapter 8 with timings for each state. This is the artifact that goes into the runbook.

## Handoff

The worksheet feeds exercise 06 — the dual-index cutover drill assumes you have decided on a vendor, an algorithm, and a sizing plan. If exercise 05 is skipped, exercise 06 is grounded in guesses rather than numbers; do not skip it.

The worksheet also feeds mod-111's retriever CI: the "acceptable `p95` latency delta" that the CI gate checks on every rebuild comes directly from the latency budget you set here.
