# Exercise 04 — Metadata Filter Pre- vs. Post-Filter Correctness Drill

**Estimated effort:** ~3 hours. Python + Docker (pgvector, Qdrant). No cloud accounts.

## Objective

Reproduce, measure, and *fix* the post-filter correctness trap from chapter 7 — the bug where a selective metadata predicate at a small `k` silently drops recall to near zero. You will build the same corpus in three engines with three filter shapes (naïve post-filter in FAISS, pgvector with iterative index scan, and Qdrant with a payload index), then sweep predicate selectivity across five orders of magnitude and plot the resulting recall curves. The deliverable is a **selectivity-vs-recall plot** with three curves and a written playbook — "for this vendor, at this selectivity, use this filter shape" — that survives contact with a production multi-tenant workload.

This is the drill that catches the class of bug most likely to burn a level-30 retrieval engineer in their first year on-call.

## Prerequisites

- Chapter 7 read end-to-end (this exercise turns its `build_and_query` sketch into a real lab).
- Chapters 2 and 3 read (you need the pre-filter-per-vendor table and the ANN-family orientation).
- Exercise 03 completed (Docker Compose for pgvector and Qdrant is already working on your machine, and you have a `psycopg` + `qdrant-client` env).
- Python ≥ 3.10 with `numpy`, `matplotlib`, `psycopg[binary]`, `pgvector`, `qdrant-client`, `faiss-cpu`.
- ~2 GB of free RAM for the corpus you will build (100 k vectors × 128 dim).

## The corpus you will build

Synthetic — the point is to control selectivity precisely, and a synthetic corpus makes ground-truth trivially cheap to compute.

- `N = 100_000` vectors in ℝ^128, sampled i.i.d. `N(0, I)`, `float32`.
- One integer payload field `tenant_id ∈ {0, 1, ..., 999}` per vector.
- 500 query vectors, sampled the same way (held out from the corpus).
- Distance: L2 (do not normalize — plain Gaussian vectors under L2 are perfectly fine for a correctness drill; the point is filter behavior, not embedding-space geometry).

Selectivity is controlled by *how you populate `tenant_id`*. For a target selectivity `s`, mark a fraction `s` of the corpus as `tenant_id = 42` (the "matching" tenant) and distribute the rest uniformly across the remaining IDs. The selectivity levels to sweep:

- `s = 0.50` (loose)
- `s = 0.20` (borderline)
- `s = 0.05` (selective)
- `s = 0.01` (very selective)
- `s = 0.001` (catastrophically selective)

At `s = 0.001` the matching subset has 100 vectors — small enough that a small-`k` post-filter is essentially guaranteed to miss.

## The measurement — what "correct" means

For each `(engine, selectivity)` cell, you retrieve the top-10 vectors that satisfy `tenant_id = 42`. Ground truth is the *exact* top-10 within the tenant-42 subset (brute-force cosine / L2 on the ~100 · N · s vectors — trivially cheap at this scale).

**`recall@10` = |ANN result ∩ ground-truth top-10| / 10**, averaged across the 500 queries.

A correctly implemented pre-filter should give `recall@10 ≥ 0.95` at every selectivity level. A naïve post-filter will crash to near zero at `s = 0.01` and below. That crash is the deliverable of Step 4.

## Steps

**Step 1 — Generate the corpus and the ground truth.**

Write `build_corpus.py` that:

- Seeds `numpy` and `random` for reproducibility (`np.random.seed(0)`).
- Emits `xb.npy` (N × 128), `tenants_{s}.npy` (N integers) for each `s` above, and `xq.npy` (500 × 128).
- For each `s`, computes `gt_{s}.npy` — a `(500, 10)` array of ground-truth top-10 IDs within the tenant-42 subset for each query. Use `np.argsort(-scores)[:10]` on the filtered subset; do not use an ANN for the ground truth.

Persist all arrays. Do not regenerate on every re-run — the exercise is a comparison across engines on identical inputs.

**Step 2 — Naïve post-filter in FAISS (the wrong shape).**

Build `faiss.IndexHNSWFlat(128, M=16)` with `efConstruction = 200`. Add all `N` vectors (payload lives outside FAISS in a `numpy` array).

Implement the query as:

1. `search(q, k=10)` — get 10 candidates over the whole corpus.
2. In Python, filter the returned IDs by `tenant_id == 42`.
3. Return whatever survives (may be `< 10` results — do not backfill).

Run this for all five selectivity levels. Record `recall@10` per level.

Also run a second variant with `k = 1000` (over-fetching), same post-filter, truncated to top-10 after the filter. This is the "just fetch more and filter" workaround — record whether it recovers recall, at what over-fetch multiplier, and what it costs in latency (measure client-side).

**Step 3 — pgvector with iterative index scan (the fix, Postgres-flavored).**

Bring up the pgvector container from exercise-03's Docker Compose. Create a table:

```sql
CREATE TABLE points (
  id        bigint PRIMARY KEY,
  embedding vector(128),
  tenant_id integer
);
CREATE INDEX ON points USING hnsw (embedding vector_l2_ops)
  WITH (m = 16, ef_construction = 200);
CREATE INDEX ON points (tenant_id);
```

Bulk-load the corpus via `psycopg.copy` (build the index *after* the load per chapter 6). Turn on iterative scan:

```sql
SET hnsw.iterative_scan = strict_order;
SET hnsw.ef_search = 80;
```

Query shape:

```sql
SELECT id
FROM points
WHERE tenant_id = 42
ORDER BY embedding <-> $1
LIMIT 10;
```

Run this for all five selectivity levels. Record `recall@10` per level.

Then repeat with `hnsw.iterative_scan = off` (pgvector's old post-filter behavior). Same predicate. Compare — this is the historical bug the iterative scan was added to fix.

**Step 4 — Qdrant with a payload index (the fix, specialist-flavored).**

Bring up the Qdrant container. Create a collection:

```python
client.create_collection(
    collection_name="points",
    vectors_config=VectorParams(size=128, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
)
client.create_payload_index(
    collection_name="points",
    field_name="tenant_id",
    field_schema=PayloadSchemaType.INTEGER,
)
```

Upload the corpus in a single `upsert` / `upload_collection` call. Query shape:

```python
client.query_points(
    collection_name="points",
    query=q,
    limit=10,
    query_filter=Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=42))
    ]),
    search_params=SearchParams(hnsw_ef=80),
)
```

Run this for all five selectivity levels. Record `recall@10` per level.

Then, as a control, drop the payload index (`client.delete_payload_index`) and re-run at `s = 0.001`. Record what happens — Qdrant may still be correct because the filter is applied inline, but the *cost model* changes (see chapter 7's "sometimes falls back to exact" note).

**Step 5 — Plot.**

Produce a single plot:

- **x-axis:** predicate selectivity (log-scale from `0.001` to `0.5`).
- **y-axis:** `recall@10` (linear, 0 to 1).
- **Series:** four curves — FAISS post-filter `k=10`, FAISS post-filter `k=1000`, pgvector iterative, Qdrant filter-aware.
- Overlay a dashed horizontal line at `recall = 0.95` to mark the "acceptable" floor.

Label axes with units. Include a legend.

**Step 6 — Write the vendor playbook.**

In ≤ 500 words:

- **The trap.** Describe what happens to FAISS post-filter at `s ≤ 0.01`, with numbers from your plot. State the theoretical explanation (chapter 7's expected-matches calculation).
- **The over-fetch workaround.** What multiplier does FAISS post-filter need at `s = 0.001` to reach `recall ≥ 0.95`? What is the latency cost?
- **The pgvector story.** At what selectivity does iterative scan noticeably slow down? Where does old-style post-filter break? Cite the exact `SET` command that flips the behavior.
- **The Qdrant story.** Is there any selectivity at which Qdrant's filter-aware traversal drops recall? Did the payload index matter for correctness, or only for cost?
- **The playbook.** A short table with rows `(vendor, selectivity regime, safe shape)`. For example: *"pgvector, `s ≤ 0.05`, require `hnsw.iterative_scan = strict_order`."*
- **The CI test you would add.** Chapter 7's step 5 said "add a CI test that runs a canary query with a selective predicate and asserts recall." Sketch that test in 10 lines of pseudo-code — what fixture, what query, what assertion.

## Acceptance criteria

- Corpus generation is deterministic (`np.random.seed(0)`), and the `gt_{s}.npy` files are exact top-10s over the filtered subset (not an ANN's opinion of top-10).
- All five selectivity levels are measured for all three engines *plus* the FAISS `k = 1000` and pgvector `iterative_scan = off` control runs.
- At `s = 0.001`, FAISS post-filter `k = 10` gives `recall@10 ≤ 0.1` (the bug is real and reproduced); Qdrant and pgvector-iterative both give `recall@10 ≥ 0.95` (the fix is real and reproduced). If your numbers do not match this shape, do not massage them — find out why. That is the whole point of the exercise.
- Plot has labelled axes, log-x, a legend, and the 0.95 acceptance line.
- Playbook is ≤ 500 words, has a per-vendor table, and includes the pseudo-code CI test.

## Stretch goals

- **Add a third specialist.** Repeat step 4 for Weaviate (Roaring-bitmap-intersected HNSW) or Milvus (`iterative` filter mode in 2.4+). Extend the plot to a fifth curve.
- **Compound predicates.** Change the filter from `tenant_id = 42` to `tenant_id = 42 AND freshness_days < 7` with `freshness_days` uniformly distributed in `{0, ..., 30}`. Rerun the sweep. Discuss where the AND-of-two-indexes cost model kicks in.
- **Selectivity by tenant, not by query.** Instead of one hardcoded tenant, run 100 queries per selectivity with a random tenant each time (constant per-tenant population size, different random tenant). Discuss whether the per-tenant recall variance matters for user experience.
- **Load-shift.** Insert an additional 10 k vectors after the initial build (all `tenant_id = 42`, so tenant-42 population jumps). Rerun the sweep without rebuilding. Compare recall pre- and post-insert. This is the incremental-insert quality drift you would encounter in a real deployment (chapter 6 hinted at it).
- **A different distance function.** Repeat one row using cosine + normalized vectors. Any surprises? Should not be — filter correctness is orthogonal to distance choice — but verify.
- **Publish-worthy plot.** Clean the plot up (title, subtitle citing chapter 7, operating-point annotations), export as PNG + PDF, and include it in your writeup. This is the plot to send to a skeptical platform lead.

## Handoff

The vendor playbook you author here is the artifact you cite in exercise-06 when you author the `pre_filter_check` step of the CI gate. It is also the artifact mod-111 references when it turns "canary query on a selective predicate" into a scheduled test. And it is the artifact mod-112 builds on when it moves from "predicate correctness" to "ACL enforcement as a security boundary."
