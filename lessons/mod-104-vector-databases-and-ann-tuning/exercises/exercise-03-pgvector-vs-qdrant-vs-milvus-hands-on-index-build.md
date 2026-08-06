# Exercise 03 — pgvector vs. Qdrant vs. Milvus: Hands-On Index Build and Comparison

**Estimated effort:** ~3 hours. Docker Compose + Python. Local; no cloud accounts needed.

## Objective

Build the same corpus in three different vector databases — pgvector, Qdrant, and Milvus — with HNSW indexes tuned to the same target, then measure the three key numbers on the same hardware: **build time, RAM footprint, and recall @ 10 at matched `p95 latency`**. The goal is to end with an experience-based intuition about how the same algorithm behaves inside different engines, and a defensible per-vendor performance profile that survives the "the docs say X but did you actually measure it?" question.

## Prerequisites

- Chapters 2 and 4 read.
- Exercise 02 completed (you have the FAISS reference numbers on SIFT1M).
- Docker + Docker Compose (local install; no cloud accounts).
- Python ≥ 3.10 with `psycopg[binary]`, `pgvector` (Python client), `qdrant-client`, `pymilvus`, `numpy`, `pandas`.

## The corpus

Use the same SIFT1M dataset from exercise 02, or a 100 k-vector sub-sample if disk / RAM is tight. Same 10 k query vectors, same ground truth. That way the numbers are directly comparable to exercise 02's FAISS baseline.

## The target configuration

Fix these parameters across all three engines:

- **Distance:** L2 (SIFT1M is L2-native).
- **HNSW:** `M = 16`, `efConstruction = 200`, `efSearch = 80`.
- **k = 10** for recall measurement.

If a vendor's parameter has a slightly different name (`m` vs. `M`, `ef` vs. `efSearch`), map it explicitly in your writeup.

## Steps

**Step 1 — Bring up the three engines.**

Write a `docker-compose.yml` that starts:

- `pgvector/pgvector:pg16` (or the current tagged image).
- `qdrant/qdrant:latest`.
- `milvusdb/milvus:latest` — Milvus is heavier; use the single-node Docker Compose from [milvus.io/docs/install_standalone-docker.md](https://milvus.io/docs/install_standalone-docker.md) which brings up etcd, MinIO, and Milvus itself.

Confirm each is reachable from your host (a `psql`, `curl`, or client-library `ping`).

**Step 2 — Build the schema / collection in each.**

- **pgvector**: create a table with `id bigint`, `embedding vector(128)`, one INT payload field `tenant_id`. Create an HNSW index with the parameters above via `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);`.
- **Qdrant**: create a collection with `size = 128`, `distance = "Euclid"`, `hnsw_config = {m: 16, ef_construct: 200}`, one payload field `tenant_id` with a payload index.
- **Milvus**: create a collection with an `INT64` primary key, a `FLOAT_VECTOR(dim=128)` field, and an `INT64` scalar field `tenant_id`. Create an HNSW index with `params = {"M": 16, "efConstruction": 200}` and `metric_type = "L2"`. Load the collection.

**Step 3 — Bulk-load the corpus.**

Load all 1 M vectors, each with a random `tenant_id` drawn uniformly from `{0, 1, ..., 9}`. Use the fastest bulk-load path each vendor offers:

- **pgvector**: `COPY` from a CSV or use `psycopg.copy` (`COPY chunks (id, embedding, tenant_id) FROM STDIN`). **Do the index build after the load**, not before (chapter 6 rule).
- **Qdrant**: `upload_collection` with `parallel > 1`.
- **Milvus**: `insert` in batches of 10 k – 50 k, then `create_index` and `load`.

**Time each load** and record it. Also record RAM footprint of the container after load + index build (via `docker stats` or the container's own metrics endpoint).

**Step 4 — Warm each engine.**

Send ~100 warm-up queries per engine before measurement. Do not include the warm-up in your latency numbers.

**Step 5 — Measure recall and latency.**

For all 10 k SIFT1M query vectors, per engine:

- Issue a top-10 query (no filter).
- Record wall-time client-side (do 3 runs, use the min of medians).
- Compute `recall@10` against SIFT1M's ground truth.

Report `p50, p95, p99 latency`, mean `recall@10`, and — as a sanity check — that all three engines report *the same* recall (they are all HNSW at the same parameters against the same data, so they should be within a percentage point of each other).

**Step 6 — Add a filter and repeat.**

For all 10 k queries, issue a top-10 query with `tenant_id = 0` (10% selectivity — moderately selective). Record the same three latencies and recall. Notice whether the recall stays high across all three engines (they should all pre-filter correctly — chapter 7).

Bump the filter selectivity to `tenant_id IN (0)` limited to only 1% of the corpus by ingesting a second collection with a skewed distribution, if you want to see the more selective case. (Optional — the 10% case is enough to confirm the pre-filter path.)

**Step 7 — Write it up.**

Produce a table:

| Engine | Load time (s) | Build time (s) | RAM (MB) | p50 (ms) | p95 (ms) | p99 (ms) | recall@10 (no filter) | recall@10 (filter, 10%) |
|---|---|---|---|---|---|---|---|---|
| pgvector | | | | | | | | |
| Qdrant | | | | | | | | |
| Milvus | | | | | | | | |

Then write ≤ 400 words covering:

- **Which engine had the lowest p95 at matched recall?** State the number.
- **Which engine had the smallest RAM footprint?** State the number.
- **Where did the numbers surprise you?** Compare to FAISS from exercise 02 — the raw algorithm cost is roughly the same; where does per-engine overhead show up?
- **Which engine would you pick for the mod-101 toolchain?** Defend, in one paragraph, given the exercise's numbers.

## Acceptance criteria

- Docker Compose brings all three engines up with one command; the file is checked into your writeup.
- The table has real numbers in every cell.
- Recall @ 10 (no filter) is within 0.02 across all three engines. If it is not, find out why *before* moving on.
- The filter case is measured; recall does not drop below 0.9 for any engine at 10% selectivity. If it does, you are looking at a chapter-7 bug and should file a note.
- The writeup names a chosen engine with a reason.

## Stretch goals

- **Add Weaviate.** Same corpus, same target config. Extend the table to four rows.
- **Sweep `efSearch` per engine.** Instead of a single point at `efSearch = 80`, produce a small Pareto plot per engine on the same axes as exercise 02. Do the engines' curves agree with FAISS's?
- **1% selectivity filter.** Ingest a second run with a skewed tenant distribution (e.g. `tenant_id = 0` gets 1% of the corpus). Repeat the filtered query. Record recall. If any engine's recall crashes, cite chapter 7 and identify whether it is a genuine post-filter or a mis-configured index.
- **Cross-restart persistence check.** Restart each engine container. Confirm queries return the same results. Note any that require a `load` or `refresh` step and how long it takes.
- **`REINDEX CONCURRENTLY` in pgvector.** Insert an additional 100 k vectors *without* rebuilding, measure recall drift, then `REINDEX CONCURRENTLY` and remeasure. Quantify the incremental-insert quality gap (chapter 6's practical note).

## Handoff

The winning engine from this exercise is a reasonable choice for the mod-101 toolchain if you have not already picked one. Exercise 04 stays inside pgvector for the metadata-filter drill; exercise 06 uses whichever engine you picked.
