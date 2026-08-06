# Chapter 6 — Index Sizing, Shard/Replica Strategy, and Update Cadence

## Why this chapter exists

Chapters 4 and 5 tuned an index on a single node. Real deployments are not one node — they are several, they have hot and cold data, they must serve one class of queries while another class rebuilds, and they must not silently give the last user of the night a stale result. This chapter is about the *shape of the running system* around the tuned index: memory footprint per million vectors, disk-vs-RAM tiering, replicas for read scaling, and the cadence you refresh the index at.

The level-40 platform engineer owns cross-DC replication and GPU-accelerated ANN (chapter 9's boundary). What you own at level 30 is the *sizing plan* — the artifact that says "for the current 12 M vectors we need X GB of RAM, Y disk, Z replicas, and we refresh nightly with a T-hour build" — and the *update cadence policy*, the one-page rule for when the index is rebuilt, when it is incrementally updated, and how new inserts appear to queries.

## Sizing — the "memory per million vectors" back-of-envelope

Every vector database has the same rough memory formula. Learn to compute it in 30 seconds on a whiteboard.

**Raw storage per vector** = `d · sizeof(dtype)` bytes.

- `d = 384` (`bge-small-en-v1.5`), `float32`: 1536 bytes → **1.5 GB / M vectors**.
- `d = 1024` (`bge-large-en-v1.5`, `text-embedding-3-large` truncated), `float32`: 4096 bytes → **4.0 GB / M vectors**.
- `d = 1536` (`text-embedding-ada-002`, `text-embedding-3-small`), `float32`: 6144 bytes → **6.0 GB / M vectors**.
- `d = 3072` (`text-embedding-3-large` full), `float32`: 12288 bytes → **12 GB / M vectors**.

**HNSW graph overhead** (chapter 4): `~2M · sizeof(neighbor_id)` bytes per vector. At `M = 32` and 4-byte IDs, roughly `~256` bytes per vector → **~0.25 GB / M vectors**. Small relative to the raw vectors; but adds up on billion-scale.

**IVF-PQ per-vector code** (chapter 5): `pq_m · pq_nbits / 8` bytes. At `pq_m = 64`, `pq_nbits = 8`: 64 bytes → **~64 MB / M vectors** for the codes alone, plus a small centroid + PQ codebook table.

**Payload / metadata.** Whatever your JSON or SQL row costs, times `N`. Easy to forget — a 500-byte payload on 10 M vectors is 5 GB.

Put those together for realistic scenarios:

| Corpus | Dim | Index | RAM footprint (raw + index) |
|---|---|---|---|
| 1 M vectors | 768 | HNSW `float32` | ~3 GB + ~0.25 GB ≈ 3.3 GB |
| 10 M vectors | 1024 | HNSW `float32` | ~40 GB + ~2.5 GB ≈ 42.5 GB |
| 100 M vectors | 1024 | HNSW `float32` | ~400 GB + ~25 GB — doesn't fit on most single nodes |
| 100 M vectors | 1024 | IVF-PQ (pq_m = 64) | ~6.4 GB codes + centroid table — fits comfortably on one node |
| 1 B vectors | 1024 | IVF-PQ (pq_m = 64) | ~64 GB codes + centroids — a single fat box |

**Rule of thumb.** If a spreadsheet says "HNSW on this corpus needs more RAM than the biggest instance we can rent," you are looking at either (a) IVF-PQ, (b) HNSW + PQ compression inside the graph, (c) DiskANN, or (d) sharding across nodes. The choice depends on your recall target and update cadence — the following sections turn that into a plan.

## Quantization as a sizing lever

The single biggest lever you have for shrinking a RAM-resident index is a lower per-vector byte count. Ordered from cheapest to most aggressive:

- **`float32` → `float16` / `bfloat16`.** 2× compression; almost no recall cost on typical text embeddings. pgvector's `halfvec` column type, Qdrant's built-in `float16` storage, FAISS `IndexFlat` with `float16` — all supported.
- **`float32` → `int8` scalar quantization.** ~4× compression; usually < 1 point recall drop after re-scoring the top candidates. Qdrant `scalar` quantization, Milvus `IVF_SQ8`, FAISS `IndexScalarQuantizer`.
- **Product quantization (PQ)** — chapter 5. `pq_m = d/8` typically gives ~32× compression with moderate recall loss (recover via refine).
- **Binary quantization (BQ).** Each dim → 1 bit. 32× compression. Recall cost is real but tolerable on some embedding models that were trained to be BQ-friendly. Qdrant BQ, Milvus `BIN_*`, Weaviate BQ.

Composed with disk tiering (below), quantization is what makes 100 M – 1 B vectors serveable on a single node. It is also what makes cheap serverless tiers possible.

## Disk-vs-RAM tiering — the on-disk index

For corpora that exceed comfortable RAM budgets, you can page most of the index to SSD and serve queries with a small RAM cache. Two shapes to know:

- **On-disk IVF** (`faiss.OnDiskInvertedLists`) — the IVF cell posting lists live on disk, streamed as `nprobe` demands. Fine when your `nprobe` is small relative to the disk IOPS you can burn.
- **DiskANN** (Subramanya et al. 2019, [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)) — a graph-based index designed from the start to serve most of the graph from SSD with a compact RAM-resident summary. Milvus's `DISKANN` index type is the widely-available packaging.

Vendor-level knobs that expose disk tiering:

- **Qdrant** — `on_disk` per-collection flag; can move vectors, payload, or both to disk.
- **Weaviate** — `vectorCacheMaxObjects` controls the warm-set size; the rest is loaded on demand.
- **Milvus** — `DISKANN` index type; can also configure per-segment memory-mapped mode.
- **pgvector** — Postgres's own buffer pool + `pg_prewarm` play this role; the graph pages get cached like any Postgres relation.

The rule: **latency drops off a cliff when your working set exceeds the buffer cache**, so measure the p99 with a realistic query mix before committing to a disk-tiered plan.

## Shard/replica strategy

Two independent axes:

- **Sharding** (writes and index size) — split the corpus across N shards, each holding a subset of the vectors. Each query fans out to all shards, results are merged. Shards let you scale **corpus size** and **index build time** horizontally.
- **Replicas** (reads) — copy of an existing shard's index on another node, kept in sync. Each replica handles a fraction of the read QPS. Replicas let you scale **query throughput** horizontally and provide **failover**.

**Where each system stands:**

| System | Shards | Replicas | Notes |
|---|---|---|---|
| pgvector | Postgres partitions or Citus | Streaming replicas | Sharding via foreign-table / Citus is DIY. |
| Pinecone | Automatic (serverless) or pod count (pod-based) | Automatic (managed) | You set target QPS and capacity; Pinecone allocates. |
| Weaviate | Sharded per-collection since v1.8 | Raft-replicated since v1.25 | Configured per-schema. |
| Qdrant | `shard_number` per collection | `replication_factor` per collection | Native and straightforward. |
| Milvus | Distributed by design (data / query / index nodes) | `replica_number` at load time | The system was built for this; the operational cost is the flip side. |
| FAISS | DIY (multiple indexes across processes) | DIY (copy the file) | You are the platform team. |
| Chroma | DIY on OSS; Cloud handles it | DIY on OSS | Not built for horizontal scale. |

**Sizing for QPS.** Estimate `queries_per_replica ≈ 1 / p50_latency_seconds` as an approximate throughput ceiling on a single replica (real throughput is lower because of tail latency). If you need 1000 QPS and one replica does ~200 QPS comfortably, you need ~5 replicas.

**Sizing for corpus.** If a single-node index fits comfortably in RAM (chapters 4–5 measured this), you probably do not need shards for size. Once the raw index exceeds ~½ node RAM (leave headroom for the query workload, OS cache, and query burst), it is time to shard.

**Replica placement.** Two flavors:

- **Node-local replicas** — same DC, different machine. Handles hardware failure; does not handle DC failure.
- **Cross-DC / cross-region replicas** — handles regional failure; adds cross-region write cost. This is where you cross into `ai-infra-senior-engineer` territory (chapter 9).

## Update cadence — the shape of "how does new data land?"

Corpora mutate. You have three broad strategies, with different operational profiles:

**1. Full rebuild on a schedule.**

- Nightly (or every few hours) job embeds the current corpus, builds a new index from scratch, atomically swaps it into the read path. Reads always see a consistent index; writes since the snapshot are invisible until the next build.
- **When to use it.** When the corpus updates slowly, the build fits in the interval, and stale reads within the interval are acceptable. Almost every RAG-over-a-wiki deployment uses this shape.
- **Pattern.** Chapter 8's dual-index cutover is the operational recipe.

**2. Incremental insert against a persistent index.**

- New vectors are inserted into the live index (`INSERT ... hnsw`, Qdrant `upsert`, etc.). Reads see them immediately.
- **When to use it.** When the corpus mutates continuously and stale-read latency is a business problem (e.g. real-time content feeds).
- **Trade-offs.** HNSW graphs degrade in quality on heavy incremental workloads (chapter 4); PQ codebooks drift on distribution shift (chapter 5). Plan a periodic clean-rebuild anyway; the incremental path is for freshness between rebuilds, not forever.

**3. Hybrid — live incremental + periodic clean rebuild.**

- Corpus writes flow into the live index continuously. On a schedule (nightly, weekly), a clean rebuild happens on the current corpus snapshot; the rebuild replaces the live index atomically.
- **When to use it.** Almost always the right answer once the corpus is non-trivial. You get freshness of (2) and quality of (1).

**Cadence-affecting question checklist:**

- What is the SLO for staleness? ("New docs must be discoverable within X minutes.")
- What is the build time on the current corpus size, on the build hardware you have?
- What is the write rate? (Vectors/sec.)
- What is the delete/update rate? (These are worse than pure inserts on HNSW.)
- Can the storage handle two indexes side by side during cutover?

Write those five answers down; the cadence policy falls out of them.

## Build-time engineering — practical notes

- **Bulk load before you build.** pgvector: `COPY` first, then `CREATE INDEX` — building HNSW during 10 M `INSERT`s is dramatically slower than building it once on the finished table. Milvus: bulk-insert then `create_index`. Qdrant: `upload_collection` with `parallel > 1`.
- **Parallelize the build.** FAISS `omp_set_num_threads`, pgvector `maintenance_work_mem` + `max_parallel_maintenance_workers`, Weaviate/Qdrant have per-server thread pools. On a 32-core box a well-tuned build is 5–10× faster than a single-core one.
- **Freeze the training sample.** For IVF-PQ, the k-means training and PQ codebook are derived from a training subset. Persist the exact subset (or a seed + selection method) so a rebuild is bit-reproducible.
- **Backup the built index.** For long build times, back up the built index files (Weaviate `backup`, Qdrant `snapshot`, Milvus `milvus-backup`, pgvector `pg_dump`/PITR) so a node restore does not require a full rebuild.

## Cutover and the "atomic swap" pattern

The read path must never see a half-built index. Two patterns:

- **Two collections, alias flip.** Build the new index in a new collection with a version tag; when it passes the CI recall check (chapter 8, mod-111), flip an alias pointer to the new collection. Weaviate aliases, Qdrant collection aliases, Elasticsearch/OpenSearch aliases, and Pinecone (indirection at the client) all support this.
- **Two Postgres tables, view flip.** Same shape for pgvector: `table_v1`, `table_v2`, a view that queries whichever is current; a single `CREATE OR REPLACE VIEW` swaps the underlying table.

Chapter 8 pairs the cutover with the versioning convention; the point here is that your sizing plan must budget for **two indexes' worth of storage** during cutover. If your target index is 40 GB, plan for at least 80 GB of storage headroom during the swap window.

## A concrete sizing worksheet — the shape exercise 05 walks you through

For a real project you sit down with the following inputs and produce the outputs:

**Inputs.**

1. `N` — vectors expected in 12 – 18 months (be honest, not conservative).
2. `d` — embedding dimensionality.
3. `dtype` — `float32`, `float16`, `int8`, or PQ-compressed.
4. Payload size per vector (bytes).
5. Target `recall@k`.
6. Target `p95` end-to-end latency.
7. Target sustained QPS.
8. Refresh cadence and per-refresh build time.

**Outputs.**

1. Per-vector RAM cost = raw vector + graph overhead + payload.
2. Total RAM = `N` · per-vector RAM · headroom factor (1.5×–2×).
3. Node count = ceil(total RAM / per-node RAM).
4. Replicas needed = ceil(target QPS / per-replica QPS).
5. Storage for two indexes side-by-side during cutover.
6. Build hardware sized to complete in ≤ ½ the refresh interval.
7. A one-page decision: which vendor / algorithm / quantization from the above.

Exercise 05 gives you a worksheet with three scenarios (10 M / 100 M / 1 B vectors) and asks you to produce a sized plan for each.

## The three sizing mistakes that break production

1. **Underestimating headroom.** RAM at 95% used pages heavily and tanks latency. Plan for 40–50% headroom in steady state.
2. **Forgetting the cutover doubling.** Two indexes coexist for the swap window; if your storage is provisioned exactly, cutover fails.
3. **Sizing to `p50` instead of `p95`.** Serving latency at replica saturation is bimodal — most requests fast, tail requests very slow. If your throughput plan assumes `p50` latency uniformly, real load will exceed it.

## What is NOT in this chapter (and where it goes)

- **Metadata filtering interaction with sizing** — chapter 7 covers filter cardinality and its effect on scanned-vector count.
- **Cross-DC replication, GPU-accelerated ANN, HA plans** — chapter 9 marks the boundary; these are `ai-infra-senior-engineer` (level 40) concerns.
- **Retriever CI and drift monitoring** — mod-111.

## What you should be able to do after this chapter

- Compute the RAM footprint of a corpus given `N`, `d`, `dtype`, and index type in 30 seconds on a whiteboard.
- Pick a quantization / disk-tier / shard strategy given a RAM budget and a recall target.
- Distinguish sharding (corpus scale) from replication (query throughput) and know which levers each vendor exposes.
- Write a one-page update cadence policy for a specific project.

The next chapter drills into metadata filtering — the correctness surface that sits on top of the tuned, sized index and is the single most common source of retrieval bugs in production.

## Further reading

- Subramanya, S. J., et al. (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019. [Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html).
- Qdrant docs: [*Storage*](https://qdrant.tech/documentation/concepts/storage/), [*Quantization*](https://qdrant.tech/documentation/guides/quantization/), [*Distributed deployment*](https://qdrant.tech/documentation/guides/distributed_deployment/).
- Weaviate docs: [*Replication architecture*](https://weaviate.io/developers/weaviate/concepts/replication-architecture), [*Sharding*](https://weaviate.io/developers/weaviate/concepts/cluster).
- Milvus docs: [*In-memory Index*](https://milvus.io/docs/index.md), [*On-disk Index*](https://milvus.io/docs/disk_index.md), [*Sizing tool*](https://milvus.io/tools/sizing/).
- Pinecone docs: [*Serverless architecture*](https://docs.pinecone.io/reference/architecture/serverless-architecture), [*Choosing an index type and size*](https://docs.pinecone.io/guides/indexes/choose-a-pod-type-and-size).
- pgvector README, [*Performance*](https://github.com/pgvector/pgvector#performance).
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks.* [arXiv:1807.05614](https://arxiv.org/abs/1807.05614) — indirectly relevant: their recall/QPS numbers are what your sizing plan defends.
