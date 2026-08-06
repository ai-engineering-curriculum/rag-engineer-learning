# Chapter 2 — Reading the Internals: pgvector, Pinecone, Weaviate, Qdrant, Milvus, FAISS, Chroma

## Why this chapter exists

A vendor comparison table you can copy from a blog is worth nothing. A comparison you can *defend* — because you have read the docs, looked at the API, tried the index build, and understood what each system exposes and hides — is a level-30 artifact. This chapter walks the seven systems against a shared rubric so exercise 01's matrix has a scaffold to hang off.

The rubric has seven cells per system:

1. **Deployment shape** — how you run it, what you page for.
2. **Persistence and durability** — what is on disk vs. RAM, replication semantics.
3. **Index families supported** — HNSW, IVF, IVF-PQ, DiskANN, ScaNN, etc.
4. **Filter story** — pre-filter or post-filter, and how predicates are pushed down.
5. **Multi-tenant isolation** — collection, namespace, or per-tenant.
6. **Quantization / memory tiering** — int8, product quantization, binary, disk-tier.
7. **Update model** — insert, delete, update cost; incremental vs. rebuild.

You are not expected to memorize each cell — you are expected to know *where to find each cell in the official docs* so a comparison built today does not go stale next quarter.

## pgvector (Postgres extension)

**Deployment shape.** A C extension for PostgreSQL 12+. Installed via `CREATE EXTENSION vector;`. Runs wherever Postgres runs — RDS, Aurora, Cloud SQL, Neon, Supabase, self-hosted. No new pager rotation; you page for Postgres.

**Persistence and durability.** WAL-backed like any Postgres table. Point-in-time recovery, logical replication, streaming replicas — inherited. Every vector-column write goes through the same MVCC machinery.

**Index families supported (as of pgvector 0.8).**

- `ivfflat` — an inverted-file index (chapter 3), fastest to build, requires a training step (`lists` clusters).
- `hnsw` — the graph-based index added in pgvector 0.5.0 (Oct 2023). Slower to build, faster to query, better recall at similar latency. This is the default choice for most workloads.
- No PQ or scalar quantization *in the index itself* (as of 0.8); you can quantize the column type (`vector` → `halfvec` for `float16`, `bit(n)` for binary quantization) and index those instead.

**Filter story.** Full SQL `WHERE`. pgvector 0.5+ added `hnsw.iterative_scan` and 0.7 improved it: for selective filters the planner iterates the ANN graph and evaluates the predicate inline, which trades some latency for correctness under selective filters (chapter 7). Old pgvector versions post-filtered — which was the correctness trap that motivated the fix.

**Multi-tenant isolation.** Whatever Postgres gives you: row-level security, separate schemas, or separate databases. There is no first-class "namespace" concept in pgvector.

**Quantization / memory tiering.** `halfvec` (float16), `sparsevec`, and `bit(n)` column types. Combined with the `<=>` (cosine), `<->` (L2), and `<#>` (inner-product) operators, you can build a binary-quantized HNSW index that fits far more vectors per gigabyte.

**Update model.** Inserts and deletes cost what Postgres inserts and deletes cost, plus HNSW graph maintenance on every insert. Bulk-loading a large table with an HNSW index is slow — the standard pattern is `COPY` first, then `CREATE INDEX`. HNSW recall on incrementally-inserted graphs can lag a from-scratch build (documented in pgvector's README); the operational answer is periodic `REINDEX`.

**Where to look.** [`github.com/pgvector/pgvector`](https://github.com/pgvector/pgvector) — the README is the primary source. Postgres docs on [index concurrency](https://www.postgresql.org/docs/current/sql-createindex.html) and [`REINDEX CONCURRENTLY`](https://www.postgresql.org/docs/current/sql-reindex.html) for the operational surface.

## Pinecone

**Deployment shape.** Managed only. Two product lines: **serverless** (2023+, storage/compute decoupled, billed by usage) and **pod-based** (older, capacity-priced). You do not choose the underlying algorithm; you choose an index type and Pinecone picks the machinery.

**Persistence and durability.** Vendor-managed; SLAs published per plan. You do not get a dump-and-restore of the index the way you do with a self-hosted engine — export is a metadata + vector export, not an index file.

**Index families supported.** Pinecone abstracts the algorithm behind index types. Serverless internally uses a proprietary variant descended from IVF-family ideas with metadata pushdown; pod-based `p1`/`p2`/`s1` pods correspond to different memory/latency profiles. You do not tune `M`, `nprobe`, etc. directly — you tune shard/replica count and the pod tier.

**Filter story.** Pre-filter via the `filter` object in the query API — metadata predicates are pushed into the retrieval path. Selectivity behavior varies by index type; the docs recommend keeping filter cardinality reasonable.

**Multi-tenant isolation.** **Namespaces.** One index can hold many namespaces; queries scope to a namespace at query time. Isolation is logical; physical isolation requires separate indexes (which have their own cost).

**Quantization / memory tiering.** Serverless uses tiered storage internally; pod-based `s1` (storage) vs. `p1` (performance) trades RAM for cost. You do not control the quantization directly.

**Update model.** Upsert-based (`upsert`), delete by ID or by filter. Under the hood, serverless handles compaction and index maintenance; you do not see reindex latency.

**Where to look.** [`docs.pinecone.io`](https://docs.pinecone.io/) — the [index types page](https://docs.pinecone.io/guides/indexes/create-an-index) and the [understanding filtering](https://docs.pinecone.io/guides/data/understanding-metadata) page are the two you cite for the rubric cells above.

## Weaviate

**Deployment shape.** Self-hosted (Docker, Kubernetes via Helm chart) or Weaviate Cloud Services. Runs as a single binary; supports multi-node clustering (Raft-based) since v1.25.

**Persistence and durability.** LSM-tree store for objects, HNSW graph persisted separately. WAL for durability; snapshot-based backups. Replication via Raft for the multi-node cluster.

**Index families supported.** HNSW (default; per-collection tunable) and a **flat** (brute-force) index for small collections. Weaviate supports **product quantization** (PQ), **binary quantization** (BQ), and **scalar quantization** on HNSW indexes as of 1.18/1.23 respectively — configured per collection.

**Filter story.** Pre-filter via GraphQL `where` clauses. Weaviate maintains a **Roaring-bitmap inverted index** on filterable properties; the vector search intersects with the bitmap so predicates are pushed into the traversal. The docs' [filters page](https://weaviate.io/developers/weaviate/concepts/filtering) explains the pre-filter cost model.

**Multi-tenant isolation.** First-class **multi-tenancy** feature: collections declared with `multiTenancyConfig` maintain a separate HNSW graph per tenant, which is stronger isolation than a namespace-filter approach and important when tenants have very different data distributions.

**Quantization / memory tiering.** PQ + HNSW, BQ + HNSW, or SQ + HNSW. The tiering knob is **`vectorCacheMaxObjects`** — you can keep a warm subset in RAM and page the rest.

**Update model.** Object-level insert/update/delete. HNSW graph maintains incrementally. Reindex is expensive; the recommended path for a schema change is dual-collection + swap (chapter 8).

**Where to look.** [`weaviate.io/developers/weaviate`](https://weaviate.io/developers/weaviate) — the *Concepts* section for the architecture pages, the *Manage data* section for CRUD, and the *Configuration* section for quantization and multi-tenancy.

## Qdrant

**Deployment shape.** Self-hosted (Docker, Kubernetes via Helm chart) or Qdrant Cloud. Written in Rust; single binary, low operational surface. Supports clustering via consensus (Raft) for horizontal scale.

**Persistence and durability.** Segment-based storage; segments are memory-mapped files. WAL for durability. Snapshots for backup.

**Index families supported.** HNSW is the primary vector index. Qdrant does not ship IVF-PQ separately — quantization is applied *on top of* HNSW: **scalar quantization** (int8), **product quantization** (PQ), and **binary quantization** (BQ). Each can be configured per-collection with an optional oversampling factor at query time to recover recall.

**Filter story.** Pre-filter is first-class. **Payload indexes** on individual fields (keyword, integer, float, geo, boolean, datetime) let the filter be pushed into segments before HNSW traversal, and the traversal itself is filter-aware — Qdrant walks the graph *inside* the filtered subset, avoiding the correctness trap of naïve post-filter. The docs' [*Filtrable HNSW*](https://qdrant.tech/articles/filtrable-hnsw/) article is the authoritative writeup.

**Multi-tenant isolation.** Per-collection isolation is the default. For tenant-per-namespace patterns, you use a `tenant_id` payload field with a payload index; the docs' [multi-tenancy guide](https://qdrant.tech/documentation/guides/multiple-partitions/) covers the trade-offs.

**Quantization / memory tiering.** Scalar / product / binary quantization, and **on-disk** payload/vector storage — the collection can be configured to keep vectors on disk and load them on demand, trading latency for RAM.

**Update model.** Point upsert/delete by ID or by filter. Segments are compacted in the background. Because HNSW is filter-aware, changing the payload distribution does not usually require a rebuild.

**Where to look.** [`qdrant.tech/documentation`](https://qdrant.tech/documentation) — the *Concepts* pages (collections, points, payload, indexing) and the *Optimization* pages (quantization, storage).

## Milvus

**Deployment shape.** Distributed by design. Self-hosted via Kubernetes (the Milvus Operator or Helm chart) or managed as **Zilliz Cloud**. Architecture is proxy → coordinator (root, data, query, index) → worker nodes (data, query, index nodes) — the heaviest system on the shelf. Milvus Lite (a single-file, embedded variant) exists for prototypes.

**Persistence and durability.** Segments written to object storage (S3, MinIO, GCS, Azure Blob) and metadata to etcd. Message-queue-backed CDC (Pulsar or Kafka) between data and index nodes. Snapshot-based backup via `milvus-backup`.

**Index families supported.** Broadest catalogue on the shelf: `FLAT`, `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ`, `HNSW`, `SCANN` (Milvus's port of Google's algorithm), `DISKANN` (large-corpus disk-tiered graph), plus GPU-backed variants (`GPU_IVF_FLAT`, `GPU_IVF_PQ`, `GPU_CAGRA`, `GPU_BRUTE_FORCE`) via cuVS/RAFT integration. Index selection is per-collection-per-field.

**Filter story.** Pre-filter via boolean expressions (`price > 10 && category == 'foo'`). Milvus maintains scalar indexes on filterable fields and pushes the predicate down; the [*Filtered Search*](https://milvus.io/docs/filtered-search.md) doc is the reference. As of Milvus 2.4, the `iterative` filter mode supersedes the older post-filter behavior for correctness on selective predicates.

**Multi-tenant isolation.** **Collections** and **partitions** (a partition is a physical shard within a collection you can address by name). Partition-per-tenant is the recommended pattern for small tenants; collection-per-tenant for large tenants; partition-key-based automatic partitioning (`partition_key_field`) for tenant fanout.

**Quantization / memory tiering.** All of `SQ8`, `PQ`, `BQ` (via `BIN_*` indexes) built in; DiskANN is the disk-tiered option for corpora that do not fit in RAM.

**Update model.** Upsert, delete by primary key or expression. Index building is async and offloaded to index nodes; querying continues on the previous segment version.

**Where to look.** [`milvus.io/docs`](https://milvus.io/docs) — the *Architecture* overview, the *In-memory Index* and *On-disk Index* pages for the index catalogue, and the *Manage Partitions* page for the tenancy story.

## FAISS

**Deployment shape.** Library — C++ core with Python bindings. You link it into your process (a FastAPI server, a batch job, a notebook). There is no server, no persistence layer, no cluster.

**Persistence and durability.** You save an index with `faiss.write_index()` and load it with `faiss.read_index()`. Persistence, replication, and durability are entirely your problem — usually the pattern is "rebuild nightly from source of truth, write to shared storage, load into the serving process."

**Index families supported.** The reference implementation for the algorithms in chapter 3: `IndexFlatL2` / `IndexFlatIP` (exact), `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`, `IndexHNSWFlat`, `IndexPQ`, `IndexBinaryFlat`, `IndexBinaryHash`, plus compositions via `index_factory("IVF1024,PQ64x8")`-style strings. GPU variants (`GpuIndexIVFPQ`, `GpuIndexFlat`) via `faiss-gpu`.

**Filter story.** No first-class filter object; you supply an **`IDSelectorBatch` / `IDSelectorRange` / `IDSelectorBitmap`** to prune candidates, or you post-filter results in Python. The pre-filter path exists but is more work than in the specialist services.

**Multi-tenant isolation.** Not a database concern — you own it. Typical pattern: one FAISS index per tenant, loaded on demand, or a single index with a per-vector ID tag and post-filter.

**Quantization / memory tiering.** The whole point. `PQ`, `SQ`, `RQ` (residual quantization), `OPQ` (optimized PQ pre-rotation), binary indexes. `OnDiskInvertedLists` for disk-tiered IVF-PQ.

**Update model.** Batch-oriented. Most `Index` classes support `add()`; only some support `remove_ids()`. The common pattern is "build once, serve, rebuild on refresh."

**Where to look.** [`github.com/facebookresearch/faiss/wiki`](https://github.com/facebookresearch/faiss/wiki) — the wiki is the primary source. Especially [*Faiss indexes*](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [*The index factory*](https://github.com/facebookresearch/faiss/wiki/The-index-factory), and [*Guidelines to choose an index*](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

## Chroma

**Deployment shape.** Ships as a Python library (`chromadb`) that runs in-process or as a **`chromadb`** server (Docker / self-hosted). Chroma Cloud is the managed offering.

**Persistence and durability.** SQLite for metadata, Parquet for objects, HNSW graph persisted alongside. Backup is a filesystem copy; there is no built-in replication (of the OSS server).

**Index families supported.** HNSW (via `hnswlib`) is the vector index. No PQ / IVF variants in OSS Chroma. Distance functions: L2, cosine, IP.

**Filter story.** Pre-filter via `where` (metadata) and `where_document` (fulltext-like substring on the document body). Metadata pushdown is on filterable fields; the algorithmic details are documented in the Chroma cookbook.

**Multi-tenant isolation.** **Collections** are the isolation boundary. There is no first-class multi-tenancy layer in OSS Chroma; the pattern is collection-per-tenant or a tenant-ID payload with `where` filtering.

**Quantization / memory tiering.** No first-class quantization in OSS Chroma as of this writing.

**Update model.** `upsert`, `delete`. HNSW maintained incrementally. Small-corpus ergonomics are excellent (`Client()` → `create_collection(...)` → `add(...)` → `query(...)` in five lines); it starts to hurt past 10⁶ vectors on a single node without external orchestration.

**Where to look.** [`docs.trychroma.com`](https://docs.trychroma.com/) — the *Guides* and *Reference* sections. The [Cookbook](https://cookbook.chromadb.dev/) has the filter and performance details.

## Reading a vendor page against the rubric — the shortcut

When a new engine appears (or an existing one ships a major release), you do not re-derive the rubric — you ask seven questions:

1. *How do I run it?* (Library / self-hosted / managed / in-database.)
2. *What is on disk and what is in RAM, and does it replicate?*
3. *What ANN index families does it support? Can I pick one, or is it hidden?*
4. *Is filtering pre- or post-scan, and are filter attributes indexed?*
5. *What is the isolation unit? Collection, namespace, partition, or per-tenant index?*
6. *What quantization / tiering knobs are exposed?*
7. *What happens to my index during an insert, delete, or schema change?*

Answering those seven for a new system is a 30-minute exercise once you have the pattern. Exercise 01 does exactly that against six of the seven above.

## A short warning about vendor benchmarks

Every vendor publishes a benchmark showing they beat the others on some slice. Almost none of those benchmarks are directly comparable — they use different corpora, different distance measures, different `k`, different recall targets, and different hardware. The one benchmark you can *use* to compare algorithms (as opposed to vendors) is [ann-benchmarks](https://github.com/erikbern/ann-benchmarks), which we introduce in chapter 5 as the tuning methodology. For vendor comparisons on *your* corpus, the answer is always: run the exercise-03 hands-on and measure the three you shortlisted.

## What you should be able to do after this chapter

- State, in one sentence per system, the deployment shape and the "why keep this system" one-liner.
- Point to the primary docs page for the index catalogue and the filter story of each system.
- Apply the seven-question rubric to a system not in this chapter (e.g. Vespa, Redis Search, LanceDB, Turbopuffer) and get a defensible profile in under an hour.

The next chapter drops one level of abstraction and covers the ANN algorithm families — HNSW, IVF, IVF-PQ, ScaNN — that these systems run inside.

## Further reading

- pgvector README: [`github.com/pgvector/pgvector`](https://github.com/pgvector/pgvector).
- Pinecone architecture overview: [*Serverless architecture*](https://docs.pinecone.io/reference/architecture/serverless-architecture).
- Weaviate concepts: [*Vector index*](https://weaviate.io/developers/weaviate/concepts/vector-index), [*Filtering*](https://weaviate.io/developers/weaviate/concepts/filtering), [*Multi-tenancy*](https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy).
- Qdrant articles: [*Filtrable HNSW*](https://qdrant.tech/articles/filtrable-hnsw/), [*Quantization*](https://qdrant.tech/documentation/guides/quantization/).
- Milvus docs: [*Architecture*](https://milvus.io/docs/architecture_overview.md), [*In-memory Index*](https://milvus.io/docs/index.md), [*Filtered Search*](https://milvus.io/docs/filtered-search.md).
- FAISS wiki: [*Guidelines to choose an index*](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), [*The index factory*](https://github.com/facebookresearch/faiss/wiki/The-index-factory).
- Chroma docs: [`docs.trychroma.com`](https://docs.trychroma.com/).
- Pan, J. J., Wang, J., & Li, G. (2024). *Survey of Vector Database Management Systems.* VLDB Journal. [arXiv:2310.14021](https://arxiv.org/abs/2310.14021) — cross-vendor systematization.
