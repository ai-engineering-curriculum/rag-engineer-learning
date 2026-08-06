# mod-104-vector-databases-and-ann-tuning: Vector Databases and ANN Index Engineering

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 18 hours

## Learning objectives

- Read the internals of the major vector databases — pgvector (Postgres HNSW / IVFFlat), Pinecone (serverless + pod), Weaviate, Qdrant, Milvus, FAISS (as a library, IndexIVFPQ / IndexHNSWFlat), and Chroma — understanding the trade-offs of managed-vs-self-hosted, hybrid-storage, filterable-attribute support, and multi-tenant isolation
- Master the ANN algorithm families used in production — HNSW (Malkov & Yashunin 2016), IVF (inverted file), IVF-PQ (product quantization), ScaNN — and know the recall-vs-latency-vs-memory frontier each occupies
- Tune HNSW parameters (M, efConstruction, efSearch) and IVF-PQ parameters (nlist, nprobe, pq_m, pq_nbits) against a target recall@k and p95 latency budget, using ann-benchmarks as the reference methodology
- Design index sizing and shard/replica strategy — memory footprint per million vectors, disk-vs-RAM tiering, replica placement for read scaling, index build vs. rebuild vs. incremental update cadence
- Implement metadata filtering correctly — pre-filter vs. post-filter, the correctness trap of post-filter with small k, and the pattern-per-vendor for pushing predicates into the index
- Establish index versioning conventions — one index per (embedding-model, chunker-version, schema-version) tuple, dual-index cutover, and the connection to retriever CI (deep-dived in mod-111)
- Cite the boundary to `ai-infra-senior-engineer` (level 40) which owns the platform-scale retrieval-serving concerns — GPU-accelerated ANN, sharded cross-DC replication, HA plans — that live above this level

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
