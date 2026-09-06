# mod-104 — Vector Databases and ANN Index Engineering

**Track:** RAG & Retrieval Engineer (role level 30, AI Engineering family)
**Estimated effort:** 18 hours
**Position in track:** fourth module. Takes the `(embedding_model, dim, quant)` tuple frozen in mod-103 and turns it into a tuned, sized, correctly-filtered, versioned index that mod-106 / mod-107 / mod-111 depend on.

## What this module does

At level 20 "the vector store" is a Chroma collection whose defaults you never touched. This module makes it an engineered artifact. By the end you can (a) read the internals of the seven vector databases retrieval teams actually pick between — pgvector, Pinecone, Weaviate, Qdrant, Milvus, FAISS, Chroma — against a shared four-axis rubric (deployment, storage, filter story, multi-tenant isolation); (b) name the ANN algorithm families used in production — HNSW, IVF, IVF-PQ, ScaNN — and place each on the recall / latency / memory frontier; (c) tune HNSW `(M, efConstruction, efSearch)` and IVF-PQ `(nlist, nprobe, pq_m, pq_nbits)` against a target `recall@k` and `p95` latency budget using the ann-benchmarks methodology; (d) produce a defensible sizing plan (memory per million vectors, disk-vs-RAM tiering, shard/replica strategy, refresh cadence) for a real project; (e) diagnose and fix the pre-filter-vs-post-filter correctness trap that silently degrades multi-tenant recall; (f) adopt an index-versioning convention keyed on `(embedding-model, chunker-version, schema-version, corpus-snapshot, ann-config)` with a dual-index cutover pattern; and (g) know where retrieval-engineer scope ends and `ai-infra-senior-engineer` platform scope begins.

## Boundary with adjacent modules

- **Below (level 20).** `llm-application-developer` mod-006 introduces "put a vector in pgvector, call `<->`." This module replaces the untuned index with an engineered one.
- **Around (this track).** mod-103 owns model selection — its output tuple is this module's input. mod-105 covers the Lucene family (Elasticsearch, OpenSearch, Solr, Vespa) and LTR; this module covers only the vector side of engines that also do keyword. mod-106 owns BM25 ⊕ dense hybrid fusion — the tuned dense index from this module is one input. mod-107 owns rerankers — this module sits below the reranker slot. mod-109 owns judgment-set methodology at depth; this module uses recall@k / p95 latency at the level of index-tuning decisions. mod-111 owns the retriever CI, drift monitoring, and orchestration integration; chapter 8's index-versioning convention is a pre-requisite for that CI. mod-112 owns ACL enforcement, structured data, graph, and multimodal at enterprise scale — chapter 7's filter correctness is a foundation for it.
- **Above (level 40).** `ai-infra-senior-engineer` owns GPU-accelerated ANN, cross-DC replication, and platform-scale HA. Chapter 9 marks the boundary explicitly with a one-page ownership contract.

## Learning objectives

- Read the internals of the major vector databases — pgvector (Postgres HNSW / IVFFlat), Pinecone (serverless + pod), Weaviate, Qdrant, Milvus, FAISS (as a library, IndexIVFPQ / IndexHNSWFlat), and Chroma — understanding the trade-offs of managed-vs-self-hosted, hybrid-storage, filterable-attribute support, and multi-tenant isolation.
- Master the ANN algorithm families used in production — HNSW (Malkov & Yashunin 2016), IVF (inverted file), IVF-PQ (product quantization), ScaNN — and know the recall-vs-latency-vs-memory frontier each occupies.
- Tune HNSW parameters (`M`, `efConstruction`, `efSearch`) and IVF-PQ parameters (`nlist`, `nprobe`, `pq_m`, `pq_nbits`) against a target `recall@k` and `p95` latency budget, using ann-benchmarks as the reference methodology.
- Design index sizing and shard / replica strategy — memory footprint per million vectors, disk-vs-RAM tiering, replica placement for read scaling, index build vs. rebuild vs. incremental update cadence.
- Implement metadata filtering correctly — pre-filter vs. post-filter, the correctness trap of post-filter with small `k`, and the pattern-per-vendor for pushing predicates into the index.
- Establish index versioning conventions — one index per `(embedding-model, chunker-version, schema-version)` tuple, dual-index cutover, and the connection to retriever CI (deep-dived in mod-111).
- Cite the boundary to `ai-infra-senior-engineer` (level 40), which owns the platform-scale retrieval-serving concerns — GPU-accelerated ANN, sharded cross-DC replication, HA plans — that live above this level.

## Prerequisites

- mod-101 toolchain green: local Postgres with pgvector reachable, Docker + Docker Compose installed, Python ≥ 3.10.
- mod-102 complete — you have a chunked corpus and a `chunker.yaml`.
- mod-103 complete — you have picked an embedding model with a frozen `(model, dim, quant)` tuple and a domain-fit judgment set from mod-103 exercise-04.
- Comfortable Python (`numpy`, `pandas`, dataclasses, `psutil` for RAM measurement), Docker Compose, and SQL. Some prior exposure to a client library for at least one of the seven databases in the module helps.
- Familiarity with the shape of the cosine / L2 / inner-product distance functions and what "top-*k* nearest neighbors" means.

## Chapters

Read chapters in order. Chapters 1–2 build the vendor-literacy foundation. Chapters 3–5 cover the ANN algorithms the vendors run inside and how to tune them. Chapters 6–8 own the operational surface (sizing, filtering, versioning) that the algorithm choice forces on you. Chapter 9 marks the boundary to the platform team.

| # | Chapter | Focus | Paired exercise |
|---|---|---|---|
| 01 | [The Vector Database Landscape: Why There Are So Many](01-vector-database-landscape-and-why-there-are-so-many.md) | Four axes: deployment, storage, filter story, multi-tenant isolation | (paired with chapter 2's exercise) |
| 02 | [Reading the Internals: pgvector, Pinecone, Weaviate, Qdrant, Milvus, FAISS, Chroma](02-reading-the-internals-of-the-seven-systems.md) | Seven-cell rubric per system; the "how do I read a new vendor's docs" pattern | [exercise-01 — vector db family comparison matrix](exercises/exercise-01-vector-db-family-comparison-matrix.md) |
| 03 | [ANN Algorithm Families: HNSW, IVF, IVF-PQ, ScaNN](03-ann-algorithm-families-hnsw-ivf-ivfpq-scann.md) | The structure, query-time work, and frontier of each family | (paired with chapters 4–5) |
| 04 | [Tuning HNSW: M, efConstruction, efSearch](04-tuning-hnsw-m-efconstruction-efsearch.md) | Graph knobs, five traps, the tuning procedure | (paired with chapter 5's exercise) |
| 05 | [Tuning IVF-PQ and the ann-benchmarks Methodology](05-tuning-ivf-pq-and-ann-benchmarks-methodology.md) | Partition + quantization knobs, refine, ann-benchmarks discipline | [exercise-02 — HNSW and IVF-PQ parameter sweep](exercises/exercise-02-hnsw-and-ivf-pq-parameter-sweep-with-ann-benchmarks.md) |
| 06 | [Index Sizing, Shard/Replica Strategy, and Update Cadence](06-index-sizing-sharding-replicas-and-update-cadence.md) | Memory per million vectors, disk tiering, refresh cadence policy | [exercise-05 — sizing worksheet](exercises/exercise-05-index-sizing-memory-latency-budget-worksheet.md) |
| — | (Cross-engine hands-on) | Same corpus, three engines, same target config | [exercise-03 — pgvector vs. Qdrant vs. Milvus hands-on](exercises/exercise-03-pgvector-vs-qdrant-vs-milvus-hands-on-index-build.md) |
| 07 | [Metadata Filtering: Pre-Filter vs. Post-Filter](07-metadata-filtering-pre-vs-post-filter.md) | The correctness trap; filter-aware traversal per vendor | [exercise-04 — pre- vs. post-filter correctness drill](exercises/exercise-04-metadata-filter-pre-vs-post-filter-correctness-drill.md) |
| 08 | [Index Versioning Conventions and Dual-Index Cutover](08-index-versioning-and-dual-index-cutover.md) | The tuple, the manifest, the green–blue cutover | [exercise-06 — dual-index cutover and versioning](exercises/exercise-06-dual-index-cutover-and-versioning-convention.md) |
| 09 | [Boundary to `ai-infra-senior-engineer`](09-boundary-to-ai-infra-senior-engineer.md) | The one-page ownership contract; the two escalation triggers | (no paired exercise) |

## How to work through it

1. Read chapters 1 and 2, then do exercise-01 (the comparison matrix). You will re-use the seven-cell rubric every time a vendor ships a major release.
2. Read chapters 3, 4, 5 as one arc. Do exercise-02 (FAISS parameter sweep on SIFT1M) — this is where the ann-benchmarks methodology gets in your muscle memory. Keep the Pareto plot; it feeds exercise-03.
3. Read chapter 6, then do exercise-03 (pgvector vs. Qdrant vs. Milvus hands-on) — the winning engine here is the one you use for exercises 04 and 06.
4. Read chapter 7, then do exercise-04 (the correctness drill). Verify that your chosen engine survives selective predicates before you rely on it for multi-tenant retrieval.
5. Do exercise-05 (the sizing worksheet) whenever you are next actually sizing a project — the artifact is small but load-bearing.
6. Read chapter 8, then do exercise-06 (dual-index cutover). The artifact is the pre-requisite for mod-111's retriever CI.
7. Read chapter 9. Fill in the one-page ownership contract for your project *before* the first production index is built.

## Structure

- `01-…md` … `09-…md`: lecture chapters.
- `exercises/`: six hands-on prompts, one per major objective.
- `labs/`: long-form hands-on labs (see `labs/README.md`).
- `quizzes/`: knowledge checks.
- `resources.md`: primary papers, official docs, and vendor pages the chapters cite.

## Where to go next

Once exercise-03's engine choice, exercise-04's filter-correctness sign-off, and exercise-06's versioning convention are committed, move to [`mod-105-lucene-family-and-ltr`](../mod-105-lucene-family-and-ltr/README.md). mod-105 covers the keyword / BM25 / LTR side of the retrieval pipeline that mod-106's hybrid fusion combines with the dense index you just tuned here.
