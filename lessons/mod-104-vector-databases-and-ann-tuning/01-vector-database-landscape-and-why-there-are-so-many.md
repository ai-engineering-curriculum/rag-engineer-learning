# Chapter 1 — The Vector Database Landscape: Why There Are So Many, and What Problem Each One Actually Solves

## Why this chapter exists

The vector-database category is loud and crowded. Every vendor's landing page claims "sub-10 ms p99," "billions of vectors," and "the fastest ANN in the industry." Read three of them back-to-back and you learn nothing about how to choose. Level-30 retrieval engineers are the ones asked to *justify* a choice — to the platform team that has to run it, to the security team that has to isolate it, to the finance team that has to pay for it, and to the next engineer who inherits the index — and to do that you need to see the market for what it is: a shelf of systems that answer the same abstract question ("give me the top-*k* nearest vectors to *q*, possibly with a filter") but *engineer* around different constraints.

This chapter draws the map. Chapter 2 walks through each system's internals against the same rubric. Chapters 3–5 cover the algorithms those systems run inside. Chapters 6–8 cover the operational surface (sizing, filtering, versioning) that the algorithm choice forces on you. Chapter 9 marks the boundary where platform-scale concerns become someone else's job.

## The abstract problem the whole shelf answers

Given a corpus of `N` vectors in ℝ^d and a query vector `q ∈ ℝ^d`, return the `k` corpus vectors most similar to `q` under a distance function (cosine, dot product, or Euclidean), *possibly* subject to a predicate on structured metadata attached to each vector. The system must serve this at low tail latency, on a corpus that grows and mutates, from a machine or cluster that has finite memory and disk.

Every extra word in that paragraph is a knob the market has split around:

- **"top-*k* most similar"** — exact search is O(N·d) per query. Nobody serves exact at scale, so every product is really an **approximate-nearest-neighbor (ANN)** engine wrapping an index that trades recall for latency.
- **"predicate on structured metadata"** — a `tenant_id = 42 AND updated_at > 2026-01-01` filter has to compose with the ANN scan without silently blowing recall (chapter 7).
- **"corpus that grows and mutates"** — insert, delete, and update paths differ dramatically across index families; some rebuild, some incrementally patch.
- **"low tail latency, finite memory"** — the memory-vs-latency-vs-recall frontier is the single most-argued-about chart in the space (chapter 3).

The vendors are not competing on *what* they solve. They are competing on *which corner of the constraint box they refuse to move on*.

## The four axes the market splits along

Rank any vector database against these four axes and the shape of the market falls out.

### 1. Deployment model — library, self-hosted service, or managed service

- **Library.** FAISS (Meta), ScaNN (Google), hnswlib (Malkov). You embed the ANN index in your own process. You own persistence, replication, sharding, backups, HA. You pay in engineering time; you get maximum control and no per-vector fee.
- **Self-hosted service.** Qdrant, Weaviate, Milvus, Vespa. A server process (or cluster of them) you run — usually a container, sometimes a Kubernetes operator (Zilliz's Milvus Operator, Weaviate's Helm chart, Qdrant's Helm chart). You get an over-the-wire API, replication, and clustering; you still own the infra.
- **Managed service.** Pinecone, Zilliz Cloud (managed Milvus), Weaviate Cloud, Qdrant Cloud, MongoDB Atlas Vector Search, AWS OpenSearch Serverless, Google Vertex AI Vector Search, Azure AI Search. You get an endpoint and a bill. The vendor owns capacity, HA, and upgrade paths; you pay a premium and accept whatever their tenancy model is.
- **In-database (retrofit).** pgvector (Postgres), MongoDB Atlas Vector, Elasticsearch/OpenSearch `dense_vector`, Redis Search + `VECTOR`, ClickHouse's `annoy` / `usearch` indexes, Cassandra 5's SAI vector index. You bolt a vector index onto a database you were going to run anyway. You get transactional metadata joins, existing operational muscle memory, and one fewer system to page for — at the cost of an ANN implementation that is younger and less-tuned than a specialist's.

The library-vs-service-vs-managed axis is the *first* question a platform team will ask, because it dictates who is on call.

### 2. Storage model — pure vector, hybrid (vector + payload), or vector-in-a-database

- **Pure vector.** FAISS, hnswlib, ScaNN. The index knows about vectors and integer IDs, nothing else. You store the payload in a side database and join by ID.
- **Vector + payload.** Pinecone, Qdrant, Weaviate, Milvus, Chroma. Each vector has a JSON-shaped payload attached; the engine stores both and lets you query the vector index with a filter on the payload. This is the shape most managed products converge on because it eliminates the "join-back-to-Postgres-per-query" round trip.
- **Vector-in-a-database.** pgvector, MongoDB Atlas, Elasticsearch, Redis, Cassandra 5. The payload is a first-class row/document/hash in the parent database; the vector is another column/field on it. You get full SQL/aggregation/transactional semantics on the payload and lose nothing compared to your existing database's tooling. Ceiling on ANN scale is lower than a specialist.

### 3. Filter-attribute story — post-filter, pre-filter, or integrated

- **Post-filter.** The engine runs ANN, gets top-*k*, then applies the predicate. Cheap to build; **breaks silently when the filter is selective and *k* is small** (chapter 7 has the drill). FAISS in its basic form is post-filter only.
- **Pre-filter (a.k.a. "filtered ANN" / "constrained ANN").** The engine intersects the predicate into the graph/scan itself, either by pruning the traversal or by keeping per-attribute indexes it AND-joins. Qdrant, Weaviate, Milvus, pgvector 0.5+, Pinecone (with `filter` parameter and modern index types), and Vespa all do variants of this. Correctness under selective filters is the value proposition.
- **Integrated with a full inverted index.** Vespa, OpenSearch, Elasticsearch. The engine is a real search engine that treats the vector field as one of many. You get keyword, tf-idf, BM25, structured queries, and vector, composed by a query planner. Highest engineering surface, most flexibility.

### 4. Multi-tenant isolation — per-collection, namespace, per-tenant index, or shared

- **Per-collection.** Every tenant lives in its own logical collection/index; the engine still colocates them physically. Qdrant collections, Weaviate classes, Milvus collections, Pinecone (pod-based) indexes, Chroma collections.
- **Namespace.** One physical index, tenants keyed by a metadata namespace. Pinecone namespaces, Weaviate multi-tenancy (`multiTenancyConfig`), Qdrant with a `tenant_id` payload filter. Cheap to run; you rely on the filter for isolation, which is a correctness (chapter 7) and a security (mod-112) concern.
- **Per-tenant index (dedicated).** One index per tenant, potentially one Postgres per tenant. Highest isolation, worst scaling story (index-per-tenant does not amortize).
- **Serverless / auto-partitioned.** Pinecone serverless, OpenSearch Serverless, MongoDB Atlas Search — the vendor handles physical layout; you address logical collections.

Ask which of these four combinations a vendor picks and you can usually predict the rest of their pricing, API, and operational shape.

## The seven systems this track cares about, sketched against the axes

Depth-per-system lands in chapter 2; this is the one-line orientation.

| System | Deployment | Storage | Filter story | Notes |
|---|---|---|---|---|
| **pgvector** (Postgres extension) | In-database | Vector-in-a-database | Pre-filter via `WHERE` on any indexed column, iterative scan with HNSW as of pgvector ≥ 0.5 | Postgres becomes a passable vector database. Ceiling for pure vector scale is well below specialists, but you get transactions, joins, backups, and one fewer system to run. |
| **Pinecone** | Managed only | Vector + payload | Pre-filter via `filter` object; serverless index type supports metadata pushdown | Serverless (2023+) decouples storage from compute; pod-based (older) is capacity-priced. No self-hosting. |
| **Weaviate** | Self-hosted or Cloud | Vector + payload (object-shaped) | Pre-filter via GraphQL/`where` clauses on inverted-indexed properties | Class-per-schema shape; HNSW default; ships modules for embedders/rerankers. |
| **Qdrant** | Self-hosted or Cloud | Vector + payload (JSON) | Pre-filter via `filter` on payload; **filterable indexes** on individual fields | Rust; strong story on filtered ANN correctness; scalar/product/binary quantization built in. |
| **Milvus** | Self-hosted (K8s) or Zilliz Cloud | Vector + payload | Pre-filter via boolean expressions; multiple index types (HNSW, IVF-*, DiskANN, SCANN, GPU_*) | Distributed by design (proxy → coordinator → worker nodes). Heaviest to operate; scales furthest. |
| **FAISS** | Library only (Python/C++) | Pure vector (ID only) | Post-filter by default; pre-filter possible with `IDSelector` | The reference implementation for `IndexHNSWFlat`, `IndexIVFFlat`, `IndexIVFPQ`. You embed it; you own the rest. |
| **Chroma** | Library or self-hosted server | Vector + payload | Pre-filter via `where` (metadata) and `where_document` (document contents) | Developer-first ergonomics; HNSW default; well-suited to prototypes and small production corpora. |

Two systems appear repeatedly in the rest of this track without being in the table above:

- **ScaNN** (Google) — a library, not a database. It ships the SOAR-partitioned IVF-AH algorithm that pushed the ann-benchmarks Pareto frontier for high-dimensional dense retrieval (Guo et al. 2020). We meet it in chapter 3 as an algorithm family and again in chapter 5 as a tuning target.
- **Vespa** (Yahoo/Verizon Media, now Vespa.ai) — a full search engine with first-class ANN. Structurally closer to Elasticsearch than to Pinecone; covered in mod-105 alongside the Lucene family, because its vector-index story is inseparable from its ranking-expression story.

## Where "and Elasticsearch/OpenSearch also do vectors" fits

Elasticsearch's `dense_vector` field, OpenSearch's `knn_vector` field, Solr's `DenseVectorField`, and Vespa's `tensor<float>(x[d])` all put HNSW (or IVF variants) alongside a full inverted index. That combination — first-class BM25 *and* first-class vectors, composed by one query planner — is what makes them the natural home for **hybrid retrieval** (BM25 ⊕ dense fused). This module covers only the vector-index side of that story; **mod-105** covers the Lucene family end-to-end, and **mod-106** covers hybrid fusion.

## The choice is downstream of two questions

For any real project, the vendor question reduces to two:

1. **What is the platform team already running?** If Postgres is already on-call, pgvector's ops cost is close to zero. If OpenSearch already indexes your logs, its `knn_vector` inherits a decade of ops muscle. The best system is often the one that does not create a new pager rotation.
2. **What is the ANN scale requirement in 18 months?** If you are heading toward ~10⁷ vectors with modest QPS, almost any of the seven works. If you are heading toward 10⁹+ vectors, filtered, with sub-100 ms p95, you are looking at Milvus, Vespa, Pinecone (serverless), or a specialist self-hosted deployment; pgvector is not the answer.

Everything else — API ergonomics, client SDKs, quantization support, hybrid features, cloud regions — is downstream of those two. Chapter 2 walks each system against a rubric that makes the trade-offs explicit; you will use that rubric in exercise 01 to build a comparison matrix that survives contact with a real project.

## What is NOT in this chapter (and where it goes)

- **Algorithm-level detail on how each of those systems indexes vectors.** Chapter 3 covers the algorithms; chapters 4–5 cover the parameters.
- **Detailed per-system internals** (paging model, replication, transaction semantics). Chapter 2.
- **The Lucene-family engines (Elasticsearch, OpenSearch, Solr, Vespa) and hybrid retrieval.** mod-105 owns the engine coverage; mod-106 owns the fusion story.
- **Platform-scale concerns — GPU-accelerated ANN, cross-DC sharding, HA plans.** That is `ai-infra-senior-engineer` (level 40) territory; chapter 9 marks the boundary explicitly.

## What you should be able to do after this chapter

- Name the four axes (deployment, storage, filter story, multi-tenant isolation) and give an example vendor at each corner.
- Sketch the seven systems this track cares about on those axes.
- Explain in one sentence per system why the market keeps it around instead of collapsing to one winner.
- Point to the two downstream questions ("what does the platform team already run?" and "what is the 18-month scale?") that drive most real choices.

The next chapter walks each of the seven systems against a shared rubric so the comparison matrix in exercise 01 has a scaffold to hang off.

## Further reading

- Han, Y., et al. (2023). *A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge.* [arXiv:2310.11703](https://arxiv.org/abs/2310.11703) — vendor-neutral survey of the shelf.
- Wang, J., et al. (2021). *Milvus: A Purpose-Built Vector Data Management System.* SIGMOD 2021. [ACM DL](https://dl.acm.org/doi/10.1145/3448016.3457550) — a good read on why "purpose-built" was worth splitting from Postgres/Elasticsearch.
- Pan, J. J., Wang, J., & Li, G. (2024). *Survey of Vector Database Management Systems.* VLDB Journal. [arXiv:2310.14021](https://arxiv.org/abs/2310.14021).
- Pinecone engineering blog, [*What is a vector database?*](https://www.pinecone.io/learn/vector-database/) — a vendor-flavored but readable orientation to the abstract shape.
- Weaviate documentation, [*Concepts: Vector index*](https://weaviate.io/developers/weaviate/concepts/vector-index) — the same orientation from a self-hosted vendor.
