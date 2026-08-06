# Chapter 4 — Dense-Vector Fields on Lucene-Family Engines: `dense_vector`, `knn_vector`, and Vespa Tensors

## Why this chapter exists

The reason you use a Lucene-family engine for retrieval — as opposed to a dedicated vector database — is almost always that you want **first-class lexical (BM25) *and* first-class dense retrieval, composed by one query planner, sharing one filter and one aggregation surface**. Elasticsearch, OpenSearch, Solr, and Vespa all ship a dense-vector field type that participates in the same queries as `text` and `keyword` fields. This chapter walks each one and marks the trade-off vs. a dedicated vector database from mod-104.

This is not the ANN-tuning chapter — that was mod-104 chapters 3–5. HNSW's `M`, `efConstruction`, `efSearch`, and IVF-PQ's `nlist`, `nprobe`, `pq_m`, `pq_nbits` all carry the same meaning on a Lucene-family engine as they did on a specialised vector database, and the tuning methodology is the same. What this chapter covers is the *engine-specific field syntax*, the *filter-composition semantics*, the *quantization options each engine ships*, and the *decision framework for choosing a Lucene-family vector field over a dedicated vector database*.

## The four engines' vector fields at a glance

| Engine | Field type | ANN engine(s) | Distance metrics | Quantization | Max dim (typical) |
|---|---|---|---|---|---|
| **Elasticsearch** | [`dense_vector`](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) | Lucene HNSW | `cosine`, `dot_product`, `l2_norm`, `max_inner_product` | `int8_hnsw` (scalar), `bbq_hnsw` (binary), `int4_hnsw` | 4096 |
| **OpenSearch** | [`knn_vector`](https://opensearch.org/docs/latest/field-types/supported-field-types/knn-vector/) | Lucene HNSW, FAISS (HNSW, IVF), NMSlib HNSW | `l2`, `cosinesimil`, `innerproduct`, `hamming` (binary) | FAISS scalar/PQ, Lucene scalar quantization (2.19+) | 16000 (FAISS) |
| **Solr** | [`DenseVectorField`](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html) | Lucene HNSW | `euclidean`, `cosine`, `dot_product` | Lucene byte-vector (int8) | 1024 (default), configurable |
| **Vespa** | [`tensor<float>(x[d])`](https://docs.vespa.ai/en/tensor-user-guide.html) with `index { hnsw }` | Native HNSW | `angular`, `euclidean`, `dotproduct`, `prenormalized-angular`, `hamming` | int8 tensor cell type, binary via `bitcell` | no fixed limit; d up to ~65k tested |

The one-line takeaway: **all four ship HNSW; OpenSearch adds FAISS engines for IVF and heavier quantization; Vespa's tensor field is uniquely expressive for multi-vector / late-interaction models**.

## Elasticsearch: `dense_vector`

### Field definition

```json
PUT /support_articles/_mapping
{
  "properties": {
    "body_embedding": {
      "type": "dense_vector",
      "dims": 768,
      "index": true,
      "similarity": "cosine",
      "index_options": {
        "type": "int8_hnsw",
        "m": 16,
        "ef_construction": 200
      }
    }
  }
}
```

Points worth naming:

- **`dims`** — must match the embedding model exactly (from mod-103). Wrong `dims` fails at ingest time, cleanly.
- **`index: true`** — build an HNSW index for this field. `false` stores the vector but supports only brute-force `script_score` scans — fine for small corpora, terrible past ~10⁵.
- **`similarity`** — `cosine`, `dot_product`, `l2_norm`, or `max_inner_product`. See mod-103 chapter 1 for which one your embedding model wants. Note: `dot_product` requires vectors to be pre-normalised to unit length; `cosine` normalises for you at query time (slightly more work per query).
- **`index_options.type`** — the quantization pick. Options as of 8.14+:
    - `hnsw` — full-precision float32.
    - `int8_hnsw` — 8-bit scalar quantization. ~4× memory reduction with ~1% recall loss. Default recommendation as of 8.14+ ([Elastic blog on int8 default](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene)).
    - `int4_hnsw` — 4-bit scalar. ~8× memory reduction; more recall loss (~3–5% depending on model).
    - `bbq_hnsw` — Better Binary Quantization ([Elastic blog on BBQ](https://www.elastic.co/search-labs/blog/rabitq-scalar-quantization-elasticsearch-lucene)). ~32× memory reduction; requires oversampling + rescoring for competitive recall.
- **`m`, `ef_construction`** — the HNSW parameters (mod-104 chapter 4). Defaults `m: 16`, `ef_construction: 100` are the Elasticsearch defaults; bump `ef_construction` to 200–400 for a corpus you index once and query millions of times.

### Query shape

Elasticsearch's [`knn` search parameter](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html) is the top-level query construct for vector retrieval:

```json
POST /support_articles/_search
{
  "knn": {
    "field": "body_embedding",
    "query_vector": [0.12, -0.03, ...],
    "k": 50,
    "num_candidates": 500,
    "filter": {
      "term": { "tenant_id": "acme" }
    }
  }
}
```

Points worth naming:

- **`num_candidates`** — the HNSW `efSearch` at query time. Higher = better recall, slower. Default is 100; go 500–1000 for high-recall workloads.
- **`filter`** — a filter that runs *during* HNSW traversal (Elasticsearch does **pre-filtering**, not post-filtering, since 8.0). The filter must be composed of standard Elasticsearch query clauses; the engine intersects the filter into the graph traversal, avoiding the small-`k` post-filter correctness trap from mod-104 chapter 7.

### Combining vector + lexical: hybrid queries in one request

The vector-plus-BM25 fusion story from mod-106 has a native Elasticsearch expression: the top-level `query` (lexical) and top-level `knn` (dense) both run and their results are fused. As of 8.9+ Elastic ships [**RRF (reciprocal rank fusion)**](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) as a first-class rank aggregation:

```json
POST /support_articles/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "multi_match": { "query": "reset password", "fields": ["title^5", "body"] }
            }
          }
        },
        {
          "knn": {
            "field": "body_embedding",
            "query_vector": [0.12, -0.03, ...],
            "k": 50,
            "num_candidates": 500
          }
        }
      ],
      "rank_window_size": 100,
      "rank_constant": 60
    }
  }
}
```

This is the pattern mod-106 will cover in depth from the fusion side. Naming it here so you know the engine supports it natively — you do not have to implement RRF client-side.

## OpenSearch: `knn_vector` and the three engines

OpenSearch's [k-NN plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/) exposes **three ANN engines**, selectable per index via the `method.engine` property:

- **`lucene`** — Lucene HNSW, same as Elasticsearch. Best filter integration (pre-filtering), smallest memory footprint for full-precision, no product quantization.
- **`faiss`** — Facebook AI Similarity Search. Supports HNSW and IVF (`ivf`, `ivf_pq`), scalar and product quantization, GPU-index build (with FAISS built with GPU support). Highest ceiling; heaviest to operate.
- **`nmslib`** — Non-Metric Space Library HNSW ([nmslib/hnswlib](https://github.com/nmslib/nmslib)). Historic default; being deprecated in favour of `faiss` and `lucene`. Do not pick for new work.

### Field definition, Lucene engine

```json
PUT /support_articles
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "body_embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": { "m": 16, "ef_construction": 200 }
        }
      }
    }
  }
}
```

### Field definition, FAISS engine with product quantization

```json
"body_embedding": {
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "name": "hnsw",
    "space_type": "innerproduct",
    "engine": "faiss",
    "parameters": {
      "m": 16,
      "ef_construction": 200,
      "encoder": { "name": "pq", "parameters": { "m": 96, "code_size": 8 } }
    }
  }
}
```

The FAISS engine's `encoder` block is where PQ (`m` sub-vectors × `code_size` bits) lives — same parameter meaning as mod-104 chapter 5. You reach for this at ~10⁸ vector scale where PQ's 32×–64× memory reduction matters.

### Query shape

```json
POST /support_articles/_search
{
  "query": {
    "knn": {
      "body_embedding": {
        "vector": [0.12, -0.03, ...],
        "k": 50,
        "filter": {
          "term": { "tenant_id": "acme" }
        }
      }
    }
  }
}
```

Filter semantics differ by engine ([OpenSearch filter docs](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/)):

- **Lucene engine** — supports **efficient pre-filtering** (correct filter-and-ANN composition).
- **FAISS engine** — supports **filtering with a Boolean filter clause** using OpenSearch's post-filter default; efficient filtering was added in 2.9 with `efSearch`-aware filtering for HNSW.
- **NMSlib engine** — post-filter only.

The rule of thumb: **for filtered ANN correctness, prefer the `lucene` engine unless you specifically need FAISS's PQ or IVF**. This is the inverse of the "FAISS is the fastest" instinct — for the *filtered* case, Lucene HNSW usually wins on correctness and is competitive on latency.

### Hybrid queries: `hybrid` query and `neural_search` plugin

OpenSearch ships a [`hybrid` query](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) that combines two sub-queries with per-query weights, plus a search pipeline for normalisation and combination (`normalization-processor`). This is OpenSearch's equivalent of Elasticsearch's RRF retriever — same idea, different DSL. mod-106 covers hybrid fusion in depth.

## Solr: `DenseVectorField`

Solr's [`DenseVectorField`](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html) is the newest of the four (Solr 9.0, 2022). It wraps Lucene HNSW; the feature set is roughly a subset of Elasticsearch's `dense_vector` — full-precision float and int8 byte-vectors, `euclidean`/`cosine`/`dot_product` similarity, but no built-in binary quantization or PQ.

Schema definition:

```xml
<fieldType name="knn_vector" class="solr.DenseVectorField"
           vectorDimension="768"
           similarityFunction="cosine"
           knnAlgorithm="hnsw"
           hnswMaxConnections="16"
           hnswBeamWidth="200"/>

<field name="body_embedding" type="knn_vector" indexed="true" stored="true"/>
```

Query syntax uses the `{!knn}` query parser:

```
q={!knn f=body_embedding topK=50}[0.12, -0.03, ...]
```

Filter composition uses `fq` (filter query) which in Solr defaults to post-filter for k-NN queries. See [Solr KNN filter docs](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html#usage-with-filter-queries) for the pre-filter escape hatch (setting `preFilter=true` in the query parser local params).

Solr's vector story is fine for basic use; it lags Elasticsearch on quantization coverage, on client-SDK maturity, and on the hybrid retrieval story. If you already run Solr, its `DenseVectorField` is a competent addition. If you are picking an engine specifically for the vector-plus-BM25 hybrid story, ES/OS/Vespa are stronger picks in 2026.

## Vespa: tensor fields and the multi-vector story

Vespa's approach is architecturally different from the Lucene family: **a vector is not a special field type; it is a tensor**. [Tensors](https://docs.vespa.ai/en/tensor-user-guide.html) in Vespa are named-dimension arrays with a rich algebra — you can express `tensor<float>(x[768])` (a fixed 768-dim vector), `tensor<float>(t{}, x[128])` (a variable-length collection of 128-dim vectors — the ColBERT shape), or `tensor<int8>(x[96])` (a scalar-quantized 96-dim vector).

### Schema (`.sd`) definition

```
schema support_article {
  document support_article {
    field body_embedding type tensor<float>(x[768]) {
      indexing: attribute | index
      attribute { distance-metric: angular }
      index {
        hnsw {
          max-links-per-node: 16
          neighbors-to-explore-at-insert: 200
        }
      }
    }
  }
  rank-profile hybrid {
    first-phase {
      expression: bm25(title) * 0.5 + closeness(field, body_embedding) * 0.5
    }
  }
}
```

Points worth naming:

- **`indexing: attribute | index`** — put the tensor in an in-memory attribute (for scoring) *and* build an HNSW index (for candidate selection). Vespa's separation of attributes and indexes is more explicit than ES/OS's.
- **`distance-metric: angular`** — Vespa's names for distance functions: `angular` (cosine), `euclidean`, `dotproduct`, `prenormalized-angular`, `hamming`.
- **Rank profile combines `bm25(title)` and `closeness(field, body_embedding)` in a single expression** — Vespa's ranking-expression story is the tightest first-class hybrid-retrieval formalism of the four engines. mod-106 uses this pattern extensively.

### The multi-vector / ColBERT case

The tensor field type is why Vespa is the reference for **multi-vector retrieval** (one vector per token, e.g., ColBERT, ColBERTv2, [Vespa's ColBERT guide](https://docs.vespa.ai/en/reference/embedding-reference.html#colbert-embedder)):

```
field colbert_tokens type tensor<int8>(t{}, x[128]) {
  indexing: attribute
  attribute { distance-metric: dotproduct }
}
```

The variable-dimension `t{}` says "an unknown number of tokens per document, each with a 128-dim vector." Rank expressions can compute MaxSim over the token dimension natively. Elasticsearch's `dense_vector` cannot express this without embedding one document as many child docs (a nested / rank_features workaround); OpenSearch's `knn_vector` has the same limitation. Vespa is the pick when you're serious about late-interaction retrieval.

## Filter semantics: pre-filter vs. post-filter across engines

mod-104 chapter 7 covered the correctness trap of post-filter with small `k` in depth. On the Lucene family:

| Engine | Pre-filter | Post-filter | Filter-and-ANN correctness at small `k` |
|---|---|---|---|
| Elasticsearch (Lucene HNSW) | Yes (default since 8.0) | — | Correct |
| OpenSearch (Lucene engine) | Yes | Fallback | Correct |
| OpenSearch (FAISS engine) | Yes (since 2.9, HNSW only) | Historic default | Correct with 2.9+ config |
| OpenSearch (NMSlib engine) | — | Only | **Trap** — do not use for selective filters |
| Solr (`DenseVectorField`) | Optional (`preFilter=true`) | Default | Trap unless explicitly configured |
| Vespa (HNSW attribute) | Yes | — | Correct |

The rule of thumb: **on any Lucene-family engine, verify pre-filtering is enabled for your k-NN queries**. The engines default badly (Solr, OpenSearch/FAISS pre-2.9) or well (Elasticsearch, OpenSearch/Lucene, Vespa) but the correctness cost of a silent post-filter with a selective filter and small `k` is large.

## The decision: Lucene-family vector field vs. dedicated vector database

The question you will be asked most often at level 30: *"We already run Elasticsearch. Do we add `dense_vector` fields, or do we stand up Qdrant/Milvus/Pinecone/pgvector?"*

**Reasons to add `dense_vector` to your existing Lucene-family engine:**

- **You want hybrid retrieval as a first-class query, not as a client-side fusion.** RRF in Elasticsearch, `hybrid` query in OpenSearch, ranking expressions in Vespa all express BM25 + dense in one request. Two engines = two roundtrips = harder fusion.
- **You already pay for the operational cost.** One fewer pager rotation is a real win. The retrieval-quality delta between "your existing ES + `dense_vector`" and "your existing ES + Qdrant next to it" is usually smaller than the operational delta.
- **You need the filter to be the same query across lexical and dense arms.** Tenant/ACL filters are a correctness concern (mod-112); one engine means one filter surface.
- **Your corpus is under ~10⁸ vectors and your embedding dim is standard (768 / 1024 / 1536).** All four engines' HNSW handles this comfortably.

**Reasons to reach for a dedicated vector database (mod-104):**

- **Your corpus is well past 10⁸ vectors.** The specialists (Milvus, Vespa, Pinecone) have deeper stories for filtered ANN at 10⁹ scale.
- **You need aggressive quantization your Lucene-family engine does not ship.** Milvus's DiskANN, Pinecone's proprietary quantization, or Qdrant's binary quantization + rescore can serve corpora that ES's `bbq_hnsw` cannot.
- **You are running multi-vector / late-interaction models seriously.** Vespa is the Lucene-family option; if you're not on Vespa, the dedicated vector databases with native ColBERT support (Vespa-cloud, Qdrant with multi-vector, Weaviate) are stronger.
- **Your existing engine is Elasticsearch on a license you cannot upgrade past a certain version.** ES 7.x has brute-force `dense_vector` only; you cannot add HNSW without a major version bump. Sometimes standing up a specialist is the smaller migration.
- **You need GPU-accelerated ANN.** Milvus's GPU indexes (mod-104 chapter 9) are the reference; ES/OS/Solr/Vespa have no first-party GPU ANN story as of 2026.

The decision is downstream of the same two questions from mod-104 chapter 1: what does the platform team already run, and what is the 18-month scale target?

## Sizing dense-vector fields on a Lucene-family engine

The sizing math from mod-104 chapter 6 carries over with one adjustment: **Lucene-family engines pay a per-document overhead for the field alongside the vector**, so a `dense_vector` field's on-disk cost is not just `dims × 4 bytes × N`. Rough sizing:

- **Full-precision `dense_vector`, 768 dims, 10M docs** — vectors alone: ~30 GB. HNSW graph: ~15% overhead. Plus the rest of the document (title, body, etc.) unchanged. So ~35 GB just for the vector field + graph.
- **`int8_hnsw` 768 dims, 10M docs** — quarter the vector cost: ~7.5 GB + 1.1 GB graph ≈ ~8.6 GB. This is why int8 is a good default.
- **`bbq_hnsw` 768 dims, 10M docs** — ~1 GB vectors + graph. Requires oversampling (`num_candidates` at 5–10× the target `k`) and often a full-precision rescore for competitive recall.

Match the sizing calc to chapter 6's shard-size heuristics: a 40-GB shard cap and 8.6 GB of vector field leaves plenty of room for the rest of the document, but a 35-GB shard of pure full-precision vectors is right at the ceiling.

## The three anti-patterns to avoid

1. **Turning on `dense_vector` without picking a quantization.** Full-precision at 10⁶+ scale is a memory bill you will regret. `int8_hnsw` is almost always the right default in 2026.
2. **Trusting post-filter with a selective filter and small `k`.** Correctness cost. Verify pre-filter is enabled (see the table above).
3. **Adding a second engine "for vectors" when the first already ships them.** Sometimes correct; often not. Defend the decision on the sizing math and the hybrid-retrieval story, not on the vendor's landing page.

## What is NOT in this chapter (and where it goes)

- **HNSW tuning parameters (`M`, `efConstruction`, `efSearch`)** — mod-104 chapter 4. The parameter meanings are engine-independent.
- **IVF-PQ tuning** — mod-104 chapter 5.
- **Metadata filtering correctness** — mod-104 chapter 7.
- **Index versioning and dual-index cutover for dense fields** — mod-104 chapter 8.
- **LTR features derived from vector scores** — chapter 5.
- **Sharding and tiering** — chapter 6.
- **Hybrid retrieval fusion (RRF, weighted-sum, learned)** — mod-106.

## What you should be able to do after this chapter

- Author a `dense_vector` / `knn_vector` / `DenseVectorField` / Vespa tensor field with correct `dims`, `similarity`, HNSW parameters, and quantization pick.
- Explain each engine's filter-composition semantics well enough to spot a post-filter correctness trap in someone else's config.
- Read the sizing math for a full-precision vs. int8 vs. binary dense-vector index and estimate the shard footprint.
- Defend the choice between "add `dense_vector` to the existing Lucene-family engine" and "stand up a dedicated vector database" on the two-question framework.
- Point to the RRF retriever (ES), hybrid query (OS), or rank expression (Vespa) as the native hybrid-retrieval construct for each engine.

## Further reading

- Elastic: [*Dense vector field type*](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) and [*k-NN search*](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
- Elastic Search Labs: [*Scalar quantization in Lucene*](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene) — the `int8_hnsw` background.
- Elastic Search Labs: [*Better Binary Quantization with RaBitQ*](https://www.elastic.co/search-labs/blog/rabitq-scalar-quantization-elasticsearch-lucene) — the `bbq_hnsw` background.
- OpenSearch: [*k-NN plugin*](https://opensearch.org/docs/latest/search-plugins/knn/index/) and [*k-NN filtering*](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/).
- Solr: [*Dense vector search*](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html).
- Vespa: [*Tensor user guide*](https://docs.vespa.ai/en/tensor-user-guide.html), [*ANN, HNSW*](https://docs.vespa.ai/en/approximate-nn-hnsw.html), [*ColBERT embedder*](https://docs.vespa.ai/en/reference/embedding-reference.html#colbert-embedder).
- Malkov, Y. A. and Yashunin, D. A. (2016). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — the HNSW paper that all four engines' ANN implementations descend from.
