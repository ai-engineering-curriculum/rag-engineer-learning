# Chapter 1 — The Lucene-Family Engine Landscape: Elasticsearch, OpenSearch, Solr, and Vespa

## Why this chapter exists

A retrieval engineer who only knows vector databases will lose an argument with any team that already runs a search cluster. Lucene-family engines — Elasticsearch, OpenSearch, Solr, and Vespa — are the systems most enterprises reach for when they need *full-text plus filters plus ranking plus vectors*, all inside one query planner. They predate the vector-database category by more than a decade, they run under a huge fraction of enterprise search and logging pipelines, and — as of 2022–2024 — they all ship first-class ANN vector fields and (in three of four cases) a learning-to-rank plugin.

That is the reason mod-105 exists in the middle of a retrieval track. The lexical index (BM25 on tokenised text) is the strong baseline every serious hybrid retrieval stack fuses with dense retrieval — and if you cannot author, tune, and operate one, you cannot build the hybrid stack that mod-106 assembles. This chapter is the map: who these four engines are, where they came from, who runs them today, and which one belongs in your stack given the constraints you inherit.

Chapter 2 walks the tokenizer/analyzer layer, chapter 3 walks BM25 mappings and scoring, chapter 4 walks the dense-vector fields those engines added, chapter 5 walks learning-to-rank on top of them, chapter 6 walks the sharding and tiering shape of a real cluster, and chapter 7 marks the boundary to level-40 and level-50 roles.

## What "the Lucene family" actually means

**Apache Lucene** ([lucene.apache.org](https://lucene.apache.org/)) is a Java library that gives you an inverted index, a scoring model, an analyzer pipeline, and a query parser. It is not a server; it has no HTTP endpoint, no cluster manager, no replication. It has been continuously developed since Doug Cutting released the first version in 1999 and remains the *only* mainstream open-source full-text index implementation of production quality.

The "Lucene family" is the set of server-shaped products that embed Lucene as their storage-and-search core and wrap it with distribution, an API, an operator experience, and additional query and aggregation surface:

- **Apache Solr** ([solr.apache.org](https://solr.apache.org/)) — the original Lucene wrapper, released by Yonik Seeley in 2004 and donated to Apache in 2006. XML/JSON HTTP API on top of Lucene. Cluster shape (SolrCloud) added Zookeeper-coordinated replicas in 2011.
- **Elasticsearch** ([elastic.co/elasticsearch](https://www.elastic.co/elasticsearch)) — Shay Banon's 2010 rewrite of the Lucene wrapper story around a JSON-native REST API, distributed shard/replica model, and the aggregations DSL. Grew into the default logging-and-search stack (ELK) through the mid-2010s and remains the market's centre of gravity.
- **OpenSearch** ([opensearch.org](https://opensearch.org/)) — the AWS-led fork of Elasticsearch 7.10.2, launched April 2021 after Elastic re-licensed Elasticsearch away from Apache 2.0 (see "The 2021 licensing split" below). Ships as Apache 2.0; feature-tracks Elasticsearch loosely; runs the AWS Managed OpenSearch service.
- **Vespa** ([vespa.ai](https://vespa.ai/)) — Yahoo!'s (later Verizon Media, now Vespa.ai's) search-and-serving engine, open-sourced in 2017 after two decades of internal use. **Not built on Lucene** — Vespa has its own C++ index, ranking, and tensor engine — but it is included in "the Lucene family" for this track because it plays the same role in retrieval architectures: a first-class inverted-index engine that also does dense retrieval, hybrid fusion, and structured queries.

All four expose the same primitives — inverted index, BM25-like scoring, structured filter query, aggregations, and (now) a dense-vector field — with different APIs, cluster shapes, and licenses. The rest of this chapter reads them against a shared rubric so the family map has a scaffold.

## The 2021 licensing split (why OpenSearch exists at all)

In January 2021 Elastic changed the license of the Elasticsearch and Kibana source code from Apache License 2.0 to a dual **Server-Side Public License (SSPL)** / **Elastic License 2.0 (ELv2)** starting with version 7.11 ([Elastic's announcement](https://www.elastic.co/blog/why-license-change-aws), January 2021). Both licenses are source-available but not OSI-approved "open source"; the practical effect for cloud vendors was that they could no longer freely offer managed Elasticsearch as a hosted service.

**AWS forked.** The [OpenSearch announcement](https://aws.amazon.com/blogs/opensource/introducing-opensearch/) (April 2021) took the last Apache-licensed Elasticsearch (7.10.2) and Kibana (7.10.2), rebranded them OpenSearch and OpenSearch Dashboards, and continued them under Apache 2.0. AWS's managed Elasticsearch Service was rebranded to "Amazon OpenSearch Service."

**Elastic re-added AGPL v3** in August 2024 ([Elastic's follow-up post](https://www.elastic.co/blog/elasticsearch-is-open-source-again)), making Elasticsearch tri-licensable as of 8.16: ELv2, SSPL v1, or AGPL v3. The company positions AGPL v3 as their formal "open source" story from 8.16 onward.

The practical map today:

| Version window | Elasticsearch license | OpenSearch license |
|---|---|---|
| ES ≤ 7.10.2 | Apache 2.0 | (OpenSearch not yet forked) |
| ES 7.11 → 8.15 | SSPL v1 / ELv2 (source-available) | Apache 2.0 |
| ES 8.16+ | SSPL v1 / ELv2 / **AGPL v3** | Apache 2.0 |

The upshot for a retrieval engineer:

- **If your org's OSS policy forbids source-available licenses**, OpenSearch is the safe pick and Elasticsearch (any version) is not — unless you accept AGPL v3, which many corporate policies also forbid. Check with legal early; do not assume "Elasticsearch is open source" from muscle memory.
- **If you already run AWS-managed Elasticsearch**, note that AWS's "Elasticsearch Service" now redirects to OpenSearch Service. Elastic Cloud is separate.
- **API compatibility between the two forks has drifted since 7.10.** They share ancestry but not features: OpenSearch has its own security plugin, its own k-NN plugin, its own LTR plugin, and its own ML Commons. Elastic ships a proprietary Elastic ML, a proprietary security X-Pack, and an integrated inference API. The APIs look similar; the extension points do not.

Solr never had a license event; it has been Apache 2.0 since 2006. Vespa has been Apache 2.0 since its 2017 open-sourcing.

## The rubric — five cells per engine

For each engine, five cells worth understanding before you pick one.

### 1. Deployment shape

- **Self-hosted (bring your own K8s / VMs).** All four. Elastic ships an [Elastic Cloud on Kubernetes (ECK) operator](https://www.elastic.co/guide/en/cloud-on-k8s/current/index.html); OpenSearch ships an [OpenSearch Kubernetes Operator](https://github.com/opensearch-project/opensearch-k8s-operator); Solr ships [`solr-operator`](https://solr.apache.org/operator/); Vespa ships a [Kubernetes deployment guide](https://docs.vespa.ai/en/vespa-quick-start-kubernetes.html) but no first-party operator.
- **Managed by the vendor.** Elastic Cloud (Elastic itself) hosts Elasticsearch; AWS OpenSearch Service and OpenSearch Serverless host OpenSearch; Vespa Cloud (Vespa.ai) hosts Vespa; no first-party managed Solr (managed offerings exist from third parties like SearchStax).
- **Managed by a cloud provider.** AWS OpenSearch (managed and serverless); GCP does not ship a native managed Elasticsearch or Solr (it partners with Elastic Cloud on Marketplace); Azure Elasticsearch is a Marketplace listing of Elastic Cloud; Bonsai/Elastic partners exist for third-party managed ES/OS.

### 2. Vector-field / ANN maturity

All four now ship dense-vector fields; the shapes and maturity differ.

- **Elasticsearch.** [`dense_vector` field](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) since 7.3 (2019, brute-force only); HNSW indexing since 8.0 (2022); scalar quantization (`int8_hnsw`) since 8.12 (2024); binary quantization (`bbq_hnsw`) since 8.14 (2024). Uses Lucene 9.x/10.x's HNSW implementation ([Lucene-9004](https://issues.apache.org/jira/browse/LUCENE-9004)) — mature and well-optimized.
- **OpenSearch.** [k-NN plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/) ships in-tree since 1.0; supports HNSW (via Lucene, FAISS, and NMSlib engines) and IVF via FAISS engine. The choice of ANN engine per index is exposed in the mapping (`method.engine` = `lucene` | `faiss` | `nmslib`). FAISS engine adds product quantization and disk-tiered storage (OpenSearch 2.14+).
- **Solr.** [`DenseVectorField`](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html) since 9.0 (2022), using Lucene HNSW. Feature-complete for basic dense retrieval; ecosystem around it (client SDKs, ranking plugins) is thinner than Elasticsearch's.
- **Vespa.** [Tensor fields](https://docs.vespa.ai/en/tensor-user-guide.html) with first-class HNSW indexing since 2020. Vespa was the earliest of the four to ship production HNSW and has the deepest support for structured tensors (multi-dimensional, named-dimension) rather than fixed-shape vectors — the shape that ColBERT-style late-interaction models want ([Vespa's ColBERT guide](https://docs.vespa.ai/en/reference/embedding-reference.html#colbert-embedder)).

The one-line takeaway: **Vespa was first on production HNSW and has the strongest multi-vector story; Elasticsearch has the most polished DX and quantization coverage; OpenSearch's k-NN is competitive but exposes three ANN engines with different behavior; Solr's is fine for basic use but is not where research is landing.**

### 3. Learning-to-rank plugin ecosystem

- **Elasticsearch.** As of Elastic 8.12 (2024), LTR is a **built-in feature** ([Elasticsearch LTR docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/learning-to-rank.html)) — no plugin install. Prior to 8.12, the community [`elasticsearch-learning-to-rank`](https://github.com/o19s/elasticsearch-learning-to-rank) plugin (o19s) was the standard. LTR supports XGBoost and LightGBM model formats.
- **OpenSearch.** [`opensearch-learning-to-rank-base`](https://github.com/opensearch-project/opensearch-learning-to-rank-base) is community-maintained under the OpenSearch project. Same model-format support (XGBoost, LightGBM).
- **Solr.** Ships an [LTR contrib module](https://solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html) directly in the Solr distribution since 6.4 (2017). Also supports XGBoost and LightGBM plus older linear/MART models.
- **Vespa.** No plugin needed — Vespa's [ranking expressions](https://docs.vespa.ai/en/ranking.html) natively support gradient-boosted trees, ONNX-model evaluation, and multi-phase ranking (first-phase, second-phase, global-phase). Vespa was designed around learned ranking from the start; ranking is not a bolted-on plugin.

### 4. Query / aggregation surface

All four support:

- Full-text queries (match, phrase, multi_match, span queries).
- Structured filters (term, terms, range, bool).
- Aggregations / facets over structured fields.
- Nested / parent-child (for one-to-many payloads).
- Dense-vector queries composed with the above.

Differences worth knowing:

- **Elasticsearch/OpenSearch** use a JSON DSL, deeply nested. Elastic's [`_search` API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html) is the reference API and the one most tooling assumes.
- **Solr** uses HTTP query parameters with a family of query parsers ([`edismax`](https://solr.apache.org/guide/solr/latest/query-guide/edismax-query-parser.html), [`dismax`](https://solr.apache.org/guide/solr/latest/query-guide/dismax-query-parser.html), [`lucene`](https://solr.apache.org/guide/solr/latest/query-guide/standard-query-parser.html), [`json`](https://solr.apache.org/guide/solr/latest/query-guide/json-query-dsl.html)). More concise for simple queries, awkward for deeply nested boolean logic.
- **Vespa** uses [YQL](https://docs.vespa.ai/en/query-language.html) — a SQL-shaped query language that composes cleanly with ranking expressions, and reads more like a query planner's input than a REST DSL.

### 5. Cluster shape and coordination

- **Elasticsearch** — cluster of nodes with roles (master-eligible, data, ingest, coordinating, ML). Cluster state managed by an internal Zen2 coordination layer (Raft-shaped consensus since 7.0).
- **OpenSearch** — inherits Elasticsearch 7.10's cluster model. Same node roles, same coordination.
- **Solr** — SolrCloud mode with ZooKeeper as the external coordination store. Non-cloud (standalone) mode also exists but is a single-node deployment.
- **Vespa** — [content clusters + container clusters](https://docs.vespa.ai/en/overview.html) with a config server (ZooKeeper-backed) for coordination. The container/content split is deeper than Elasticsearch's — the container is where you plug in Java components (searchers, document processors) that ES/OS lack.

## The four systems, one-line orientation

| Engine | Core | License (2026) | Managed offering | Vector maturity | LTR story | Distinctive strength |
|---|---|---|---|---|---|---|
| **Elasticsearch** | Lucene | ELv2 / SSPL v1 / AGPL v3 | Elastic Cloud (native) | HNSW + int8/binary quantization | Built-in from 8.12 | Largest ecosystem, best DX, deepest observability integrations |
| **OpenSearch** | Lucene (forked from ES 7.10) | Apache 2.0 | AWS OpenSearch Service (managed + serverless) | HNSW via Lucene/FAISS/NMSlib engines | Community plugin | AWS-blessed, fully open source, three ANN engines to pick from |
| **Solr** | Lucene | Apache 2.0 | SearchStax and other third parties; no first-party | HNSW `DenseVectorField` (Lucene-backed) | Contrib module in-tree | Longest-running production track record, Apache governance |
| **Vespa** | Own C++ engine | Apache 2.0 | Vespa Cloud | HNSW + multi-vector tensors (ColBERT, ColBERT-quantized) | Native ranking expressions (no plugin) | Best fit for large-scale personalization, multi-phase ranking, multi-vector retrieval |

Two more one-liners that come up often enough to name:

- **"Just use Elasticsearch"** is the default answer inside most enterprises that already run it, because the operational cost of adding a second engine outweighs the retrieval-quality delta between engines for most workloads. This is often correct, but you should defend it — chapter 6 gives you the sizing conversation, exercise 01 gives you the family-map artifact.
- **"Just use OpenSearch"** is the default inside most orgs on AWS, because AWS OpenSearch Service is the least-friction managed option. Same defense-required rule applies.

## Which one belongs in your stack — the four questions

The picks below are heuristics for the level-30 conversation. They are not endorsements; there are shops that swap any of these on other axes.

1. **What license posture does your org accept?**
   - If Apache 2.0 only: OpenSearch, Solr, or Vespa. Not Elasticsearch (any recent version).
   - If AGPL v3 is acceptable but SSPL is not: Elasticsearch 8.16+ under AGPL, plus the Apache-licensed three.
   - If SSPL/ELv2 source-available is acceptable: any of the four.
2. **What is your platform team already running?**
   - Already ELK-shaped: probably Elasticsearch or OpenSearch. Do not add a third engine to page for.
   - Already Solr for other workloads: Solr for the retrieval workload too, unless you specifically need modern LTR features (in which case, ES/OS/Vespa).
   - Nothing already: pick on the other three questions.
3. **What are your ANN and multi-vector requirements?**
   - Single-vector-per-document HNSW, up to ~10⁸ vectors: any of the four.
   - Multi-vector-per-document (ColBERT-style late interaction), or tensor-shaped signals: Vespa is the strong pick.
   - Aggressive quantization (int8, binary) with mature tooling: Elasticsearch has the most polished story as of 8.12+.
   - Choice of ANN engines exposed in the mapping: OpenSearch (Lucene / FAISS / NMSlib).
4. **How mature is your LTR story?**
   - No LTR yet: any of the four works; Vespa has the shallowest learning curve *if* you already speak ranking expressions.
   - Existing XGBoost/LightGBM models: Elasticsearch (built-in) or Solr (contrib module) are the smoothest imports.
   - Planning a large personalization / multi-signal ranking system: Vespa's multi-phase ranking + tensor ranking is the class-leading design.

You will use these four questions to structure the family-map artifact in **exercise 01**. They are not the *only* questions, but they cover ~80% of real vendor-selection conversations at this level.

## What is NOT in this chapter (and where it goes)

- **The analyzer pipeline** — tokenizers, filters, ICU, stemmers, synonyms. Chapter 2.
- **Field mappings, boosting, BM25 tuning, function_score** — the mechanics of authoring a production lexical index. Chapter 3.
- **Dense-vector field definitions and ANN parameters per engine** — chapter 4.
- **Learning-to-rank pipelines, feature engineering, model training, deployment** — chapter 5.
- **Shard sizing, replicas, tiering** — chapter 6.
- **The line to level-40 / level-50 roles** — chapter 7.

## What you should be able to do after this chapter

- Name the four engines in the Lucene family (three literally Lucene-backed, one Lucene-family-in-spirit) and give a one-sentence orientation for each.
- Explain the 2021 Elastic license split, the OpenSearch fork, and Elastic's 2024 AGPL v3 re-addition — well enough to advise a security/legal review.
- Read a project's constraints (license posture, platform incumbency, ANN needs, LTR ambition) and defend an engine choice on the four questions above.
- Point to the specific downstream chapter that covers each detail this chapter deferred.

## Further reading

- Apache Lucene documentation: [lucene.apache.org](https://lucene.apache.org/) — the underlying library.
- Elastic's licensing history: [*Doubling down on open, Part II*](https://www.elastic.co/blog/why-license-change-aws) (2021) and [*Elasticsearch is open source, again*](https://www.elastic.co/blog/elasticsearch-is-open-source-again) (August 2024).
- AWS's fork announcement: [*Introducing OpenSearch*](https://aws.amazon.com/blogs/opensource/introducing-opensearch/) (April 2021).
- Apache Solr documentation: [solr.apache.org/guide](https://solr.apache.org/guide/) — the reference for Solr internals.
- Vespa documentation: [docs.vespa.ai](https://docs.vespa.ai/) — the reference for the non-Lucene member of this family.
- Grand, A. and Muir, R. (2015). *Bringing Learning-to-Rank to Solr and Elasticsearch.* Berlin Buzzwords 2015 — historical origin of the LTR plugins that all four engines now support.
