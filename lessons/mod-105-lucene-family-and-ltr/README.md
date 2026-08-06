# mod-105-lucene-family-and-ltr: Lucene-Family Engines and Lexical / Learning-to-Rank Retrieval

**Estimated effort:** 14 hours

Lucene-family engines — Elasticsearch, OpenSearch, Solr, and Vespa — are the systems most enterprises reach for when they need full-text plus filters plus ranking plus vectors, composed by one query planner. This module walks the family end-to-end: the engine landscape and license map, the analyzer pipeline that decides recall, the field-mapping-and-BM25 baseline every hybrid stack fuses with, dense-vector fields alongside the inverted index, classical learning-to-rank on top of a lexical index, and the shard-sizing / tiering shape of a real cluster.

## Learning objectives

- Read the Lucene-family engine landscape — Elasticsearch, OpenSearch, Solr, and Vespa — and know which sits where (managed-vs-self-hosted, licensing, ANN maturity, LTR plugin ecosystem).
- Author a production BM25 index — analyzer choice (ICU, language-specific stemmers, custom synonyms), field mappings, boosting, function scoring — as the strong lexical baseline every hybrid stack needs.
- Extend the lexical index with dense-vector fields — Elasticsearch `dense_vector`, OpenSearch k-NN, Vespa tensor / HNSW — and understand the trade-off vs. a dedicated vector database.
- Implement classical learning-to-rank on top of a lexical index — LambdaMART / LambdaRank / RankNet via LightGBM ranking or XGBoost pairwise — using the Elasticsearch LTR built-in / OpenSearch LTR plugin.
- Reason about shard sizing (Elastic's shard-size heuristics, 20–40 GB shard cap), replica placement, hot-warm-cold tiering, and the query-latency implications of each.
- Cite the boundary to `staff-ml-engineer` (level 50) and `ai-infra-senior-engineer` (level 40) for web-scale sharding, cross-DC replication, and novel-ranking-model concerns.

## Lecture chapters

1. [Chapter 1 — The Lucene-family engine landscape](./01-lucene-family-landscape.md) — Elasticsearch, OpenSearch, Solr, Vespa; the 2021 license split; the four-question engine-selection framework.
2. [Chapter 2 — The analyzer pipeline](./02-analyzers-and-normalization.md) — character filters, tokenizers, ICU normalisation, language-specific stemmers, curated synonym strategies, testing with `_analyze`.
3. [Chapter 3 — Field mappings, BM25, boosting, and function scoring](./03-bm25-mappings-boosting-and-function-scoring.md) — the production lexical baseline: explicit mappings, BM25 tuning, `multi_match`, `function_score`, `rescore`.
4. [Chapter 4 — Dense-vector fields on Lucene-family engines](./04-dense-vector-fields.md) — ES `dense_vector`, OS k-NN engines, Solr `DenseVectorField`, Vespa tensor + HNSW; quantization picks; filter semantics; when to add vs. reach for a dedicated vector database.
5. [Chapter 5 — Classical learning-to-rank](./05-learning-to-rank.md) — RankNet → LambdaRank → LambdaMART; LightGBM/XGBoost training; feature engineering; ES/OS LTR plugin deployment; training/serving skew and the anti-patterns.
6. [Chapter 6 — Shard sizing, replica placement, and hot-warm-cold tiering](./06-shard-sizing-replicas-and-tiering.md) — the 20–40 GB shard heuristic, shard-count immutability, replicas, ILM policies, refresh cost, tiering economics.
7. [Chapter 7 — Boundaries to level 40 and level 50](./07-boundaries-to-level-40-and-level-50.md) — what `ai-infra-senior-engineer` and `staff-ml-engineer` own that you don't; the escalation triggers; the ownership contract.

## Exercises

- [Exercise 01 — Elasticsearch vs. OpenSearch vs. Vespa vs. Solr family map](./exercises/exercise-01-elasticsearch-vs-opensearch-vs-vespa-vs-solr-family-map.md)
- [Exercise 02 — BM25 analyzer and field-mapping workshop](./exercises/exercise-02-bm25-analyzer-and-field-mapping-workshop.md)
- [Exercise 03 — Dense-vector fields in Elasticsearch and OpenSearch](./exercises/exercise-03-dense-vector-fields-in-elasticsearch-and-opensearch.md)
- [Exercise 04 — LambdaMART LTR with the Elasticsearch LTR plugin](./exercises/exercise-04-lambdamart-ltr-with-elasticsearch-ltr-plugin.md)
- [Exercise 05 — Shard sizing and hot-warm-cold tiering drill](./exercises/exercise-05-shard-sizing-and-hot-warm-cold-tiering-drill.md)

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs (scaffolded; not authored in this module).
- `quizzes/`: knowledge checks (scaffolded; not authored in this module).
- `resources.md`: external references, primary docs, and reading list.
