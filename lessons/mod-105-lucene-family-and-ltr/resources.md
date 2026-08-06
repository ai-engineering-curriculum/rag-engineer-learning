# Resources for mod-105-lucene-family-and-ltr

Curated primary references for the topics covered by this module. Every link here is either a primary vendor doc, a peer-reviewed paper, or a canonical text — no vendor marketing pages.

## Apache Lucene

- Apache Lucene project home: [lucene.apache.org](https://lucene.apache.org/).
- Lucene documentation: [lucene.apache.org/core/9_10_0/](https://lucene.apache.org/core/9_10_0/) (adjust the version path as needed).
- Cutting, D. (1999). Original Lucene announcement (historical). See [`lucene.apache.org/core/`](https://lucene.apache.org/core/) for the project timeline.

## Elasticsearch and OpenSearch

- Elasticsearch reference documentation: [elastic.co/guide/en/elasticsearch/reference/current/index.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html).
- Elasticsearch analyzer anatomy: [elastic.co/guide/en/elasticsearch/reference/current/analyzer-anatomy.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/analyzer-anatomy.html).
- Elasticsearch language analyzers: [elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html).
- Elasticsearch ICU analysis plugin: [elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html).
- Elasticsearch Kuromoji plugin (Japanese): [elastic.co/guide/en/elasticsearch/plugins/current/analysis-kuromoji.html](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-kuromoji.html).
- Elasticsearch Nori plugin (Korean): [elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html).
- Elasticsearch SmartCN plugin (Chinese): [elastic.co/guide/en/elasticsearch/plugins/current/analysis-smartcn.html](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-smartcn.html).
- Elasticsearch mapping types: [elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html).
- Elasticsearch BM25 similarity: [elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html).
- Elasticsearch multi-match query: [elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html).
- Elasticsearch function score: [elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html).
- Elasticsearch dense_vector: [elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
- Elasticsearch k-NN search: [elastic.co/guide/en/elasticsearch/reference/current/knn-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
- Elasticsearch RRF: [elastic.co/guide/en/elasticsearch/reference/current/rrf.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html).
- Elasticsearch Learning to Rank overview: [elastic.co/guide/en/elasticsearch/reference/current/learning-to-rank.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/learning-to-rank.html).
- Elasticsearch shard-size guidance: [elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html).
- Elasticsearch data tiers: [elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html).
- Elasticsearch index lifecycle management (ILM): [elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html).
- Elasticsearch cross-cluster replication: [elastic.co/guide/en/elasticsearch/reference/current/xpack-ccr.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/xpack-ccr.html).
- Elastic Search Labs: [*Scalar quantization in Lucene*](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene).
- Elastic Search Labs: [*Better Binary Quantization with RaBitQ*](https://www.elastic.co/search-labs/blog/rabitq-scalar-quantization-elasticsearch-lucene).
- Elastic licensing history: [*Doubling down on open, Part II*](https://www.elastic.co/blog/why-license-change-aws) (2021); [*Elasticsearch is open source, again*](https://www.elastic.co/blog/elasticsearch-is-open-source-again) (2024).
- OpenSearch project home and docs: [opensearch.org/docs/latest/](https://opensearch.org/docs/latest/).
- OpenSearch k-NN plugin: [opensearch.org/docs/latest/search-plugins/knn/index/](https://opensearch.org/docs/latest/search-plugins/knn/index/).
- OpenSearch k-NN filter search: [opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/).
- OpenSearch k-NN vector field: [opensearch.org/docs/latest/field-types/supported-field-types/knn-vector/](https://opensearch.org/docs/latest/field-types/supported-field-types/knn-vector/).
- OpenSearch hybrid query: [opensearch.org/docs/latest/query-dsl/compound/hybrid/](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/).
- OpenSearch LTR plugin: [github.com/opensearch-project/opensearch-learning-to-rank-base](https://github.com/opensearch-project/opensearch-learning-to-rank-base).
- OpenSearch Index State Management (ISM): [opensearch.org/docs/latest/im-plugin/ism/index/](https://opensearch.org/docs/latest/im-plugin/ism/index/).
- OpenSearch fork announcement (April 2021): [aws.amazon.com/blogs/opensource/introducing-opensearch/](https://aws.amazon.com/blogs/opensource/introducing-opensearch/).

## Apache Solr

- Solr project home: [solr.apache.org](https://solr.apache.org/).
- Solr reference guide: [solr.apache.org/guide/solr/latest/](https://solr.apache.org/guide/solr/latest/).
- Solr language analysis: [solr.apache.org/guide/solr/latest/indexing-guide/language-analysis.html](https://solr.apache.org/guide/solr/latest/indexing-guide/language-analysis.html).
- Solr synonym graph filter: [solr.apache.org/guide/solr/latest/indexing-guide/filters.html#synonym-graph-filter](https://solr.apache.org/guide/solr/latest/indexing-guide/filters.html#synonym-graph-filter).
- Solr eDisMax query parser: [solr.apache.org/guide/solr/latest/query-guide/edismax-query-parser.html](https://solr.apache.org/guide/solr/latest/query-guide/edismax-query-parser.html).
- Solr function queries: [solr.apache.org/guide/solr/latest/query-guide/function-queries.html](https://solr.apache.org/guide/solr/latest/query-guide/function-queries.html).
- Solr dense vector search: [solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html).
- Solr Learning to Rank: [solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html](https://solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html).
- Solr distributed search / SolrCloud: [solr.apache.org/guide/solr/latest/deployment-guide/cluster-types.html](https://solr.apache.org/guide/solr/latest/deployment-guide/cluster-types.html).

## Vespa

- Vespa project home: [vespa.ai](https://vespa.ai/).
- Vespa documentation: [docs.vespa.ai](https://docs.vespa.ai/).
- Vespa tensor user guide: [docs.vespa.ai/en/tensor-user-guide.html](https://docs.vespa.ai/en/tensor-user-guide.html).
- Vespa ANN and HNSW: [docs.vespa.ai/en/approximate-nn-hnsw.html](https://docs.vespa.ai/en/approximate-nn-hnsw.html).
- Vespa ranking framework: [docs.vespa.ai/en/ranking.html](https://docs.vespa.ai/en/ranking.html).
- Vespa BM25 rank feature: [docs.vespa.ai/en/reference/bm25.html](https://docs.vespa.ai/en/reference/bm25.html).
- Vespa LightGBM ranking: [docs.vespa.ai/en/lightgbm.html](https://docs.vespa.ai/en/lightgbm.html).
- Vespa ColBERT embedder: [docs.vespa.ai/en/reference/embedding-reference.html#colbert-embedder](https://docs.vespa.ai/en/reference/embedding-reference.html#colbert-embedder).
- Vespa YQL query language: [docs.vespa.ai/en/query-language.html](https://docs.vespa.ai/en/query-language.html).
- Vespa content cluster reference: [docs.vespa.ai/en/reference/services-content.html](https://docs.vespa.ai/en/reference/services-content.html).

## Analyzers, tokenizers, and Unicode

- Unicode Text Segmentation (UAX #29): [unicode.org/reports/tr29/](https://unicode.org/reports/tr29/).
- ICU project home: [icu.unicode.org](https://icu.unicode.org/).
- Porter, M. F. (1980). *An algorithm for suffix stripping.* Program, 14(3), 130–137. [tartarus.org/martin/PorterStemmer](https://tartarus.org/martin/PorterStemmer/def.txt).
- Snowball stemmers: [snowballstem.org](https://snowballstem.org/).
- Kuromoji Japanese tokenizer: [github.com/atilika/kuromoji](https://github.com/atilika/kuromoji).
- WordNet: [wordnet.princeton.edu](https://wordnet.princeton.edu/).

## BM25 and probabilistic IR

- Robertson, S. and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval. [City U London PDF](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf).
- Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Free HTML edition](https://nlp.stanford.edu/IR-book/).
- Turnbull, D. and Berryman, J. (2016). *Relevant Search: With Applications for Solr and Elasticsearch.* Manning Publications.

## Learning to rank

- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research TR MSR-TR-2010-82. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/).
- Burges, C., et al. (2005). *Learning to Rank using Gradient Descent.* ICML 2005 (RankNet paper). [Microsoft Research](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
- Wu, Q., Burges, C. J., Svore, K. M., & Gao, J. (2010). *Adapting boosting for information retrieval measures.* Information Retrieval, 13(3). [Microsoft Research](https://www.microsoft.com/en-us/research/publication/adapting-boosting-for-information-retrieval-measures/).
- Joachims, T., Swaminathan, A., & Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased Feedback.* WSDM 2017. [ACM DL](https://dl.acm.org/doi/10.1145/3018661.3018699).
- Chapelle, O. and Zhang, Y. (2009). *A Dynamic Bayesian Network Click Model for Web Search Ranking.* WWW 2009. [ResearchGate](https://www.researchgate.net/publication/221153934_A_dynamic_bayesian_network_click_model_for_web_search_ranking).
- Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval.* Foundations and Trends in IR. Springer / preprint widely available.
- LightGBM: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/).
- XGBoost Learning-to-Rank tutorial: [xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
- OpenSearch Blog: [*Introducing Learning to Rank*](https://opensearch.org/blog/introducing-learning-to-rank/).
- Community LTR resource: [o19s.com/learn](https://o19s.com/learn/) — OpenSource Connections write-ups on LTR practice.

## Dense retrieval frontier (for context beyond this module)

- Malkov, Y. A. and Yashunin, D. A. (2016). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906).
- Guo, R., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020 (ScaNN). [arXiv:1908.10396](https://arxiv.org/abs/1908.10396).

## Ecosystem tooling

- Elasticsearch Python client: [elasticsearch-py.readthedocs.io](https://elasticsearch-py.readthedocs.io/).
- OpenSearch Python client: [opensearch-project.github.io/opensearch-py/](https://opensearch-project.github.io/opensearch-py/).
- SolrJ (Java) and pysolr: [github.com/django-haystack/pysolr](https://github.com/django-haystack/pysolr).
- Vespa Python client (`pyvespa`): [pyvespa.readthedocs.io](https://pyvespa.readthedocs.io/).
- [`o19s/elasticsearch-learning-to-rank`](https://github.com/o19s/elasticsearch-learning-to-rank) — the community LTR plugin history that preceded ES 8.12's built-in LTR.

## Reading order suggestion

If you are new to the space, read in this order: BM25 paper (Robertson & Zaragoza) → LTR overview (Burges 2010) → your target engine's reference docs (ES or OS depending on stack) → engine-specific LTR plugin docs → *Relevant Search* (Turnbull & Berryman) for the operator's playbook.
