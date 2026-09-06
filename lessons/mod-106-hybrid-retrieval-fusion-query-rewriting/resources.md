# Resources for mod-106-hybrid-retrieval-fusion-query-rewriting

Curated primary references for the topics covered by this module. Every link here is either a primary vendor doc, a peer-reviewed paper, or a canonical text — no vendor marketing pages.

## Fusion — the RRF default and its lineage

- Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009. [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the two-page origin paper for RRF. Read it once; it's short and the whole module's default rests on it.
- Bruch, S., Nardini, F. M., Nardo, C., Perego, R., and Tonellotto, N. (2023). *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM Transactions on Information Systems 41(4). [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) — the modern empirical study across BEIR and open-domain benchmarks, motivating rank-based over score-based fusion in the general case.
- Fox, E. A., and Shaw, J. A. (1994). *Combination of Multiple Searches.* TREC-2 proceedings. [NIST TREC](https://trec.nist.gov/pubs/trec2/papers/txt/22.txt) — the CombSUM / CombMNZ / CombMAX family; origin of the weighted-sum school.
- Aslam, J. A., and Montague, M. (2001). *Models for metasearch.* SIGIR 2001. [ACM DL](https://dl.acm.org/doi/10.1145/383952.384007) — the classical treatment of Borda, Condorcet, and rank-combination methods in information retrieval.
- Manmatha, R., Rath, T., and Feng, F. (2001). *Modeling Score Distributions for Combining the Outputs of Search Engines.* SIGIR 2001. [ACM DL](https://dl.acm.org/doi/10.1145/383952.384005) — the score-distribution modeling paper that justifies (or refutes) careful weighted-sum normalization strategies.
- Montague, M., and Aslam, J. A. (2002). *Condorcet fusion for improved retrieval.* CIKM 2002. [ACM DL](https://dl.acm.org/doi/10.1145/584792.584881) — the Condorcet-fusion baseline that RRF is measured against.
- Cormack, G. V., and Lynam, T. R. (2006). *Statistical precision of information retrieval evaluation.* SIGIR 2006. [ACM DL](https://dl.acm.org/doi/10.1145/1148170.1148236) — the underpinning for how many queries you actually need to see a fusion delta.

## Hybrid retrieval — landscape and benchmarks

- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the benchmark most modern fusion papers report on.
- BEIR project repository: [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir) — datasets, evaluation scripts, and leaderboards.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — the DPR paper; canonical dense-retriever reference.
- Wang, X., Macdonald, C., Tonellotto, N., and Ounis, I. (2021). *Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval.* ICTIR. [arXiv:2106.11251](https://arxiv.org/abs/2106.11251) — a dense-side treatment of the recall gap that motivates hybrid.
- Formal, T., Piwowarski, B., and Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* SIGIR. [arXiv:2107.05720](https://arxiv.org/abs/2107.05720) — the SPLADE reference paper; the learned-sparse arm option for three-arm hybrid.
- Formal, T., Lassance, C., Piwowarski, B., and Clinchant, S. (2021). *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.* [arXiv:2109.10086](https://arxiv.org/abs/2109.10086) — SPLADE v2 refinements.
- Lin, J., and Ma, X. (2021). *A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques.* [arXiv:2106.14807](https://arxiv.org/abs/2106.14807) — the uniCOIL discussion; useful frame for where learned-sparse sits between BM25 and dense.
- SPLADE reference implementation: [github.com/naver/splade](https://github.com/naver/splade).
- Ma, X., Guo, R., Wang, J., Lin, J., and Ma, X. (2021). *A Replication Study of Dense Passage Retriever.* [arXiv:2104.05740](https://arxiv.org/abs/2104.05740) — for the reader who wants to see the reproducibility discussion behind DPR-style baselines.

## Engine-native hybrid and RRF support

- Elasticsearch: [*Reciprocal rank fusion*](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — the native `rrf` retriever documentation.
- Elasticsearch: [*Retrievers*](https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html) — composable-retriever framework that hosts RRF, standard, kNN, and text-similarity sub-retrievers.
- Elasticsearch: [*Filter for kNN search*](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-search-filter-example) — the pre-filtered ANN shape used by chapter 6.
- Elasticsearch: [*Function score query*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html) — for soft-boost-not-filter preferences.
- OpenSearch: [*Hybrid query*](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) — the native two-arm hybrid shape.
- OpenSearch: [*Hybrid search with score normalization*](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) — the search-pipeline processor that provides RRF and min-max/L2/arithmetic-mean/geometric-mean/harmonic-mean fusion.
- OpenSearch: [*Filtered k-NN search*](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/) — pre-filtered ANN for OpenSearch.
- Vespa: [*Ranking framework*](https://docs.vespa.ai/en/ranking.html) and [*Reciprocal Rank Fusion in Vespa*](https://blog.vespa.ai/hybrid-search-in-vespa-with-reciprocal-rank-fusion/) — Vespa's rank-profile-based fusion story.
- Weaviate: [*Hybrid search*](https://weaviate.io/developers/weaviate/search/hybrid) — Weaviate's native BM25 + vector fusion with `alpha` and RRF options.
- Qdrant: [*Query API — Hybrid queries*](https://qdrant.tech/documentation/concepts/hybrid-queries/) — Qdrant's fusion (`rrf`, `dbsf`) between named vectors.
- Milvus: [*Hybrid search*](https://milvus.io/docs/multi-vector-search.md) — multi-vector search with RRF and weighted-sum rankers.
- pgvector: [pgvector README](https://github.com/pgvector/pgvector) and [pgvector's iterative index scans](https://github.com/pgvector/pgvector#iterative-index-scans) — the postgres-side answer to the tight-filter-with-HNSW recall problem.
- Pinecone: [*Hybrid search*](https://docs.pinecone.io/guides/search/hybrid-search) — Pinecone's sparse-dense hybrid with per-arm alpha.

## Filtered ANN and filter semantics

- Weaviate: [*Filters*](https://weaviate.io/developers/weaviate/search/filters) — the `where` semantics for filtered vector search.
- Qdrant: [*Filtering*](https://qdrant.tech/documentation/concepts/filtering/) — Qdrant's payload-index and filter-estimator model.
- Wang, X., Zhou, Y., and Deng, F. (2023). *Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters.* Microsoft Research. [arXiv:2401.01715](https://arxiv.org/abs/2401.01715) — the algorithms literature behind filter-aware ANN.
- Malkov, Y. A., and Yashunin, D. A. (2016). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — the HNSW paper; useful as background for why filtered ANN is a research problem at all.

## Query rewriting

- Gao, L., Ma, X., Lin, J., and Callan, J. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL 2023. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) — the HyDE (Hypothetical Document Embeddings) paper.
- Zheng, C., Mishra, S., Chen, X., Cheng, H.-T., Chi, E. H., Le, Q. V., and Zhou, D. (2023). *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.* [arXiv:2310.06117](https://arxiv.org/abs/2310.06117) — the step-back prompting paper.
- Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M. (2022). *Measuring and Narrowing the Compositionality Gap in Language Models.* [arXiv:2210.03350](https://arxiv.org/abs/2210.03350) — the self-ask decomposition paper.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP. [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) — the classic multi-hop QA benchmark whose decomposition pattern LLMs now emulate.
- Ma, X., Gong, Y., He, P., Zhao, H., and Duan, N. (2023). *Query Rewriting for Retrieval-Augmented Large Language Models.* EMNLP. [arXiv:2305.14283](https://arxiv.org/abs/2305.14283) — a systematic study of learned query rewriting; useful for the fine-tuned-rewriter alternative to prompt-engineered rewriting.
- Wang, L., Yang, N., and Wei, F. (2023). *Query2doc: Query Expansion with Large Language Models.* EMNLP. [arXiv:2303.07678](https://arxiv.org/abs/2303.07678) — a Query2doc / HyDE-adjacent expansion technique.
- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2022). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.* [arXiv:2212.10509](https://arxiv.org/abs/2212.10509) — IRCoT; the interleaved retrieve-then-reason pattern that decomposition subsumes.
- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2022). *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL. [arXiv:2108.00573](https://arxiv.org/abs/2108.00573) — a multi-hop QA benchmark useful for evaluating decomposition rewriters.
- Jagerman, R., Zhuang, H., Qin, Z., Wang, X., and Bendersky, M. (2023). *Query Expansion by Prompting Large Language Models.* [arXiv:2305.03653](https://arxiv.org/abs/2305.03653) — a controlled study of LLM-driven query expansion patterns.
- Rocchio, J. J. (1971). *Relevance feedback in information retrieval.* In *The SMART Retrieval System — Experiments in Automatic Document Processing.* Prentice-Hall. Historical origin for pseudo-relevance feedback (PRF), the classical antecedent to multi-query fanout.

### Framework references for rewriting patterns

- LangChain: [MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) — LLM-driven multi-query fanout.
- LangChain: [Multi-vector retriever](https://python.langchain.com/docs/how_to/multi_vector/) — decomposition / sub-query patterns.
- LangChain: [HyDE](https://api.python.langchain.com/en/latest/chains/langchain.chains.hyde.base.HypotheticalDocumentEmbedder.html) — reference HyDE chain.
- LlamaIndex: [SubQuestionQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/) — decomposition query engine.
- LlamaIndex: [Query transformations](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/advanced/query_transformations/) — step-back and HyDE transformers.
- Haystack: [Query decomposition and rewriting components](https://docs.haystack.deepset.ai/docs/query-decomposition) — Haystack 2's pipeline nodes for rewrite patterns.

## Semantic routing

- Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — the standard reference for the embedding backbone used in nearest-centroid routing.
- Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) — the reference model class for small intent classifiers.
- Guo, D., Sun, Y., Tang, D., Duan, N., Yin, J., Chi, H., Cao, J., Chen, P., and Zhou, M. (2018). *Question generation from SQL queries improves neural semantic parsing.* EMNLP. [ACL Anthology](https://aclanthology.org/D18-1188/) — background for intent classification with data augmentation.
- Bekker, J., and Davis, J. (2020). *Learning from positive and unlabeled data: a survey.* Machine Learning 109. [arXiv:1811.04820](https://arxiv.org/abs/1811.04820) — for the reader building routers with sparse labels; PU-learning frame for one-vs-rest training.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) — temperature scaling; the standard method for confidence calibration on the router's outputs.
- semantic-router (Aurelio Labs): [github.com/aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router) — an OSS implementation of the embedding-centroid routing pattern chapter 5 describes.
- LangChain: [Routing (RouterRunnable)](https://python.langchain.com/docs/how_to/routing/) — LangChain's routing primitive; useful as a comparator to a hand-rolled cascade.
- LlamaIndex: [RouterQueryEngine](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/) — LlamaIndex's LLM-router primitive.
- OpenAI: [Function calling / tool use](https://platform.openai.com/docs/guides/function-calling) — the mechanism most LLM-router fallbacks lean on.

## Score distributions, normalization, and paired significance testing

- Sakai, T. (2014). *Statistical Reform in Information Retrieval?* SIGIR Forum 48(1). [ACM DL](https://dl.acm.org/doi/10.1145/2641383.2641394) — a rigorous discussion of paired significance testing in IR eval; useful for the fusion-vs-fusion comparison discipline this module requires.
- Smucker, M. D., Allan, J., and Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM. [ACM DL](https://dl.acm.org/doi/10.1145/1321440.1321528) — the paper that argues for the randomization test / paired bootstrap over the t-test for IR metrics.
- Efron, B., and Tibshirani, R. J. (1994). *An Introduction to the Bootstrap.* Chapman and Hall / CRC. Foundational text for the bootstrap-confidence-interval technique used in the recall-at-k eval loops the exercises specify.

## Reranking, evaluation, and downstream context (for background)

- Nogueira, R., and Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the canonical cross-encoder reranker; important context for chapter 1's "fusion is about recall, not precision" claim.
- Nogueira, R., Lin, J., and Epistemic AI. (2020). *Document Ranking with a Pretrained Sequence-to-Sequence Model.* Findings of EMNLP. [arXiv:2003.06713](https://arxiv.org/abs/2003.06713) — monoT5 / duoT5 reranker.
- Khattab, O., and Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — the late-interaction retriever that sometimes competes with a two-arm fusion baseline.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the RAG-triad evaluation framework the peer track (`applied-ai-engineer`) uses at response granularity.
- Chen, T., Wang, H., Chen, S., Yu, W., Ma, K., Zhao, X., Zhang, H., and Yu, D. (2024). *Dense X Retrieval: What Retrieval Granularity Should We Use?* [arXiv:2312.06648](https://arxiv.org/abs/2312.06648) — the chunk-granularity discussion; useful cross-reference when a fusion experiment's failure turns out to be a granularity problem, not a fusion problem.

## Canonical textbooks and long-form references

- Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Free HTML edition](https://nlp.stanford.edu/IR-book/) — the foundational textbook; chapters 6, 8, and 11 are the closest fit to this module.
- Robertson, S., and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval. [PDF](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — the definitive BM25 monograph; carried over from mod-105 as the score-scale-mismatch argument's foundation.
- Bruch, S., Nardini, F. M., Perego, R., and Tonellotto, N. (2024). *Efficient and Effective Tree-Based and Neural Learning to Rank.* Foundations and Trends in Information Retrieval. [arXiv:2305.08869](https://arxiv.org/abs/2305.08869) — modern monograph covering fusion and LTR context; the reader who wants the wider frame around chapters 2–3.

## Ecosystem tooling

- `rank_bm25`: [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) — the reference Python BM25 used by the exercises' in-memory setup.
- `sentence-transformers`: [sbert.net](https://www.sbert.net/) — the reference Python embedding library.
- FAISS: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) — Facebook's ANN library; the default in-memory dense-retriever backend for exercise 01.
- Chroma: [docs.trychroma.com](https://docs.trychroma.com/) — a low-friction local vector store for prototyping.
- Ranx: [github.com/AmenRa/ranx](https://github.com/AmenRa/ranx) — Python library for IR evaluation and fusion; ships RRF, CombSUM, CombMNZ, Borda, Bordafuse, etc. Useful comparator for exercises 01–02.
- TREC eval scripts: [github.com/usnistgov/trec_eval](https://github.com/usnistgov/trec_eval) — the canonical recall@k / MRR / nDCG / MAP evaluator.

## Reading order suggestion

If you are new to the space, read in this order: RRF paper (Cormack 2009) → BEIR paper (Thakur 2021) → the score-scale-mismatch discussion in chapter 1 → Bruch et al. 2023 for the modern fusion-function comparison → HyDE (Gao 2022) and step-back (Zheng 2023) for the rewriting lever → your target engine's hybrid docs (Elasticsearch RRF retriever or OpenSearch hybrid pipeline) for the wire format. Save Manning / Raghavan / Schütze as the reference you keep coming back to for the classical framing.
