# Resources — mod-101: Retrieval-Engineering Foundations and Role Position

Curated primary sources for the chapters and exercises. Every reference below has been used or cited by the module's chapters; links go to the primary source (arXiv, publisher, or official documentation) rather than a summary blog.

## Canonical framing papers

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* v5. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — the direct predecessor Lewis 2020 builds on.
- Guu, K., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* ICML 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909).

## Classical information retrieval

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Free HTML edition](https://nlp.stanford.edu/IR-book/). Chapters 1 (Boolean retrieval), 2 (dictionaries), 6 (scoring and vector-space model), 8 (evaluation), 11 (probabilistic retrieval and BM25) are the ones this module refreshes.
- Robertson, S. (1977). *The Probability Ranking Principle in IR.* Journal of Documentation 33(4). [DOI:10.1108/eb026647](https://www.emerald.com/insight/content/doi/10.1108/eb026647/full/html).
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval 3(4). [DOI:10.1561/1500000019](https://dl.acm.org/doi/10.1561/1500000019).
- Elasticsearch engineering blog: [*Practical BM25* (parts 1–3)](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — walks through `k1`, `b`, and Elasticsearch's `idf` variant.

## Pipeline shape — auxiliary reading

- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the motivation for the *Assemble* stage's ordering and compression.
- Anthropic engineering (2024). *Introducing Contextual Retrieval.* [Anthropic blog post](https://www.anthropic.com/news/contextual-retrieval) — a public engineering case study of the exact shape-shift from feature-RAG to engineered-system-RAG that chapter 3 names.

## Toolchain — official documentation

- Python: [docs.python.org](https://docs.python.org/3/) · [`uv` docs](https://docs.astral.sh/uv/) · [Poetry docs](https://python-poetry.org/docs/).
- LlamaIndex: [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- sentence-transformers: [sbert.net](https://www.sbert.net/) · [Hugging Face Hub](https://huggingface.co/models).
- PyTorch install matrix: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
- pgvector (extension): [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
- pgvector-python (client): [github.com/pgvector/pgvector-python](https://github.com/pgvector/pgvector-python).
- Elasticsearch install: [official install docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
- OpenSearch install: [official install docs](https://opensearch.org/docs/latest/install-and-configure/).
- `ir_measures`: [ir-measur.es](https://ir-measur.es/en/latest/) · [getting started](https://ir-measur.es/en/latest/getting-started.html) — the retrieval-only evaluation library used from mod-101 onward.
- `rank_bm25` (used in exercise 02): [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25).

## Track-internal cross-references

- [`../../CURRICULUM.md`](../../CURRICULUM.md) — the track's ownership rule and module map (chapter 4 references it).
- [`../../PREREQUISITES.md`](../../PREREQUISITES.md) — assumed entry skills and adjacent-track links.
- [`../../JOB_REQUIREMENTS.md`](../../JOB_REQUIREMENTS.md) — the requirements-to-coverage map from the plan-authoring pass.
