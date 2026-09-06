# Resources for mod-107-reranking-and-context-assembly

Curated primary references for the topics covered by this module. Every link is either a primary vendor doc, a peer-reviewed paper, a canonical text, or the reference implementation of a technique the chapters describe — no vendor marketing pages.

## The two-stage pipeline — reranking motivation and landscape

- Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the paper that put cross-encoder reranking on the map for IR. The reference for "why a two-stage retrieve-then-rerank pipeline beats a one-stage retriever."
- Nogueira, R., Lin, J., and Epistemic AI (2020). *Document Ranking with a Pretrained Sequence-to-Sequence Model.* Findings of EMNLP. [arXiv:2003.06713](https://arxiv.org/abs/2003.06713) — monoT5 / duoT5; the sequence-to-sequence reranker alternative to encoder-only cross-encoders.
- Craswell, N., Mitra, B., Yilmaz, E., Campos, D., and Voorhees, E. M. (2020). *Overview of the TREC 2019 Deep Learning Track.* [arXiv:2003.07820](https://arxiv.org/abs/2003.07820) — the TREC-DL benchmark that most reranker numbers report on alongside MS MARCO.
- Microsoft: [*MS MARCO Passage Ranking Leaderboard*](https://microsoft.github.io/msmarco/) — the reference public leaderboard for passage-ranking numbers.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the zero-shot IR benchmark most modern reranker papers report on.
- BEIR project repository: [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir) — datasets, evaluation scripts, and baselines.
- Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Free HTML edition](https://nlp.stanford.edu/IR-book/) — the foundational textbook; chapter 8 (Evaluation) and chapter 11 (Probabilistic IR) are the classical framing behind the two-stage pipeline.

## Cross-encoder rerankers

- Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — introduces the bi-encoder shape whose contrast defines the cross-encoder.
- Sentence Transformers: [*Cross-Encoder documentation*](https://www.sbert.net/examples/applications/cross-encoder/README.html) — the canonical Python library reference for open-weight cross-encoders.
- Sentence Transformers: [*Pretrained Cross-Encoders*](https://www.sbert.net/docs/pretrained-cross-encoders.html) — comparison numbers for the `ms-marco-MiniLM-*` family on BEIR + MS MARCO.
- FlagEmbedding (BGE): [*Reranker documentation*](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker) — the BAAI reranker library reference. Includes `bge-reranker-v2-m3`, `bge-reranker-large`, and the multilingual variants.
- Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding.* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) — the BGE embedding-and-reranker family paper.
- Khattab, O. and Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — the late-interaction reranker; a middle point between bi- and cross-encoders.
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2021). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) — the ColBERT follow-up with quantized index and compression.
- Rosa, G., Bonifacio, L., Jeronymo, V., Abonizio, H., Fadaee, M., Lotufo, R., and Nogueira, R. (2022). *In Defense of Cross-Encoders for Zero-Shot Retrieval.* [arXiv:2212.06121](https://arxiv.org/abs/2212.06121) — reports N-sweep behaviour and cross-domain generalisation of cross-encoder rerankers.

## Hosted rerank APIs

- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank-overview) — Cohere's own overview and API reference for `rerank-english`, `rerank-multilingual`, and the `rerank-v3.5` family.
- [Voyage AI Reranker documentation](https://docs.voyageai.com/docs/reranker) — Voyage's `rerank-2` model docs and pricing.
- [Jina Reranker documentation](https://jina.ai/reranker/) — Jina's hosted rerank offering (`jina-reranker-v2-base-multilingual`, `jina-reranker-v1-turbo-en`) plus the open-weight variants.
- Mixedbread AI: [mxbai-rerank model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) — an open-weight cross-encoder often offered as a hosted API alternative.
- Amazon Bedrock: [*Rerank models in Bedrock*](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html) — AWS's hosted-in-your-VPC option; the enterprise / compliance-driven path.

## LTR (Learning-to-Rank) rerankers

- Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval.* Foundations and Trends in Information Retrieval. [nowpublishers.com](https://www.nowpublishers.com/article/Details/INR-016) — the canonical LTR survey; still the reference for the pointwise / pairwise / listwise taxonomy.
- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research technical report. [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) — the LambdaMART reference paper; underpins the LightGBM and XGBoost LTR objectives.
- LightGBM: [*Learning to Rank* documentation](https://lightgbm.readthedocs.io/en/latest/Features.html#lambdarank) — `lambdarank` objective and query-grouping API.
- XGBoost: [*Learning to Rank* documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) — the equivalent for XGBoost with `rank:ndcg` / `rank:pairwise` objectives.
- Bruch, S., Nardini, F. M., Perego, R., and Tonellotto, N. (2024). *Efficient and Effective Tree-Based and Neural Learning to Rank.* Foundations and Trends in Information Retrieval. [arXiv:2305.08869](https://arxiv.org/abs/2305.08869) — a modern monograph covering tree-based LTR and its relationship to neural rerankers.
- Joachims, T., Swaminathan, A., and Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased Feedback.* WSDM. [arXiv:1608.04468](https://arxiv.org/abs/1608.04468) — the reference on debiasing click-based training data; important if you build an LTR reranker over click logs.

## LLM-as-reranker

- Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., and Ren, Z. (2023). *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.* EMNLP. [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) — the RankGPT paper; the reference for listwise-LLM reranking.
- Pradeep, R., Sharifymoghaddam, S., and Lin, J. (2023). *RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.* [arXiv:2309.15088](https://arxiv.org/abs/2309.15088) — the open-weight counterpart to RankGPT.
- Pradeep, R., Sharifymoghaddam, S., and Lin, J. (2023). *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!* [arXiv:2312.02724](https://arxiv.org/abs/2312.02724) — RankZephyr; a stronger open-weight LLM-reranker.
- Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., Shen, J., Liu, T., Liu, J., Metzler, D., Wang, X., and Bendersky, M. (2024). *Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.* NAACL. [arXiv:2306.17563](https://arxiv.org/abs/2306.17563) — pairwise-LLM reranking prompt design; a more compute-efficient alternative to full listwise.

## Funnel tuning (N-to-K) and evaluation metrics

- Voorhees, E. M. and Harman, D. K., eds. (2005). *TREC: Experiment and Evaluation in Information Retrieval.* MIT Press. Canonical reference for nDCG, MAP, MRR, and the metric conventions the chapters use.
- Järvelin, K. and Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM Transactions on Information Systems 20(4). [ACM DL](https://dl.acm.org/doi/10.1145/582415.582418) — the origin paper for nDCG; useful for understanding what the K in "nDCG@K" is doing to the metric.
- Sakai, T. (2014). *Statistical Reform in Information Retrieval?* SIGIR Forum 48(1). [ACM DL](https://dl.acm.org/doi/10.1145/2641383.2641394) — paired significance testing and eval-set-size guidance for the funnel-sweep exercise.
- TREC eval scripts: [github.com/usnistgov/trec_eval](https://github.com/usnistgov/trec_eval) — the canonical recall@k / MRR / nDCG / MAP evaluator.
- Ranx: [github.com/AmenRa/ranx](https://github.com/AmenRa/ranx) — a Python IR evaluation library with per-query bootstrap CIs; convenient for the plots in exercise 02.

## Contextual compression

- LangChain: [*Contextual Compression*](https://python.langchain.com/docs/how_to/contextual_compression/) — the framework's own guide including `LLMChainExtractor`, `EmbeddingsFilter`, `LLMChainFilter`, and `DocumentCompressorPipeline`.
- LlamaIndex: [*Node Postprocessors*](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/) — the equivalent postprocessor abstraction, including `SentenceEmbeddingOptimizer`, `LLMRerank`, and the metadata-preserving `NodeWithScore` contract.
- LlamaIndex: [*Query Pipelines*](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/) — the compose-multiple-postprocessors reference.
- Xu, Y., Lai, V., and Callison-Burch, C. (2023). *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.* [arXiv:2310.04408](https://arxiv.org/abs/2310.04408) — a research reference on RAG-side compression, with an extractive-vs-abstractive comparison.
- Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., and Qiu, L. (2023). *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP. [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) — a specific compression technique for LLM prompts; applies to retrieved-context compression.
- Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. (2023). *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) — the long-context extension; relevant to the K=20+ regime.

## Context assembly — dedup, ordering, citation anchors

- Carbonell, J. and Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. [CMU PDF](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) — the origin paper for Maximal Marginal Relevance; the reference for the diversity-vs-relevance trade-off.
- LangChain: [*Retrieval + Answer Question with Sources*](https://python.langchain.com/docs/how_to/qa_sources/) — the framework's own guide on citation-anchor patterns for source-grounded answers.
- LlamaIndex: [*Citation Query Engine*](https://docs.llamaindex.ai/en/stable/examples/query_engine/citation_query_engine/) — the framework's citation-first query engine reference.
- OpenAI: [`tiktoken`](https://github.com/openai/tiktoken) — the tokenizer for OpenAI models. `cl100k_base` for GPT-3.5/4, `o200k_base` for GPT-4o.
- Anthropic: [*Count Tokens API*](https://docs.claude.com/en/api/messages-count-tokens) — the tokenizer for Claude models; use for token-budget arithmetic.
- Hugging Face Transformers: [`AutoTokenizer`](https://huggingface.co/docs/transformers/main_classes/tokenizer) — the tokenizer for open-weight models.

## Long context and lost-in-the-middle

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the reference for the U-shape attention finding across long-context LLMs. Motivates the first-and-last ordering (chapter 7), the K peak (chapter 5), and the "reranking still matters" argument (chapter 8).
- Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) — a broader long-context benchmark suite; shows most models' *effective* context length is meaningfully shorter than the advertised one.
- Kamradt, G. (2023). *Needle in a Haystack — Pressure Testing LLMs.* [github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the reproducible test harness that popularised the format.
- Anthropic: [*Introducing Claude 3.5 Sonnet*](https://www.anthropic.com/news/claude-3-5-sonnet) — vendor-reported long-context needle-in-a-haystack numbers; useful for calibration.
- OpenAI: [*Models overview*](https://platform.openai.com/docs/models) — vendor-side long-context claims for the GPT-4 family; verify current numbers directly.
- Google: [*Gemini API — models*](https://ai.google.dev/gemini-api/docs/models/gemini) — Gemini 1.5 Pro / Flash context-window and long-context claims.
- Zhu, D., Yang, N., Wang, L., Song, Y., Wu, W., Wei, F., and Li, S. (2024). *LongEmbed: Extending Embedding Models for Long Context Retrieval.* [arXiv:2404.12096](https://arxiv.org/abs/2404.12096) — the embedder-side of the long-context push; relevant to whole-document retrieval shapes.

## Evaluation — RAG triad and response-granularity metrics

- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the RAG-triad (faithfulness / answer relevance / context precision) methodology the exercises use.
- RAGAS: [docs.ragas.io](https://docs.ragas.io/) — the reference implementation of the RAG triad.
- TruLens: [*RAG Triad evaluation guide*](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/) — the peer-track eval methodology reference used by `applied-ai-engineer`.
- DeepEval: [docs.confident-ai.com](https://docs.confident-ai.com/) — an alternative response-granularity eval framework.
- Saad-Falcon, J., Khattab, O., Potts, C., and Zaharia, M. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) — an alternative to RAGAS with a trained-judge model.

## Framework and infrastructure references

- LangChain: [*Retrievers overview*](https://python.langchain.com/docs/concepts/retrievers/) — the framework's retriever abstraction that all its rerankers and compressors compose over.
- LlamaIndex: [*Response modes*](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes/) — how the framework assembles retrieved context for the generator (`compact`, `refine`, `tree_summarize`).
- Haystack 2: [*Rankers*](https://docs.haystack.deepset.ai/docs/rankers) — Haystack's reranker components (`TransformersSimilarityRanker`, `LostInTheMiddleRanker` — a built-in first-and-last reorderer).
- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — the shape of the four contract clauses in chapter 9.

## The peer track and adjacent modules in this track

- [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) — the peer track that consumes `AssembledContext`; chapter 9 draws the boundary.
- Mod-106 in this track (Hybrid Retrieval, Fusion, Query Rewriting) — the retriever whose output this module reranks and assembles.
- Mod-108 in this track (Retrieval-Specific Prompting) — the immediate downstream module that turns `AssembledContext` into an LLM prompt.
- Mod-109 in this track (Retrieval-Quality Evaluation) — the eval methodology behind every "measure on your data" instruction in this module's exercises.
- Mod-111 in this track (Retrieval MLOps) — the versioning, deployment, and drift-monitoring for the reranker + compressor + assembler stack.

## Reading order suggestion

If you are new to reranking specifically, read in this order: Nogueira & Cho 2019 (why cross-encoders) → Liu et al. 2023 (lost-in-the-middle — the reason the reranker is not obviated by long context) → the Sentence Transformers cross-encoder docs (the concrete library) → Cohere Rerank docs (the concrete API) → Carbonell & Goldstein 1998 (MMR — the diversity primitive) → LangChain's contextual-compression guide → RAGAS paper (the eval framework you will measure everything with). Keep Manning / Raghavan / Schütze as the classical reference you keep returning to.
