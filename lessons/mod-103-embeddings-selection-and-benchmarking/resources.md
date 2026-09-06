# Resources — mod-103: Embeddings: Selection, Benchmarking, and Domain Fit

Primary sources for the chapters and exercises. Links go to official docs, source repos, or canonical papers — not to third-party summary blogs, except where a vendor's own engineering post is the canonical practitioner reference for a technique (e.g., Cohere's binary-embeddings blog).

## Embedding family shapes (chapter 1)

- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — the canonical bi-encoder retrieval paper.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — the training-recipe canon; `sentence-transformers` implements this line.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the cross-encoder rerank shape.
- Nogueira, R., Jiang, Z., & Lin, J. (2020). *Document Ranking with a Pretrained Sequence-to-Sequence Model.* [arXiv:2003.06713](https://arxiv.org/abs/2003.06713) — monoT5, the seq2seq reranker.
- Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* [arXiv:2004.12832](https://arxiv.org/abs/2004.12832).
- Santhanam, K., et al. (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* [arXiv:2112.01488](https://arxiv.org/abs/2112.01488).
- Santhanam, K., et al. (2022). *PLAID: An Efficient Engine for Late Interaction Retrieval.* [arXiv:2205.09707](https://arxiv.org/abs/2205.09707).
- Formal, T., Piwowarski, B., & Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* [arXiv:2107.05720](https://arxiv.org/abs/2107.05720). Follow-ups: *SPLADE v2* [arXiv:2109.10086](https://arxiv.org/abs/2109.10086); *SPLADE++ / Hard Negative Sampling* [arXiv:2205.04733](https://arxiv.org/abs/2205.04733).
- Sturua, S., et al. (2024). *jina-embeddings-v3: Multilingual Embeddings With Task LoRA.* [arXiv:2409.10173](https://arxiv.org/abs/2409.10173) — current vendor-view discussion of dense vs. late-interaction trade-offs.
- `sentence-transformers` — [*Pretrained Models*](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html), [*Cross-Encoders*](https://sbert.net/docs/cross_encoder/pretrained_models.html), [*Information Retrieval*](https://www.sbert.net/examples/applications/information-retrieval/README.html).
- RAGatouille, [`AnswerDotAI/RAGatouille`](https://github.com/AnswerDotAI/RAGatouille) — practitioner wrapper around ColBERT with PLAID indexing.
- Stanford ColBERT, [`stanford-futuredata/ColBERT`](https://github.com/stanford-futuredata/ColBERT) — reference ColBERTv2 implementation.

## Benchmarks — MTEB and BEIR (chapters 2, 3)

- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316).
- Muennighoff, N., et al. (2024). *MMTEB: Massive Multilingual Text Embedding Benchmark.* [arXiv:2502.13595](https://arxiv.org/abs/2502.13595).
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* [arXiv:2104.08663](https://arxiv.org/abs/2104.08663). NeurIPS Datasets and Benchmarks.
- MTEB leaderboard, [Hugging Face Spaces](https://huggingface.co/spaces/mteb/leaderboard) — the live leaderboard; sort by Retrieval nDCG@10 for this track.
- MTEB repo, [`embeddings-benchmark/mteb`](https://github.com/embeddings-benchmark/mteb). Browse [`mteb/models/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models) for the reference model-adapter shape and [`mteb/tasks/Retrieval/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks/Retrieval) for the built-in tasks.
- BEIR repo, [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — the reference harness, corpus / qrels download scripts, canonical BM25 baseline scores.
- Pyserini BEIR reproduction docs, [castorini.github.io/pyserini/2cr/beir](https://castorini.github.io/pyserini/2cr/beir.html) — a widely-cited BM25 baseline reference for BEIR datasets.
- `ir_measures` — [docs](https://ir-measur.es/en/latest/), [*Getting started*](https://ir-measur.es/en/latest/getting-started.html) — alternative retrieval-metric library to MTEB's built-ins.

## Hosted embedding APIs (chapter 4)

- OpenAI, [*Embeddings guide*](https://platform.openai.com/docs/guides/embeddings), [*Embeddings API reference*](https://platform.openai.com/docs/api-reference/embeddings), [*New embedding models and API updates*](https://openai.com/blog/new-embedding-models-and-api-updates), [*Rate limits*](https://platform.openai.com/docs/guides/rate-limits), [*Pricing*](https://openai.com/api/pricing/).
- Cohere, [*Embed API reference*](https://docs.cohere.com/reference/embed), [*Embeddings guide*](https://docs.cohere.com/docs/embeddings), [*Introducing Embed v3*](https://cohere.com/blog/introducing-embed-v3), [*Pricing*](https://cohere.com/pricing).
- Voyage AI, [*Embeddings docs*](https://docs.voyageai.com/docs/embeddings), [*voyage-3 announcement*](https://blog.voyageai.com/2024/09/18/voyage-3/), [*Pricing*](https://www.voyageai.com/pricing).
- Anthropic, [*Embeddings (recommend Voyage AI)*](https://docs.anthropic.com/en/docs/build-with-claude/embeddings), [*Voyage AI embeddings on Amazon Bedrock*](https://www.anthropic.com/news/voyage-ai-embeddings-on-amazon-bedrock).
- Google Cloud, [*Get text embeddings*](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings), [*Text embeddings API — task types*](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api), [*Vertex AI pricing*](https://cloud.google.com/vertex-ai/generative-ai/pricing).
- AWS Bedrock, [*Amazon Titan Text Embeddings*](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html), [*Supported models*](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html), [*Bedrock pricing*](https://aws.amazon.com/bedrock/pricing/).

## Self-hosted open embedders (chapter 5)

- Xiao, S., et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding.* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) — the BGE line.
- Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
- Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) — E5.
- Wang, L., et al. (2024). *Improving Text Embeddings with Large Language Models.* [arXiv:2401.00368](https://arxiv.org/abs/2401.00368) — e5-mistral-7b-instruct.
- Nussbaum, Z., et al. (2024). *Nomic Embed: Training a Reproducible Long Context Text Embedder.* [arXiv:2402.01613](https://arxiv.org/abs/2402.01613).
- Li, Z., et al. (2023). *Towards General Text Embeddings with Multi-stage Contrastive Learning.* [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) — the GTE line.
- Li, C., et al. (2024). *Making Text Embedders Few-Shot Learners.* [arXiv:2409.15700](https://arxiv.org/abs/2409.15700) — GTE-Qwen instruct.
- Merrick, L., et al. (2024). *Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models.* [arXiv:2405.05374](https://arxiv.org/abs/2405.05374). Snowflake announcement: [*Introducing Snowflake Arctic Embed*](https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/).
- Mixedbread AI, [*Introducing mxbai-embed-large-v1*](https://www.mixedbread.ai/blog/mxbai-embed-large-v1).
- Hugging Face model cards — [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2), [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large), [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5), [`Alibaba-NLP/gte-Qwen2-7B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct), [`Snowflake/snowflake-arctic-embed-l`](https://huggingface.co/Snowflake/snowflake-arctic-embed-l), [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3), [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
- SPLADE model cards — [`naver/splade-v3`](https://huggingface.co/naver/splade-v3), [`naver/splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil), [`prithivida/Splade_PP_en_v2`](https://huggingface.co/prithivida/Splade_PP_en_v2).
- Elastic, [*ELSER — Elastic Learned Sparse EncodeR*](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html) — Elastic's shipped SPLADE-family model.

## Serving stacks

- Hugging Face, [*Text Embeddings Inference (TEI)*](https://github.com/huggingface/text-embeddings-inference) — reference OSS embedding server; supports BGE, E5, GTE, Nomic, Arctic, Jina, mxbai. Model support and throughput reference: [`text-embeddings-inference` README](https://github.com/huggingface/text-embeddings-inference#supported-models).
- vLLM, [*Pooling / Embedding Models*](https://docs.vllm.ai/en/latest/models/pooling_models.html) — for LLM-backbone embedders (`gte-Qwen2-*-instruct`, `e5-mistral-7b-instruct`).
- `sentence-transformers`, [*Docs home*](https://www.sbert.net/) — in-process embedding for offline jobs and simple services.

## Cost, latency, and quality methodology (chapter 6)

- Provider pricing pages (bookmark, do not memorize — these move): OpenAI, Cohere, Voyage, Vertex AI, Bedrock (linked above).
- ANN Benchmarks, [`erikbern/ann-benchmarks`](https://github.com/erikbern/ann-benchmarks) — reference recall-vs-QPS methodology, extended in mod-104.
- Cormack, G. V., & Grossman, M. R. (2017). *Navigating Imprecision in Relevance Assessments on the Road to Total Recall.* SIGIR — one of the citations for why sub-noise quality differences should not drive cost decisions.

## Domain-fit benchmark design and significance testing (chapter 7)

- Thakur, N., et al. (2021). *BEIR* [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — benchmark design your domain-fit run mirrors at 1/50th scale.
- Muennighoff, N., et al. (2022). *MTEB* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — task metadata and per-task scoring shape (chapter 3 reuses the harness).
- Sakai, T. (2014). *Metrics, Statistics, Tests.* PROMISE '14. [PDF via TETSUYASAKAI.info](https://www.f.waseda.jp/tetsuya/pubs/promise2014-sakai.pdf) — canonical reference for paired significance testing on retrieval metrics.
- Craswell, N., et al. (2021). *Overview of the TREC 2020 deep learning track.* [arXiv:2102.07662](https://arxiv.org/abs/2102.07662) — public "same corpus + fixed qrels + rotating models" shape reference.
- Anthropic engineering, [*Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — public engineering case study of a domain-fit-shape chunk + embedding + reranker ablation.
- SciPy, [`scipy.stats.wilcoxon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) — non-parametric paired test used in the exercise-04 tie-breaker.

## Dimensionality, quantization, Matryoshka (chapter 8)

- Kusupati, A., et al. (2022). *Matryoshka Representation Learning.* [arXiv:2205.13147](https://arxiv.org/abs/2205.13147). NeurIPS.
- OpenAI, [*New embedding models and API updates*](https://openai.com/blog/new-embedding-models-and-api-updates) — MRL-native `text-embedding-3-*` with the `dimensions` parameter.
- Google Cloud, [*Text embeddings API — output dimensionality*](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_parameters).
- Cohere, [*Int8 & Binary Embeddings — Store 32x More Data*](https://cohere.com/blog/int8-binary-embeddings) — practitioner reference on binary embeddings + rescore.
- pgvector, [`pgvector/pgvector` README](https://github.com/pgvector/pgvector) — `halfvec`, `vector`, `bit`, `sparsevec` types and indexes.
- Qdrant, [*Quantization docs*](https://qdrant.tech/documentation/guides/quantization/) — scalar, product, and binary quantization with rescore.
- Weaviate, [*Vector index compression*](https://weaviate.io/developers/weaviate/configuration/compression) — PQ, SQ, and binary compression.
- Milvus, [*Vector index types*](https://milvus.io/docs/index.md) — FLAT, IVF variants, HNSW, and binary indexes.
- FAISS, [*Indexes*](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [`ScalarQuantizer` reference](https://faiss.ai/cpp_api/struct/structfaiss_1_1ScalarQuantizer.html) — scalar-quantizer and binary-index implementations you can benchmark locally.

## Fine-tuning handoff (chapter 9)

- `fine-tuning-engineer-learning` peer track — [`ml-engineering-curriculum/fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) (level 30, ML Engineering family).
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
- `sentence-transformers`, [*Training Overview*](https://sbert.net/docs/sentence_transformer/training_overview.html), [*Losses*](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) — including [`MultipleNegativesRankingLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) and [`GISTEmbedLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss).
- Wang, K., Reimers, N., & Gurevych, I. (2021). *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning.* [arXiv:2104.06979](https://arxiv.org/abs/2104.06979).
- Gao, T., Yao, X., & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* [arXiv:2104.08821](https://arxiv.org/abs/2104.08821).
- Wang, K., et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.* [arXiv:2112.07577](https://arxiv.org/abs/2112.07577).
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Hugging Face, [`peft`](https://github.com/huggingface/peft) — the PEFT / LoRA library the peer track uses.

## Track-internal cross-references

- [`../mod-101-retrieval-engineering-foundations/README.md`](../mod-101-retrieval-engineering-foundations/README.md) — the seven-stage pipeline this module drills into.
- [`../mod-102-document-ingestion-and-chunking/README.md`](../mod-102-document-ingestion-and-chunking/README.md) — the chunked corpus this module embeds.
- [`../mod-104-vector-databases-and-ann-tuning/README.md`](../mod-104-vector-databases-and-ann-tuning/README.md) — where the chosen `(model, dim, quant)` tuple is tuned into an ANN index.
- [`../mod-105-lucene-family-and-ltr/README.md`](../mod-105-lucene-family-and-ltr/README.md) — where SPLADE / ELSER get their production serving story.
- [`../mod-106-hybrid-retrieval-fusion-query-rewriting/README.md`](../mod-106-hybrid-retrieval-fusion-query-rewriting/README.md) — where BM25 ⊕ dense fusion is engineered.
- [`../mod-107-reranking-and-context-assembly/README.md`](../mod-107-reranking-and-context-assembly/README.md) — where the cross-encoder / ColBERT reranker slot is owned.
- [`../mod-109-retrieval-quality-evaluation/README.md`](../mod-109-retrieval-quality-evaluation/README.md) — deeper judgment-set methodology and paired significance testing.
- [`../mod-111-retrieval-mlops-versioning-cicd-drift/README.md`](../mod-111-retrieval-mlops-versioning-cicd-drift/README.md) — where `index_version_key` (with `embedding_model_id`, `dim`, `quant`) is owned end-to-end.
- [`../../CURRICULUM.md`](../../CURRICULUM.md) — track ownership rule and full module map.
- [`../../PREREQUISITES.md`](../../PREREQUISITES.md) — assumed entry skills.
