# mod-103-embeddings-selection-and-benchmarking: Embeddings: Selection, Benchmarking, and Domain Fit

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 16 hours

## Learning objectives

- Distinguish bi-encoder, cross-encoder, ColBERT-style late-interaction, and sparse-neural (SPLADE) embedding families — when each shape fits, and the retrieval/reranking role each plays
- Read the MTEB (Massive Text Embedding Benchmark) leaderboard and the BEIR benchmark critically — understand what tasks measure what, how leaderboard scores translate (or fail to translate) to your corpus, and how to run MTEB locally on a custom task
- Compare hosted embedding APIs (OpenAI text-embedding-3, Cohere Embed v3, Voyage voyage-3, Google text-embedding-005, Anthropic voyage-* via Amazon Bedrock) against self-hosted open models (BGE, E5, Nomic, GTE, Snowflake Arctic Embed, Alibaba GTE-Qwen, Jina) on latency, cost, dimensionality, and quality-per-dollar
- Design a domain-fit benchmark for a specific corpus — pin the eval set, evaluate ≥5 embedding models under identical chunker + retriever conditions, and rank on recall@k / MRR
- Reason about embedding dimensionality, quantization (int8, binary, Matryoshka), and the storage/latency/quality trade-off; understand how MRL (Matryoshka Representation Learning) enables adaptive dimensions
- Cite the boundary to `fine-tuning-engineer` (peer, level 30) for embedding SFT / LoRA / PEFT — this module selects and evaluates; the peer track owns the training craft (sentence-transformers TSDAE, MultipleNegativesRankingLoss, GISTEmbedLoss)

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
