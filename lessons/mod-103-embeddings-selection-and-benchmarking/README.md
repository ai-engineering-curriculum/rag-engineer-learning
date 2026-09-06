# mod-103 — Embeddings: Selection, Benchmarking, and Domain Fit

**Track:** RAG & Retrieval Engineer (role level 30, AI Engineering family)
**Estimated effort:** 16 hours
**Position in track:** third module. Sits between mod-102 (chunked corpus) and mod-104 (vector index) — it decides *which embedding model* turns the chunks into vectors.

## What this module does

At level 20 "the embedding model" is a single dropdown. This module makes it a defensible choice. By the end you can (a) name the four neural-retrieval families (bi-encoder, cross-encoder, ColBERT / late-interaction, SPLADE / sparse-neural) and place each in the retrieve-then-rerank pipeline, (b) read MTEB and BEIR leaderboards critically and translate them into a shortlist for *your* corpus, (c) survey the current hosted embedding APIs (OpenAI, Cohere, Voyage, Google, Bedrock-hosted variants) against the current self-hosted open families (BGE, E5, Nomic, GTE, Snowflake Arctic, Jina, mxbai), (d) design and run a domain-fit benchmark on your own data with paired significance testing, (e) reason about the storage / latency / quality trade-off using MRL truncation and int8 / binary quantization, and (f) recognize when to hand off to the peer `fine-tuning-engineer` track and produce the brief that makes the handoff productive.

## Boundary with adjacent modules

- **Below (level 20).** `llm-application-developer` mod-006 introduces "call an embeddings API, put the vector in pgvector, do cosine top-K." This module replaces the single dropdown with a defensible model-selection decision.
- **Around (this track).** mod-102 owns the chunker — its output is this module's input. mod-104 owns the ANN index — this module's output is its input. mod-105 owns the Lucene / SPLADE serving stack; this module names SPLADE as a family but does not tour the Lucene ops. mod-106 owns hybrid retrieval fusion; this module gives you the pieces that get fused. mod-107 owns rerankers; this module names the cross-encoder / ColBERT reranker families but the reranker slot lives there. mod-109 owns judgment-set methodology and paired significance testing at depth; this module runs a domain-fit benchmark scoped to the *embedding* choice.
- **Peer (level 30).** `fine-tuning-engineer` owns the *training* craft — contrastive losses, hard-negative mining, LoRA / PEFT, TSDAE / SimCSE / GPL, distillation. This module selects and evaluates already-trained models; when off-the-shelf is not enough, chapter 9 is the handoff shape.

## Learning objectives

- Distinguish bi-encoder, cross-encoder, ColBERT-style late-interaction, and sparse-neural (SPLADE) embedding families — when each shape fits, and the retrieval / reranking role each plays.
- Read the MTEB (Massive Text Embedding Benchmark) leaderboard and the BEIR benchmark critically — understand what tasks measure what, how leaderboard scores translate (or fail to translate) to your corpus, and how to run MTEB locally on a custom task.
- Compare hosted embedding APIs (OpenAI text-embedding-3, Cohere Embed v3, Voyage voyage-3, Google text-embedding-005, Anthropic-recommended Voyage via Amazon Bedrock) against self-hosted open models (BGE, E5, Nomic, GTE, Snowflake Arctic Embed, Alibaba GTE-Qwen, Jina, mxbai) on latency, cost, dimensionality, and quality-per-dollar.
- Design a domain-fit benchmark for a specific corpus — pin the eval set, evaluate ≥5 embedding models under identical chunker + retriever conditions, and rank on Recall@k / MRR / nDCG@10.
- Reason about embedding dimensionality, quantization (int8, binary, Matryoshka), and the storage / latency / quality trade-off; understand how MRL (Matryoshka Representation Learning) enables adaptive dimensions.
- Cite the boundary to `fine-tuning-engineer` (peer, level 30) for embedding SFT / LoRA / PEFT — this module selects and evaluates; the peer track owns the training craft (sentence-transformers TSDAE, MultipleNegativesRankingLoss, GISTEmbedLoss).

## Prerequisites

- The mod-101 toolchain exercise (`exercise-05-local-toolchain-bootstrap-and-smoke-test`) is green: pgvector reachable, Elasticsearch/OpenSearch reachable, `sentence-transformers`, `mteb`, and `ir_measures` importable.
- mod-102 complete — you have a chunked corpus with a pinned `chunker.yaml` and (ideally) a small hand-labeled qrels set from `exercise-03`.
- Comfortable Python (`numpy`, `pandas`, typed dataclasses, packaging) and Docker.
- Familiarity with cosine similarity and the difference between a dense vector and an inverted index at the level of mod-005 (level 20) or mod-101 chapter 2 (level 30).

## Chapters

Read chapters in order. Chapters 1–3 build the shape + benchmark-literacy foundation. Chapters 4–6 survey the models and put them on the same cost / latency axes. Chapters 7–8 own the on-your-data decision + storage engineering. Chapter 9 is the handoff to the peer training track.

| # | Chapter | Focus | Paired exercise |
|---|---|---|---|
| 01 | [Embedding Family Map: Bi-Encoder, Cross-Encoder, ColBERT, SPLADE](01-embedding-family-map.md) | The four architectural shapes and which slot each fills | [exercise-01 — bi/cross/ColBERT/SPLADE family map](exercises/exercise-01-bi-cross-colbert-splade-family-map.md) |
| 02 | [Reading MTEB and BEIR Critically](02-mteb-and-beir-critical-read.md) | Task taxonomies, subset-reporting traps, prompt sensitivity | (paired with chapter 3's exercise) |
| 03 | [Running MTEB Locally on a Custom Retrieval Task](03-running-mteb-locally-on-custom-task.md) | The runner, adapters, `AbsTaskRetrieval` subclass | [exercise-02 — MTEB / BEIR critical read + local task run](exercises/exercise-02-mteb-and-beir-critical-read-and-local-task-run.md) |
| 04 | [Hosted Embedding APIs: OpenAI, Cohere, Voyage, Google, Bedrock](04-hosted-embedding-apis-survey.md) | Provider surface, `input_type` / `task_type` routing, compliance | (paired with chapter 6's exercise) |
| 05 | [Self-Hosted Open Embeddings: BGE, E5, Nomic, GTE, Arctic, Jina](05-self-hosted-open-embeddings-survey.md) | Open families, prompt-sensitive models, licensing traps, TEI / vLLM serving | (paired with chapter 6's exercise) |
| 06 | [Latency, Cost, Dimensionality, and Quality-per-Dollar](06-latency-cost-quality-per-dollar.md) | The four cost surfaces, the quality-floor-first ranking rule | [exercise-03 — hosted vs. self-hosted sweep](exercises/exercise-03-hosted-vs-self-hosted-embedding-cost-latency-quality-sweep.md) |
| 07 | [Designing a Domain-Fit Benchmark](07-designing-a-domain-fit-benchmark.md) | Five design decisions; the sweep + significance test + recommendation shape | [exercise-04 — domain-fit benchmark on your own corpus](exercises/exercise-04-domain-fit-benchmark-on-your-own-corpus.md) |
| 08 | [Dimensionality, Quantization, and Matryoshka Representation Learning](08-dimensionality-quantization-matryoshka.md) | MRL truncation, int8 scalar quantization, binary + rescore | [exercise-05 — quantization and Matryoshka storage/latency trade-off](exercises/exercise-05-quantization-and-matryoshka-storage-latency-tradeoff.md) |
| 09 | [Boundary to `fine-tuning-engineer`: When to Hand Off Embedding Training](09-boundary-to-fine-tuning-engineer.md) | Failure modes, three fine-tuning shapes, the handoff artifact | [exercise-06 — fine-tuning brief to the peer track](exercises/exercise-06-hand-off-fine-tuning-brief-to-fine-tuning-engineer-track.md) |

## How to work through it

1. Read chapter 1, then do exercise-01 (touch one representative of each of the four families) — you will refer back to this muscle memory for the rest of the module.
2. Read chapters 2 and 3 together, then do exercise-02 (MTEB harness + custom task). You will reuse the harness for exercises 03 and 04.
3. Read chapters 4, 5, 6 as one survey arc, then do exercise-03 (hosted vs. self-hosted sweep). You will now know what the tables in front of you at an architecture review look like.
4. Read chapter 7, then do exercise-04 — the capstone. This is the artifact every subsequent module in the track assumes exists.
5. Read chapter 8, then do exercise-05 (quantization + Matryoshka frontier). Apply the winning operating point to the model you selected in exercise-04.
6. Read chapter 9, then do exercise-06 — write the fine-tuning brief. Even if you would not actually pursue fine-tuning on your corpus, the artifact is the same shape.

## Structure

- `01-…md` … `09-…md`: lecture chapters.
- `exercises/`: six hands-on prompts, one per major objective.
- `labs/`: long-form hands-on labs (see `labs/README.md`).
- `quizzes/`: knowledge checks.
- `resources.md`: primary papers, official docs, and vendor pages the chapters cite.

## Where to go next

Once exercise-04's ranking table and exercise-05's operating point are committed, move to [`mod-104-vector-databases-and-ann-tuning`](../mod-104-vector-databases-and-ann-tuning/README.md). The winning `(embedding_model, dim, quant)` tuple from this module is the input to mod-104's ANN parameter sweep — HNSW `M` / `efConstruction` / `efSearch`, IVF-PQ variants, and the recall-vs-latency curve you tune against the same benchmark you froze here.
