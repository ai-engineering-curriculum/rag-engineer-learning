# Chapter 5 — Self-Hosted Open Embeddings: BGE, E5, Nomic, GTE, Arctic, Jina

## Why this chapter exists

The reason to self-host is not saving money (though on a large corpus, it is). It is **control**: over the model version (a hosted vendor can deprecate; a local weights file is yours forever), over the data path (nothing leaves your VPC), over the tokenizer (no `input_type` surprises you can't inspect), and over the ops surface (you know exactly what happens on the GPU because you wrote the server).

The trade you make: someone (you) has to run the inference. A 1B-parameter embedder on a busy GPU is a real service that needs batching, autoscaling, health checks, warm cache, tokenizer version-pinning, and a memory / VRAM plan. `sentence-transformers` will run in-process for offline embedding jobs; text-embeddings-inference (TEI) or vLLM is what you serve behind a real product.

This chapter is the tour of the open-model families you will actually be pointed at, why each exists, and how they differ operationally. Pair it with chapter 4 (hosted) and chapter 6 (comparing the two on the same axes) to justify a choice; pair it with chapter 7 (domain-fit benchmark) to actually pick one.

## The families

| Family | Provenance | Sizes commonly used | Distinguishing shape |
|---|---|---|---|
| **BGE** (BAAI General Embedding) | Beijing Academy of AI | `bge-{small,base,large}-en-v1.5`, `bge-m3` | Strong general-purpose; `bge-m3` is dense + multi-vector + sparse in one model |
| **E5** | Microsoft Research | `e5-{small,base,large}-v2`, `multilingual-e5-*`, `e5-mistral-7b-instruct` | Weakly-supervised contrastive pretraining; needs `query: ` / `passage: ` prefixes |
| **Nomic** | Nomic AI | `nomic-embed-text-v1.5` | Long context (8k), open training data + weights + config; MRL trained |
| **GTE** | Alibaba DAMO | `gte-{small,base,large}-en-v1.5`, `gte-Qwen2-7B-instruct`, `gte-Qwen2-1.5B-instruct` | Strong on MTEB retrieval; large LLM-backbone variants for max quality |
| **Snowflake Arctic Embed** | Snowflake | `snowflake-arctic-embed-{xs,s,m,l,l-v2.0}` | Retrieval-focused training; competitive at small sizes |
| **Jina** | Jina AI | `jina-embeddings-v3`, `jina-colbert-v2`, `jina-embeddings-v2-base-code` | Long context (8k), multilingual, code-specialized variants |
| **mxbai** | Mixedbread AI | `mxbai-embed-large-v1` | Strong general-purpose; commonly cited in mid-2024 retrieval work |
| **Instructor / GIST family** | various | `hkunlp/instructor-large`, `GIST-Embedding-v0` | Instruction-tuned; you pass a task instruction alongside the text |

Not exhaustive — but this list covers what you will actually see cited in production retrieval postmortems and on the MTEB retrieval top-30.

## BGE — the BAAI General Embedding line

**Models.** `BAAI/bge-small-en-v1.5` (33M params, 384 dim, 512 tokens), `BAAI/bge-base-en-v1.5` (109M, 768 dim, 512), `BAAI/bge-large-en-v1.5` (335M, 1024 dim, 512), `BAAI/bge-m3` (multilingual, ~570M, 1024 dim + late-interaction + sparse). Paper: Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding* ([arXiv:2309.07597](https://arxiv.org/abs/2309.07597)). BGE-M3: Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216)).

**Why it is the default open recommendation.**

- **Strong on MTEB retrieval at every size.** `bge-large-en-v1.5` has been a top-quality open embedder since 2023 and the whole family is well-documented.
- **BGE-M3 is unusual.** In one forward pass it produces a dense vector, a set of per-token late-interaction vectors (ColBERT-shaped), and a sparse-neural (SPLADE-shaped) vector. You get chapter 1's three retrieval families from one model at index time — pick whichever you want to serve, or fuse them.
- **BAAI ships reranker weights (`bge-reranker-*`) with matched training methodology.** If your stack is `bge-large` retrieval + `bge-reranker-large` rerank, you inherit a well-tested pair.

**Serving.** `SentenceTransformer("BAAI/bge-large-en-v1.5")` for offline embed jobs; TEI (below) with the Hugging Face model ID for a service.

**Prompts.** Retrieval queries recommended prefix: `"Represent this sentence for searching relevant passages: "` (per the model card; check yours). Documents: no prefix. Skip the prefix and quality drops.

**When it fits.** Anything general-purpose English; almost anything Chinese (BGE is Chinese-authored and Chinese is a first-class training language); multilingual when using `bge-m3`; when you want the option of hybrid retrieval from one model (`bge-m3`).

**License.** MIT for the `v1.5` line; check the specific model card.

## E5 — the Microsoft weakly-supervised line

**Models.** `intfloat/e5-{small,base,large}-v2` (~33M / 109M / 335M params, 384 / 768 / 1024 dim, 512 tokens), `intfloat/multilingual-e5-{small,base,large}` (multilingual variants; 100+ languages), `intfloat/e5-mistral-7b-instruct` (7B LLM backbone; 4096 dim; 32k tokens). Paper: Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* ([arXiv:2212.03533](https://arxiv.org/abs/2212.03533)). E5-mistral: Wang et al., *Improving Text Embeddings with Large Language Models* ([arXiv:2401.00368](https://arxiv.org/abs/2401.00368)).

**Why it is interesting.**

- **`query: `/`passage: ` prefix contract.** E5 is the family that made "prompt sensitivity" a topic. Every input must be prefixed at both train and inference time; skipping the prefix drops MTEB retrieval numbers noticeably. Wrappers that forget this are a common bug in the wild.
- **`e5-mistral-7b-instruct` was the first well-cited 7B-parameter embedder that beat everything else on MTEB retrieval at the time of its release.** It is also expensive to serve — one Mistral-7B forward pass per input, per query. Reserve for reranker-slot or highest-quality-tier retrievers, not for general index-scale embedding.
- **Long track record on multilingual.** `multilingual-e5-*` is a strong open multilingual baseline; competes with Cohere multilingual and Google's `text-multilingual-embedding-002`.

**Serving.** `SentenceTransformer` handles the small/base/large; the 7B needs vLLM or a dedicated embedding server with more VRAM planning.

**When it fits.** Multilingual open-source stacks; teams that want a well-documented, reproducible training recipe; when you can pay for a 7B embedder in the reranker slot.

**License.** MIT for the sentence-transformer sizes; check `e5-mistral-7b-instruct` card for its licensing.

## Nomic Embed

**Models.** `nomic-ai/nomic-embed-text-v1.5` (137M params, 768 dim, 8192 tokens; MRL-trained so you can truncate to 512 / 256 / 128 / 64). Paper: Nussbaum et al., *Nomic Embed: Training a Reproducible Long Context Text Embedder* ([arXiv:2402.01613](https://arxiv.org/abs/2402.01613)).

**Why it is interesting.**

- **Open training data.** Nomic published the training data (Nomic Atlas) alongside the weights and the code. If reproducibility is a compliance requirement, this is the family that goes furthest.
- **Long context.** 8192 tokens matches OpenAI on max input length — unusual for a small open embedder (most cap at 512).
- **MRL out of the box.** Truncating to any of 64, 128, 256, 512, or 768 dims retains predictable quality. Compare with OpenAI's `text-embedding-3-*` (chapter 4) for the hosted analogue.

**Serving.** `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)` (Nomic uses a custom modeling class). Task-typed prefix (`"search_query: "` / `"search_document: "` / `"classification: "` / `"clustering: "`) is expected — same shape as Cohere / Voyage's `input_type`.

**When it fits.** Long-chunk stacks (>512 tokens); reproducibility-first shops; when you want MRL from an open weight; when Nomic Atlas fits your data-tooling.

**License.** Apache-2.0.

## GTE — Alibaba DAMO general text embeddings

**Models.** `Alibaba-NLP/gte-{small,base,large}-en-v1.5` (33 / 137 / 434M params, 384 / 768 / 1024 dim, 8192 tokens), `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (1.5B, 1536 dim), `Alibaba-NLP/gte-Qwen2-7B-instruct` (7B, 3584 dim). Papers: Li et al., *Towards General Text Embeddings with Multi-stage Contrastive Learning* ([arXiv:2308.03281](https://arxiv.org/abs/2308.03281)); Li et al., *Making Text Embedders Few-Shot Learners* ([arXiv:2409.15700](https://arxiv.org/abs/2409.15700)) for the GTE-Qwen instruct line.

**Why it is interesting.**

- **`gte-*-en-v1.5` has 8192-token max input** in a small (~400M-param) package — Nomic-shaped long-context without the large-model cost.
- **`gte-Qwen2-7B-instruct` has been a repeated top-of-MTEB entry** since late 2024. It is an LLM-backbone embedder: a 7B decoder-only model with a special "last-token" pooling head. Very good quality; very expensive to serve.
- **Task instructions.** The Qwen-instruct variants take an English instruction alongside the text (e.g., `"Given a web search query, retrieve relevant passages that answer the query"`), fed to the LLM. Getting the instruction right measurably affects the score.

**Serving.** For the small/base/large sizes, `SentenceTransformer` with `trust_remote_code=True`. For the Qwen sizes, you want vLLM or a dedicated GPU with enough VRAM to load a 7B model with attention.

**When it fits.** When you have GPU headroom to run a 7B embedder for max quality (Qwen-7B); when you want long context in a compact model (`gte-large-en-v1.5`); Chinese and multilingual work.

**License.** Apache-2.0 for the standard GTE line; check individual model cards, especially for the Qwen-based variants.

## Snowflake Arctic Embed

**Models.** `Snowflake/snowflake-arctic-embed-{xs,s,m,l}` (22M / 33M / 109M / 335M params, various dims), plus `snowflake-arctic-embed-l-v2.0` and `snowflake-arctic-embed-m-v2.0`. Paper / launch: Merrick et al., *Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models* ([arXiv:2405.05374](https://arxiv.org/abs/2405.05374)); Snowflake's [*Introducing Arctic Embed*](https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/) blog is the announcement.

**Why it is interesting.**

- **Explicitly optimized for retrieval quality per parameter.** The training paper argues the training-recipe choices (harder negatives, curriculum, contrastive pretraining on cleaner query-document pairs) matter more than parameter count. Empirically, `arctic-embed-l` is a strong open retriever at ~335M parameters.
- **Available on Snowflake Cortex** as a hosted call for teams already in Snowflake — so it appears on both this chapter and the "hosted API" surface of chapter 4 depending on how you consume it.

**Serving.** Standard `SentenceTransformer`. No prompt required for the v1.0 family; check v2.0's card.

**License.** Apache-2.0.

## Jina Embeddings

**Models.** `jinaai/jina-embeddings-v3` (~570M params, 1024 dim, up to 8192 tokens, multilingual, MRL-trained, five task-typed adapters), `jinaai/jina-embeddings-v2-base-code` (code-specialized), `jinaai/jina-colbert-v2` (multilingual ColBERT — chapter 1). Papers: Sturua et al., *jina-embeddings-v3: Multilingual Embeddings With Task LoRA* ([arXiv:2409.10173](https://arxiv.org/abs/2409.10173)).

**Why it is interesting.**

- **Task LoRA adapters.** `jina-embeddings-v3` ships with five LoRA adapters (retrieval query, retrieval passage, separation, classification, text-matching). At inference you specify the task and Jina swaps in the right adapter, giving per-task specialization without shipping five separate models.
- **Long-context multilingual open weights.** 8192 tokens on a multilingual model with published weights is a rare combination; competes with `bge-m3` and Cohere multilingual on that axis.
- **Ships a matching reranker (`jina-reranker-v2-base-multilingual`)** and a ColBERT (`jina-colbert-v2`) — you can build a full-family stack.

**Serving.** `SentenceTransformer` with `trust_remote_code=True`. Pass `task="retrieval.query"` or `task="retrieval.passage"` on `encode`.

**License.** CC-BY-NC 4.0 for `jina-embeddings-v3` — **non-commercial** by default. Commercial use needs a Jina license. This is a common trap; check the model card before deploying to production. Older Jina models (`v2`) have different licenses; some are Apache-2.0.

## mixedbread `mxbai-embed-large-v1`

**Models.** `mixedbread-ai/mxbai-embed-large-v1` (~335M, 1024 dim, 512 tokens). Blog: [*Introducing our first embedding model — mxbai-embed-large-v1*](https://www.mixedbread.ai/blog/mxbai-embed-large-v1).

**Why it is worth naming.** It is a well-cited general-purpose open embedder in the ~335M-parameter class that has repeatedly been competitive on MTEB retrieval. Similar shape and cost to `bge-large-en-v1.5`; different training recipe. Useful as an alternative when you want to A/B two well-trained ~1024-dim models against your corpus.

**License.** Apache-2.0.

## Serving open embedders in production

`sentence-transformers`'s in-process `SentenceTransformer("...").encode(texts)` is fine for offline embed jobs on a workstation. In production you almost certainly want one of:

- **Text Embeddings Inference (TEI)** — Hugging Face's [`huggingface/text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference) — the reference OSS serving stack for embedding models. Supports BGE, E5, GTE, Nomic, Arctic, Jina, mxbai and more. Handles dynamic batching, ONNX/CUDA graph optimization, and exposes an OpenAI-compatible `/embeddings` endpoint. This is the "TEI" name you will see in retrieval-service Kubernetes manifests.
- **vLLM** — vLLM's embedding server support ([`vllm/embeddings`](https://docs.vllm.ai/en/latest/models/pooling_models.html)) — needed for LLM-backbone embedders like `gte-Qwen2-7B-instruct` and `e5-mistral-7b-instruct` where you want PagedAttention throughput.
- **Custom FastAPI + `sentence-transformers`** — fine for internal / low-traffic; will lose to TEI on tail latency once you have any real concurrent load.

**Batching and throughput.** Small embedders (BGE / E5 / GTE small–large) get near-linear throughput scaling to a saturated GPU with batch sizes of 64–256. LLM-backbone embedders (7B) are memory-bound at much smaller batches. Measure on your GPU; do not extrapolate.

**Quantization at inference.** ONNX-runtime int8 quantization on the encoder, or bitsandbytes 8-bit for the LLM-backbone variants, both give measurable throughput lifts with modest quality cost. Chapter 8 covers *output-vector* quantization; this is the *model-weight* quantization sibling.

**Version pinning.** `sentence-transformers` and the underlying `transformers` library can affect model behavior across versions. Pin `sentence-transformers==...`, pin the specific model revision (git commit SHA on Hugging Face), and record both in the index metadata alongside `embedding_model_id`.

## The open vs. hosted mental model

You are not choosing "open embedders" vs. "hosted embedders" once, for your whole stack. You are choosing **per role**, and per **traffic tier**:

- Offline corpus embed (batch, throughput-bounded, one-shot): a self-hosted open model on a rented GPU is often cheapest by an order of magnitude at large corpus sizes.
- Online query embed (interactive, latency-bounded, low volume per user): hosted API is often the right call — a per-query cost is trivial, and you avoid the ops surface of running the GPU service for the query path.
- Reranker slot (top-K rescoring, latency-bounded, low volume per user): often self-hosted cross-encoder or ColBERT — small models run fast, no per-token cost.
- Domain-specialized retriever: a `voyage-code-3` / `voyage-law-2` hosted call for the vertical, or a fine-tuned open BGE (peer track owns the training; chapter 9 boundary) for full control.

Chapter 6 is the methodology to *measure* the trade-off; chapter 7 is the domain-fit benchmark that decides *which* model in each slot.

## What is NOT in this chapter (and where it goes)

- **Fine-tuning any of these.** Peer track `fine-tuning-engineer`, chapter 9 boundary.
- **Cross-encoder rerankers.** mod-107; chapter 1 named the family, chapter 5 does not tour the reranker cards.
- **Multimodal embedders (CLIP / SigLIP / ColPali).** mod-112.
- **The Elasticsearch / OpenSearch text-expansion (ELSER) sparse-neural model.** Chapter 1 named SPLADE and mentioned ELSER; the *serving* story lives in mod-105 (Lucene family) — self-hosting ELSER means running Elastic's ML nodes, which is Elastic-shaped ops rather than TEI-shaped.

## What you should be able to do after this chapter

- Name at least six open embedding families and give the flagship model in each.
- Explain BGE-M3's unusual property (dense + late-interaction + sparse from one model).
- Explain why E5, Nomic, GTE-Qwen, and Jina all require some form of task-typed prefix or instruction and what happens when you forget it.
- Point out that Jina v3 is CC-BY-NC (non-commercial) and check every model card's license *before* production.
- Pick a serving stack (TEI, vLLM, or in-process `sentence-transformers`) matched to the model size and traffic tier.
- Name at least two levers (batch size, model-weight quantization) you use to squeeze throughput out of a self-hosted embedder.

## Further reading

- Xiao, S., et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding.* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) (BGE).
- Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
- Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) (E5).
- Wang, L., et al. (2024). *Improving Text Embeddings with Large Language Models.* [arXiv:2401.00368](https://arxiv.org/abs/2401.00368) (E5-mistral).
- Nussbaum, Z., et al. (2024). *Nomic Embed: Training a Reproducible Long Context Text Embedder.* [arXiv:2402.01613](https://arxiv.org/abs/2402.01613).
- Li, Z., et al. (2023). *Towards General Text Embeddings with Multi-stage Contrastive Learning.* [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) (GTE).
- Li, C., et al. (2024). *Making Text Embedders Few-Shot Learners.* [arXiv:2409.15700](https://arxiv.org/abs/2409.15700) (GTE-Qwen instruct).
- Merrick, L., et al. (2024). *Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models.* [arXiv:2405.05374](https://arxiv.org/abs/2405.05374); Snowflake, [*Introducing Snowflake Arctic Embed*](https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/).
- Sturua, S., et al. (2024). *jina-embeddings-v3: Multilingual Embeddings With Task LoRA.* [arXiv:2409.10173](https://arxiv.org/abs/2409.10173).
- Mixedbread AI, [*Introducing mxbai-embed-large-v1*](https://www.mixedbread.ai/blog/mxbai-embed-large-v1).
- Hugging Face, [*Text Embeddings Inference (TEI)*](https://github.com/huggingface/text-embeddings-inference) — the reference OSS embedding server.
- vLLM, [*Pooling / Embedding Models*](https://docs.vllm.ai/en/latest/models/pooling_models.html) — for LLM-backbone embedders.
- `sentence-transformers`, [*Pretrained Models*](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) — the maintained list of publicly available bi-encoders and their reference cards.
