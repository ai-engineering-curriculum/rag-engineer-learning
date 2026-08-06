# Chapter 2 — Cross-Encoder Rerankers

## Why this chapter exists

The most common second-stage reranker is a **cross-encoder** — a transformer that takes the query and a single candidate as a joint input and outputs a scalar relevance score. Chapter 1 named cross-encoders as the default reranker family; this chapter is why the default is a good default and how you deploy one.

Cross-encoders are almost always where you should start. The families this chapter covers — BGE Reranker (BAAI), MS-MARCO MiniLM (Sentence Transformers), Mixedbread's `mxbai-rerank`, Jina Reranker's open-weight variants — give you strong reranking quality with weights you can download, run on your own hardware, and iterate on. You reach for a hosted API (chapter 3), an LTR reranker (chapter 4), or an LLM-as-reranker (chapter 4) when a cross-encoder's specific limitations start to matter for your stack. If you have not deployed a cross-encoder yet, deploy one first.

## The bi-encoder vs. cross-encoder distinction

A first-stage dense retriever is a **bi-encoder**:

```
query  ─►  encoder  ─►  q_vector  ┐
                                   ├─► cosine(q, c) = score
chunk  ─►  encoder  ─►  c_vector  ┘
```

The two encoders run *independently* on the query and the chunk. The chunk vector can be precomputed once at index time and stored in a vector database (mod-104); at query time you only encode the query and do an ANN lookup. That is why bi-encoders scale to millions of documents — the expensive work is amortised.

A cross-encoder runs the two inputs *together*:

```
[CLS] query [SEP] chunk [SEP]  ─►  encoder  ─►  score
```

Every layer of the transformer sees both the query and the chunk tokens attending to each other. That joint attention is why cross-encoders are more accurate — the model can decide relevance by looking at *which query token matches which chunk token* — and why they cannot precompute the chunk representation. Every (query, chunk) pair is a fresh forward pass.

The cost implication that follows: at query time you do **N forward passes** for a top-N candidate set. A cross-encoder with a 128–512-token window at N=100 is ~50–200 ms on a single GPU with a small model (MiniLM-class) and ~200–800 ms with a larger one (BGE-large, DeBERTa-based). That is affordable at the second stage precisely because N=100, not N=10-million.

The three properties that follow from the joint-attention shape:

1. **Higher precision than any bi-encoder retriever at the same top-K.** The literature is consistent on this — Nogueira & Cho 2019 ([arXiv:1901.04085](https://arxiv.org/abs/1901.04085)) is the reference.
2. **Non-precomputable.** You cannot cache "the chunk score" across queries; the score is per-pair.
3. **Sensitive to the joint-input length.** The model's input length is (query length + chunk length + special tokens). Longer chunks eat the budget quickly.

## The open-weight families you should know

The open-weight rerankers listed here all follow the (query, chunk) → score cross-encoder shape and all publish weights under permissive licences. Numbers on any benchmark move quickly; treat the notes below as pointers to the model card, not fixed rankings.

### BGE Reranker (BAAI)

- **Repository:** [FlagEmbedding on GitHub](https://github.com/FlagOpen/FlagEmbedding); models on Hugging Face under `BAAI/bge-reranker-*`.
- **Model sizes:** `bge-reranker-base` (~278 M params), `bge-reranker-large` (~560 M params), `bge-reranker-v2-m3` (multilingual, ~568 M params, XLM-RoBERTa base), `bge-reranker-v2-gemma` (large, decoder-based). Newer variants ship regularly — always check the [model card](https://huggingface.co/BAAI) for the current recommended default.
- **Strengths:** strong English + multilingual performance, actively maintained, wide adoption in production RAG stacks. The v2-m3 variant handles most non-English corpora without extra work.
- **When to use:** general-purpose starting point when you want an open-weight reranker with strong out-of-the-box behavior on heterogeneous corpora.
- **License:** MIT for BGE reranker base/large; check the specific model card for v2 variants (some are Apache-2.0).

### MS-MARCO MiniLM (Sentence Transformers)

- **Repository:** [sentence-transformers on GitHub](https://github.com/UKPLab/sentence-transformers); models on Hugging Face under `cross-encoder/ms-marco-*`.
- **Model sizes:** `ms-marco-MiniLM-L-6-v2` (~23 M params, fastest), `ms-marco-MiniLM-L-12-v2`, `ms-marco-electra-base`, `ms-marco-TinyBERT-*` variants.
- **Strengths:** very fast, very well-known, the reference implementation used in the sentence-transformers documentation. The MiniLM-L-6 variant is a common latency-first pick when your corpus is English and you cannot afford BGE-large.
- **When to use:** you have tight latency and you can eat some quality loss vs. BGE. The trained-on-MS-MARCO detail matters — if your domain looks nothing like MS MARCO (short queries, web passages), a domain-specific fine-tune (see below) will beat this on your queries.
- **License:** Apache-2.0.

### Mixedbread `mxbai-rerank`

- **Repository:** [Mixedbread on Hugging Face](https://huggingface.co/mixedbread-ai); documentation at [mixedbread.ai](https://www.mixedbread.ai/).
- **Model sizes:** `mxbai-rerank-xsmall`, `-small`, `-base`, `-large` — several sizes on the same architecture family. Later `v2` releases update the training data and license terms; always check the model card for the current variant.
- **Strengths:** actively maintained, publishes evaluation numbers on BEIR, positions itself as a drop-in replacement for the older MS-MARCO MiniLM models with better quality per parameter.
- **When to use:** as a modern alternative to `cross-encoder/ms-marco-MiniLM-L-6-v2` when you want a small model but not a stale one.

### Jina Reranker (open-weight variants)

- **Repository:** [jina-ai on Hugging Face](https://huggingface.co/jinaai); documentation at [jina.ai/reranker](https://jina.ai/reranker/).
- **Model sizes:** `jina-reranker-v1-turbo-en` (very small, ~38 M params), `jina-reranker-v1-base-en`, `jina-reranker-v2-base-multilingual`.
- **Strengths:** turbo variant is one of the smallest usable cross-encoders (fits on CPU inference for very small batch sizes), multilingual variant handles code + prose. Jina also runs a hosted rerank API (chapter 3) with the same model family, which makes hosted-vs-self-hosted testing straightforward.
- **When to use:** you want cheap inference on modest hardware, or you want to A/B the open-weight vs. hosted variants of the same model.

### ColBERT / ColBERTv2 (as a reranker)

- **Repository:** [stanford-futuredata/ColBERT on GitHub](https://github.com/stanford-futuredata/ColBERT). Paper: Khattab & Zaharia 2020 ([arXiv:2004.12832](https://arxiv.org/abs/2004.12832)); ColBERTv2, Santhanam et al. 2021 ([arXiv:2112.01488](https://arxiv.org/abs/2112.01488)).
- **Architecture:** late-interaction, not a pure cross-encoder. The query and chunk are encoded separately (like a bi-encoder), but the score is computed as a max-sim over per-token embeddings rather than a single vector cosine. That gives you *some* of a cross-encoder's fine-grained matching quality without paying a full joint forward pass per pair.
- **When to use:** you need cross-encoder-ish quality at bi-encoder-ish latency. ColBERT can also serve as a first-stage retriever, not just a reranker; whether that fits your stack depends on the index shape (a per-token index is larger than a per-chunk one).
- **Note:** ColBERT is more of an operational lift than a pure cross-encoder because of the per-token index. For most teams, "start with a cross-encoder, revisit ColBERT if latency-quality tension is unresolved" is the right sequencing.

## The interface — how to call one from Python

The `sentence-transformers` and `FlagEmbedding` libraries expose near-identical shapes.

**With `sentence-transformers` (works for MS-MARCO MiniLM and any cross-encoder on the Hub):**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

query = "What is the capital of France?"
candidates = [
    "Paris is the capital and most populous city of France.",
    "France is a country in Western Europe with a rich history.",
    "The Eiffel Tower is located in Paris.",
    "London is the capital of the United Kingdom.",
]

pairs = [(query, c) for c in candidates]
scores = reranker.predict(pairs)   # returns np.ndarray of shape (N,)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
```

**With `FlagEmbedding` (BGE Reranker):**

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

pairs = [[query, c] for c in candidates]
scores = reranker.compute_score(pairs)   # returns list[float]
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
```

Two production-relevant details:

- **Batching matters.** Both APIs accept the full list of pairs and run them as batches on the GPU. Calling `predict` in a Python loop over pairs is the slow path; calling it once with all N pairs is the fast path. On a T4/L4-class GPU, batch-of-100 for MiniLM-L-6 is well under 100 ms.
- **`use_fp16` (or `torch_dtype=torch.float16`) roughly halves latency on modern GPUs** with essentially no quality loss on the standard rerank benchmarks. Always enable it in production; do not enable it in unit tests that check exact score values (small numeric drift will break exact-value asserts).

## Deploying a cross-encoder: the four operational levers

Once you have picked a model, the four things you tune to fit a latency budget:

1. **Model size.** Bigger is better on quality up to a plateau. Small models (23–100 M params) fit on CPU inference for small batches and can hit p95 well under 100 ms on a single GPU for N=100. Large models (500 M–1 B params) hit better nDCG@10 but need a GPU and careful batching.
2. **Max sequence length.** The default (typically 512 tokens) is the *joint* (query + chunk) budget. Shrinking it (to 256, or even 128 for very short chunks) speeds up inference roughly quadratically in the transformer's attention layers. If your chunks are mostly < 200 tokens, capping `max_length` at 256 is close to free and cuts latency substantially.
3. **Precision (fp16, int8, quantised).** fp16 is nearly always a win. Int8 quantisation via [ONNX Runtime](https://onnxruntime.ai/) or [BetterTransformer](https://huggingface.co/docs/optimum/en/bettertransformer/overview) trades a small quality drop for 2–3× throughput; measure on your eval set before shipping.
4. **Batch size and concurrency.** Batching N=100 candidates for one query is easy. Batching *across queries* (multiplexing several concurrent user queries into one GPU batch) is a real engineering lift but recovers latency-under-load at high QPS. Frameworks: [vLLM](https://github.com/vllm-project/vllm) for LLM serving (not directly applicable to cross-encoders), [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) from Hugging Face (which does support cross-encoder / rerank endpoints), or a custom Triton Inference Server setup.

## When you fine-tune a cross-encoder

Off-the-shelf cross-encoders trained on MS MARCO transfer well to most English text-search domains. The domain gap starts to matter when:

- Your queries look nothing like MS MARCO (very long queries, code queries, legal / medical / SQL queries).
- Your relevance definition is not the same as generic web-passage relevance (e.g., "this passage supports the *specific claim* in the query," or "this passage is a *citation-worthy* source, not just topically related").
- You have real click-through or human-labeled preference data that the off-the-shelf model has not seen.

The fine-tuning path is the standard sentence-transformers `CrossEncoder.fit()` with pairwise or triplet loss on `(query, positive, negative)` triples, or a listwise loss (e.g., ListNet, LambdaLoss) if you have per-query graded relevance judgments. Details are in the [sentence-transformers cross-encoder training documentation](https://sbert.net/docs/cross_encoder/training/overview.html) and, for the loss functions, Burges et al. 2005 ([*Learning to Rank using Gradient Descent*](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningTo_BurgesEtAl.pdf)).

Fine-tuning is often the right lever *before* you jump to a hosted API — a fine-tuned MiniLM on 5–20K of your own labeled pairs will frequently beat a much larger off-the-shelf model on your specific queries, at a fraction of the inference cost. Mod-111 covers the MLOps of maintaining a fine-tuned reranker (retraining cadence, drift monitoring, eval-set versioning).

## Cost and latency ballparks

Numbers move constantly; treat these as order-of-magnitude anchors.

| Model                                        | Parameters   | Latency for N=100 (single GPU, fp16, batched) | Deployment |
|----------------------------------------------|--------------|-----------------------------------------------|------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2`       | ~23 M        | ~30–60 ms                                     | Fits on CPU for small batch; ideal on T4 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2`      | ~33 M        | ~50–100 ms                                    | Small-GPU sweet spot |
| `BAAI/bge-reranker-base`                     | ~278 M       | ~100–200 ms                                   | Single GPU |
| `BAAI/bge-reranker-large`                    | ~560 M       | ~200–400 ms                                   | Single GPU with headroom or a smaller model on CPU |
| `BAAI/bge-reranker-v2-m3`                    | ~568 M       | ~200–400 ms                                   | Single GPU |
| `jina-reranker-v1-turbo-en`                  | ~38 M        | ~30–70 ms                                     | Small GPU or CPU |

Chapter 3 puts the hosted-API latencies alongside these for direct comparison. Chapter 5 puts them into a p95-budget worksheet.

## When cross-encoders are the wrong pick

Three cases:

1. **Latency budget is under ~30 ms for the whole rerank step.** Even the smallest cross-encoder at N=100 will not fit under a very tight budget. LTR (chapter 4) is the alternative — a GBDT scores 100 candidates in single-digit milliseconds because the "features" are already computed at retrieval time.
2. **You need a score that's calibrated across queries.** Cross-encoder scores are query-relative. If your product logic needs an absolute "is this chunk relevant *at all*?" threshold (e.g., to trigger a "no results" fallback), you need either an LLM-judge classifier or a calibration layer. Chapter 4 discusses.
3. **The relevance definition is complex and hard to capture in a training set** — e.g., "this chunk is *authoritative*, not just topical," or "this chunk *contradicts* the query claim." An LLM-as-reranker (chapter 4) can carry that instruction in a prompt in a way a fine-tuned cross-encoder can only approximate through the labeled data.

For everything else — start here.

## What you should be able to do after this chapter

- Explain in one paragraph why a cross-encoder is more accurate than a bi-encoder and why that accuracy costs a per-pair forward pass.
- Name three cross-encoder families with strong open-weight releases (BGE, MS-MARCO MiniLM, Mixedbread `mxbai-rerank`, Jina Reranker) and the situations each fits.
- Deploy an off-the-shelf cross-encoder with `sentence-transformers` or `FlagEmbedding`, use batching + fp16, and stay inside a 200 ms p95 for N=100 on a single modern GPU.
- Enumerate the four deployment levers (model size, max sequence length, precision, batch/concurrency) and know which one to pull first when latency is over budget.
- Recognise the three cases where a cross-encoder is the wrong first pick (ultra-tight latency, cross-query calibration, complex relevance definition).

## Further reading

- Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the reference paper for cross-encoder reranking in IR.
- Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — introduces the bi-encoder shape by contrast; the sentence-transformers library flows from this paper.
- Khattab, O. and Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — late-interaction rerankers as a middle point between bi- and cross-encoders.
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2021). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) — the ColBERT follow-up.
- Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding.* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) — the BGE embedding-and-reranker family paper.
- The [BEIR benchmark](https://github.com/beir-cellar/beir) and Thakur et al. 2021 ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)) — the standard zero-shot IR benchmark most reranker papers report on.
- Sentence Transformers: [Cross-Encoder documentation](https://www.sbert.net/examples/applications/cross-encoder/README.html) — the canonical library reference for open-weight cross-encoders.
- FlagEmbedding: [reranker documentation](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker) — the BGE reranker library reference.
