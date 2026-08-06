# Chapter 8 — Dimensionality, Quantization, and Matryoshka Representation Learning

## Why this chapter exists

Chapter 6's cost formulas made the point in the abstract: `dim` is a multiplier on every storage cost downstream, and float32 is not the cheapest datatype. This chapter is the concrete engineering of that trade-off — three levers (dimensionality reduction, scalar quantization, binary/bit quantization) and one training-time trick (Matryoshka Representation Learning) that lets you change the served dimension after training without re-embedding.

Everything here is a **post-embedding transform**. You embed once at the model's native dim in float32; you *store* and *serve* at a smaller dim and/or a smaller datatype. The choice affects storage, ANN latency, and quality — usually monotonically, and often by an order of magnitude.

## The storage / quality frontier at a glance

A rough table of what each lever costs and buys, at a 1024-dim dense embedder over 100M chunks (~410 GB float32 baseline before ANN overhead):

| Transform | Bytes per vector | 100M-chunk raw storage | Typical quality cost | Native support |
|---|---|---|---|---|
| Full float32 | 4096 | 410 GB | baseline | universal |
| Float16 | 2048 | 205 GB | ~0 (negligible) | universal in modern ANN libs |
| Int8 (scalar quantization) | 1024 | 100 GB | typically small (couple % on retrieval quality) | pgvector, Qdrant, Weaviate, Milvus, FAISS |
| Binary (1 bit per dim) | 128 | 12.5 GB | larger (5–15 %) mitigated by rescore | pgvector 0.7+, Qdrant, Milvus, FAISS |
| MRL truncation 1024 → 512 | 2048 | 205 GB | small on MRL-native models | requires MRL-trained model |
| MRL truncation 1024 → 256 | 1024 | 100 GB | moderate on MRL-native models | requires MRL-trained model |
| MRL 512 + int8 stacked | 512 | 50 GB | additive of the two costs | supported where int8 is |
| MRL 256 + binary stacked | 32 | 3.2 GB | large but often survivable with rescore | supported |

These numbers are order-of-magnitude illustrations, not measured on your data. Rerun the sweep on your own corpus (chapter 7 protocol; chapter 6 methodology).

The 100× storage compression at the bottom of the table is not a stretch scenario — teams operating at hundreds of millions of chunks are running MRL-truncated int8 or binary embeddings, with a rescore step to recover the quality cost, and paying an order of magnitude less for their vector infrastructure than they would with plain float32. mod-104 covers the ANN-side implementation; this chapter is the model / vector-transform side.

## Dimensionality — the axis you cannot ignore

**What "dim" is.** The width of the vector the encoder emits. For dense bi-encoders it is a fixed model property — `bge-small-en-v1.5` emits 384-dim, `bge-large-en-v1.5` emits 1024-dim, `text-embedding-3-large` emits 3072-dim.

**Why dim matters.**

- **Storage.** Linear in `dim`.
- **ANN memory.** HNSW graph is linear in `dim` for the vectors + a constant for the graph edges.
- **ANN latency.** Distance computation is linear in `dim` per candidate; HNSW visits `~O(log N) × efSearch` candidates per query.
- **Fusion / arithmetic.** Any downstream math on the vector (RRF fusion inputs, reranker features, etc.) is linear in `dim`.

**Why you cannot arbitrarily truncate a non-MRL model.** For a model *not* trained with Matryoshka, the components of the embedding vector are not ordered by importance. Dropping components 512..1023 of a 1024-dim `bge-large` vector destroys quality faster than dropping components 512..1023 of an MRL-trained `nomic-embed-text-v1.5` vector, because the second was trained to survive that truncation.

Non-MRL alternatives if you need lower dim without training: **PCA** after embedding, or a linear **projection** learned on your corpus. Both work; both add an offline step, and both interact awkwardly with future retriever updates.

## Matryoshka Representation Learning (MRL)

**What it is.** MRL is a training regularizer that packs the embedding so that any prefix of the vector is itself a good embedding. Reference: Kusupati, A., et al. (2022). *Matryoshka Representation Learning* ([arXiv:2205.13147](https://arxiv.org/abs/2205.13147)). The name is a Russian nesting doll — a 1024-dim MRL vector "contains" a 512-dim vector, which contains a 256-dim vector, which contains a 128-dim vector, and so on, each usable independently.

**How it works (short version).** During contrastive training the loss is a sum: retrieval loss on the full 1024-dim vector, plus retrieval loss on the 512-dim prefix, plus 256-dim prefix, plus 128-dim, plus 64-dim. The model is forced to put the most retrieval-discriminative information in the leading dimensions, because dropping the trailing dimensions must not break the shorter-prefix losses.

**What it lets you do at deployment.** Embed once at the native dim. Serve at whichever prefix your quality/cost budget allows. You can even serve **different products at different dims from the same index** (a low-latency mobile endpoint at 256 dim; a research desktop client at 1024 dim), truncating at query time.

**MRL-native models you can pick from today.**

- OpenAI `text-embedding-3-small` (native 1536; truncate via `dimensions` API param).
- OpenAI `text-embedding-3-large` (native 3072; truncate via `dimensions`).
- Google `text-embedding-005` (native 768; truncate via `outputDimensionality`).
- Amazon `titan-embed-text-v2:0` (native 1024; truncate via `dimensions`).
- Nomic `nomic-embed-text-v1.5` (native 768; supports 512/256/128/64 prefixes per the model card).
- Snowflake `snowflake-arctic-embed-l-v2.0` (per the v2.0 model card).
- Mixedbread `mxbai-embed-large-v1` (per the model card; check current status).
- Alibaba `gte-Qwen2-*-instruct` variants document MRL support in later revisions — check the model card.

**MRL is a *training-time* property.** You cannot retrofit MRL onto a non-MRL model by truncating its output — the leading dimensions were not trained to be sufficient. If your model of choice is not on the list, either accept its native dim, do explicit PCA, or fine-tune with the MRL loss (peer track).

**MRL is not free.** Contemporary MRL-trained models routinely give up 1–3 nDCG points on retrieval tasks vs. an identically-sized non-MRL model, in exchange for the flexibility. If you always deploy at native dim, MRL costs you a small quality tax you may not want to pay. If you deploy at half-native or below, MRL wins by a mile.

**Concrete: MRL truncation with OpenAI.**

```python
from openai import OpenAI
client = OpenAI()

# Index-time: get the full-native embedding, plus a truncated view.
r = client.embeddings.create(
    model="text-embedding-3-large",
    input=["chunk text"],
    dimensions=512,           # server-side truncation via MRL
)
vec_512 = r.data[0].embedding
```

Note that the *cost* is the same — you pay for the full compute — but you store only 512 floats. If you later decide to serve at 1024, you must re-embed to get the wider vector.

**Concrete: MRL truncation with a local `sentence-transformers` model.**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
full = model.encode("search_query: what is HNSW", normalize_embeddings=True)
# Truncate to the prefix of the desired dim; renormalize on the shorter vector.
short = full[:256]
short = short / np.linalg.norm(short)
```

Always L2-normalize *after* truncation. The unnormalized prefix has a different norm than the unnormalized full vector; cosine similarity on unnormalized truncated vectors is wrong.

## Scalar quantization — int8 embeddings

**What it is.** Represent each component of the vector with an 8-bit integer instead of a 32-bit float. You need to store the (min, max) scale per vector (or per index shard) so you can reconstruct approximate float values at query time — this is the "scalar quantization" that FAISS, Qdrant, pgvector, and Weaviate all support natively.

**The gain.** 4× storage reduction over float32 (12× over float64, if anyone still had that). ANN distance computations on int8 are also faster because they hit smaller memory bandwidth per vector.

**The cost.** Reintroduces a small quantization error per component. Empirically, well-trained modern embedders (BGE, Cohere v3, `text-embedding-3-large`) tolerate int8 with a very small nDCG drop — often within noise on retrieval eval. Cohere's Embed v3 was explicitly trained with quantization in mind and offers int8 output natively via the API.

**Concrete: pgvector's `halfvec` (fp16) and int8 support.** As of pgvector 0.7 the extension supports `halfvec` (fp16) columns and bit / int8 quantized indexes. See [`pgvector` README](https://github.com/pgvector/pgvector). Example:

```sql
-- fp16 storage; halves memory vs. float32.
ALTER TABLE chunks ADD COLUMN embedding_h halfvec(1024);
UPDATE chunks SET embedding_h = embedding::halfvec(1024);
CREATE INDEX ON chunks USING hnsw (embedding_h halfvec_l2_ops);
```

Similar patterns on [Qdrant scalar quantization](https://qdrant.tech/documentation/guides/quantization/), [Weaviate PQ / SQ](https://weaviate.io/developers/weaviate/configuration/compression), and [FAISS ScalarQuantizer](https://faiss.ai/cpp_api/struct/structfaiss_1_1ScalarQuantizer.html).

**Concrete: Cohere Embed v3 int8 output.**

```python
resp = co.embed(
    texts=chunks,
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float", "int8"],   # get both
)
vec_float = resp.embeddings.float[0]
vec_int8  = resp.embeddings.int8[0]     # 4× smaller on the wire and on disk
```

## Binary quantization — 1 bit per dimension

**What it is.** For each component, keep only the sign — `1` if positive, `0` if negative. A 1024-dim vector becomes 1024 bits = 128 bytes, a 32× storage reduction over float32.

**How you score.** Similarity between binary vectors is **Hamming distance** (count of differing bits) instead of dot product / cosine. It is a very fast bit-count instruction on modern CPUs.

**The cost.** Non-trivial. A well-trained embedder loses roughly ~5–15 % of its retrieval quality when quantized to pure binary — the sign of each component alone throws away most of the magnitude information. On tail queries the loss can be larger.

**Why binary is still popular.** Two mitigations turn it from lossy to production-shippable:

1. **Rescore with the original float vector on top-K.** Retrieve the top-K (e.g., top-1000) with fast binary Hamming search; rescore those K with the original float embedding to reorder to the final top-10. You keep binary's storage and search speed for the recall step, and float's quality for the ranking step.
2. **Train the model for binary quantization** — some embedders (Cohere v3 documents this, and MRL-trained + binary-fine-tuned open models exist) explicitly optimize for the sign-only representation to retain quality better than a naive `sign(vec)`.

Reference: Cohere's engineering post [*Binary Embeddings: Store and Retrieve 32x More Data with Vector Search*](https://cohere.com/blog/int8-binary-embeddings) is the canonical shape argument; MyScale, Qdrant, and Weaviate have similar public writeups.

**Concrete: Cohere binary embeddings.**

```python
resp = co.embed(
    texts=chunks,
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["binary"],
)
# Each element is an int representing the packed bits; 1024-dim binary -> 128 bytes on the wire.
vec_binary = resp.embeddings.binary[0]
```

**Concrete: FAISS binary index.**

```python
import faiss, numpy as np

d = 1024
# Convert your float32 vectors to a binary matrix of shape (n, d/8) bytes.
def to_binary(vecs: np.ndarray) -> np.ndarray:
    return np.packbits(vecs > 0, axis=1)

xb_bin = to_binary(xb_float32)               # (n, 128) uint8
xq_bin = to_binary(xq_float32)               # (m, 128) uint8

index = faiss.IndexBinaryHNSW(d, 32)          # HNSW for binary vectors, Hamming distance
index.add(xb_bin)
_, top = index.search(xq_bin, k=1000)         # top-1000 by Hamming — rescore next
```

## Stacking MRL and quantization

MRL and quantization are orthogonal — you can do both. The typical stack for a very cost-constrained index:

1. Embed with an MRL-native model at native dim (e.g., 1024).
2. Truncate to a chosen dim (512, 256, 128).
3. Quantize the truncated vector to int8 (or to binary).
4. Optionally: index the small quantized vector for retrieval; keep the full float vector on disk (or in a slower tier) to rescore the top-K.

At 256-dim binary + rescore, a 100M-chunk index that would be 410 GB in float32 becomes ~3.2 GB in the ANN index (with the float vectors held on colder storage for rescore). Same corpus, ~100× less RAM.

You should reach for this stack when storage is the binding constraint. If your corpus is small enough that everything fits in a laptop's RAM, do not compress; you are trading precious quality for cost you do not have.

## Choosing the shape — decision tree

Rough guidance; the domain-fit benchmark (chapter 7) is what actually decides:

- **Small corpus (<1M chunks) or dev/prototype:** native dim, float32. Storage is cheap, quality is precious.
- **Medium corpus (1M–10M):** float16 or int8 is a straightforward win — small quality cost, 2–4× storage saving. Skip binary unless your ANN is memory-bound.
- **Large corpus (10M–100M):** MRL truncation to half native + int8 is the common default. If the model is not MRL-native, consider a different model rather than PCA'ing yourself into a corner.
- **Very large corpus (>100M) or memory-constrained ANN (e.g., in-memory HNSW at scale):** MRL to 256 or below + binary + rescore top-K with float. Test quality carefully; some corpora tolerate this well and some do not.
- **Rerank slot (mod-107):** never quantize the reranker's input signals. Rerankers benefit from the full float representation; they run over top-K only and the cost is bounded.

## The interaction with the ANN index

Everything in this chapter is about the *vector* you feed to the ANN. mod-104 covers what the ANN does with it — HNSW parameters (M, efConstruction, efSearch), IVF (nlist, nprobe), IVF-PQ (pq_m, pq_nbits). Two interactions to keep in mind:

- **IVF-PQ is *another* quantization on top of yours.** If you MRL-truncate to 256 dim then serve on IVF-PQ, you are quantizing twice. Sometimes this is fine; often it costs more quality than expected. Prefer MRL+SQ, or float+IVF-PQ — not the stack of both.
- **Binary indexes have a different metric.** Cosine and dot-product ANN implementations do not directly apply to binary vectors — you need a Hamming-distance ANN (FAISS `IndexBinaryHNSW`, Milvus `BIN_FLAT` / `BIN_IVF_FLAT`, Qdrant binary quantization). Not every vector DB supports it natively — check before designing the stack.

## Reproducibility: capture the quantization in the index key

The mod-111 index-version key must record the quantization choices alongside the model:

```
index_version_key = f"{embedding_model_id}@{revision}:dim={served_dim}:quant={quant}"
# e.g., "text-embedding-3-large@1:dim=512:quant=int8"
```

If you swap int8 for binary, that is a new index. If you truncate from 1024 to 512, that is a new index. Both require a re-embed (or re-transform of the stored float vector) and a re-build of the ANN index.

## What is NOT in this chapter (and where it goes)

- **ANN parameter tuning (HNSW, IVF-PQ).** mod-104.
- **The actual index-version cutover machinery.** mod-111.
- **Learning your own quantizer as a fine-tuning step.** Peer track `fine-tuning-engineer`.
- **Sparse vectors (SPLADE, BM25).** Storage of sparse vectors is a different game — you pay for the number of non-zeros, not for `dim`. Chapter 1 named SPLADE; mod-105 covers Lucene-based sparse storage.

## What you should be able to do after this chapter

- Give the four levers of embedding storage (dim, float precision, scalar quantization, binary quantization) and pick a defensible combination for a given corpus size.
- Name at least four MRL-native models (hosted or open) and explain why MRL is a training-time property.
- Explain the mechanics of scalar quantization (int8), what it costs (small quality tax), and what it buys (4× storage).
- Explain the mechanics of binary quantization, why it needs a rescore step to recover quality, and cite a vector DB that supports it.
- Draw the storage / quality frontier and pick the shape at any point on it.
- Add the quantization choice to the index-version key so mod-111's re-embed / cutover machinery has a stable ID.

## Further reading

- Kusupati, A., et al. (2022). *Matryoshka Representation Learning.* [arXiv:2205.13147](https://arxiv.org/abs/2205.13147). NeurIPS.
- OpenAI, [*New embedding models and API updates*](https://openai.com/blog/new-embedding-models-and-api-updates) — the announcement of MRL-native `text-embedding-3-*` with the `dimensions` parameter.
- Google Cloud, [*Text embeddings API — output dimensionality*](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_parameters) — MRL truncation on Vertex.
- Nomic, [*nomic-embed-text-v1.5 model card*](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — documented MRL prefix lengths.
- Cohere, [*Int8 & Binary Embeddings — Store 32x More Data*](https://cohere.com/blog/int8-binary-embeddings) — the practitioner reference on binary embeddings + rescore.
- pgvector, [`pgvector/pgvector` README](https://github.com/pgvector/pgvector) — `halfvec`, `vector`, `bit`, `sparsevec` types and index families.
- Qdrant, [*Quantization docs*](https://qdrant.tech/documentation/guides/quantization/) — scalar, product, and binary quantization with rescore.
- Weaviate, [*Vector index compression*](https://weaviate.io/developers/weaviate/configuration/compression) — PQ, SQ, and binary compression in Weaviate.
- Milvus, [*Vector index types*](https://milvus.io/docs/index.md) — FLAT, IVF variants, HNSW, and binary indexes.
- FAISS, [*Indexes*](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) — scalar-quantizer and binary-index implementations you can benchmark locally.
