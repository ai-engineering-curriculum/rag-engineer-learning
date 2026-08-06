# Chapter 6 — Latency, Cost, Dimensionality, and Quality-per-Dollar

## Why this chapter exists

Chapters 4 and 5 introduced two dozen embedding options. Chapters 2 and 3 gave you the quality lens (MTEB, BEIR, your own domain-fit run). Missing so far: the *cost lens*. Real production choices are made on a two- or three-axis frontier — quality on one axis, latency on another, dollar cost on the third — and picking the model at the top of one axis is almost never right on the others.

This chapter is the framework you use to draw that frontier honestly. It covers the four cost surfaces that matter (per-token embedding cost, storage cost, query-time cost, retraining/re-embedding cost), the latency budgets your model choice slots into, and the "quality-per-dollar" ratio that lets you compare a hosted API to a self-hosted GPU rental without waving hands.

## The four cost surfaces

Every embedding model choice imposes cost in four places. Miss one and your comparison is wrong.

### 1. Offline embed cost (corpus → vectors)

This is a one-shot cost per corpus rebuild. Rebuilds happen (mod-111 sizes the frequency) — every schema change, every chunker version bump, every embedding-model swap.

**Hosted formula:**

```
cost_offline = tokens_per_chunk × n_chunks × price_per_million_tokens / 1e6
```

Concrete: 1M chunks × 300 tokens/chunk × $0.020 / 1M tokens = **$6** for a small OpenAI embed. Same corpus with a Cohere-v3 embed at $0.10 / 1M tokens = **$30**. Same corpus with `text-embedding-3-large` at $0.13 / 1M tokens = **$39**. These are order-of-magnitude anchors; check current [OpenAI pricing](https://openai.com/api/pricing/), [Cohere pricing](https://cohere.com/pricing), [Voyage pricing](https://www.voyageai.com/pricing), and [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) before quoting them.

**Self-hosted formula:**

```
cost_offline = (corpus_tokens / throughput_tokens_per_sec) × gpu_hourly_rate
```

Concrete: 300M corpus tokens; `bge-large-en-v1.5` at ~50k tokens/sec on an A10G (~$0.75/hr on AWS spot / g5.xlarge); ≈ 100 minutes ≈ **$1.25**. Same corpus with `e5-mistral-7b-instruct` on an A100 (~$3/hr spot for a single 40GB A100); throughput ~2k tokens/sec, ≈ 40 hours ≈ **$120**. Small open embedders win offline embed cost decisively on large corpora; 7B-parameter embedders undo that decisively. Measure throughput on *your* GPU / *your* model with TEI or vLLM before believing anyone's numbers, including these.

### 2. Storage cost (vectors → index → disk / memory)

Storage compounds forever — you pay for every vector for as long as the index is alive, and it dominates cost on large corpora.

**Per-chunk raw storage:**

```
bytes_per_vector = dim × bytes_per_component
```

- `dim=384, float32` → 1,536 bytes.
- `dim=1024, float32` → 4,096 bytes.
- `dim=3072, float32` → 12,288 bytes.
- `dim=1024, int8`   → 1,024 bytes (4× reduction).
- `dim=1024, binary` → 128 bytes (32× reduction).

At 1M chunks:

- 384-dim float32 = **1.5 GB**.
- 1024-dim float32 = **4.1 GB**.
- 3072-dim float32 = **12.3 GB**.

HNSW roughly adds another ~2× on top of the raw vectors for graph edges. IVF-PQ compresses well below raw. mod-104 owns the ANN-side arithmetic; the point here is that **`dim` is a multiplier on every storage cost downstream**. Chapter 8 shows how MRL and quantization cut this cost by 2–32× with modest quality cost.

At **100M** chunks (a realistic enterprise scale) the arithmetic scales up: 1024-dim float32 = **410 GB** of raw vectors before index overhead. Ram-resident HNSW at that size costs real money in EC2 memory instances or in managed-vector-DB (Pinecone, Weaviate Cloud, Qdrant Cloud) storage tiers. This is where quantization is not optional.

### 3. Query-time embed cost (query → vector, per request)

Every user query encodes to one vector before you hit the ANN index. That cost accrues per-query and pays out over the whole lifetime of the deployed service — so it dominates when your query volume is large.

**Hosted formula:**

```
cost_query_embed = tokens_per_query × price_per_million_tokens / 1e6
```

At $0.02 / 1M tokens and 30 tokens per query: **$0.0000006 per query** — trivial. At 10M queries/month: **$6/month**. At 1B queries/month: **$600/month**.

**Self-hosted formula:**

```
cost_query_embed = gpu_hourly_rate / (queries_per_hour)
```

A self-hosted small embedder on a GPU that costs $0.75/hr and can serve 100 queries/second → ~$0.75 / 360,000 queries/hr = **$0.0000021 / query** — of similar order to the hosted per-query price, but with the upfront ops burden. Below a few M queries/month, the hosted API is almost always cheaper; above that, the self-hosted GPU wins if you can keep it busy.

**Query-time latency (not cost, but same axis).** For a hosted API the RTT + provider first-token latency dominates: OpenAI / Cohere / Voyage typically 30–150 ms for a small batch depending on batch size and region. A local `sentence-transformers` or TEI call on a warm GPU is single-digit-ms — often ~5 ms for a small BGE. If your product has a p95 latency budget of 200 ms and the embedding call is 100 ms of it, that is half the budget spent before ANN — self-hosting the query encoder can be worth it for latency alone.

### 4. Re-embed cost (model swap, chunker swap, MRL truncation change)

The one cost engineers under-count. Every time you swap embedding model, or bump chunker version, or change the MRL truncation dim, you re-run **offline embed** (surface 1) on the whole corpus. If your corpus is 100M chunks and you switch from `text-embedding-3-small` to `text-embedding-3-large`, you pay $6 → $39 = **$33 delta**, plus the ANN rebuild cost, plus the dual-index cutover cost (mod-111).

The mitigation: **decouple the coupling** where you can. MRL (chapter 8) lets you change the *served dim* without re-embedding. Chunker versioning (mod-102 chapter 7) lets you re-embed only the changed chunks. But swapping the model itself is always a full re-embed. Choose deliberately; do not swap on a whim.

## The latency budget

Your product has one end-to-end latency budget (typically 200–1000 ms for interactive chat, more for backfill jobs). Embedding is one line in the budget:

```
p95_end_to_end
  = query_embed_ms
  + ann_search_ms                     (mod-104)
  + rerank_ms (K candidates)          (mod-107)
  + context_assembly_ms
  + llm_generation_ms                 (owned by mod-108, but reserves the tail of the budget)
```

Order-of-magnitude anchors:

| Stage | Typical p95 (well-tuned) |
|---|---|
| Query embed — hosted API (small batch) | 50–150 ms |
| Query embed — self-hosted small BGE (warm GPU) | 3–10 ms |
| Query embed — self-hosted 7B (warm A100) | 30–80 ms |
| ANN search — HNSW at 1M vectors, `efSearch=100` | 5–20 ms |
| Rerank — cross-encoder MiniLM on top-50 | 30–100 ms |
| Rerank — hosted (Cohere / Voyage / Jina) on top-50 | 100–300 ms |
| Rerank — LLM-as-reranker | 500 ms+ |
| LLM generation (first token) | 200–1000+ ms |

The embedding call is a small line on the budget once you get past the naive "the whole latency is inference" framing. But it is *not zero*, and choosing a hosted embedder for the query path adds ~100 ms of provider RTT that a self-hosted small model does not. On a chat product with an aggressive latency SLO, that is a real design trade.

## Quality-per-dollar — how to actually rank

The comparison you want to make is not "model A costs $X and gets quality Y." It is "model A gets quality-per-dollar ρ_A, model B gets ρ_B, and the difference is worth spending Z engineering hours on."

The naive formula:

```
ρ = nDCG@10 / cost_per_query_all_in
```

But that ranks a slightly-cheaper mediocre model above an expensive great one, which is not what you want. In practice the framework is:

1. **Set a quality floor.** "We will not deploy an embedder that gets <X nDCG@10 on our domain-fit benchmark." Chapter 7 makes this concrete. Below the floor, cost is irrelevant.
2. **Above the floor, pick the cheapest.** Cost = all-in per-query cost + amortized offline cost (over the expected corpus lifetime and re-embed frequency).
3. **Only pay for quality above the floor if the incremental quality is measurable and business-relevant.** A 0.02 nDCG lift for a 3× cost is almost never worth it; a 0.10 lift for a 3× cost frequently is. mod-109 covers paired significance testing on the eval delta.

A worked comparison table (fictional but shape-correct — plug in your own numbers):

| Model | Dim | nDCG@10 (domain) | Offline embed $/1M chunks | Query embed p95 (ms) | Query embed $/1M queries | Storage $/M chunks/month |
|---|---|---|---|---|---|---|
| `bge-small-en-v1.5` (self-hosted) | 384 | 0.58 | ~$0.30 | 5 | ~$0.06 (GPU amortized) | $0.05 (float32) |
| `text-embedding-3-small` (hosted) | 512 (MRL) | 0.60 | ~$6 | 80 | $20 | $0.07 (float32) |
| `text-embedding-3-large` (hosted) | 1024 (MRL) | 0.63 | ~$39 | 120 | $130 | $0.13 (float32) |
| `voyage-3` (hosted) | 1024 | 0.65 | ~$25 | 100 | $80 | $0.13 (float32) |
| `bge-large-en-v1.5` (self-hosted) | 1024 | 0.61 | ~$1.50 | 10 | ~$0.06 | $0.13 (float32) |
| `gte-Qwen2-7B-instruct` (self-hosted A100) | 3584 | 0.68 | ~$120 | 80 | ~$0.30 | $0.46 (float32) |

If your floor is 0.60 nDCG:

- `bge-small` fails the floor.
- `text-embedding-3-small`, `bge-large`, `text-embedding-3-large`, `voyage-3`, `gte-Qwen2-7B` all pass.
- Amongst the survivors, `bge-large` self-hosted is the cheapest (~$1.50 offline, ~$0.06 query, ~$0.13 storage), and is 0.02 nDCG below `voyage-3`.
- Question becomes: is 0.02 nDCG worth ~$25 offline + ~$80/M queries + slightly more storage? On a 100M-query/month product, that is ~$8k/month of API cost for a 0.02 nDCG lift. On a 10k-query/month internal tool, that is $0.80/month for 0.02 nDCG lift — obviously worth it. Cost/benefit is scale-dependent.

The table above is what a defensible embedding-choice document looks like. Chapter 7 gives you the domain-fit run that fills in the `nDCG@10` column honestly.

## Latency and throughput at index time

For offline embed jobs, throughput matters more than per-call latency. The knobs:

- **Batch size.** All open embedders benefit from batches of 32–512 depending on GPU memory. Hosted APIs cap at their own batch limit (OpenAI: 2048 inputs per call; Cohere: 96; Voyage: 128; Google: 250) — pack every call to the cap unless individual inputs are huge.
- **Concurrency.** Hosted APIs will typically accept 5–50 concurrent connections before rate limiting. Pipeline your embed job to keep the provider saturated without triggering `429`s. Self-hosted TEI can drive an A10G to full utilization at ~4 concurrent client threads with batch=64.
- **Long inputs are slow.** Encoding a 8k-token chunk costs ~16× as much as a 512-token chunk on a transformer (quadratic attention). If throughput matters, keep chunks short; if quality on long chunks matters, budget for the throughput hit.

## Latency and throughput at query time

- **Warm-up matters.** The first call after service start pays cold-start cost (~2–10s on a container startup). Prewarm every replica.
- **Query embedding cache.** Deduplicate queries at the service edge. Anecdotally, in customer-facing search a small percentage of queries make up the majority of volume; caching them at 1000-entry LRU can halve embedding load.
- **Batched query embedding when possible.** If your product supports "search across N sub-queries" (mod-106 covers multi-query fanout), batch those N encodings into one call. Hosted APIs charge per token, not per call — batching does not lower cost, but it dramatically lowers latency.
- **Streaming responses do not help embedding latency.** Embedding is not a token-by-token operation; you pay the whole call latency before you get any vector. Do not architect the product path expecting to stream from the embedder.

## Dimensionality as a first-class cost knob

Dim shows up in all four cost surfaces:

- **Offline embed:** unchanged (same tokens; wider vectors).
- **Storage:** linear in `dim`.
- **Query embed:** unchanged.
- **ANN search:** roughly linear in `dim` per distance computation, and HNSW graph memory is linear too.
- **Fusion / arithmetic downstream:** linear in `dim`.

Halving the dim (from 1024 → 512 via MRL truncation) halves storage and halves the per-comparison distance cost in ANN. On a 100M-vector index that is a real number in dollars per month. Chapter 8 makes the trade-off concrete: MRL truncation to 512 dims on OpenAI `text-embedding-3-*` typically retains most retrieval quality — a small quality cost for a 2× storage saving is a straightforward call at large scale.

Not every embedder supports MRL truncation gracefully. BGE / E5 / GTE were mostly not trained with MRL; truncating them arbitrarily degrades quality much faster. Nomic, OpenAI text-embedding-3, Google text-embedding-005 were. Chapter 8 lists the current MRL-native models.

## A working template: the cost-latency-quality table

For every embedding candidate you shortlist in chapter 7, fill out this table:

```markdown
| Model | dim (served) | nDCG@10 (domain) | Offline embed $/M chunks | Query embed p95 (ms) | Query embed $/M queries | Storage GB per 1M chunks | Notes |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | (license, per-role adapter, MRL support) |
```

Populate every column. If you cannot fill a column, note *why* — an unknown latency because you have not benchmarked on your GPU is a real gap that your architect will call you on. Chapter 7's exercise makes the domain-fit `nDCG@10` numbers a deliverable; this template is where you put them alongside the cost model.

## What is NOT in this chapter (and where it goes)

- **Actually running the sweep across models.** Chapter 7 (design) and its exercise.
- **ANN cost tuning (HNSW / IVF-PQ trade-offs).** mod-104.
- **Reranker cost (cross-encoder / hosted rerank / LLM-as-reranker).** mod-107.
- **Ingestion cost.** mod-102.
- **Ongoing MLOps cost of running the retrieval service.** mod-111.

## What you should be able to do after this chapter

- Draw the four cost surfaces (offline embed, storage, query embed, re-embed) and give a formula for each.
- Compute a rough monthly all-in cost for embedding a corpus of size N at query volume Q for both a hosted and a self-hosted candidate.
- Recognize that `dim` is a multiplier on storage and ANN cost, not a free knob.
- State the "quality floor first, cost second" ranking rule for choosing between candidates.
- Fill in the cost-latency-quality table for every model in chapter 7's shortlist.
- Point out the latency vs. cost trade of running the query embedder locally instead of hitting a hosted API on a chat product with an aggressive p95.

## Further reading

- OpenAI, [*Pricing*](https://openai.com/api/pricing/); Cohere, [*Pricing*](https://cohere.com/pricing); Voyage AI, [*Pricing*](https://www.voyageai.com/pricing); Google Cloud, [*Vertex AI generative-AI pricing*](https://cloud.google.com/vertex-ai/generative-ai/pricing); AWS, [*Bedrock pricing*](https://aws.amazon.com/bedrock/pricing/). Bookmark, do not memorize — these move.
- Hugging Face, [*Text Embeddings Inference README*](https://github.com/huggingface/text-embeddings-inference#supported-models) — throughput numbers per model / GPU that you can use as a starting point (and should re-measure).
- ANN Benchmarks, [`erikbern/ann-benchmarks`](https://github.com/erikbern/ann-benchmarks) — the reference recall vs. QPS methodology, extended to embedding-cost analysis in mod-104.
- Muennighoff et al., *MTEB* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — the quality axis half of the trade-off.
- Cormack, G. V., & Grossman, M. R. (2017). *Navigating Imprecision in Relevance Assessments on the Road to Total Recall.* SIGIR — one of the citations for why quality numbers below noise should not drive cost decisions.
