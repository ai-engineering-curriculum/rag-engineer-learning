# Chapter 5 — Retrieval Cost and Latency Optimization

## Why this chapter exists

Every retrieval system has three cost line items and three latency line items. Any team that has run one for more than a few months knows what they are and roughly what each contributes; teams that have not run one yet almost always underestimate one of them. This chapter is the cost-and-latency mental model plus the optimization levers, from cheapest to most-disruptive.

**Cost line items:**

1. **Embedding cost** — tokens sent to the embedding API (per doc at build/backfill time; per query at retrieval time).
2. **Vector-DB storage + compute** — bytes stored per million vectors, plus the DB's compute for queries.
3. **Reranker / generator cost** — reranker API tokens (or self-hosted GPU) per query; generator tokens per query.

**Latency line items:**

1. **Query-embedding** — the round-trip to the embedding API to turn the query into a vector.
2. **ANN search** — the vector-DB's search time; depends heavily on ANN parameters (mod-104 chapter 4).
3. **Reranker + generator** — the reranker's forward pass (cross-encoder or API) and, if in the same request, the generator's decode.

Every optimization in this chapter and the next moves one of these six numbers. The rule of engagement is: **measure first, then optimize**. Guessing which line item dominates is how teams waste a quarter tuning the wrong thing.

## The cost model — a per-query decomposition

Author this early in the project and keep it up to date. Illustrative shape:

```
Cost per retrieval request =
    (query_embedding_tokens × embedding_rate_per_token) +          # query embed
    (vector_db_query_amortized) +                                   # storage + compute
    (reranker_top_N × reranker_rate_per_pair)                       # rerank cost

Cost per retrieved doc at build/backfill =
    (chunk_tokens × embedding_rate_per_token) +                     # embed at ingest
    (vector_db_storage_amortized_per_vector)                         # storage per year
```

Where each rate is:

- **`embedding_rate_per_token`** — vendor-published (OpenAI `text-embedding-3-small` / `-3-large` [pricing](https://openai.com/api/pricing/), Voyage AI [pricing](https://docs.voyageai.com/docs/pricing), Cohere [pricing](https://cohere.com/pricing), Anthropic — no first-party embeddings as of writing). Self-hosted models replace this with amortized GPU-hour cost per million tokens; get it by benchmarking your specific model on your specific hardware.
- **`vector_db_query_amortized`** — total DB cost per month / total queries per month. Depends on the vendor (Pinecone charges [per pod / per RU](https://www.pinecone.io/pricing/); Weaviate [Cloud](https://weaviate.io/pricing) has serverless + dedicated tiers; Qdrant Cloud, Milvus/Zilliz, etc. all have their own shapes; self-hosted is CPU + RAM + disk amortized).
- **`vector_db_storage_amortized_per_vector`** — total storage cost per month / total vectors. Chapter 6 of mod-104 has the sizing math; the short version is that a 1024-dim float32 vector is 4 KB, plus payload; a million such vectors is ~4 GB before quantization.
- **`reranker_rate_per_pair`** — cross-encoder API rate (Cohere Rerank API [pricing](https://cohere.com/pricing), Voyage [rerank pricing](https://docs.voyageai.com/docs/pricing)), or self-hosted GPU cost per million rerank pairs.

Every retrieval PR should be able to update the cost model with the new tuple's numbers.

## The per-query cost dashboard

The cost model on its own is theory. The *dashboard* on top makes it actionable.

Instrument every retrieval request to emit:

- `embedding_tokens` (query embedding)
- `retriever_latency_ms`
- `retriever_top_n`
- `reranker_input_pairs` (if the reranker was called)
- `reranker_latency_ms`
- `cache_hits` (chapter 6 — count per tier)

Aggregate to per-day / per-tenant / per-endpoint. A Grafana dashboard with the four numbers (embedding cost, DB cost amortized, reranker cost, cache hit rate) per day is what makes cost regressions obvious. When the number moves 20%, someone should be paged; when the number climbs 3% a week for a month, someone should notice the trend.

Two things this dashboard is *not*: it is not the general LLM-cost dashboard (that is `ai-eval-engineer`'s territory, and it lives alongside this one), and it is not the vector-DB vendor's dashboard (which reports engine-side metrics, not per-request). Both should be linked from the retrieval dashboard for cross-reference.

## Embedding cost — the biggest optimization surface

Embedding cost dominates in most retrieval systems for two reasons:

1. **Every corpus doc gets embedded once** (at ingest) plus **once per backfill** (mod-104 chapter 8's tuple bumps). A model upgrade re-embeds the whole corpus.
2. **Every query gets embedded** at retrieval time. Volume × rate = a nontrivial monthly line item.

Levers, from cheapest to most-disruptive:

### Cheap: cache the query embedding

Chapter 6 goes into cache design. Short version: for repeated queries (which are very common in production — think "how do I reset my password"), cache the query embedding by query-text key. Hit rates of 20–60% are typical for user-facing surfaces.

### Cheap: pick a smaller embedding model when you can afford the recall trade

mod-103 selected the embedding model on a recall-vs-cost curve. Revisit it periodically. The rate difference between `text-embedding-3-small` and `text-embedding-3-large` (or between `bge-small-en` and `bge-large-en`) is roughly an order of magnitude in token cost. If the recall delta is smaller than your product needs, downshift.

### Medium: batch the embedding pass at build time

The embedding pass at build/backfill time should batch as high as the vendor supports (chapter 4). Wire in and monitor batch efficiency — a pass that runs many small requests can pay per-request overhead 10× larger than a well-batched pass.

### Medium: incremental, not full, embedding

Chapter 4's incremental-vs-full decision. Every full rebuild that could have been incremental is a proportional amount of embedding cost wasted. The retriever CI's tuple-change detector (chapter 3) is the discipline that prevents accidental full rebuilds.

### Disruptive: switch to a self-hosted embedding model

At sufficient volume, a self-hosted embedding model on your own GPU becomes cheaper than a hosted API. The threshold depends on many factors — model size, GPU utilization, personnel cost. Compare vendor per-token pricing to your amortized GPU cost per million tokens; the crossover is often somewhere between billions and hundreds of billions of tokens per month.

### Disruptive: quantize the embedding storage

Doesn't reduce embedding cost per se, but reduces the *storage* line item — see the vector-DB storage section.

## Vector-DB storage cost — quantization is the big lever

Vector-DB storage cost is a function of (vector count) × (bytes per vector) × (storage rate). The vector count is fixed by the corpus; the bytes per vector is what quantization moves.

### float32 vs. int8 vs. binary — the shape of the trade

A 1024-dim vector:

- **float32** — 4 KB per vector. Full precision. Baseline recall.
- **int8** ([scalar quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/)) — 1 KB per vector. ~4× smaller. Typical recall drop is small (single-digit-percent range on most benchmarks); measure on your judgment set before you ship.
- **PQ (product quantization)** — sub-KB per vector at 8× compression is common; 32×–64× compression is possible with more aggressive PQ settings. Recall drop is larger and more workload-dependent than scalar quantization.
- **binary quantization** (1 bit per dim) — 128 bytes per vector for 1024-dim. ~32× smaller than float32. Recall drop can be substantial; often paired with a re-ranking pass over the un-quantized vectors for top-K.

References for the concrete tradeoffs on real models: Cohere published a [binary + int8 benchmark](https://cohere.com/blog/int8-binary-embeddings) with their embed-v3 models; the Pinecone [scalar quantization docs](https://docs.pinecone.io/docs/vector-quantization) discuss their default int8 path; Weaviate has [PQ / SQ / BQ docs](https://weaviate.io/developers/weaviate/configuration/compression). Every vendor supports at least one form of quantization; the specific config knob differs.

**How to choose the quantization level.** Run the mod-109 judgment set at each level:

| Config | Storage / M vectors | Query latency | recall@10 | nDCG@10 |
|---|---|---|---|---|
| float32 (baseline) | *your baseline number* | *your baseline* | *baseline* | *baseline* |
| int8 (scalar) | *~4× less* | *similar* | *measure* | *measure* |
| PQ-16 | *~32× less* | *slightly faster* | *measure* | *measure* |
| binary | *~32× less* | *fastest* | *measure* | *measure* |

Pick the smallest that stays within your recall budget. Publish the table alongside the tuple change PR (chapter 3's CI catches the regression if you go too far).

### Two-stage retrieval with quantization

A common shape: quantize for the primary ANN search (fast, cheap), then re-rank the top-K by exact / higher-precision distance. This gives most of the storage / latency win with most of the recall preserved. Weaviate's [rescore mechanism](https://weaviate.io/developers/weaviate/configuration/compression#rescoring) and Pinecone's [rerank-with-original](https://docs.pinecone.io/docs/vector-quantization#reranking) are the vendor-specific instances of this pattern.

## ANN parameters as latency levers

mod-104 chapters 4 and 5 tuned the ANN parameters (HNSW `M`, `efConstruction`, `efSearch`; IVF-PQ `nlist`, `nprobe`) primarily against recall. This chapter is where those knobs re-appear as **latency** and **cost** levers.

- **`efSearch` (HNSW) / `nprobe` (IVF-PQ)** — per-request knobs. Higher = more recall, more compute per query. Lowering by 20% often cuts query latency by 15–30% at a small recall cost. Chapter 6's cache tiers amplify the win because a fraction of queries never hit the ANN at all.
- **`M` (HNSW) / `nlist` (IVF-PQ)** — build-time knobs. Move only via a full rebuild (chapter 4). Optimize once at build; do not re-tune casually.
- **Hardware** — every vector DB has a memory-vs-disk story. HNSW indexes typically require the graph in RAM; IVF-PQ is more disk-friendly (mod-104 chapter 5). Right-sizing the instance is the cheapest latency win you'll ever get.

The two-panel dashboard: **recall@k vs. `efSearch`** and **p95 latency vs. `efSearch`** for your workload — chapter 5 of mod-104 has the recipe. Keep this chart current; it is the primary reference when someone asks "can we get p95 latency down by 20 ms?"

## Latency line items — the round-trip breakdown

Every request goes through, roughly:

```
    client → (network) → API gateway → (network) → retriever service →
      → embedding API (~10-100 ms depending on vendor + region)
      → ANN search (~5-50 ms depending on config)
      → (optional) reranker (~50-500 ms for a cross-encoder over top-N)
      → generator (~200-2000 ms depending on model + streaming)
```

Rough shape only — measure on your setup. A few common wins:

- **Colocate embedding API + retriever region.** Cross-region embedding calls add 50-200 ms round-trip. Use the vendor's nearest region.
- **Cache the query embedding.** Chapter 6. When the cache hits, embedding-latency goes to zero.
- **Cap the reranker window.** mod-107 chapter 5 (funnel tuning): shrinking the rerank window from 50 → 20 cuts reranker latency proportionally, usually with a small recall cost.
- **Stream the generator.** Streaming responses (SSE) starts the first token to the user in a fraction of the total generation time. Not a retrieval concern per se, but often the largest wall-clock component in the full request.

## The three-way trade — cost vs. latency vs. quality

Every optimization moves you along a Pareto surface with three axes. Any move helps at most two; the third gets worse or stays flat.

- **Quantize aggressively.** Cost ↓, latency ↓, quality ↓.
- **Lower `efSearch`.** Cost ↓ (slightly, via less compute), latency ↓, quality ↓.
- **Raise reranker window.** Quality ↑, cost ↑, latency ↑.
- **Cache more.** Cost ↓, latency ↓, quality ⊚ (unchanged when cache is fresh; ↓ when stale — chapter 6's TTL is the defense).
- **Switch to a smaller embed model.** Cost ↓, latency ↓, quality ↓ (usually).
- **Switch to a bigger embed model.** Quality ↑, cost ↑, latency ↑.

The tuple you settle on is the one that fits your product's price + latency + quality budget. It is not fixed; it moves as pricing changes, as new models ship, and as the corpus + query mix drift.

## Concrete example — an optimization sweep

Assume the mod-109 judgment set + a tuple `(bge-large-en-v1.5, chunker-3, schema-2, snap-2026-08-01, hnsw-M32-ef200)` on a 1M-vector corpus. A cost + latency sweep looks like:

| Change | recall@50 | nDCG@10 | p95 latency | cost / month |
|---|---|---|---|---|
| baseline | *baseline* | *baseline* | *baseline* | *baseline* |
| add query-embedding cache | ⊚ | ⊚ | *lower* | *lower* |
| lower `efSearch` 200→100 | *slightly lower* | *slightly lower* | *lower* | *lower* |
| quantize to int8 (scalar) | *slightly lower* | *slightly lower* | *~same* | *storage lower* |
| lower rerank window 50→20 | *lower on tail queries* | *slightly lower* | *lower* | *lower* |
| downshift embed model bge-large → bge-base | *lower* | *lower* | *lower* | *significantly lower* |

Every row is one PR through the retriever CI (chapter 3). Every row has a per-query record so the significance test (mod-109 chapter 4) can tell you whether the "lower" is significant. Every row updates the cost model. Ship the ones whose deltas are inside your product's budget.

## Two failures the discipline defends against

### Failure A: "our embedding bill 4×'d and no one knows why"

A retriever code change caused *every* query to invalidate the cache (a new query-normalization added a whitespace that broke exact-match lookups). Cache hit rate fell from 40% to 3%. Embedding-API cost quadrupled overnight. Nobody had a per-day cache-hit dashboard, so the correlation was invisible.

**Defense.** The cost dashboard's cache-hit-rate panel with an alert on ≥ 20% relative drop. Add that alert on the first day.

### Failure B: "we quantized and shipped and users started complaining about wrong answers"

An `int8` scalar quantization shipped after a two-hour eval on a 30-query smoke set. The full 120-query judgment set (from mod-109) was skipped because "quantization is well-known to not hurt much." On the production query mix, the recall drop was concentrated in one specific query type (numerically-dense queries where the quantization broke the distance ordering). Users noticed within a week.

**Defense.** Every cost / latency PR runs the same CI gate as any other retriever PR (chapter 3). No skipping. The eval time is a rounding error compared to the incident cost.

## What you should be able to do after this chapter

- Author a per-query cost model with the three cost line items (embed / DB / rerank) and derive an amortized cost per query.
- Instrument a retrieval service to emit the six per-request fields (embedding tokens, retriever latency, reranker pairs, reranker latency, cache hits, latency breakdown) and build a Grafana / dashboard panel per line item.
- Name the four quantization levels (float32, int8, PQ, binary) and their approximate storage / recall trade-off shapes without inventing specific numbers.
- Identify the three-way trade among cost, latency, and quality; predict which axes an optimization will move.
- Point at mod-104 chapter 5's ANN-latency curve and mod-107 chapter 5's funnel curve as the two upstream calibration artifacts this chapter re-uses.
- Recognize when a change requires a full mod-109 eval vs. when a smoke set is enough (short answer: always full).

## Further reading

- OpenAI [API pricing](https://openai.com/api/pricing/) — for the embedding token rates the cost model reads.
- Cohere [pricing](https://cohere.com/pricing) — for the alternative embedding and rerank rates.
- Voyage AI [pricing](https://docs.voyageai.com/docs/pricing) — the third mainstream embedding vendor.
- Pinecone: [*Vector Quantization*](https://docs.pinecone.io/docs/vector-quantization) — the vendor-side documentation for the default int8 path and the two-stage rerank pattern.
- Weaviate: [*Compression*](https://weaviate.io/developers/weaviate/configuration/compression) — the PQ / SQ / BQ knobs and rescore mechanism.
- Qdrant: [*Quantization*](https://qdrant.tech/documentation/guides/quantization/) — Qdrant's scalar / product / binary quantization docs.
- Cohere blog: [*int8 & binary embeddings*](https://cohere.com/blog/int8-binary-embeddings) — a vendor-published benchmark of the storage-vs-recall trade at the quantization levels this chapter names.
- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). [DOI:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57) — the primary source for PQ.
- Guo, R., Sun, P., Lindgren, E., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN).* ICML. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396) — ScaNN as an alternate quantization / retrieval path.
- Malkov, Y. A., & Yashunin, D. A. (2020). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 42(4). [DOI:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473) — the HNSW paper; the latency-vs-recall curve mod-104 chapter 4 derives from.
- Kim, S., et al. (2023). *SqueezeLLM: Dense-and-Sparse Quantization.* [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) — an example of the quantization literature broadly; the concepts translate to embedding storage.
- Google SRE Book, Ch. 25 (*Data-Processing Pipelines*). [Full text](https://sre.google/sre-book/data-processing-pipelines/) — the general perspective this chapter's cost / latency instrumentation specializes.
