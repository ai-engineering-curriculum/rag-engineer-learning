# Exercise 05 — Quantization and Matryoshka: Storage / Latency / Quality Trade-Off

**Estimated effort:** ~3 hours.

## Objective

Chapter 8 argued three levers cut storage and ANN cost by 4–100× while costing only single-digit percent nDCG: **MRL truncation**, **int8 scalar quantization**, and **binary quantization + rescore**. This exercise makes that table concrete on your own corpus. You will pick one MRL-native embedder, embed once at native dim, then generate a grid of five to eight `(dim, quant)` variants from that single native embedding, and measure each variant on:

- retrieval quality (nDCG@10, Recall@10 against exercise-04's eval set — or SciFact if you skipped it),
- storage footprint per 1M chunks,
- ANN query-time latency (p50 / p95) on the same index shape.

The output is a **frontier plot and a defensible operating point** — the shape a reviewer expects when asking "why 512 dim int8 and not 1024 dim float32?"

## Prerequisites

- Chapter 8 read once. Chapter 6 (cost surfaces) helpful.
- Exercise-04 complete, or exercise-02's `TinyDomainRetrieval` task available as a stand-in. Reuse the same corpus + qrels; the exercise is about *transforming* the embeddings, not producing a new benchmark.
- Python deps: `sentence-transformers`, `numpy`, `faiss-cpu` (or `faiss-gpu`), `ir_measures`, `pandas`, `matplotlib`.
- One MRL-native embedder chosen from chapter 8's list:
  - Hosted: OpenAI `text-embedding-3-large` (native 3072, MRL-reducible), Google `text-embedding-005` (native 768, MRL-reducible), Amazon `titan-embed-text-v2:0` (native 1024, MRL-reducible).
  - Open: `nomic-ai/nomic-embed-text-v1.5` (native 768, supports 512/256/128/64), `Snowflake/snowflake-arctic-embed-l-v2.0` (per model card), `mixedbread-ai/mxbai-embed-large-v1` (per model card).
- A vector DB with quantization support, **or** FAISS locally. pgvector 0.7+ (`halfvec`, bit), Qdrant, Weaviate, Milvus all work. FAISS is easiest for an exercise; it exposes both float scalar-quantizer and binary indexes without a service.

## Part A — Embed once at native dim (30 min)

- Read the frozen `data/corpus/chunks.jsonl` from exercise-04 (or from mod-102 if you are running this earlier).
- Embed the whole corpus at the model's **native** dimension in float32.
- Persist to `data/vectors/native_float32.npy` with a matching `chunk_ids.tsv` so you can rebuild any variant without re-calling the API / GPU.
- Log the wall-clock and (for hosted) the token count you paid for.

Chapter 8 emphasized that embedding cost is unchanged by later dim / quant transforms. Prove that in your run — you should embed *once* here, and every other variant is a NumPy transform of this array.

## Part B — Build the grid of variants (45 min)

Generate at minimum the following variants from the native array. Persist each as `data/vectors/{name}.npy` and record the size on disk:

| Variant name | Transform | Bytes per vec (dim=1024 example) |
|---|---|---|
| `native_f32` | full float32 | 4096 |
| `f16` | cast to float16 | 2048 |
| `int8` | scalar quantization per-vector min/max → int8 | 1024 |
| `mrl_512_f32` | truncate to leading 512 dims, renormalize, float32 | 2048 |
| `mrl_256_f32` | truncate to leading 256 dims, renormalize, float32 | 1024 |
| `mrl_512_int8` | truncate to 512 then scalar-quantize | 512 |
| `mrl_256_binary` | truncate to 256 then sign() → packed bits | 32 |
| `native_binary` | sign() → packed bits at native dim | dim / 8 |

Rules:

- **Renormalize after truncation.** Chapter 8 was explicit: cosine similarity on unnormalized truncated vectors is wrong. `x = x[:k]; x = x / np.linalg.norm(x)`.
- **Scalar quantization** — per-vector `(min, max)` scale is enough for this exercise. Persist the (min, max) pair alongside each int8 vector so you can dequantize at query time. FAISS's `ScalarQuantizer` implements this if you want a reference.
- **Binary** — `np.packbits(vec > 0, axis=1)` (chapter 8's snippet).
- **Do not attempt MRL on a non-MRL model.** If the model in Part A does not appear in chapter 8's MRL-native list, either switch models or note in the doc that "MRL rows would degrade quickly on this base — see chapter 8." The exercise assumes MRL support; if you deliberately want to see the failure mode, add a non-MRL variant (`bge-large-en-v1.5` truncated to 256) as a stretch row.

Persist a manifest `docs/variants.md`:

```markdown
| Variant | Dim | Bytes/vec | Bytes for 1M chunks | Notes |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
```

## Part C — Build an ANN index per variant (45 min)

Use FAISS locally (simplest) or pgvector / Qdrant / Weaviate if you already have one from mod-101:

- Dense (float32, float16, int8, MRL-truncated): `IndexHNSWFlat(dim, M=32)` — FAISS's default HNSW.
- Binary (`native_binary`, `mrl_256_binary`): `IndexBinaryHNSW(dim, M=32)` with Hamming distance.

**Rescore step for binary.** Chapter 8 said binary alone loses 5–15% and rescore recovers most of it. For each binary variant, implement a two-stage retriever:

1. Binary top-K (K = 500 for a 1M-chunk corpus; K = 100 for smaller).
2. Fetch the corresponding **native float32** vectors from `data/vectors/native_float32.npy`.
3. Rescore with cosine similarity, keep the top-10.

For every dense variant, keep top-10 directly. This is the fair-fight setup: dense keeps top-10 in one pass; binary keeps top-10 after rescore. That is how production stacks compare.

## Part D — Measure quality (30 min)

Reuse exercise-04's queries + qrels. Embed queries once at native dim, then for each variant embed the query into the variant's shape (truncate + renormalize for MRL rows; quantize with the same scale for int8 rows; sign for binary rows). Retrieve top-10 (or top-500 + rescore for binary rows). Score with `ir_measures`:

```python
metrics = ir_measures.calc_aggregate(
    [ir_measures.nDCG @ 10, ir_measures.R @ 10, ir_measures.R @ 100],
    qrels,
    run_for_variant,
)
```

Persist to `docs/quantization-quality.md`:

```markdown
| Variant | nDCG@10 | Recall@10 | Recall@100 | Δ vs. native_f32 (nDCG@10) |
|---|---|---|---|---|
| native_f32 | ... | ... | ... | 0.000 |
| f16 | ... | ... | ... | ... |
| int8 | ... | ... | ... | ... |
| mrl_512_f32 | ... | ... | ... | ... |
| mrl_256_f32 | ... | ... | ... | ... |
| mrl_512_int8 | ... | ... | ... | ... |
| mrl_256_binary | ... | ... | ... | ... |
| native_binary | ... | ... | ... | ... |
```

## Part E — Measure ANN query latency (30 min)

For each variant's index, measure query-time latency:

- Warm the index (10 discarded queries).
- Run 500 queries. Record p50 and p95 latency, and mean QPS at single-thread concurrency.
- For binary variants, record latency for the **two-stage** retrieve+rescore path, not the pure binary path. The two-stage is what you would deploy.

Persist to `docs/quantization-latency.md`:

```markdown
| Variant | Index size on disk | ANN p50 (ms) | ANN p95 (ms) | Notes |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
```

Methodology pitfalls to name in a `## Notes` section:

- FAISS in-memory HNSW latency is dominated by cache locality; a 100M-vector index on the same hardware will not extrapolate linearly.
- Binary indexes are usually faster than float per-comparison but pay two-stage overhead; if the reranker fetches native vectors from disk you must include the fetch time.
- Query encoding is *not* the variant's cost — that is chapter 6 / exercise-03 territory. Do not include it in the "ANN latency" column here.

## Part F — Draw the frontier and pick an operating point (20 min)

Merge Parts D and E into `docs/frontier.md`:

- One scatter plot: X = storage bytes per vector, Y = nDCG@10, one point per variant. Save as `docs/frontier.png`.
- One table with (variant, storage, latency, quality) — the shape a reviewer will read.

Then write two closing paragraphs:

1. **Which variant would you ship for a 100M-chunk index at aggressive p95 latency?** Name it. Cite the numbers.
2. **Which variant would you ship for a 1M-chunk index where quality matters and cost is negligible?** Name it. Cite the numbers.

Chapter 8's decision tree is the frame — verify your two picks agree with the small-corpus / medium-corpus / large-corpus rules the chapter gave, or explain the disagreement.

Finally, extend the `index_version_key` you would write into mod-111's index metadata with the winning variant's parameters:

```
index_version_key = f"{model_id}@{revision}:dim=512:quant=int8"
```

Commit that string in `docs/index-key.md`. This is the artifact mod-111's cutover machinery reads.

## Deliverables

- `data/vectors/native_float32.npy` + one `.npy` per variant + `chunk_ids.tsv`.
- `scripts/build_variants.py`, `scripts/build_indexes.py`, `scripts/eval_quality.py`, `scripts/eval_latency.py`.
- `docs/variants.md`, `docs/quantization-quality.md`, `docs/quantization-latency.md`, `docs/frontier.md` (with `frontier.png`), `docs/index-key.md`.

## Acceptance criteria

- One `native_float32.npy` was produced by exactly one embed pass. Every other `.npy` is a NumPy transform of it — no additional embed calls.
- The grid has ≥6 variants including one binary variant and at least two MRL truncations.
- Every dense variant retrieves top-10 in a single ANN pass; every binary variant retrieves top-K then rescores with the native float vectors — not top-10 in one binary pass.
- The quality table names the native_float32 row as the reference and reports `Δ vs. native_f32` for every row. No row is missing from the table.
- The frontier plot is committed and shows all variants; the two chosen operating points are called out in text with their numbers.
- The `index_version_key` includes the served `dim` and `quant` fields.

## Stretch goals

- Repeat the same grid on a **non-MRL** model (e.g., `bge-large-en-v1.5`) and compare the truncation curves. Chapter 8 predicted the non-MRL model degrades much faster below 512 dim; verify.
- Add an **IVF-PQ** row (FAISS `IndexIVFPQ` at `m=64, nbits=8`) as a *different* quantization family. Chapter 8 warned about stacking MRL + IVF-PQ — verify by adding both an MRL-then-IVF-PQ and a float-then-IVF-PQ row.
- Push to a real vector DB — pgvector 0.7+ `halfvec` and bit columns, or Qdrant scalar / binary quantization — and confirm the on-disk sizes match your local FAISS numbers. Note any DB-specific overhead.
- If you used a hosted model with native int8 / binary output (Cohere Embed v3 with `embedding_types=["int8"]` / `["binary"]`), add rows for the **provider-quantized** vectors alongside your post-hoc quantized ones. Are they the same quality? Cohere argues the provider version was trained for quantization and should be better; measure it.
- Extend the frontier plot with an *equal-quality contour* — draw a horizontal line at (`native_f32` − 0.02 nDCG) and highlight every variant on or above it. Everything on that contour is "safe" quality-wise; the leftmost (smallest storage) point on it is the ship candidate.
