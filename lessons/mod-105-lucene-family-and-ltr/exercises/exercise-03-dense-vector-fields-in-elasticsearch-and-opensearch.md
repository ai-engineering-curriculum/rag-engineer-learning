# Exercise 03 — Dense-Vector Fields in Elasticsearch and OpenSearch

**Estimated effort:** ~3 hours. Hands-on Elasticsearch **and** OpenSearch, plus a pre-computed embedding corpus and a small judgment set.

## Objective

Extend the lexical index from Exercise 02 with a dense-vector field, on both Elasticsearch and OpenSearch, and prove three things end-to-end:

1. Filtered ANN is **correct** (pre-filter, not post-filter) on the config you deploy.
2. `int8_hnsw` (or the OpenSearch equivalent scalar-quantized field) costs roughly a quarter of a full-precision field's storage with a small, measured recall delta.
3. A native hybrid retriever — Elasticsearch's `rrf` retriever and OpenSearch's `hybrid` query — beats the BM25-only baseline from Exercise 02 on your judgment set.

The point is to build the pattern muscle you use every time you extend a Lucene-family index with vectors: pick the field type, pick the quantization, verify filter semantics, and measure the hybrid retrieval win on an eval set before shipping.

## Prerequisites

- Chapter 4 read.
- Exercise 02 complete — you have a working lexical index, a corpus of 1,000–10,000 English documents, and a judgment set of ~20–50 (query, relevant-doc-ids) pairs.
- A running Elasticsearch 8.14+ single-node cluster and a running OpenSearch 2.13+ single-node cluster. Docker Compose files ship with both projects; running both side by side on a laptop is fine.
- An embedding model you can call locally or hosted — pick one from mod-103 you already trust (a 384-dim MiniLM, a 768-dim bge-base, or a hosted 1536-dim model). Same model for both clusters.
- Python with the `elasticsearch`, `opensearch-py`, and `sentence-transformers` (or your hosted-embedding SDK) libraries installed.
- A modest tenant / ACL flag on each doc for the filtered-ANN test — e.g., `tenant_id` set to one of `acme`, `globex`, `initech`, distributed so ~10–20% of docs belong to each tenant. Fabricate it if your corpus does not have one.

## Steps

**Step 1 — Embed the corpus once (~20 min).** Compute an embedding per document (a natural pick is title + first ~256 tokens of body concatenated) and store it in a local file (`embeddings.jsonl`) so both clusters ingest identical vectors. This isolates the vector-field-and-engine variables from the embedding variable.

**Step 2 — Elasticsearch: full-precision dense_vector (~20 min).**

Add a `body_embedding` field to your Exercise 02 mapping:

```json
"body_embedding": {
  "type": "dense_vector",
  "dims": <your model's dim>,
  "index": true,
  "similarity": "cosine",
  "index_options": { "type": "hnsw", "m": 16, "ef_construction": 200 }
}
```

Reindex. Capture the on-disk size of the shard (`_cat/allocation?v` or `_stats`).

Run a **pure kNN** query for each of your ~20 judgment-set queries. Record `recall@10` vs. the qrels.

**Step 3 — Elasticsearch: int8_hnsw quantization (~20 min).**

Create a second index (`support_articles_int8`) identical to the above except `"index_options": { "type": "int8_hnsw", "m": 16, "ef_construction": 200 }`. Reindex. Capture shard size again.

Re-run the same kNN queries. Record `recall@10`. You want to see:

- Roughly **~4× reduction** in the vector-field's on-disk footprint (do the math: full vector footprint ≈ `dims × 4 bytes × N_docs`; int8 ≈ `dims × 1 byte × N_docs`; the HNSW graph itself is unchanged).
- Recall loss under ~2 percentage points on your set. If you see more, note it in the writeup — it is a real property of your embedding model and corpus.

**Step 4 — Filtered-ANN correctness drill (~25 min).**

Pick a tenant that owns ~10% of the corpus. Run the query in two shapes:

- **A. Filter inside the `knn` clause** (Elasticsearch's pre-filter path, since 8.0):

  ```json
  { "knn": { "field": "body_embedding", "query_vector": [...], "k": 10,
             "num_candidates": 100,
             "filter": { "term": { "tenant_id": "acme" } } } }
  ```

- **B. Filter as a post-filter** on a `knn` clause without an internal filter (use `post_filter: {...}` at the top level).

Compare the **top-10** returned by each. Compute a brute-force ground truth for that tenant (a `script_score` over only that tenant's docs) and compute `recall@10` for A and B against it. Write down: at `k=10, num_candidates=100`, with a 10%-selective filter, which shape lost documents that the ground truth said should be there? By how much? Sweep `num_candidates` up to 1000 and note where B closes the gap.

The point of this drill is to produce evidence — on your data — for the correctness claim in chapter 4 and mod-104 chapter 7.

**Step 5 — OpenSearch: knn_vector, Lucene engine (~20 min).**

On the OpenSearch cluster, define a `knn_vector` field with the `lucene` engine:

```json
"body_embedding": {
  "type": "knn_vector",
  "dimension": <your model's dim>,
  "method": { "name": "hnsw", "space_type": "cosinesimil",
              "engine": "lucene",
              "parameters": { "m": 16, "ef_construction": 200 } }
}
```

Set `"settings": { "index.knn": true }` on the index. Ingest the same embeddings. Run the same 20 queries. Record `recall@10`.

**Step 6 — OpenSearch: knn_vector, FAISS engine with PQ (~25 min).**

Author a second OpenSearch index using the FAISS engine and a PQ encoder (`{"name": "pq", "parameters": {"m": <dims/8 or 96>, "code_size": 8}}`). Ingest. Record on-disk size and `recall@10`.

Expect **another order of magnitude** in vector-storage reduction vs. int8, and a larger recall hit (5–15 percentage points is normal at your corpus scale — PQ pays off at ~10⁸ vectors, not 10⁴). Do not "fix" this by tuning; the point is to feel the trade-off, not chase a good PQ config on a small corpus. Note it in the writeup as *"PQ is not the right pick at this scale; the observation confirms chapter 4's guidance."*

**Step 7 — Hybrid retrieval: Elasticsearch RRF (~20 min).**

Wire your best-config lexical query from Exercise 02 and your best-config `knn` query into a top-level `retriever` block using `rrf` (chapter 4 has the shape). Sweep `rank_window_size ∈ {50, 100, 200}` and `rank_constant ∈ {20, 60, 100}` — a 3×3 grid, 9 runs. Record `recall@10` and `nDCG@10` (0–4 eyeball grading is fine, mod-109 covers rigorous scoring).

Compare the winning RRF config against:

- Exercise 02's BM25-only best.
- This exercise's pure-kNN best.

The RRF config should beat both. If it doesn't, use `_explain` on the queries where it lost to diagnose. Common causes: the BM25 baseline is very strong on your corpus (RRF is helping tail queries but hurting head queries), or `num_candidates` is too low for the kNN arm.

**Step 8 — Hybrid retrieval: OpenSearch `hybrid` query (~15 min).**

On OpenSearch, wire an equivalent hybrid query using OpenSearch's [`hybrid` query](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) and a `normalization-processor` search pipeline (min-max or L2 normalization; try both). Report `recall@10` and `nDCG@10`. Do not expect it to match Elasticsearch RRF one-for-one — the fusion algorithm is genuinely different — but the delta vs. either pure arm should be positive.

**Step 9 — Sizing math and shard reality check (~15 min).**

Using the numbers you captured in steps 2, 3, and 6, extrapolate to a 10-million-document corpus with the same embedding model. Show:

- Full-precision vector-field footprint at 10M docs.
- `int8_hnsw` vector-field footprint at 10M docs.
- FAISS-PQ footprint at 10M docs.
- HNSW graph overhead you observed (as a percentage of the raw vector bytes).
- How many 30-GB shards each of the three configs would need (bare vector field only; the rest of the document unchanged).

This connects step 3's storage numbers to chapter 6's shard-sizing math, and it is the answer to the sizing question your platform team will ask before approving any dense-vector rollout.

**Step 10 — Write it up (~20 min).** Author `writeup.md` covering:

- The final mappings (Elasticsearch full-precision, Elasticsearch int8_hnsw, OpenSearch Lucene, OpenSearch FAISS-PQ) checked into your work directory.
- A results table: for each of the four dense configs, `recall@10`, on-disk footprint of the vector field, and (where measured) query `p50` latency at `num_candidates=100` over the 20-query set.
- A results table for the RRF and OpenSearch hybrid sweeps.
- The filtered-ANN evidence table from step 4.
- The sizing extrapolation table from step 9.
- One paragraph on each: "which config would I ship for a 100K-doc corpus?", "which for 10M?", "which for 500M?" — cite your numbers.

## Acceptance criteria

- Both clusters (Elasticsearch 8.14+ and OpenSearch 2.13+) hold the corpus with a working dense-vector field and pass a basic kNN query.
- At least four dense configurations measured: ES full-precision, ES int8, OS Lucene, OS FAISS-PQ. Each row includes on-disk footprint and `recall@10`.
- Filtered-ANN evidence table shows the recall gap between pre-filter (A) and post-filter (B) at `k=10, num_candidates=100`, and the `num_candidates` at which B closes the gap.
- Hybrid retrieval evaluated on both engines. At least one hybrid config beats both the BM25-only baseline and the pure-kNN baseline on `recall@10`. If it does not, the writeup diagnoses why with `_explain` output on the losing queries.
- Sizing table extrapolates all three storage configs to 10M docs and derives shard counts against the 30-GB target from chapter 6.
- No unverified vendor performance claims in the writeup — every latency, footprint, and recall number is one you measured on your own hardware.

## Stretch goals

- **Add `bbq_hnsw` (binary quantization) on Elasticsearch 8.14+.** Measure the footprint (should be ~1/32 of full precision) and the recall loss. Add a full-precision rescore stage (`rescore_vector` in ES 8.16+ or a `script_score` rescore) and measure the recovered recall.
- **Vespa comparison.** If you have Vespa running (Docker image is straightforward), define an equivalent `tensor<float>(x[dims])` field with `distance-metric: angular` and an HNSW attribute; author a rank profile that computes `bm25(title) * 0.5 + closeness(field, body_embedding) * 0.5`; run the same 20-query eval. Report where Vespa's ranking expression wins or loses vs. RRF and the OpenSearch hybrid on your corpus.
- **Selective-filter cliff.** Redo step 4 at three filter selectivities: 50% of docs, 10% of docs, 1% of docs. Plot `recall@10` for pre-filter vs. post-filter as filter selectivity drops. This is the practical evidence for mod-104 chapter 7's "post-filter fails hardest on the most selective filters" claim.
- **Multi-tenant fairness stretch.** Verify that pre-filter results for each tenant are correct *and roughly equal-quality* — sometimes a badly-composed graph traversal returns fine results for the majority tenant and poor results for the smallest tenant. Group `recall@10` by tenant and flag any per-tenant gap > 10 percentage points; sketch why.

## Notes

- Do not tune HNSW's `m` or `ef_construction` in this exercise. That is mod-104's chapter. Use `m: 16, ef_construction: 200` on every engine and every config so your comparisons are apples-to-apples.
- If your embeddings are 1536-dim (OpenAI `text-embedding-3-small` sized) and your Solr cluster is on defaults, you will hit the [`DenseVectorField` 1024-dim default cap](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html) — this exercise skips Solr for that reason; the pattern extends but the config is fiddlier.
- Two clusters running side-by-side on a laptop each need heap headroom. If either OOMs during reindex, reduce shard count to 1 and `-Xmx` per cluster to ~2 GB; the corpus is tiny.
- Every latency number in this exercise is single-node, single-user, laptop-scale — treat it as directional, not production-representative. The pattern muscle you build is the point, not the numbers.
