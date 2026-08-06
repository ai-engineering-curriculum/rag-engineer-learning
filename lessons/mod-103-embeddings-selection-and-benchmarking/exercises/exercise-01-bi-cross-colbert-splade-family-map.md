# Exercise 01 — Bi-encoder, Cross-encoder, ColBERT, SPLADE: One Query, Four Shapes

**Estimated effort:** ~2 hours.

## Objective

Instantiate one representative model from each of the four families in chapter 1 (bi-encoder, cross-encoder, ColBERT / late-interaction, SPLADE) and run the same query against a tiny shared corpus. The output is not a benchmark — it is muscle memory. You should leave this exercise able to *feel* how each family responds to the same input differently: what a bi-encoder returns, what changes when a cross-encoder rescores it, what a per-token MaxSim gives you that a single dot product cannot, and what a sparse-neural expansion writes out that BM25 would not.

The point is that when chapter 6, 7, and mod-107 talk about "we could put a cross-encoder here" or "SPLADE would help on this tail query," you have a concrete shape in your head — not just a name from the leaderboard.

## Prerequisites

- Chapter 1 (*Embedding Family Map*) read once.
- The mod-101 toolchain is up (Python env, `sentence-transformers`, `torch`).
- Dependencies: `sentence-transformers>=3`, `torch`, `transformers`, plus the two extras below for the ColBERT and SPLADE sides:
  - ColBERT: `pip install RAGatouille` (a maintained wrapper that handles the ColBERT model + PLAID index details).
  - SPLADE: `pip install torch scipy` — you will load the model with `transformers` directly and inspect the sparse output; no vector DB is required for this exercise.
- A modern CPU is enough. A GPU is faster (especially for the ColBERT index), not required.

## The tiny shared corpus

Create `data/tiny_corpus.jsonl` with these ten passages (or ten of your own choosing that cover a couple of distinct topics — the point is that some are obviously off-topic for your test queries, some are near-topic, and some are on-topic):

```
{"id": "d1", "text": "Kubernetes schedules containers across a cluster of nodes and is the reference orchestration platform."}
{"id": "d2", "text": "The Python asyncio event loop coordinates coroutines and handles awaitables in a single thread."}
{"id": "d3", "text": "HNSW is a graph-based approximate-nearest-neighbor index used in most modern vector databases."}
{"id": "d4", "text": "BM25 is a bag-of-words scoring function derived from probabilistic relevance theory and used in Elasticsearch."}
{"id": "d5", "text": "k8s is a common abbreviation for Kubernetes; both refer to the same container orchestration system."}
{"id": "d6", "text": "IVF-PQ combines inverted-file partitioning with product quantization for compact billion-scale vector indexes."}
{"id": "d7", "text": "Cross-encoders concatenate query and document, feed them through a transformer, and return a scalar relevance."}
{"id": "d8", "text": "SPLADE learns sparse term expansions over the BERT vocabulary and scores dot products on inverted indexes."}
{"id": "d9", "text": "Sentence-BERT extends BERT with siamese networks to produce fixed-dimension sentence embeddings for retrieval."}
{"id": "d10", "text": "The Postgres pgvector extension stores dense vectors and supports HNSW and IVFFlat indexes."}
```

Two test queries:

- Q1 (single-intent): `"how does the kubernetes scheduler pick a node?"`
- Q2 (jargon-heavy): `"what is ivf-pq and how does it beat plain hnsw on storage?"`

## Part A — Bi-encoder retrieval (30 min)

Use `sentence-transformers` with `BAAI/bge-small-en-v1.5`. Encode the corpus once, encode each query, return the top-3 by cosine similarity.

Requirements:

- Use `normalize_embeddings=True` and take the dot product.
- Print `(rank, doc_id, score, text[:80])` for each hit.
- Write your script as `scripts/family_bi_encoder.py`.

Expected shape: on Q1 you should see `d1`, `d5`, and something from the k8s/orchestration cluster. On Q2 you should see `d6` and probably `d3`.

## Part B — Cross-encoder reranking (30 min)

Take the top-5 from Part A and rescore with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Compare the rank changes.

Requirements:

- Use `sentence_transformers.CrossEncoder.predict(pairs)` where each pair is `(query, doc_text)`.
- Print a small table: `rank_before | rank_after | doc_id | bi_score | ce_score`.
- Write as `scripts/family_cross_encoder.py`.
- In `docs/family-notes.md`, write one sentence per query: did the cross-encoder move the ranking, and if so which doc moved where?

You are *not* expected to see the cross-encoder change every ranking — on a 10-doc corpus with clear topics, the bi-encoder is often already close to right. What you *are* expected to see is that the cross-encoder scores are on a different scale (they are logits, not cosines) and that the *deltas* between docs may be larger or smaller than the bi-encoder's — that scale mismatch is chapter 6's motivation for reciprocal-rank fusion (mod-106).

## Part C — ColBERT retrieval and MaxSim inspection (45 min)

Use RAGatouille to index the corpus and query it with `colbert-ir/colbertv2.0` (or `answerdotai/answerai-colbert-small-v1` for a faster CPU experience).

Requirements:

- Build an index from the ten documents.
- Run Q1 and Q2 and print the top-3.
- For Q1's #1 hit, dump the **per-query-token MaxSim contributions** — for each query token, print the document token that matched and its cosine similarity contribution. RAGatouille exposes the `search` result with token-level details; if not directly available in your version, load the underlying ColBERT checkpoint with `transformers`, encode query and doc separately, and compute the MaxSim by hand (~20 lines).
- Write as `scripts/family_colbert.py`.

In `docs/family-notes.md`, add: which query token contributed the largest MaxSim to the top hit? Does it match a term that BM25 would have caught? (This is the "late interaction gives you per-token attributability" point from chapter 1.)

## Part D — SPLADE sparse expansion (30 min)

Load `naver/splade-v3` (or `naver/splade-cocondenser-ensembledistil` if the v3 model is unavailable) with the `transformers` library. Encode Q2 as a sparse vector, print the **top-20 non-zero terms** with their learned weights.

Requirements:

- Use the model's MLM head to produce logits, take `log(1 + relu(logits))` and max-pool over tokens (that is the standard SPLADE aggregation).
- Print the top-20 `(term, weight)` pairs sorted by weight descending.
- Encode `d6` the same way and compute `dot(q_sparse, d_sparse)` — this is the score SPLADE would return.
- Compare against a plain `rank_bm25` score of the same `(query, doc)` pair: which terms does SPLADE expand into that BM25 does not have? (Look for `k8s` → `kubernetes` type expansions, or acronym expansions.)
- Write as `scripts/family_splade.py`.

In `docs/family-notes.md`, add: name at least two terms SPLADE added that were not literally in the query. This is the "learned expansion" property from chapter 1.

## Part E — The comparison note (15 min)

Complete `docs/family-notes.md` with a short "when I would reach for each family" paragraph in your own words, per family. Do not paraphrase the chapter. Write from what you observed in Parts A–D.

## Deliverables

- `data/tiny_corpus.jsonl` — the ten passages.
- `scripts/family_bi_encoder.py`, `scripts/family_cross_encoder.py`, `scripts/family_colbert.py`, `scripts/family_splade.py` — one per family.
- `docs/family-notes.md` — per-part notes + closing "when to reach for each" paragraph.
- Any generated indexes (`.ragatouille/`, etc.) can be `.gitignore`d.

## Acceptance criteria

- All four scripts run end-to-end on a laptop CPU (allow multi-minute runs for the ColBERT indexing step) and print their expected outputs.
- The bi-encoder and cross-encoder scripts return sensible top-3 for both queries (Q1 finds Kubernetes docs; Q2 finds `d6` and `d3`).
- The ColBERT script prints per-query-token MaxSim contributions for Q1's top hit.
- The SPLADE script prints ≥20 non-zero terms for Q2 with weights, and calls out ≥2 expansion terms not literally in the query.
- `docs/family-notes.md` has all five sections filled and the "when to reach for each" paragraph is written by you, not the chapter.

## Stretch goals

- Add BGE-M3 to Part A and use its single call to emit *dense + sparse + colbert-shape* outputs; compare the sparse output against the SPLADE output from Part D — do they agree on the expansion terms?
- Swap the cross-encoder to `BAAI/bge-reranker-large` and compare — is the rank order different?
- Replace RAGatouille with the raw ColBERT library and confirm you can index and query without the wrapper.
- Feed the SPLADE sparse vector into Elasticsearch (or OpenSearch) using the text-expansion query and score against a real inverted index. This is a preview of the mod-105 machinery.
