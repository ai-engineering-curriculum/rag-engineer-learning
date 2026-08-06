# Exercise 02 — Classical IR fundamentals refresher: inverted index, tf-idf, BM25

**Estimated effort:** ~2 hours. Half paper-and-pen, half short Python script.

## Objective

You will build a five-document inverted index by hand, compute tf-idf and BM25 scores for one query against that mini-corpus, and then reproduce the same numbers in a short Python script using an off-the-shelf BM25 implementation. The goal is not to become a BM25 implementer — it is to make chapter 2's mental model *sticky*, so that when mod-105 asks you to justify `k1`, `b`, or an analyzer choice, you feel the mechanics rather than trust the vendor default.

## Prerequisites

- Chapter 2 (*Classical IR Fundamentals*) — read once.
- Python with `pip install rank_bm25` (or `elasticsearch` if you have the local stack from exercise 05 running and prefer to use its BM25).
- Nothing more.

## The mini-corpus

Use exactly these five documents. Do not modify them — hand-scoring hinges on the exact tokenization.

```
D1: "vector databases store dense embeddings for retrieval"
D2: "inverted indexes store terms mapped to postings for retrieval"
D3: "reciprocal rank fusion combines lexical and dense retrieval"
D4: "bm25 is the classical scoring function for lexical retrieval"
D5: "cross encoder rerankers improve retrieval quality"
```

The query:

```
Q: "lexical retrieval"
```

## Part A — Build the inverted index by hand (30 min)

Tokenize by lowercase-and-split-on-whitespace. (Yes, a real analyzer would do more — stemming, stopword removal, punctuation. Skipping that keeps the arithmetic tractable.) For every term that appears in any document, write down the posting list as `[doc_id, tf]` pairs.

**Deliverable A:** a table with columns `term | df | posting list`. Verify that summing `tf` across your postings for a given term matches the term's total occurrences in the corpus.

## Part B — Compute tf-idf for the query on each document (30 min)

Using the classical tf-idf formula from chapter 2 with `idf(t) = log(N / df(t))`:

- Compute `idf` for each query term.
- For each document, compute `score(Q, d) = Σ_{t ∈ Q ∩ d} tf(t, d) · idf(t)`.
- Rank the five documents by tf-idf score.

**Deliverable B:** a table with columns `doc | tf-idf score | rank`. Show your work for D1 through D5.

## Part C — Compute BM25 for the same query on each document (30 min)

Use the BM25 formula from chapter 2 with defaults `k1 = 1.2`, `b = 0.75`. You will need:

- `N` (corpus size) and `df(t)` per term (from Part A).
- `|d|` per document (token count) and `avgdl` (average document length).
- Per-query-term, per-document `tf(t, d)` (from Part A).

Compute `score(Q, d)` for each document and rank.

Then:

- **Sensitivity check.** Recompute the ranking with `b = 0` (no length normalization). Note which documents' ranks change and explain why in one sentence.
- **Saturation check.** Suppose we duplicated the word `"retrieval"` five extra times in D1 so `tf("retrieval", D1)` = 6. Compute the BM25 delta and confirm the saturation you see matches chapter 2's description.

**Deliverable C:** three ranked tables — default BM25, `b = 0`, and the D1-spammed variant — plus a two-sentence commentary per variant.

## Part D — Reproduce the ranking in Python (30 min)

Write a short script (≤ 30 lines) that:

1. Encodes the five documents and the query.
2. Uses `rank_bm25.BM25Okapi` (or Elasticsearch BM25 if you prefer) to score.
3. Prints the ranked list with scores.

Compare against your hand-computed BM25 ranking from Part C.

The point is not that the *scores* match to four decimal places — different implementations use slightly different `idf` variants (Elasticsearch uses `log(1 + (N − df + 0.5) / (df + 0.5))`, `rank_bm25` uses the classical form). The point is that the *ranking* matches, and if it does not, you can point at exactly which formula difference explains it.

**Deliverable D:** the script (as `bm25_smoke.py`) plus a one-paragraph note comparing the machine ranking with your hand ranking.

## Acceptance criteria

- Deliverables A, B, C, D exist.
- Your hand-computed BM25 ranking (Part C, default) agrees with the `rank_bm25` (or Elasticsearch) ranking on the top-2 documents. If it does not, note which of `idf` formula, tokenization, or `avgdl` computation explains the discrepancy.
- Your sensitivity check (`b = 0`) and saturation check (D1 spammed) each identify one specific document whose position changes and give a one-sentence *why*.

## Stretch goals

- Add an analyzer step (stemming with `nltk.PorterStemmer`, stopword removal) and repeat Parts A–C. Notice which query/document pairs newly match and which stop matching. In one paragraph, defend an analyzer choice for a corpus that mixes prose and code identifiers.
- Read Elasticsearch's blog post [*Practical BM25*](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) and, using the same mini-corpus, sweep `k1 ∈ {0.5, 1.2, 2.0, 5.0}` and `b ∈ {0.0, 0.5, 0.75, 1.0}`. Chart the rank of the correct top document (D3 or D4) as a function of each. Explain the shapes you see using the formula, not the plot.
- Convert the five documents into a real Elasticsearch / OpenSearch index using the local stack from exercise 05 and confirm the ranking. Note any differences vs. `rank_bm25` and trace them to the `idf` variant Elasticsearch uses.
