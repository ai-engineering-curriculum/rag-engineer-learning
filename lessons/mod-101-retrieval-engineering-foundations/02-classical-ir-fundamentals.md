# Chapter 2 — Classical IR Fundamentals: The Mental Model Every Retrieval Decision Hangs Off

## Why this chapter exists

The vector-database vendors would like you to believe modern retrieval began with `text-embedding-ada-002`. It did not. Every design choice a retrieval engineer makes — chunk size, top-*k*, weighted-sum vs. reciprocal-rank fusion, why we care about pre-filter vs. post-filter, why BM25 remains the strong baseline hybrid stacks fuse against — descends from classical information-retrieval work formalized in the 1970s–1990s and taught in Manning-Raghavan-Schütze's *Introduction to Information Retrieval* (Cambridge, 2008; [free HTML edition](https://nlp.stanford.edu/IR-book/)). This chapter refreshes the parts you will reach for weekly. If any of it is genuinely new, work through MRS chapters 1–8 before mod-102.

## The inverted index

An **inverted index** maps each term to a **posting list** — the set of documents containing that term, usually with per-document positional and frequency information.

```
term            posting list
────────────────────────────────────────────────
"retrieval"  →  [(doc_17, tf=3, positions=[41, 89, 402]),
                 (doc_92, tf=1, positions=[6]),
                 ...]
"vector"     →  [(doc_2, tf=5, ...), (doc_17, tf=2, ...), ...]
"index"      →  [(doc_5, tf=4, ...), (doc_17, tf=6, ...), ...]
```

A query like `"retrieval AND vector AND index"` becomes an intersection of the three posting lists — a linear-in-list-length scan, not a linear-in-corpus scan. The inverted index is why lexical search is fast on billion-document corpora and why Lucene-family engines (Elasticsearch, OpenSearch, Solr, Vespa) still dominate real production stacks. Chapters 1–5 of MRS build this data structure from scratch — read them if the compression tricks (variable-byte, front-coding, skip pointers) are new.

Two implications you will reach for every day:

1. **Analyzer choice is not a footnote.** How you tokenize, lowercase, stem, remove stopwords, expand synonyms, and handle CJK/agglutinative languages determines what ends up in the term dictionary. Two engineers indexing the same corpus with different analyzers will get different recall on the same query.
2. **A posting list is a filter, not just a score input.** Metadata fields indexed alongside the text (`tenant_id`, `permission_scope`, `updated_at`) let the engine intersect a filter into the retrieval before scoring — the correctness-shape you will see again as **pre-filter** in ANN systems in mod-104.

## Tf-idf

**Tf-idf** scores a `(term, document)` pair by two intuitions:

- Terms that appear often in *this* document are more descriptive of it — **term frequency** (`tf`).
- Terms that appear in *many* documents are less discriminating — **inverse document frequency** (`idf`).

The classic form (Salton & McGill, 1983) for a query *q* and document *d* is:

```
score(q, d) = Σ_{t ∈ q ∩ d} tf(t, d) · idf(t)

with       idf(t) = log( N / df(t) )    (N = corpus size, df(t) = # docs containing t)
```

Tf-idf is the ancestor of every scoring function since. Weights the same term differently depending on how rare it is across the corpus — the reason `"and"` contributes nothing and `"HNSW"` contributes a lot.

## BM25

Tf-idf's flaws are visible once corpora contain long documents: raw `tf` grows without bound, and a single-page ad brochure that spams the query term outranks a substantive report that mentions it once. Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* ([DOI:10.1561/1500000019](https://dl.acm.org/doi/10.1561/1500000019)), formalized **Okapi BM25** to fix both:

```
                    ┌            (k1 + 1) · tf(t, d)                    ┐
score(q, d) = Σ idf(t) · │ ─────────────────────────────────────────────── │
              t ∈ q ∩ d  │  k1 · ( 1 − b + b · |d| / avgdl ) + tf(t, d)   │
                    └                                                    ┘
```

The two knobs that matter in practice:

- **k1 (default ≈ 1.2)** — term-frequency saturation. As `tf` grows, the numerator saturates, so the 20th occurrence of a term contributes barely more than the 10th.
- **b (default ≈ 0.75)** — length normalization. Long documents get their scores discounted so a 100-page manual does not automatically beat a one-page FAQ.

BM25 is the strong lexical baseline modern hybrid stacks fuse against (mod-106). If your dense-only retriever cannot beat a properly-analyzed BM25 index on your corpus, you have a chunker, embedding, or analyzer problem — not a "we need a bigger model" problem. Read MRS chapter 11 for the derivation and chapter 6 for the tf-idf → vector-space model story that leads into it.

## The Probability Ranking Principle

Every scoring function above (and every embedding-similarity function you will meet) is trying to approximate the same idea, laid down by Robertson (1977), *The Probability Ranking Principle in IR*:

> If a retrieval system's response to each query is a ranking of the documents in the collection in order of decreasing probability of relevance to the user who submitted the query, the effectiveness of the system to its user will be the best that is obtainable on the basis of that data.

The PRP is why we talk about **ranked retrieval** and not just Boolean match, why **top-*k*** is the interface every retriever exposes, and why every subsequent evaluation metric — MRR, MAP, nDCG — cares more about *the order of the top few results* than about "did the right doc appear anywhere in the corpus." It is also why "just return everything with cosine similarity > 0.8" is a shape you will never see in production: a *ranking* subject to a truncation is strictly better-defined than a *thresholded set*.

## Precision, recall, F1 — and why they are the floor, not the ceiling

You have seen these before. The retrieval-engineering framing:

- **Precision@k** — of the top *k* results returned, what fraction were relevant?
- **Recall@k** — of *all* relevant documents in the corpus for this query, what fraction appeared in the top *k*?
- **F1** — harmonic mean of precision and recall, useful when you must pick one number for classification-shaped tasks.

Two observations that trip up newcomers:

1. **Precision and recall are set-based**; they ignore the *order* within the top *k*. Two rankings with the same members but reversed order have identical precision@k and recall@k. Once you care about order — and in RAG you always do, because the generator reads top-down — reach for MRR, MAP, and nDCG@k (depth: mod-109).
2. **In RAG, recall@k usually matters more than precision@k**, because a downstream reranker + LLM can filter out the noise you retrieved but cannot recover a chunk you never retrieved. The mod-104 tuning loop is largely "pick the smallest *k* and cheapest ANN config that still hits your recall@k target."

## From classical IR to modern retrieval — what carries, what does not

| Classical IR (MRS, Lucene) | Modern dense retrieval | Same idea? |
|---|---|---|
| Inverted index on terms | ANN index on dense vectors | Same shape: **fast lookup instead of scan** |
| tf-idf / BM25 scoring | Cosine / dot-product similarity | Same shape: **rank by decreasing estimated relevance** |
| Analyzer (tokenize, lowercase, stem, ...) | Embedding model | Same shape: **map text to comparable representation** |
| Query expansion (synonyms, thesauri) | Query rewriting (HyDE, decomposition, step-back) | Same shape: **make the query look more like the target text** |
| Learning-to-rank (LambdaMART, RankNet) | Cross-encoder reranker | Same shape: **slower, better second-pass scorer** |
| Relevance judgments (TREC/Cranfield) | Judgment sets on your corpus | Same shape: **ground eval in labeled `(q, d, rel)` triples** |

The story is not "classical is dead, learn embeddings." The story is **the moving parts have the same names**. Practicing retrieval engineers reach for both lexical and dense at the same time (mod-106) because each captures the other's blind spots.

## What you should be able to do after this chapter

- Sketch an inverted index for a three-document corpus.
- Explain, in one sentence each, what `k1` and `b` in BM25 control.
- State the Probability Ranking Principle and name one metric that respects rank order.
- Distinguish precision@k from recall@k and explain why RAG systems usually optimize the latter first.
- Point to which of your future modules will replace which classical part with its modern analogue.

The refresher exercise for this chapter walks you through building a five-document inverted index by hand and computing tf-idf and BM25 scores for one query on that mini-corpus.

## Further reading

- Manning, Raghavan, Schütze (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Free HTML edition](https://nlp.stanford.edu/IR-book/). Chapters 1 (Boolean retrieval), 2 (dictionaries), 6 (scoring and vector-space model), 8 (evaluation), 11 (probabilistic retrieval / BM25).
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval 3(4). [DOI:10.1561/1500000019](https://dl.acm.org/doi/10.1561/1500000019).
- Robertson, S. (1977). *The Probability Ranking Principle in IR.* Journal of Documentation 33(4). [DOI:10.1108/eb026647](https://www.emerald.com/insight/content/doi/10.1108/eb026647/full/html).
- Elasticsearch guide, [*Practical BM25*](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — Elastic's engineering-flavored walk-through of `k1` and `b`.
