# Chapter 1 — Why Hybrid Retrieval Exists and the Score-Scale Mismatch That Shapes It

## Why this chapter exists

Modules 103, 104, and 105 built three retrievers that all claim to answer the same question — "which documents are most relevant to this query?" — from three very different signals:

- A **dense** retriever (mod-103, mod-104): passages whose embedding is close (cosine / dot-product / L2) to the query's embedding.
- A **lexical** retriever (mod-105): documents whose tokens overlap the query's tokens with high BM25 score.
- Optionally, a **learned sparse** retriever (SPLADE, uniCOIL, TILDE): expanded-token-weight vectors that plug into the same inverted index as BM25.

Each is a strong baseline in isolation and each fails in ways the others don't: dense misses out-of-distribution or rare-string queries ("order number Q-77841"); lexical misses paraphrase and synonymy ("cancel my order" vs. "revoke my purchase"); learned-sparse hits the middle but is more expensive to train and serve. The literature — Bruch et al. 2023 ([arXiv:2210.11934](https://arxiv.org/abs/2210.11934)), Wang et al. 2021 ([arXiv:2106.14807](https://arxiv.org/abs/2106.14807)), and every serious benchmark on [BEIR](https://github.com/beir-cellar/beir) since 2021 — reports that a *fused* result of dense + lexical outperforms either arm alone on most heterogeneous corpora, often by a wide enough margin to justify the extra index.

This module is about *how* you fuse them. This chapter frames the problem so the next three chapters can offer the three main answers: reciprocal-rank fusion (chapter 2), weighted-sum fusion (chapter 3), and Borda-count fusion (also chapter 3). Chapter 4 tackles the *query* side (rewriting) — a different lever on the same recall problem. Chapter 5 covers routing — when to run which retriever at all. Chapter 6 covers filter-heavy queries where the naïve fusion approach silently degrades. Chapter 7 marks the boundary to `applied-ai-engineer` — the peer track that *consumes* the retriever this module produces.

## The recall gap that fusion is meant to close

A useful shape to hold: on any real corpus, the top-*k* dense results and the top-*k* lexical results overlap somewhere between **10% and 60%** for the same query. The exact number depends on the corpus, the embedding model, the analyzer, and *k*, but the two lists are rarely near-identical. That overlap gap is both the reason fusion works and the reason it is not trivial.

The recall arithmetic that makes fusion attractive:

- Suppose dense-retrieved top-100 has recall@100 = 0.72 (72% of relevant docs are in the top-100 dense list).
- Suppose lexical-retrieved top-100 has recall@100 = 0.68.
- Suppose the two lists overlap in ~40% of the returned docs.

Then the *union* of the two lists has recall@100 roughly in the 0.82–0.88 band — a real jump that a downstream reranker (mod-107) can then convert into precision at top-10. Fusion is how you turn that union recall into a single ranked list without asking the reranker to score 200 candidates every time.

The union-recall gain is the *upper bound* on fusion's value. A good fusion algorithm captures most of it; a bad one captures very little and can even lose ground vs. a strong single arm. That is the level-30 skill this module trains.

## What "hybrid retrieval" means precisely

The industry has settled on **hybrid retrieval** as the umbrella term for "run two or more retrievers whose signals differ meaningfully, then combine the ranked outputs into one list." The three levels of hybridity you will see:

1. **BM25 + dense.** The most common shape. One lexical arm (BM25 or DFR on a Lucene-family index) and one dense arm (HNSW / IVF-PQ / IVF-Flat on a vector database or a Lucene-family engine's `dense_vector` field).
2. **BM25 + dense + learned-sparse.** BM25 + a dense arm + a SPLADE-family learned-sparse arm ([SPLADE](https://github.com/naver/splade), Formal et al. 2021 [arXiv:2109.10086](https://arxiv.org/abs/2109.10086), or uniCOIL, Lin & Ma 2021 [arXiv:2106.14807](https://arxiv.org/abs/2106.14807)). The learned-sparse arm captures the semantic-lexical middle: it produces sparse token weights that ride an inverted index but are learned from a pretrained language model.
3. **BM25 + dense + rewriting fanout.** BM25, dense, plus *k* rewrites of the query (chapter 4). This adds retrievers implicitly — every rewrite is another retrieval call.

Everything downstream of this chapter assumes at least (1). SPLADE and rewrites are optional extensions.

**What "hybrid" is not:** two dense retrievers on two different embedding models. That is a dense ensemble, not a hybrid — you get some diversification but the signal shape is the same, so the recall gain is smaller than a dense + lexical stack. Ensemble a dense model with a dense model only when you have a specific reason (e.g., a cross-lingual model paired with a monolingual model on a multilingual corpus).

## The score-scale mismatch problem

If you are going to combine two ranked lists, the natural first idea is a weighted sum:

```
score_hybrid(d, q) = w_dense · score_dense(d, q) + w_lex · score_lex(d, q)
```

This looks fine on paper and fails in practice, for one specific reason: **the two scores live on different scales that are not directly comparable, and their scales shift per query.**

Concretely:

- **BM25 scores are unbounded above and have no natural scale.** A short-tail-term query might have top-1 BM25 ≈ 8; a common-terms query on the same corpus might have top-1 BM25 ≈ 22. The relationship between raw BM25 and "how relevant" is not linear and is corpus-dependent (chapter 3 of mod-105 covers the k1/b knobs that shift the scale further).
- **Dense scores are bounded (typically cosine ∈ [−1, 1] or dot-product with model-dependent range).** Top-1 cosine on a normalized-embedding model is usually in 0.4–0.9; a specialty vector database might return a *distance* rather than a similarity, which flips the ordering.
- **The score-to-relevance mapping is not linear on either side.** Two docs with BM25 15 and BM25 10 are not "1.5× as relevant" on any well-defined scale; two docs with cosine 0.82 and 0.75 are much closer in relevance than the score gap suggests near the top.

The concrete failure mode this produces:

**Query A** returns top-1 BM25 = 8, top-1 cosine = 0.45. **Query B** returns top-1 BM25 = 22, top-1 cosine = 0.71. With any fixed `w_lex, w_dense` you pick, one arm dominates on query A and the other dominates on query B — the effective weights *are* per-query, and you have no principled way to set them.

The workarounds are all imperfect:

- **Min-max normalization per query** — rescale each arm's top-N scores to [0, 1] before combining. Loses information (a query with only one strong hit and a long tail of noise looks the same as a query with ten strong hits) and is sensitive to the choice of N.
- **Z-score normalization per query** — subtract mean, divide by standard deviation. Assumes roughly Gaussian score distributions, which BM25 emphatically is not (BM25 has a long tail; z-score compresses it in ways that don't reflect relevance).
- **CDF / rank normalization** — replace each score with its rank in the arm's output. Discards the absolute score, but that is often the right choice; the *rank* is a signal that survives per-query scale drift.

That last workaround — throw away the scores, keep only the ranks — is the intuition behind **reciprocal-rank fusion (RRF)**, the fusion default this module recommends. Chapter 2 is that method.

If you *are* going to fight the score-scale problem and use a weighted sum, you do so with per-arm score normalization, per-query, and you eyeball or hyperparameter-tune the arm weights on an eval set. Chapter 3 covers when that is worth doing (it sometimes is — usually when you have a lot of eval-set labels and the corpus is stable).

## Recall vs. precision: what fusion actually optimizes

A subtlety worth stating up front: fusion at the retrieval stage is about **top-k recall**, not top-1 precision. You are producing the candidate pool that a reranker (mod-107) will then score with a heavier model to pick the top-3 or top-10 the LLM sees. If your fusion is producing recall@100 = 0.88 while a single dense arm produces recall@100 = 0.72, you have handed the reranker a much better shot at precision at k=10.

That framing has two consequences:

1. **You measure fusion with recall@k (usually k = 100, sometimes k = 50 or k = 200 depending on the reranker budget), not with precision@10 or nDCG@10.** The precision metric belongs to the reranker chapter.
2. **A fusion algorithm that occasionally misranks the top-3 among the retrieved candidates is fine, as long as those three candidates are in the top-100.** The reranker will fix the ordering. This gives you a lot of room to prefer a robust, well-behaved fusion (RRF) over a marginally-better-tuned but brittle one (weighted-sum with fragile normalization).

Mod-109 covers the eval methodology (recall@k, MRR, nDCG, paired significance testing) rigorously. This chapter and the next two use recall@100 as the working metric.

## When hybrid loses to a single arm

Hybrid does not always win. The three settings where a single arm is the right pick:

- **Very narrow corpora with a rich lexicon.** If your corpus is a codebase indexed by symbol and your queries are function names, BM25 (or exact-match on a symbol-keyword field) is already recall-saturated. Adding a dense arm adds noise and cost without meaningful recall gain.
- **Corpora where the dense model was trained on nearly-identical data.** If your embedding model was fine-tuned on your exact domain and evaluated at recall@100 > 0.95, the room for BM25 to add is small; the cost of a second index may not be worth the 1–2 percentage-point recall gain. This is uncommon in practice because most orgs use off-the-shelf embedding models.
- **Latency-critical stacks with a rigid budget.** Two retrievers plus fusion adds ~2× the retrieval latency (they run in parallel, but the slower arm sets the floor). If your p99 budget is 40 ms and dense-only is 30 ms while BM25 alone is 25 ms, the second arm may not fit. Measure before you commit.

These are the escape hatches. In every other setting — enterprise search, product search, RAG on heterogeneous corpora, agent-tool retrieval — start with hybrid.

## The three levers this module gives you

The rest of the module is organised around three levers you can pull on a hybrid retriever:

1. **Fusion algorithm** — how the ranked outputs from each arm combine. Chapters 2 and 3.
2. **Query rewriting** — how the query is transformed *before* it hits each arm. Chapter 4.
3. **Routing** — whether to run all arms on every query, or route to a subset. Chapter 5.

Chapter 6 layers on the filter-heavy composition case (structured predicates on top of hybrid). Chapter 7 marks the boundary to `applied-ai-engineer`.

## What is NOT in this chapter (and where it goes)

- **The RRF formula and its parameters** — chapter 2.
- **Weighted-sum and Borda-count fusion mechanics** — chapter 3.
- **HyDE, multi-query fanout, step-back, decomposition** — chapter 4.
- **Semantic routing with intent classification, embedding clusters, LLM router** — chapter 5.
- **Filter-heavy composition with structured predicates** — chapter 6.
- **The retriever-interface contract the peer track consumes** — chapter 7.
- **Neural rerankers (cross-encoders, ColBERT, LLM-as-judge)** — mod-107.
- **Recall@k / MRR / nDCG / paired significance testing methodology** — mod-109.

## What you should be able to do after this chapter

- Explain in one paragraph why BM25 + dense usually outperforms either arm alone on heterogeneous corpora, and cite the ballpark recall gap the union captures.
- Name the score-scale mismatch problem and give one concrete failure mode of naïve weighted-sum fusion without normalization.
- Distinguish "hybrid retrieval" (different signals — dense + lexical) from "dense ensemble" (two dense models) and defend which one to reach for on a given corpus.
- List the three settings where a single-arm retriever beats hybrid.
- Recognise the three levers this module covers (fusion, rewriting, routing) and know which chapter owns each.

## Further reading

- Bruch, S., Nardini, F. M., Nardo, C., Perego, R., and Tonellotto, N. (2023). *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM Transactions on Information Systems 41(4). [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) — the fusion-methodology paper that motivates preferring rank-based over score-based fusion in the general case.
- Wang, X., Macdonald, C., Tonellotto, N., and Ounis, I. (2021). *Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval.* ICTIR. [arXiv:2106.11251](https://arxiv.org/abs/2106.11251) — dense-side view of the same recall-gap discussion.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the benchmark most fusion papers report on.
- Formal, T., Piwowarski, B., and Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* SIGIR. [arXiv:2107.05720](https://arxiv.org/abs/2107.05720) — the SPLADE reference paper for the third-arm learned-sparse retriever named above.
- Lin, J. and Ma, X. (2021). *A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques.* [arXiv:2106.14807](https://arxiv.org/abs/2106.14807) — the uniCOIL discussion; useful frame for where learned-sparse sits between BM25 and dense.
