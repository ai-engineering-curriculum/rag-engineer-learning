# mod-106-hybrid-retrieval-fusion-query-rewriting: Hybrid Retrieval, Fusion, and Query Rewriting

**Estimated effort:** 12 hours

The first five modules of the track built three retrievers — dense (mod-103, mod-104), lexical BM25 (mod-105), and optionally learned-sparse (SPLADE, uniCOIL) — each of which fails in ways the others don't. This module is about *composing* them: fusing their ranked lists into one result, rewriting the query to exercise them harder, routing the query to the right subset, and keeping structured filters from silently collapsing the whole stack to a single arm. The output is a hybrid retriever with a stable interface that the peer track (`applied-ai-engineer`) can consume.

## Learning objectives

- Compose hybrid retrieval — BM25 + dense (or BM25 + dense + SPLADE) — with reciprocal-rank fusion (Cormack et al. 2009) as the well-tested default, and know when weighted-sum or Borda-count fusion fits better.
- Reason about the score-scale mismatch problem that makes weighted-sum fusion brittle, and why RRF's rank-based scoring sidesteps it.
- Implement query rewriting patterns — HyDE (Hypothetical Document Embeddings, Gao et al. 2022), query decomposition (multi-hop), step-back prompting, and multi-query fanout — with the eval cost of each.
- Design semantic routing — direct a query to the right index / retriever / prompt template based on intent classification, embedding-nearest-cluster, or an LLM router; know the reliability-vs-latency budget.
- Handle filter-heavy queries (structured predicates + semantic query) and the compose-with-metadata pattern that keeps hybrid results from collapsing to the lexical or dense side.
- Cite the boundary to `applied-ai-engineer` (peer, level 30) — the peer track *consumes* a hybrid retriever with a fixed interface; this module *builds and tunes* the retriever.

## Lecture chapters

1. [Chapter 1 — Why hybrid retrieval exists and the score-scale mismatch that shapes it](./01-hybrid-retrieval-motivation-and-score-scale-mismatch.md) — the recall gap between BM25 and dense; why naïve weighted-sum fails; the three levers this module covers.
2. [Chapter 2 — Reciprocal Rank Fusion: the default fusion algorithm](./02-reciprocal-rank-fusion.md) — the Cormack et al. 2009 formula, the `k=60` constant, the reference Python implementation, and the caveats that motivate chapter 3.
3. [Chapter 3 — Weighted-sum and Borda-count fusion: when they beat RRF](./03-weighted-sum-and-borda-fusion.md) — the normalization strategies that make weighted-sum tractable, Borda-count mechanics, and the regimes where each earns its keep.
4. [Chapter 4 — Query rewriting: HyDE, multi-query fanout, step-back, and decomposition](./04-query-rewriting-patterns.md) — four rewriting patterns with their per-query LLM cost, when each helps, and the eval methodology that keeps you from lying to yourself.
5. [Chapter 5 — Semantic routing: getting the query to the right retriever](./05-semantic-routing.md) — the rules → embedding-nearest-cluster → LLM-fallback cascade, calibration and confidence, and the reliability-vs-latency budget.
6. [Chapter 6 — Filter-heavy hybrid: structured predicates without collapsing to one arm](./06-filter-heavy-hybrid-composition.md) — the four filter-integration points, the compose-with-metadata pattern, ACL-aware retrieval, and the anti-patterns.
7. [Chapter 7 — The boundary to `applied-ai-engineer`: the retriever interface contract](./07-boundary-to-applied-ai-engineer.md) — what shape of interface the peer track consumes, what stays yours to own, and the escalation triggers back and forth.

## Exercises

- [Exercise 01 — Reciprocal Rank Fusion implementation and analysis](./exercises/exercise-01-reciprocal-rank-fusion-implementation-and-analysis.md)
- [Exercise 02 — Weighted-sum vs. RRF vs. Borda comparison on one eval](./exercises/exercise-02-weighted-sum-vs-rrf-vs-borda-comparison-on-one-eval.md)
- [Exercise 03 — HyDE and multi-query fanout rewriting workshop](./exercises/exercise-03-hyde-and-multi-query-fanout-rewriting-workshop.md)
- [Exercise 04 — Semantic router with embedding clusters and LLM fallback](./exercises/exercise-04-semantic-router-with-embedding-clusters-and-llm-fallback.md)
- [Exercise 05 — Filter-heavy hybrid composition drill](./exercises/exercise-05-filter-heavy-hybrid-composition-drill.md)

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs (scaffolded; not authored in this module).
- `quizzes/`: knowledge checks (scaffolded; not authored in this module).
- `resources.md`: external references, primary docs, and reading list.
