# mod-106-hybrid-retrieval-fusion-query-rewriting: Hybrid Retrieval, Fusion, and Query Rewriting

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 12 hours

## Learning objectives

- Compose hybrid retrieval — BM25 + dense (or BM25 + dense + SPLADE) — with reciprocal-rank fusion (Cormack et al. 2009) as the well-tested default, and know when weighted-sum or Borda-count fusion fits better
- Reason about the score-scale mismatch problem that makes weighted-sum fusion brittle, and why RRF's rank-based scoring sidesteps it
- Implement query rewriting patterns — HyDE (Hypothetical Document Embeddings, Gao et al. 2022), query decomposition (multi-hop), step-back prompting, and multi-query fanout — with the eval cost of each
- Design semantic routing — direct a query to the right index / retriever / prompt template based on intent classification, embedding-nearest-cluster, or an LLM router; know the reliability-vs-latency budget
- Handle filter-heavy queries (structured predicates + semantic query) and the compose-with-metadata pattern that keeps hybrid results from collapsing to the lexical or dense side
- Cite the boundary to `applied-ai-engineer` (peer, level 30) — the peer track *consumes* a hybrid retriever with a fixed interface; this module *builds and tunes* the retriever

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
