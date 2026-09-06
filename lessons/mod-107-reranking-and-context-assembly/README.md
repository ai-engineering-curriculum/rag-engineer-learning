# mod-107-reranking-and-context-assembly: Reranking and Context Assembly

**Estimated effort:** 14 hours

Mod-106 handed you a hybrid retriever tuned for recall@100 — a top-100 pool in which the answer-supporting chunks are somewhere, but not yet at the top. This module is the precision half of the two-stage RAG pipeline: **rerank** the top-N to a tightly ordered top-K, optionally **compress** each chunk to the sentences that actually address the query, and **assemble** the surviving chunks into a token-budgeted context with citation anchors the generator can quote back. Chapters 5 and 8 hold the two knobs (N-K funnel; lost-in-the-middle) that make this a measurement discipline rather than a defaults-copy exercise. Chapter 9 draws the interface to `applied-ai-engineer` (peer, level 30), which consumes the `AssembledContext` this module produces.

## Learning objectives

- Choose a reranker for the job — cross-encoder (bge-reranker, ms-marco MiniLM), hosted API (Cohere Rerank v3, Voyage rerank-2, Jina Reranker), LTR reranker (LightGBM / XGBoost), or LLM-as-reranker — with the latency and cost trade-off documented.
- Tune the retrieve-top-N → rerank-to-K funnel — N and K choice, the diminishing-returns curve, and the p95 latency budget it eats.
- Implement contextual compression (LlamaIndex ContextualCompressor, LangChain LLMChainExtractor) to shrink the reranked set to what actually fits the context window without losing citation anchors.
- Design context assembly for the generator — dedup by source, order by score (or by document structure for long-doc RAG), include per-chunk citation metadata, and reserve token budget for the answer.
- Reason about context-window management — long-context models (200K+, 1M+) don't remove the reranking step; the 'lost-in-the-middle' effect (Liu et al. 2023) and cost-per-token still make a tight context the right shape.
- Cite the boundary to `applied-ai-engineer` (peer, level 30) — the peer's mod-103 covers *consuming* a reranked context at the product surface; this module builds the reranker + compressor + assembler.

## Lecture chapters

1. [Chapter 1 — Why reranking exists and the precision job it owns](./01-why-reranking-exists.md) — the two-stage pipeline, the recall/precision division of labor, and the four levers this module tunes.
2. [Chapter 2 — Cross-encoder rerankers](./02-cross-encoder-rerankers.md) — the default reranker family; `bge-reranker`, `ms-marco-MiniLM`, deployment shape, and quality/latency arithmetic.
3. [Chapter 3 — Hosted API rerankers](./03-hosted-api-rerankers.md) — Cohere Rerank, Voyage `rerank-2`, Jina Reranker; the trade-off vs. self-hosted (ops, tunability, network-hop p95).
4. [Chapter 4 — LTR and LLM-as-reranker: the two ends of the spectrum](./04-ltr-and-llm-as-reranker.md) — LightGBM/XGBoost LambdaMART for tight latency budgets; RankGPT / RankZephyr for the last-mile of hard queries.
5. [Chapter 5 — The retrieve-top-N → rerank-to-K funnel](./05-funnel-tuning-n-to-k.md) — the two diminishing-returns curves, the latency budget worksheet, the N-K joint tuning recipe, and the anti-patterns.
6. [Chapter 6 — Contextual compression](./06-contextual-compression.md) — the four compression strategies, the citation-anchor and answer-completeness failure modes, and when compression is worth the budget.
7. [Chapter 7 — Context assembly for the generator](./07-context-assembly.md) — the four assembly jobs (dedup, ordering, citation anchors, token budgeting), the `AssembledContext` shape, and the greedy-pack default.
8. [Chapter 8 — Long-context models, lost-in-the-middle, and why rerankers don't retire](./08-lost-in-the-middle-and-long-context.md) — Liu et al. 2023 in operational terms; the cost-per-token argument; the three cases where a reranker can defensibly be skipped.
9. [Chapter 9 — The boundary to `applied-ai-engineer`: the `AssembledContext` contract](./09-boundary-to-applied-ai-engineer.md) — the four contract clauses (precision SLO, latency, token-count accuracy, anchor stability) and the three anti-patterns at this boundary.

## Exercises

- [Exercise 01 — Reranker family selection with a latency + cost matrix](./exercises/exercise-01-reranker-family-selection-with-latency-cost-matrix.md)
- [Exercise 02 — Retrieve-top-N → rerank-to-K funnel tuning](./exercises/exercise-02-retrieve-top-n-rerank-to-k-funnel-tuning.md)
- [Exercise 03 — Contextual compression workshop](./exercises/exercise-03-contextual-compression-workshop.md)
- [Exercise 04 — Context assembly with dedup, ordering, and citation anchors](./exercises/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors.md)
- [Exercise 05 — Lost-in-the-middle and long-context trade-off drill](./exercises/exercise-05-lost-in-the-middle-and-long-context-tradeoff-drill.md)

## Structure

- `01-…md` … `09-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs (scaffolded; not authored in this module).
- `quizzes/`: knowledge checks (scaffolded; not authored in this module).
- `resources.md`: external references, primary docs, and reading list.
