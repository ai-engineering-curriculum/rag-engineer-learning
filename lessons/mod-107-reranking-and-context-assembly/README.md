# mod-107-reranking-and-context-assembly: Reranking and Context Assembly

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Choose a reranker for the job — cross-encoder (bge-reranker, ms-marco MiniLM), hosted API (Cohere Rerank v3, Voyage rerank-2, Jina Reranker), LTR reranker (LightGBM / XGBoost), or LLM-as-reranker — with the latency and cost trade-off documented
- Tune the retrieve-top-N → rerank-to-K funnel — N and K choice, the diminishing-returns curve, and the p95 latency budget it eats
- Implement contextual compression (LlamaIndex ContextualCompressor, LangChain LLMChainExtractor) to shrink the reranked set to what actually fits the context window without losing citation anchors
- Design context assembly for the generator — dedup by source, order by score (or by document structure for long-doc RAG), include per-chunk citation metadata, and reserve token budget for the answer
- Reason about context-window management — long-context models (200K+, 1M+) don't remove the reranking step; the 'lost-in-the-middle' effect (Liu et al. 2023) and cost-per-token still make a tight context the right shape
- Cite the boundary to `applied-ai-engineer` (peer, level 30) — the peer's mod-103 covers *consuming* a reranked context at the product surface; this module builds the reranker + compressor + assembler

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
