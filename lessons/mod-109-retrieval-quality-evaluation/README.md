# mod-109-retrieval-quality-evaluation: Retrieval-Quality Evaluation and Judgment Sets

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Build a retrieval-quality judgment set — pairs of (query, relevant-doc-IDs) — with the TREC / BEIR shape, using either curated relevance labels or LLM-assisted labeling with human spot-check
- Compute the core retrieval metrics — recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate — with the ir_measures / trec_eval / pytrec_eval toolchain; know when each metric answers what
- Reason about statistical significance for retrieval comparisons — paired t-test, Wilcoxon signed-rank on per-query score deltas, bootstrap confidence intervals — grounded in Sakai's 'Metrics, Statistics, Tests' (2014)
- Design ablation harnesses — chunker × embedding × retriever × reranker as an orthogonal grid — and reason about which axis actually moved the number
- Run BEIR / MTEB retrieval subsets and derive a corpus-specific delta from the leaderboard baseline
- Cite the boundary to `ai-eval-engineer` (peer, level 30) which explicitly owns application-layer RAG eval (RAGAS, TruLens, DeepEval, RAG triad — faithfulness / answer relevance / context precision-recall). This module owns retrieval-*only* eval — the metrics that don't need the generator in the loop.
- Cite the boundary to `model-evaluation-engineer` (peer, level 30) for benchmark-methodology depth — the RAG track uses paired-significance testing; full derivation and calibration methodology-of-methodology lives there

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
