# mod-109-retrieval-quality-evaluation: Retrieval-Quality Evaluation and Judgment Sets

**Estimated effort:** 14 hours

## What this module is about

Every earlier module in this track produced knobs — chunkers (mod-102), embedders (mod-103), ANN indexes (mod-104), Lucene indexes (mod-105), hybrid fusion (mod-106), rerankers and funnel widths (mod-107), retrieval-specific prompt fragments (mod-108). This module is the discipline that turns those knobs from vibes into evidence: a **judgment set** on your corpus, a **metric harness** that scores a retriever's ranked list against it, a **paired significance test** that says whether a delta is real, an **orthogonal ablation grid** that attributes the delta to a single axis, and a **BEIR / MTEB corpus-fit delta** that anchors the whole thing to a public baseline.

The retriever-only slice — recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate — is what this module owns. Application-layer RAG metrics that need the generator in the loop (RAGAS faithfulness, answer relevance, context precision/recall, TruLens RAG triad) belong to `ai-eval-engineer` at level 30. Benchmark methodology-of-methodology — deriving new significance tests, contamination detection, cross-benchmark curation — belongs to `model-evaluation-engineer` at level 30. Chapter 7 draws both boundaries explicitly.

## Learning objectives

- Build a retrieval-quality judgment set — pairs of (query, relevant-doc-IDs) — with the TREC / BEIR shape, using either curated relevance labels or LLM-assisted labeling with human spot-check.
- Compute the core retrieval metrics — recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate — with the `ir_measures` / `trec_eval` / `pytrec_eval` toolchain; know when each metric answers what.
- Reason about statistical significance for retrieval comparisons — paired t-test, Wilcoxon signed-rank on per-query score deltas, bootstrap confidence intervals — grounded in Sakai's *Metrics, Statistics, Tests* (2014).
- Design ablation harnesses — chunker × embedding × retriever × reranker as an orthogonal grid — and reason about which axis actually moved the number.
- Run BEIR / MTEB retrieval subsets and derive a corpus-specific delta from the leaderboard baseline.
- Cite the boundary to `ai-eval-engineer` (peer, level 30) which explicitly owns application-layer RAG eval (RAGAS, TruLens, DeepEval, RAG triad — faithfulness / answer relevance / context precision-recall). This module owns retrieval-*only* eval — the metrics that don't need the generator in the loop.
- Cite the boundary to `model-evaluation-engineer` (peer, level 30) for benchmark-methodology depth — the RAG track uses paired-significance testing; full derivation and calibration methodology-of-methodology lives there.

## Chapters

- `01-why-retrieval-quality-evaluation-exists.md` — Why this discipline exists: the four operational questions retrieval metrics answer, the two failure modes (un-defensible ship, mean-without-CI), and the peer-track boundaries.
- `02-judgment-sets-trec-beir-shape-and-llm-assisted-labeling.md` — The TREC / BEIR qrels shape, protocol A (curated labels) and protocol B (LLM-assisted labels with mandatory human spot-check), pooling from multiple retrievers, doc-ID schemes, and the ≥ 80% accuracy / κ ≥ 0.6 deployment bar.
- `03-core-retrieval-metrics-with-ir-measures.md` — Recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate with the `ir_measures` / `pytrec_eval` / `trec_eval` toolchain; the "which metric answers what" decision table.
- `04-paired-significance-and-bootstrap-ci.md` — Per-query score arrays as random variables, paired Wilcoxon signed-rank vs. paired t-test, BCa bootstrap CI on the mean delta, Benjamini-Hochberg FDR correction, the sample-size back-of-envelope, grounded in Sakai (2014).
- `05-orthogonal-ablation-grid-and-attribution.md` — Chunker × embedding × retriever × reranker as an orthogonal grid, one-axis-at-a-time discipline, reverse-cost ordering, factorial extensions for suspected interactions, the six confounders that make a grid dishonest, and how to read the grid into a decision memo.
- `06-beir-mteb-subset-runs-and-corpus-delta.md` — Running BEIR / MTEB subsets locally, matching the leaderboard (prefix / score-function / pooling / split / field concatenation), computing the corpus-fit delta, and the six-item leaderboard-mismatch checklist.
- `07-boundaries-to-ai-eval-engineer-and-model-evaluation-engineer.md` — The two peer-track boundaries: what this module owns vs. what `ai-eval-engineer` owns (application-layer metrics) vs. what `model-evaluation-engineer` owns (benchmark methodology-of-methodology). Includes the hand-off artifact list and a boundary-aware evaluation-report template.

## Exercises

- `exercises/exercise-01-build-a-judgment-set-with-curated-and-llm-assisted-labels.md` — Build a ≥ 60-query judgment set on your corpus (or a BEIR stand-in) with pooled candidates from ≥ 3 retrievers, LLM-assisted labeling, and a stratified 10% human spot-check. Ships `eval/qrels.tsv` + `eval/queries.jsonl` + `eval/VERSION`.
- `exercises/exercise-02-compute-recall-mrr-ndcg-with-ir-measures.md` — Run ≥ 2 retrievers, produce TREC-shape run files, compute aggregate + per-query metrics with `ir_measures`, cross-check against `pytrec_eval` / `ranx` / `trec_eval`. Ships `report/aggregate.md` + `report/per-query.csv`.
- `exercises/exercise-03-paired-significance-and-bootstrap-ci-on-per-query-deltas.md` — Compute paired Wilcoxon + paired t-test + BCa bootstrap CI + Benjamini-Hochberg FDR + sample-size back-of-envelope on exercise 02's per-query arrays. Ships `report/significance.md` and a regression-alert primitive mod-111 will wrap.
- `exercises/exercise-04-orthogonal-ablation-grid-and-attribution.md` — Author an 8-12-row one-axis-at-a-time grid over chunker × embedder × retriever × reranker; evaluate every row against the frozen qrels; run FDR-corrected paired significance; publish the attributed table and a ship / defer / refuse decision memo.
- `exercises/exercise-05-beir-mteb-subset-run-and-corpus-delta-report.md` — Run 2-3 BEIR / MTEB Retrieval datasets locally with the same embedder, reproduce each leaderboard number within ±0.01 nDCG@10, compute the corpus-fit delta, and append a one-paragraph infrastructure-sanity note to exercise 04's decision memo.

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
