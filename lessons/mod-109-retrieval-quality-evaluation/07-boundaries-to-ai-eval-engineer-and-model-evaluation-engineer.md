# Chapter 7 — The Boundaries to `ai-eval-engineer` and `model-evaluation-engineer`

## Why this chapter exists

Chapter 1 named two peer tracks at level 30 that sit next to this one:

1. **`ai-eval-engineer` (peer, level 30)** owns *application-layer RAG eval* — RAGAS, TruLens, DeepEval, and the RAG triad (faithfulness / answer relevance / context precision-recall). Anything that requires the *generator* to be in the loop.
2. **`model-evaluation-engineer` (peer, level 30)** owns *benchmark-methodology depth* — deriving new significance tests, calibrating them, meta-evaluating metrics, curating cross-benchmark artifacts. This module *uses* paired significance; the peer *vouches* for it.

Chapters 2-6 stayed inside the retriever-only slice. This chapter is the contract when the retriever-only slice exchanges artifacts with the two peer tracks. Getting the boundary wrong produces two symmetric failures: the rag-engineer either over-reaches (re-deriving RAGAS metrics, arguing about the Wilcoxon's α-calibration) or under-delivers (handing off metrics without the per-query records, publishing significance results without documenting which test was used).

## Where this module sits between the two peers

```
                              rag-engineer (this track)                             ai-eval-engineer (peer, level 30)

    ingest → chunk → embed → index → retrieve → rerank → assemble  ─► AssembledContext ──►  generator prompt → generate →
    ────────────────────────────────────────────────────────►                                parsed answer → verify citations → 
                                                                                            faithfulness / answer relevance /
    plus:  judgment set + qrels  ────────────────────────────►                              context precision & recall (RAG triad)
           retriever-only metrics                                                           
           (recall@k, nDCG@k, MRR, MAP, hit-rate)  ──────────────►    metrics + qrels ────► inputs to RAGAS' context-precision / -recall
           paired significance + bootstrap CI                                               (which computes the SAME metrics from a
           orthogonal ablation grid                                                          judgment-set-shaped input, then wraps
           BEIR / MTEB baseline runs                                                        answer-relevance and faithfulness on top)

                                                                                                                       ▲
                                                                                                                       │
                                                                                    model-evaluation-engineer (peer, level 30)
                                                                                    ────────────────────────────────────────────
                                                                                    - derives / validates new significance tests
                                                                                    - meta-evaluates metric-metric correlation
                                                                                    - calibrates false-positive rates
                                                                                    - curates cross-benchmark artifacts (like MTEB)
                                                                                    - contamination detection
                                                                                    
    this module USES the peer's outputs (Wilcoxon,       ◄──────────────
    bootstrap CI, MTEB) as tools. It does not re-derive.
```

The rag-engineer produces the retrieval-only slice. The two peers consume different parts of it: the eval peer consumes the qrels + metrics (and adds the generator layer); the model-eval peer curates the tools this module uses.

## Boundary 1: to `ai-eval-engineer`

### What this module owns

Everything in chapters 2-6. Concretely: judgment set (qrels), per-query metric arrays, paired significance tests, ablation grid, BEIR corpus-fit delta. **None of these require the generator to be in the loop.** They score the retriever's ranked list against the qrels; they never call the LLM.

### What the eval peer owns

Application-layer RAG eval — the metrics that need the LLM's output. Specifically:

- **Faithfulness** — is every claim in the LLM's answer supported by the retrieved context? Requires the LLM's answer.
- **Answer relevance** — does the LLM's answer address the user's question? Requires the LLM's answer + the question.
- **Context precision** (RAGAS's version) — of the retrieved chunks, what fraction were used by the LLM in the answer? Requires the LLM's answer + the retrieved chunks.
- **Context recall** (RAGAS's version) — of the ground-truth answer's atomic claims, what fraction have a supporting retrieved chunk? Requires a ground-truth answer + the retrieved chunks + an LLM to atomize.

The tools that operationalize these — RAGAS, TruLens, DeepEval, Ragflow — belong on the peer's side of the boundary. See the [RAGAS paper](https://arxiv.org/abs/2309.15217), the [TruLens RAG triad docs](https://www.trulens.org/), and the [DeepEval faithfulness metric](https://docs.confident-ai.com/docs/metrics-faithfulness) for the peer track's canonical references.

### Where the boundary lives in the file tree

- **This module produces:** `eval/qrels.tsv`, `eval/queries.jsonl`, `runs/*.run`, `eval/per-query-scores.csv`, `eval/significance-report.md`, `eval/ablation-grid.md`.
- **Peer track produces:** `eval/faithfulness.json`, `eval/answer-relevance.json`, `eval/context-precision-recall.json`, `eval/rag-triad-report.md`.

The two directories sit alongside; each is the input the *other* would need for downstream analysis. When retrieval and application-layer metrics move in opposite directions ("retriever's nDCG@10 went up, answer-relevance went down"), that is the signal for a joint investigation — often the retriever surfaced longer / more diverse chunks that helped precision on ranked eval but confused the generator on faithfulness eval.

### Confusable metrics that live on this side of the boundary

Two RAGAS metrics have names that look retrieval-only but are *application-layer*:

- **RAGAS `context_precision`** is **not** the same as this module's `precision@k`. RAGAS's version calls the LLM to judge whether each retrieved chunk was "useful" to the answer; this module's precision@k checks whether the chunk is in the qrels' relevant set. Different metric, different owner.
- **RAGAS `context_recall`** is **not** the same as this module's `recall@k`. RAGAS's version atomizes the ground-truth answer with an LLM and checks whether each atom has a supporting chunk; this module's recall@k checks whether the qrels' relevant docs are in the top-k. Different metric, different owner.

When a stakeholder says "context precision dropped," always clarify which — you may be being asked about a metric on the peer's side of the boundary.

### Hand-off shape

The eval peer needs three artifacts from this module:

1. **The qrels file (`eval/qrels.tsv`).** The peer's RAGAS `context_recall` and `context_precision` can operate against ground-truth answers, but for cross-checks against retrieval-only metrics the peer will re-use this module's qrels.
2. **The retriever's per-query top-k output.** So the peer can call the generator on the same ranked list this module scored. Persist run files (chapter 3) — the peer reads them, not just the aggregate.
3. **The `AssembledContext` shape (from mod-107 chapter 4).** The rendered, deduplicated, cite-anchored bundle. Peer track's faithfulness / answer-relevance metrics run on top of this exact shape.

Return direction: the peer publishes a **faithfulness rate** and a **grounding rate** per retriever config. When those diverge from the retrieval-only metrics ("retriever nDCG@10 improved, faithfulness dropped"), open a joint investigation.

## Boundary 2: to `model-evaluation-engineer`

### What this module owns

*Applying* significance tests: chapter 4's paired Wilcoxon, paired t-test, bootstrap CI. Running BEIR / MTEB subsets and reporting per-dataset numbers. Reading leaderboards.

### What the model-eval peer owns

*Vouching for* the tests and benchmarks — the methodology-of-methodology.

- **Deriving new significance tests.** If a new IR metric needs a bespoke variance estimator or a permutation-test scheme, the peer builds it. This module uses off-the-shelf paired Wilcoxon.
- **Calibrating false-positive rates.** Empirical validation that Wilcoxon holds its nominal α = 0.05 on IR-shaped per-query distributions across query counts (n = 30 to n = 500) and metrics (nDCG@k, MAP, MRR, ERR, RBP). This module trusts Sakai (2014) and Smucker et al. (2007).
- **Meta-evaluation of metrics.** How well do offline retrieval metrics (nDCG@10) correlate with downstream measures (user satisfaction, click-through, RAG triad faithfulness)? This is the peer's meta-evaluation project.
- **Contamination detection.** Whether an embedder was trained on data overlapping with an MTEB task and whether its reported score is thereby inflated. This module reports the leaderboard number as-is; the peer publishes a contamination-adjusted version.
- **Curating new benchmarks.** MTEB itself is a peer-track artifact. If your corpus is important enough that a new benchmark is warranted, the peer designs it.

### Where the boundary lives in a decision

**Inside this module's scope:**

- "We compared bge-large vs. bm25 with Wilcoxon on paired nDCG@10 deltas. `p = 0.0004`, CI `[+0.041, +0.108]`. Ship bge-large."
- "We ran NFCorpus locally with bge-base and got 0.368 nDCG@10, which is within 0.005 of the leaderboard's 0.372. Our infrastructure is correctly configured."
- "Our corpus-fit delta vs. NFCorpus is +0.044 — our corpus is easier for this retriever than NFCorpus."

**Outside this module's scope; hand to the peer:**

- "Is Wilcoxon the right test for paired nDCG@10 deltas on n = 30 queries with a bimodal distribution?" (calibration question)
- "Is the delta bge-large vs. bge-base of +0.03 on MTEB Retrieval reproducible, or is bge-base contaminated with an MTEB task?" (contamination question)
- "Should we introduce a new IR metric for our corpus's specific behavior (e.g., a class-weighted recall that penalizes missing a low-frequency claim)?" (metric-design question)
- "Does nDCG@10 on the retriever correlate with the peer's answer-relevance score on the generator?" (meta-evaluation question — actually crosses *both* boundaries; a joint effort with the eval peer too)

### Hand-off shape

The model-eval peer needs, from this module:

1. **Full per-query metric arrays** for any comparison the peer is being asked to validate. Aggregates lose the distributional shape.
2. **The exact ablation configs** (chapter 5) — reference row, one-at-a-time rows, seeds, tokenization, prompts. Anything that could confound.
3. **The BEIR / MTEB run configs** (chapter 6) — model name + version, score function, pooling, prefix, split. So the peer can reproduce and check contamination.

Return direction: the peer publishes a **methodology brief** for the significance-testing approach and (occasionally) a **contamination advisory** for specific model-benchmark pairs. This module cites those briefs in its evaluation reports.

## Two failures the boundary defends against

### Failure A: rag-engineer publishes RAGAS numbers

Reasoning: "we can just install RAGAS and get faithfulness / answer-relevance too; more data is better." Effect: the rag-engineer's evaluation report now has an application-layer metric next to retrieval-only metrics. When the numbers move in opposite directions, no one knows who owns the diagnosis. When RAGAS changes its metric definitions between releases (they have, several times), the retrieval report is now sensitive to a peer-owned dependency the rag-engineer isn't tracking.

**Defense.** Publish retrieval-only metrics. Cite the peer track and their reports for application-layer metrics. If a stakeholder wants both in one dashboard, that dashboard is the peer track's — this module's evaluation report is a scoped artifact the peer can pull from.

### Failure B: rag-engineer argues about the Wilcoxon's calibration

Reasoning: "the Wilcoxon assumes symmetric per-query deltas; ours are skewed; therefore the p-value is wrong." Effect: three weeks of the rag-engineer re-deriving significance-test theory from scratch — a project the peer track has already published on and can hand off in an afternoon.

**Defense.** If you have a distributional concern that makes you doubt the Wilcoxon, either (a) run it *plus* the paired t-test *plus* a permutation test with `scipy.stats.permutation_test` and check whether all three agree — usually they do, and you can move on — or (b) hand the question to the peer track. Do not re-derive.

## Concrete: what a boundary-aware evaluation report looks like

Chapter 4's aggregate-comparison-with-significance script produces the retrieval-side of a report. The report itself, published in a shared docs repo, has this shape:

```markdown
# Retrieval Evaluation: bge-large vs. bm25 on internal runbook corpus

## Judgment set
- eval-set version: v2024-11-a (120 queries, LLM-assisted + 12% spot-check, κ = 0.71)
- doc-id scheme: stable passage anchor (`<slug>#chunk=<n>`)

## Retrieval-only metrics (this module's scope)

| Metric | bm25 | bge-large | Δ | 95% CI (BCa) | Wilcoxon p | n |
|---|---|---|---|---|---|---|
| nDCG@10 | 0.412 | 0.487 | +0.075 | [+0.041, +0.108] | 0.0004 | 118 |
| recall@50 | 0.741 | 0.803 | +0.062 | [+0.031, +0.094] | 0.0007 | 118 |
| MRR@10 | 0.523 | 0.612 | +0.089 | [+0.048, +0.132] | 0.0002 | 118 |

Method: paired Wilcoxon signed-rank on per-query deltas (Sakai 2014).
BCa bootstrap 95% CI over 10,000 resamples. n = queries with non-null score in both runs.
Multiple-comparison correction: not applied (single system-vs-baseline comparison).

## BEIR corpus-fit sanity (this module's scope)
Local BEIR NFCorpus run with bge-large: nDCG@10 = 0.402 (leaderboard: 0.408; delta -0.006, within noise → correctly configured).
Corpus-fit delta: +0.085 vs. NFCorpus (our corpus is easier for bge-large than NFCorpus).

## Application-layer metrics (see peer report)
See [ai-eval-engineer's evaluation report](../eval-application/bge-large-vs-bm25.md) for RAGAS faithfulness, answer-relevance, and RAG-triad context-precision/recall on this same comparison. Retrieval-only numbers here are one of the inputs to that report; they do NOT substitute for the application-layer eval.

## Ship recommendation
Ship bge-large on retrieval quality alone. Await peer's faithfulness and answer-relevance numbers before final approval; if faithfulness regresses > 3%, escalate for joint investigation.
```

Every clause of that report is on the correct side of the boundary. The two peer tracks each read their own section without needing to re-derive anything from another's.

## What you should be able to do after this chapter

- Name what this module owns (retrieval-only metrics + judgment set + paired significance + ablation grid + BEIR corpus-fit) and what it does not (application-layer RAG metrics, benchmark methodology derivation).
- Distinguish retrieval-only `precision@k` and `recall@k` from RAGAS's application-layer `context_precision` and `context_recall`.
- Name the three artifacts this module hands to the eval peer (qrels, per-query top-k, `AssembledContext` shape) and the three it hands to the model-eval peer (per-query arrays, ablation configs, BEIR run configs).
- State when a question is inside this module's scope (applying an off-the-shelf test) and when it belongs to the peer (deriving / calibrating a test, curating a new benchmark, checking contamination).
- Author a boundary-aware evaluation report that cites the peer for application-layer metrics rather than publishing them itself.

## Further reading

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — canonical reference for the metrics on the eval peer's side of the boundary.
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) — ARES; another application-layer eval framework worth knowing about.
- TruLens documentation: [*RAG Triad*](https://www.trulens.org/) — the eval peer's canonical framework.
- DeepEval, [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) — the eval peer's canonical harness; [Faithfulness metric docs](https://docs.confident-ai.com/docs/metrics-faithfulness).
- Sakai, T. (2014). *Metrics, Statistics, Tests.* [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — the paired-significance-test reference this module uses (chapter 4).
- Sakai, T., & Zeng, Z. (2023). *On Estimating Variances for Topic Set Size Design.* [arXiv:2304.12063](https://arxiv.org/abs/2304.12063) — the model-eval peer's canonical treatment of query-count design.
- Muennighoff, N., et al. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — MTEB as an artifact curated by the model-eval peer.
- Zhu, K., Wang, J., Zhou, J., et al. (2024). *Benchmarking Large Language Models in Retrieval-Augmented Generation.* AAAI. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431) — RGB benchmark; joint retrieval + generation quality; a canonical example of cross-boundary work.
- Adjacent-track pointers:
  - [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) — the eval peer's curriculum.
  - [`model-evaluation-engineer-learning`](https://github.com/ai-engineering-curriculum/model-evaluation-engineer-learning) — the model-eval peer's curriculum.
