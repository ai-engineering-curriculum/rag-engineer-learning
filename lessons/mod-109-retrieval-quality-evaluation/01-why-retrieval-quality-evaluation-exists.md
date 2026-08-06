# Chapter 1 — Why Retrieval-Quality Evaluation Exists

## Why this chapter exists

Every module before this one produced a knob. Mod-102 gave you chunk-size, overlap, hierarchical-vs-flat. Mod-103 gave you a shortlist of embedders. Mod-104 gave you HNSW `M` / `efSearch`, IVF-PQ centroids. Mod-105 gave you BM25 field weights and boosts. Mod-106 gave you fusion weight or RRF constant `k`. Mod-107 gave you top-N and rerank-to-K funnel widths. Each of those knobs, individually plausible, joins a combinatorial space that no amount of intuition tunes.

The only defensible way through that space is a **judgment set** — a corpus-native set of (query, relevant-doc-IDs) labels — and a **metric harness** that turns a retriever run into a number, run after run, in the same shape. Everything downstream — the retriever CI gate in mod-111, the ablation harness for shipping a new embedding model, the paired significance test that says "this change is real, not noise" — all of it depends on the judgment set + metric pair this module builds.

Two properties make this a discipline rather than a spreadsheet exercise:

1. **Retrieval scores are per-query random variables.** A single retriever run over 100 queries produces 100 nDCG@10 numbers, not one. The mean nDCG@10 across them is a *statistic* on a *sample*, and any comparison between two retrievers must respect that. Sakai's *Metrics, Statistics, Tests* (2014) is the field's reference for how — this module takes that reference literally.
2. **Metric choice is task-shaped, not preference.** MRR answers "is the top hit good?" nDCG@10 answers "is the top-10 ranked well?" recall@100 answers "did we get the relevant docs into the reranker's window?" Choosing the wrong one hides regressions and inflates wins.

By the end of this module you will have a judgment set on your own corpus, a metric harness that produces nDCG@10 / recall@k / MRR / MAP / hit-rate through `ir_measures`, a significance test that says whether run B beats run A at p < 0.05, an ablation grid that attributes deltas to a single axis, and a BEIR / MTEB subset run that produces a defensible corpus-fit delta from the leaderboard number. Every downstream module in this track reads one of those artifacts.

## What this module is not

The `rag-engineer` role sits next to two peer tracks whose work overlaps at the edges:

- **`ai-eval-engineer` (peer, level 30)** owns *application-layer RAG eval*. That means RAGAS, TruLens, DeepEval, and the "RAG triad" (faithfulness, answer relevance, context precision-recall) — the metrics that require the *generator* to be in the loop. Chapter 7 draws that boundary explicitly.
- **`model-evaluation-engineer` (peer, level 30)** owns *benchmark methodology-of-methodology*. That means calibration of judgment procedures, meta-evaluation of metrics, contamination detection, the full derivation of paired significance tests, and the multi-benchmark curation that produces new artifacts like MTEB itself. The RAG track uses those tools; the peer track builds and vouches for them. Chapter 7 draws that boundary too.

This module owns the **retriever-only** slice: metrics that need only a ranked list of doc-IDs and a set of labels. If the metric requires the generator's output, it is the eval peer's. If the metric requires a new significance-test derivation, it is the model-eval peer's. If the metric operates on a ranked list from the retriever and a set of labels — recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate — it is this module's.

```
                                                                  ┌─ RAGAS faithfulness
                                                                  ├─ RAGAS answer relevance
        ┌─ application-layer RAG eval ─────────────►               ├─ context precision-recall
        │                                                          └─ TruLens RAG triad
        │                                                                 (owned by ai-eval-engineer)
Retrieval                                                                 (peer track, level 30)
eval  ──┤
        │                                                                     ┌─ recall@k
        │                                                                     ├─ precision@k
        └─ retriever-only eval ────────────────────────────────────►          ├─ MRR / MAP
                                                                              ├─ nDCG@k / hit-rate
                                                                              └─ paired significance on deltas
                                                                                    (owned by rag-engineer)
                                                                                    (this module)
```

Both slices belong in a shipped system. This module is the retrieval slice.

## The four properties this module teaches you to measure

Retrieval quality decomposes cleanly into four operational questions. Each of the metrics in chapter 3 answers one of them; picking the metric that matches your product's question is the first skill this module builds.

1. **Did the relevant docs enter the top-k window?** The retriever's job is to hand a candidate list to the reranker (or the generator). If the relevant docs are not in the top-k, no downstream stage can rescue them. This is **recall@k** and **hit-rate**; it is what mod-107's reranker depends on and what mod-106's fusion protects.
2. **Is the top-1 (or the first few) correct?** For UX with a single-answer feel — a chatbot's "the answer is X" — the top position matters disproportionately. This is **MRR** (reciprocal rank of the first relevant hit) and **precision@1**.
3. **Is the ranking of the top-k any good?** For UX that shows a ranked list — search results, citation panel — the *order* of hits inside the top-k matters. A relevant doc at rank 1 is worth more than a relevant doc at rank 10. This is **nDCG@k**, and it is the field's default aggregate metric.
4. **How many relevant docs did we surface, on average, across the ranked list?** For exploratory / batch use — pulling every relevant regulation, every case citation — the average precision along the list matters. This is **MAP**.

Every retrieval evaluation report starts by naming which of the four questions is the product's real question, and picking the metric family accordingly.

## The judgment set is the artifact

The judgment set — sometimes called qrels (query-relevance file), following the TREC convention — is the artifact this module is organized around. Its shape is standard:

```
q1 0 d17 1
q1 0 d92 1
q1 0 d5  2
q2 0 d43 1
...
```

Four fields per line: query-ID, iteration (unused, always `0`), doc-ID, relevance grade (`0` = irrelevant, `1` = relevant, higher = more relevant for graded-relevance metrics). This is the same shape TREC uses, the same shape BEIR uses, the same shape `trec_eval` / `pytrec_eval` / `ir_measures` consume. Once you have a judgment set in this shape, every metric implementation in the field takes it directly.

Chapter 2 covers how you build one — from small curated sets (subject-matter expert labels) up to LLM-assisted labeling with a mandatory human spot-check. Chapter 3 covers what to compute from it.

## Two failures that motivate the discipline

Two well-documented failure modes give this module its urgency, and every subsequent chapter is a defense against one of them.

### Failure 1: "we shipped it because the demo felt better"

The team swaps the embedder from `all-MiniLM-L6-v2` to `bge-large-en-v1.5`, runs ten hand-picked queries, and everyone agrees the results "feel better." Six weeks later, when a user complaint surfaces on a specific query pattern, no one can tell whether the swap made it worse or whether it was already broken. There is no baseline, no metric harness, no per-query record. The upgrade was un-defensible on the day it shipped and un-diagnosable afterward.

This failure is what chapters 2 and 3 stop. A ~100-query judgment set + a `ir_measures` run + a per-query record checked into git is a five-hour investment that turns retrieval decisions from vibes into evidence. The rest of the track — mod-111's retriever CI gate, mod-104's dual-index cutover, mod-103's model shortlist — cannot function without it.

### Failure 2: "the mean went up but I can't tell if the change is real"

The team runs A/B: retriever A averages nDCG@10 = 0.412, retriever B averages 0.418. Ship B? The mean went up, but if the per-query standard deviation is 0.15 and there are 50 queries, the confidence interval on that 0.006 delta overlaps zero comfortably. Shipping on the mean alone is confusing noise for signal — one of the most-cited failure modes in the applied-IR literature (Sakai 2014's opening motivation).

Chapter 4 is the defense: a paired significance test (Wilcoxon signed-rank on per-query score deltas is the field default), a bootstrap confidence interval on the mean delta, and the discipline of never citing a mean without a `p` or a CI. It is short, mostly mechanical, and it is the single largest quality-of-decisions upgrade the module offers.

## The rest of the module

- **Chapter 2** builds the judgment set: TREC / BEIR qrels shape, curated labeling protocols, LLM-assisted labeling, the mandatory human spot-check that stops LLM-only labels from silently drifting.
- **Chapter 3** turns the judgment set into numbers: `ir_measures` / `pytrec_eval` / `trec_eval` for recall@k, precision@k, MRR, MAP, nDCG@k, hit-rate; the mental model for when each metric answers what.
- **Chapter 4** defends the numbers: paired t-test, Wilcoxon signed-rank, bootstrap CI on per-query deltas, grounded in Sakai's *Metrics, Statistics, Tests* (2014).
- **Chapter 5** attributes the numbers: the orthogonal ablation grid (chunker × embedding × retriever × reranker), the discipline of moving one axis per row, and the trap of confounded ablations.
- **Chapter 6** sanity-checks the numbers: how to run a BEIR / MTEB subset alongside your corpus, derive a corpus-fit delta from the leaderboard baseline, and read the delta.
- **Chapter 7** marks the boundary to the two peer tracks (`ai-eval-engineer`, `model-evaluation-engineer`) and lists the hand-off artifacts each expects from this one.

## What you should be able to do after this chapter

- Name the four operational questions retrieval metrics answer (top-of-window, top-1, ranked-order, average-precision-along-list) and match a product surface to the right one.
- Explain why retrieval scores are per-query random variables and why a single mean is not enough to compare two retrievers.
- Draw the boundary between retriever-only eval (this module) and application-layer RAG eval (peer `ai-eval-engineer`).
- Draw the boundary between using paired significance tests (this module) and deriving / calibrating them from scratch (peer `model-evaluation-engineer`).
- State the two motivating failures (un-defensible ship, mean-without-CI) and which chapter is the defense against each.

## Further reading

- Sakai, T. (2014). *Metrics, Statistics, Tests.* In *Bridging Between Information Retrieval and Databases* (PROMISE Winter School), Lecture Notes in Computer Science 8173. [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — the field's reference for per-query significance testing; chapter 4 uses it as the anchor.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Chapter 8, "Evaluation in Information Retrieval"](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — the textbook derivation of recall, precision, MAP, nDCG.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the reference for the qrels shape this module uses and the domain-shift motivation for chapter 6.
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — the retrieval-subset baseline chapter 6 derives from.
- Voorhees, E. M. (2001). *The Philosophy of Information Retrieval Evaluation.* In *Evaluation of Cross-Language Information Retrieval Systems* (CLEF 2001), Lecture Notes in Computer Science 2406. [DOI:10.1007/3-540-45691-0_31](https://doi.org/10.1007/3-540-45691-0_31) — the TREC-era treatment of why "the mean" is insufficient and what test-collection eval actually promises.
