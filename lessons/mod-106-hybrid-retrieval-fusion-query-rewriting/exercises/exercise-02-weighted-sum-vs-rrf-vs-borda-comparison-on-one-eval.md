# Exercise 02 — Weighted-Sum vs. RRF vs. Borda-Count: Fusion Bake-Off on One Eval Set

**Estimated effort:** ~3 hours. Python + the same two-arm retrieval stack from exercise 01, plus a labeled eval set with a held-out split.

## Objective

Run a controlled bake-off of the three fusion algorithms from chapters 2 and 3 (RRF, weighted-sum with per-query normalization, Borda-count) on a single labeled eval set. Tune weighted-sum on a training split; report all three on a held-out test split with paired significance testing. The point is not "which fusion wins" — it is *the discipline of running a fusion bake-off correctly*, so when you make the fusion choice for a real project you can defend it.

## Prerequisites

- Chapters 2 (RRF) and 3 (weighted-sum, Borda) read.
- Exercise 01 completed — you already have the two-arm retriever and an RRF implementation.
- A labeled eval set of **at least 200 queries** with relevance judgments. Weighted-sum's tuning needs enough queries to avoid noise-fitting. If you don't have 200, use a BEIR dataset that does: [SciFact](https://github.com/beir-cellar/beir) (300 test queries) or FiQA (~650 test queries) are the two most accessible.
- Familiarity with the paired *t*-test or the paired randomization test (mod-109 rigorous treatment). Quick reference: for each query, compute `metric_A - metric_B`; the paired test asks whether the mean of that per-query difference is significantly different from zero.

## Steps

**Step 1 — Split the eval set (10 min).** Split your queries into 70% train / 30% test *by query, not by document*. Use a fixed random seed. Save both splits to disk as `queries_train.json` and `queries_test.json`. Every subsequent step reports two numbers: train recall@100 (for tuning) and test recall@100 (for reporting).

**Step 2 — Establish baselines on the test split (15 min).** Compute test-split recall@100 for:

- BM25 alone.
- Dense alone.
- RRF with `k = 60` and `top_n = 100` (chapter 2 default).

This is the "if I skip this exercise" reference. Every alternative below must beat one of these to justify the added complexity.

**Step 3 — Tune weighted-sum on the training split (60 min).** Implement weighted-sum fusion with a chosen normalizer. Do both of the following:

- **Normalizer sweep.** Try three normalizers: min-max per query, z-score per query, and sum (divide each score by the arm's sum-of-top-N scores). For each, sweep `w_dense ∈ {0.0, 0.1, 0.2, ..., 1.0}` with `w_lex = 1 - w_dense`. Report training recall@100 for each of the 33 configurations (3 normalizers × 11 weights).
- **Pick the winner.** Take the (normalizer, weight) tuple with the best training recall@100.

Write one paragraph on what you learned about the normalizer choice: which normalizer's curve is smoothest? Which is most peaked? Any degenerate behavior (e.g., z-score blowing up when a query has only a few candidates)?

**Step 4 — Tune Borda-count on the training split (20 min).** Implement Borda-count fusion. Sweep `top_n ∈ {50, 100, 200}` — that is the main knob (Borda has no `k`). Report training recall@100 for each. Pick the winner.

**Step 5 — Head-to-head on the test split (30 min).** On the *held-out* test split, compute recall@100 for:

- RRF with `k = 60`.
- Weighted-sum with the (normalizer, weight) tuple from step 3.
- Borda-count with the `top_n` from step 4.

Report all three in a single table.

Now compute paired significance:

- For each query in the test split, compute per-query recall@100 for each of the three configurations.
- Run a paired *t*-test between RRF and weighted-sum, and between RRF and Borda. Alternatively, use a paired bootstrap: resample queries 1000 times, compute the mean of the per-query difference in each resample, and report the fraction of resamples where each contender beats RRF.

Write one paragraph interpreting the results:

- Which fusion wins on the test split?
- Is the delta statistically significant (p < 0.05, or paired-bootstrap fraction > 95%)?
- How does the train recall compare to the test recall? A large train-test gap for weighted-sum is the signature of eval-set overfit (chapter 3 warned about this).

**Step 6 — Per-query-class analysis (30 min).** Split the test queries by two axes and compare fusion behavior:

- **Query length.** Short (< 5 tokens) vs. long (≥ 5 tokens).
- **Arm-agreement.** For each query, compute the overlap between the top-10 BM25 and the top-10 dense (Jaccard). Split into "high agreement" (Jaccard ≥ 0.3) and "low agreement" (Jaccard < 0.3).

For each of the four buckets, report recall@100 for the three fusion algorithms. Look for buckets where one algorithm dominates. A common finding: RRF wins overall but weighted-sum wins on the "large arm-quality-gap" subset. If you see this, you have discovered a routing opportunity (chapter 5): route queries to different fusion algorithms based on a fast-computed feature.

**Step 7 — Write it up (~20 min).** Author `writeup.md` with:

- The step-2 baselines table (BM25, dense, RRF-default on test split).
- The step-3 normalizer sweep table (33 configs, training recall).
- The step-5 head-to-head table with paired-significance results.
- The step-6 per-query-class table.
- A recommendation paragraph: which fusion would you ship on this corpus, and why? Defend it against the two failure modes chapter 3 named (eval-set overfit for weighted-sum, tail-flatness for Borda).

## Acceptance criteria

- Train / test split is 70/30, seeded, and each metric reports on the test split unless labeled `(train)`.
- Weighted-sum normalizer sweep across at least 3 normalizers × 11 weights = 33 configurations reported.
- Borda-count `top_n` sweep across at least 3 values reported.
- Head-to-head table on the test split with paired significance (paired *t*-test or paired bootstrap).
- Per-query-class breakdown across at least one axis (length or arm-agreement).
- The recommendation names the pick and defends it in ≤ 200 words.
- If weighted-sum wins on train but loses on test, the writeup explicitly names this as the eval-set-overfit failure mode from chapter 3.

## Stretch goals

- **Weighted-RRF.** Implement `Σ_r w_r / (k + rank_r(d))`. Sweep `w_dense` on the training split, evaluate on the test split. Weighted-RRF is a middle ground: per-arm weighting without the score-normalization headache. Does it beat unweighted RRF? Does it match or beat weighted-sum?
- **CombSUM and CombMNZ.** Implement the Fox and Shaw 1994 classical fusion functions: CombSUM (sum of normalized scores) and CombMNZ (CombSUM multiplied by the number of arms that returned the doc). Add them to the head-to-head. CombMNZ has an "agreement bonus" that RRF hides — see if it wins on high-arm-agreement queries.
- **Statistical power calculation.** Given your eval-set size and observed effect sizes, compute the statistical power of your bake-off. If the true weighted-sum vs. RRF delta is 1 percentage point, would you have detected it? At what eval-set size *would* you? (Mod-109 covers this rigorously; this is a first-pass exercise.)
- **Cost model.** Estimate the incremental compute cost of the tuning loop for weighted-sum vs. RRF (the "engineer-hours + eval-set-collection cost" side of the choice). If your best weighted-sum only beats RRF by 0.5 recall@100 points, is that gain worth the ongoing maintenance cost of the tuning pipeline? Write a paragraph.
- **Swap the corpus.** Run the same three-way bake-off on a *second* corpus (SciFact if you started on FiQA, or vice versa). The pick that wins on one corpus may lose on the other; that is the "know when each fusion fits" lesson chapter 3 wanted.

## Notes

- Do not tune weighted-sum weights on the test split. That is not a fusion bake-off; that is finding the "best number you can report." The point of the train/test split is defending your production choice against overfit.
- Paired-metric testing is more sensitive than unpaired testing on the same eval set (the per-query pairing removes query-level variance from the noise). Use paired if you have paired data — which you do, since every fusion runs on the same queries.
- A "significant win" of 1 percentage point recall@100 that costs a full tuning + maintenance pipeline is often *not* worth shipping. Judgment on when to ship a marginal-but-real gain is a level-30 skill, not an eval-set output.
- If your winner is RRF-default, that is a valid outcome and a valuable result. RRF-default winning is not "the exercise didn't work" — it is the answer chapter 2 predicted for most workloads. Report it and defend it.
