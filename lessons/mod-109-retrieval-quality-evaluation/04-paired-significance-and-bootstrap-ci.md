# Chapter 4 — Paired Significance and Bootstrap Confidence Intervals

## Why this chapter exists

Chapter 3 gave you an aggregate table: bge-large's nDCG@10 = 0.487, bm25's = 0.412. The delta is 0.075 — plausibly large. Ship? Not yet. What chapter 3 hid, and what this chapter reveals, is that the aggregate is the mean of a per-query array with real variance. On 50 queries, a 0.075 delta might be a signal so strong you'd bet on it — or it might be within the confidence interval of "no difference at all" because a few queries swung wildly and the rest were tied.

This chapter is the field's answer: **paired significance testing** on the per-query score deltas plus a **bootstrap confidence interval** on the mean delta. Both are grounded in Sakai's *Metrics, Statistics, Tests* (2014), the IR community's reference for how comparisons are actually done. Skipping this step is the single most common source of "shipped a retriever change that turned out not to matter" — the failure that chapter 1 warned about, and this chapter's discipline is the defense.

## Why paired, and why per-query

Two retrievers over the same 50 queries produce two aligned per-query score arrays:

```
query    bm25 nDCG@10    bge nDCG@10    delta
q1       0.40           0.51           +0.11
q2       0.63           0.68           +0.05
q3       0.22           0.20           -0.02
...
q50      0.45           0.72           +0.27
```

The natural question is "is the *delta* systematically greater than zero?" — not "is bge's mean greater than bm25's mean, treating the samples as independent." An unpaired test throws away the pairing (each query is a matched sample) and loses statistical power. A paired test uses the pairing and is the correct default.

Sakai (2014) makes this precise: retrieval evaluation is **matched-pair on queries** because both systems saw exactly the same queries, and the variability across queries is much larger than the variability between systems. A test that models the pairing has orders of magnitude more power than one that doesn't.

## The two paired tests

The IR community reaches for two paired tests, both computed on the per-query delta array.

### Paired t-test

Compute the delta for each query, then test whether the mean of the deltas is significantly different from zero, assuming the deltas are approximately normally distributed.

```python
from scipy.stats import ttest_rel

deltas = [bge_scores[q] - bm25_scores[q] for q in queries]  # aligned by query
t_stat, p_value = ttest_rel(bge_scores, bm25_scores)
# equivalently: ttest_1samp(deltas, 0)
```

**When it's the right test.** When per-query deltas are approximately symmetrically distributed with no severe outliers. On graded metrics like nDCG@10 with a reasonably sized query set (n ≥ 40), the Central Limit Theorem is generous — the paired t-test is usually fine even when the deltas aren't strictly normal.

**When it's the wrong test.** When per-query deltas are heavily skewed (one system dominates on a handful of queries) or bounded in a way that produces asymmetric distributions (all deltas are in a narrow range except a few big wins). In practice, RR (reciprocal rank) is often heavy-tailed enough that the t-test loses honesty.

### Wilcoxon signed-rank test

Non-parametric alternative. Test whether the *median* of the paired differences is zero. Ranks the absolute deltas, signs them by direction, sums, tests. Robust to outliers and non-normal distributions.

```python
from scipy.stats import wilcoxon

t_stat, p_value = wilcoxon(bge_scores, bm25_scores, zero_method="wilcox")
```

**When it's the right test.** As the paired default in the IR community. Sakai (2014) recommends it as the primary paired test for IR because it is robust to the distributional assumptions the t-test relies on, and IR metrics (especially binary ones and MRR) routinely violate those assumptions.

**When it's the wrong test.** Almost never for retrieval work. Two edge cases: (a) very small query counts (n < 20) where the Wilcoxon's exact distribution is chunky; (b) when many query deltas are exactly zero — the standard Wilcoxon drops zero-deltas, which can bias the result if there are many. `scipy.stats.wilcoxon(zero_method="pratt")` handles zeros differently and is worth knowing about.

### Which to report

The RAG track's convention (matching Sakai's recommendation) is:

- **Wilcoxon signed-rank** as the primary paired test.
- **Paired t-test** as a secondary consistency check; if the two disagree, investigate — usually there's a small number of extreme-outlier queries that the t-test is over-weighting.
- **Bootstrap CI** on the mean delta as the third artifact — always report the CI alongside the p-value; it answers "how big is the delta?" while `p` answers only "is it not zero?"

## Bootstrap confidence intervals

The paired test tells you "is the delta zero or not?" The confidence interval tells you "what's a plausible range for the delta?" Both matter. A `p < 0.001` on a delta whose 95% CI is `[+0.001, +0.010]` means the change is significant but tiny; not worth shipping. A `p = 0.08` on a delta whose 95% CI is `[+0.005, +0.100]` means the change is uncertain but possibly meaningful; worth more evaluation.

Bootstrap is the standard way to produce this CI without assuming a distribution. Efron & Tibshirani (1993) is the reference; Smucker et al. (2007) applied it to IR eval and it has been the community default since.

**Algorithm.** Given per-query deltas `d_1, ..., d_n`:

1. Sample `n` deltas with replacement from the array → bootstrap sample `b^(1)`.
2. Compute the mean of `b^(1)` → one bootstrap mean.
3. Repeat B times (B = 10,000 is typical; 1,000 is a quick-and-dirty lower bound).
4. The 2.5th and 97.5th percentiles of the B bootstrap means form the **95% CI**.

```python
import numpy as np

deltas = np.array([bge_scores[q] - bm25_scores[q] for q in queries])

def bootstrap_ci(deltas, n_boot=10_000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    n = len(deltas)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return deltas.mean(), (lo, hi)

mean_delta, (lo, hi) = bootstrap_ci(deltas, rng=42)
print(f"Mean delta = {mean_delta:+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]")
```

`scipy.stats.bootstrap` since SciPy 1.7 does the same thing with a nicer API and bias-corrected-and-accelerated (BCa) intervals available — BCa is more accurate for skewed distributions and is worth using when the histogram of bootstrap means is visibly asymmetric.

```python
from scipy.stats import bootstrap
res = bootstrap((deltas,), np.mean, n_resamples=10_000,
                confidence_level=0.95, method="BCa", random_state=42)
lo, hi = res.confidence_interval
```

**Reading the CI.** The CI is for the **population mean delta** you would see if you ran the two systems on infinitely many queries drawn from the same distribution as your judgment set. If `0` is not in the interval, you have significance at that confidence level; the interval width tells you the effect-size uncertainty.

## Reporting the triple

Every retriever comparison this module produces should report **all three** numbers:

| System vs. baseline | Metric | Mean delta | 95% CI | Wilcoxon `p` |
|---|---|---|---|---|
| bge-large vs. bm25 | nDCG@10 | +0.075 | [+0.041, +0.108] | 0.0004 |
| bge-large vs. bm25 | recall@50 | +0.062 | [+0.031, +0.094] | 0.0007 |
| bge-large vs. bm25 | MRR@10 | +0.089 | [+0.048, +0.132] | 0.0002 |

The CI + `p` together give the honest story. The reviewer's checklist for a "we changed the retriever" write-up:

- Was the eval done on a fixed judgment set (chapter 2)?
- Are per-query values reported (not just aggregates)?
- Is the test paired (Wilcoxon or paired t)?
- Is a CI reported alongside the p-value?
- Is a triple of metrics reported (top-of-window + ranked-order + top-1)?

Any answer of "no" is a rewrite request, not a nit.

## Sample size and power

How many queries do you need to detect a delta of a given size?

Sakai (2020) and follow-ups on TREC "topic set size design" give the calibrated answer: for a `nDCG@10` delta of ~0.02 (a moderate practical effect) at 80% power at α = 0.05, expect to need **~100 queries** on typical text-retrieval workloads. For a `nDCG@10` delta of ~0.05, ~40 queries usually suffice. For a `nDCG@10` delta of ~0.01, expect > 400.

A back-of-envelope check without running Sakai's full topic-set-size calculation: if your per-query nDCG standard deviation is `σ_q` and you want to detect a delta `d`, you need roughly `n ≥ 8 (σ_q / d)^2` queries at 80% power, α = 0.05. On chapter 2's typical judgment sets, `σ_q ≈ 0.20` and detecting `d = 0.02` gives `n ≥ 800`; detecting `d = 0.05` gives `n ≥ 128`. This is why one-shot ~50-query judgment sets can only reliably detect large deltas.

If your significance test is coming out inconclusive (`p > 0.05` on a delta you're pretty sure is real), the fix is usually more queries in the judgment set, not more retriever tuning. Chapter 2's protocol B (LLM-assisted with spot-check) is the scalable way to grow it.

## Multiple comparisons

If you compare 6 retrievers to a baseline with 5 metrics each, you have run 30 significance tests. At α = 0.05, you expect ~1.5 false positives just by chance. The two standard corrections:

- **Bonferroni** — divide α by the number of tests. Conservative; often over-corrects.
- **Benjamini-Hochberg (FDR)** — controls the false discovery rate; usually the right choice for exploratory sweeps like chapter 5's ablation grid.

`statsmodels.stats.multitest.multipletests(pvals, method="fdr_bh")` implements BH. Apply it whenever your evaluation report has > 5 significance tests on the same set of queries.

## What paired testing does NOT solve

- **Bias in the qrels.** If your labeler (human or LLM) systematically prefers verbose passages, a retriever that returns verbose passages will look good regardless of whether that's actually better for users. Paired testing does not correct for this — chapter 2's spot-check does.
- **Distribution shift between eval and production.** Your queries and corpus are a snapshot; production is a stream. A significant win on the eval set is a *necessary* condition for shipping, not sufficient. Mod-111's drift monitor is the production-side complement.
- **Missing metrics.** A significant win on nDCG@10 with an insignificant loss on recall@100 is a *possibly harmful trade*. Reporting the triple (chapter 3) surfaces this; paired testing alone does not.

## Boundary to `model-evaluation-engineer`

The peer track at level 30 owns *methodology-of-methodology* — deriving new significance tests, calibrating them, proving their false-positive rates on IR distributions, curating cross-benchmark meta-evaluations. This module *uses* the paired Wilcoxon and bootstrap CI as tools; the peer track *vouches* for them and extends them.

Where the boundary shows up in practice:

- This module: "we compared bge-large vs. bm25 with Wilcoxon on paired nDCG@10 deltas; `p = 0.0004`, CI `[+0.041, +0.108]`."
- Peer track: "we validated on 200 synthetic IR distributions that Wilcoxon holds its α = 0.05 nominal false-positive rate on paired nDCG@10 deltas as query counts scale from 30 to 500."

If you find yourself needing the second kind of statement — either because a stakeholder is challenging the *test itself* or because you're introducing a new metric whose statistical properties are unknown — hand off to the peer track. If you are running the paired test as-published and reporting `p` + CI, you are inside this module's scope.

## Concrete: the end-to-end script

Chapter 3's example ended with an aggregate table. Chapter 4's follow-on:

```python
import ir_measures
from ir_measures import nDCG, RR, R
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, bootstrap

qrels = ir_measures.read_trec_qrels("eval/qrels.tsv")
run_a = ir_measures.read_trec_run("runs/bm25.run")
run_b = ir_measures.read_trec_run("runs/bge-large.run")

MEASURES = [nDCG@10, RR@10, R@50]

def per_query(qrels, run, measures):
    from collections import defaultdict
    by = defaultdict(dict)
    for m in ir_measures.iter_calc(measures, qrels, run):
        by[m.measure][m.query_id] = m.value
    return by

a = per_query(qrels, run_a, MEASURES)
b = per_query(qrels, run_b, MEASURES)

for measure in MEASURES:
    common = sorted(set(a[measure]) & set(b[measure]))
    va = np.array([a[measure][q] for q in common])
    vb = np.array([b[measure][q] for q in common])
    d  = vb - va

    _, p_wilcoxon = wilcoxon(vb, va, zero_method="pratt")
    _, p_ttest    = ttest_rel(vb, va)

    res = bootstrap((d,), np.mean, n_resamples=10_000,
                    confidence_level=0.95, method="BCa",
                    random_state=42)
    lo, hi = res.confidence_interval

    print(f"{measure}: mean_delta={d.mean():+.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"Wilcoxon p={p_wilcoxon:.4f}  t-test p={p_ttest:.4f}  "
          f"n={len(common)}")
```

The output is the shape every retriever comparison in this track produces. Chapter 5's ablation grid runs this over many system pairs; mod-111's CI gate runs it on every PR.

## What you should be able to do after this chapter

- Explain why paired tests are the right default for retrieval comparisons (matched-pair on queries; per-query variance dominates).
- Choose Wilcoxon signed-rank as the primary test and paired t-test as the secondary; explain when they disagree and what to do.
- Compute a bootstrap 95% CI on the mean per-query delta and read the result correctly.
- Report a retriever comparison as the triple (mean delta, CI, `p`) with the qrels version.
- Estimate the query-count needed to detect a given delta using either Sakai's calibrated tables or the `n ≥ 8(σ_q/d)^2` back-of-envelope.
- Apply Benjamini-Hochberg FDR when comparing many systems at once.
- Draw the boundary between using these tests (this module) and deriving / calibrating new ones (peer `model-evaluation-engineer`).

## Further reading

- Sakai, T. (2014). *Metrics, Statistics, Tests.* In *Bridging Between Information Retrieval and Databases,* Lecture Notes in Computer Science 8173. [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — the anchor reference; every claim in this chapter traces to it.
- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A Comparison of Statistical Significance Tests for Information Retrieval Evaluation.* CIKM. [DOI:10.1145/1321440.1321528](https://doi.org/10.1145/1321440.1321528) — the empirical study behind Wilcoxon-as-default and bootstrap-as-CI-default.
- Sakai, T. (2016). *Statistical Significance, Power, and Sample Sizes: A Systematic Review of SIGIR and TOIS, 2006-2015.* SIGIR. [DOI:10.1145/2911451.2911492](https://doi.org/10.1145/2911451.2911492) — how the IR community actually reports significance (and where it under-reports).
- Sakai, T. (2020). *On the Significance of the Non-Significance of the Top-Ranked Systems.* SIGIR (short paper). [DOI:10.1145/3397271.3401266](https://doi.org/10.1145/3397271.3401266) — a modern reminder of the trap the mean-only report falls into.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall. [DOI:10.1201/9780429246593](https://doi.org/10.1201/9780429246593) — the reference for the bootstrap.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society Series B, 57(1). [DOI:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) — the FDR paper.
- Carterette, B. (2012). *Multiple Testing in Statistical Analysis of Systems-Based Information Retrieval Experiments.* ACM TOIS 30(1). [DOI:10.1145/2094072.2094076](https://doi.org/10.1145/2094072.2094076) — multiple-comparison correction, specifically for IR eval.
- SciPy, [`scipy.stats.wilcoxon` docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [`scipy.stats.ttest_rel` docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [`scipy.stats.bootstrap` docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — implementation references for the three tests.
