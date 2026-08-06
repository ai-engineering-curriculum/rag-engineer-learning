# Exercise 03 — Paired Significance and Bootstrap CI on Per-Query Deltas

**Estimated effort:** ~3 hours. Python, the per-query CSV from exercise 02, `scipy` and `numpy`. No LLM API and no GPU required.

## Objective

Turn exercise 02's per-query score arrays into a defensible ship / no-ship decision. Compute paired Wilcoxon signed-rank, paired t-test, and a BCa bootstrap 95% CI for the mean per-query delta. Report the **triple** (mean delta, CI, `p`) that chapter 4 requires. Estimate the query-count needed to resolve a delta smaller than yours. Apply Benjamini-Hochberg FDR correction to the multi-metric comparison.

At the end you have `report/significance.md`, a runnable significance-testing script that mod-111's CI will wrap, and the sample-size back-of-envelope for growing the judgment set.

## Prerequisites

- Chapter 4 read.
- Exercise 02 complete — `report/per-query.csv` exists with per-query metric values for ≥ 2 runs × ≥ 3 metrics (recall@N, nDCG@10, MRR@10 as the minimum triple).
- Python 3.10+, `pip install pandas scipy statsmodels`.

## Steps

**Step 1 — Load per-query deltas (15 minutes).** In `significance.py`:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("report/per-query.csv")
# columns: run, measure, query_id, value

# Reshape: index=(query_id, measure), columns=run, values=value
wide = df.pivot_table(index=["query_id", "measure"], columns="run",
                      values="value").reset_index()

# For each metric, compute per-query delta (challenger - baseline).
BASELINE, CHALLENGER = "bm25", "bge-large"
wide["delta"] = wide[CHALLENGER] - wide[BASELINE]
```

Sanity: for each metric, `wide[wide.measure == "nDCG@10"].shape[0]` should equal your query count. Verify no queries were dropped.

**Step 2 — Paired Wilcoxon + paired t-test per metric (30 minutes).** In `significance.py`:

```python
from scipy.stats import wilcoxon, ttest_rel

results = []
for measure, g in wide.groupby("measure"):
    a = g[BASELINE].to_numpy()
    b = g[CHALLENGER].to_numpy()
    d = b - a
    # Use zero_method="pratt" to keep zero-deltas rather than drop.
    _, p_w = wilcoxon(b, a, zero_method="pratt")
    _, p_t = ttest_rel(b, a)
    results.append({
        "measure":    measure,
        "n":          len(d),
        "mean_delta": d.mean(),
        "median_delta": np.median(d),
        "p_wilcoxon": p_w,
        "p_ttest":    p_t,
    })

sig = pd.DataFrame(results)
print(sig.to_string(index=False))
```

**Step 3 — Bootstrap BCa 95% CI (30 minutes).** In `significance.py`:

```python
from scipy.stats import bootstrap

def bootstrap_ci(d, n_boot=10_000, ci=0.95, seed=42):
    res = bootstrap((d,), np.mean, n_resamples=n_boot,
                    confidence_level=ci, method="BCa",
                    random_state=seed)
    return res.confidence_interval.low, res.confidence_interval.high

sig["ci_lo"], sig["ci_hi"] = zip(*[
    bootstrap_ci(wide[wide.measure == m]["delta"].to_numpy())
    for m in sig["measure"]
])
```

Sanity: for a metric where `p_wilcoxon < 0.05`, `0` should not be inside `[ci_lo, ci_hi]`. If it is, either the test or the CI is mis-computed; debug before proceeding.

**Step 4 — Benjamini-Hochberg FDR correction (15 minutes).** In `significance.py`:

If you're running > 3 metrics or comparing > 1 challenger to the baseline, correct the p-values:

```python
from statsmodels.stats.multitest import multipletests
_, p_adj, _, _ = multipletests(sig["p_wilcoxon"].tolist(), method="fdr_bh")
sig["p_wilcoxon_bh"] = p_adj
```

For a 2-retriever × 3-metric comparison, correction usually shifts p-values by a small amount; on a 5×5 grid it matters much more.

**Step 5 — Sample-size back-of-envelope (20 minutes).** In `significance.py`:

For each metric, estimate the query-count needed to detect a delta of 0.02 nDCG@10 (a common "meaningful but small" effect) at 80% power at α = 0.05:

```python
def n_for_delta(sigma_q, target_delta):
    # chapter 4's back-of-envelope: n >= 8 * (sigma_q / target_delta)^2
    return int(np.ceil(8 * (sigma_q / target_delta) ** 2))

for _, row in sig.iterrows():
    m = row["measure"]
    sigma_q = wide[wide.measure == m]["delta"].std()
    for target in (0.01, 0.02, 0.05):
        print(f"{m}: to detect Δ={target:.2f} with σ_q={sigma_q:.3f}, "
              f"need n ≥ {n_for_delta(sigma_q, target)}")
```

If your current query count is well below the number needed for your target delta, that is your action item: grow the judgment set (exercise 01 protocol B is the scalable path) before you draw stronger conclusions.

**Step 6 — Bootstrap distribution plot (20 minutes; optional but instructive).** In `report/bootstrap-hist.png`:

Plot the histogram of bootstrap means for your primary metric (nDCG@10). If the histogram is visibly asymmetric or long-tailed, BCa's asymmetric correction is doing real work — cite that as your reason for choosing BCa over percentile.

```python
import matplotlib.pyplot as plt

d = wide[wide.measure == "nDCG@10"]["delta"].to_numpy()
rng = np.random.default_rng(42)
boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10_000)])

plt.hist(boot, bins=60, edgecolor="black", alpha=0.7)
plt.axvline(d.mean(), linestyle="--", label=f"observed mean = {d.mean():+.4f}")
plt.axvline(np.percentile(boot, 2.5),  linestyle=":", label="percentile 2.5%")
plt.axvline(np.percentile(boot, 97.5), linestyle=":", label="percentile 97.5%")
plt.axvline(0, linestyle="-", label="no-effect line")
plt.legend(); plt.xlabel("bootstrap mean delta (nDCG@10)"); plt.title(f"{CHALLENGER} vs {BASELINE}")
plt.tight_layout(); plt.savefig("report/bootstrap-hist.png", dpi=120)
```

**Step 7 — Write the significance report (30 minutes).** In `report/significance.md`:

- Header: qrels version, baseline, challenger, query count.
- The triple table (chapter 4's default shape):

    | Metric | Mean delta | 95% CI (BCa) | Wilcoxon `p` | t-test `p` | Wilcoxon `p_BH` |
    | --- | --- | --- | --- | --- | --- |

- One paragraph per metric interpreting the row:
  - "nDCG@10 improved by X points; CI [lo, hi]; `p = ...` — this is a *real* effect / *within noise*."
  - Any disagreement between Wilcoxon and t-test — investigate and explain (usually outlier queries).
- The sample-size table from step 5 with a recommendation: "to detect a 0.02 delta on our corpus at 80% power we would need N queries; we have M; the gap is..."
- **Ship recommendation** with the qualifier from chapter 7 — "on retrieval quality alone; await peer's application-layer numbers."

**Step 8 — Add the "regression alert" mini-check (10 minutes).** In `significance.py`:

Add a mode where the baseline is the *current shipping retriever* and the challenger is a candidate. Print a red-flag if:

- Any *primary* metric regresses with Wilcoxon `p < 0.05` and negative delta.
- Any *secondary* metric regresses by > 5% with Wilcoxon `p < 0.10`.

Mod-111's retriever CI runs this exact check on every PR; you're pre-building the interface.

## Acceptance criteria

- `significance.py` computes paired Wilcoxon, paired t-test, and BCa 95% bootstrap CI per metric.
- `report/significance.md` reports the triple (mean delta, CI, `p`) for at least 3 metrics.
- FDR correction applied when > 3 comparisons.
- Sample-size back-of-envelope reports the query counts needed to detect deltas of 0.01, 0.02, 0.05.
- Ship recommendation follows chapter 7's boundary — "on retrieval quality alone; peer's application-layer eval pending."
- Regression-alert check runs and prints a clear pass / fail on primary + secondary metrics.

## Stretch goals

- **Permutation test as third leg.** Add `scipy.stats.permutation_test(..., permutation_type="samples")` on the paired deltas. If Wilcoxon, t-test, and permutation-test agree, the finding is robust; if they disagree, dig in — usually there's a small number of high-influence queries.
- **Per-stratum significance.** If exercise 01's stretch goal produced difficulty strata, run the Wilcoxon separately on each stratum. Some retriever wins concentrate in the "hard" stratum (dense embedders beating BM25 there) or the "easy" stratum (BM25 doing fine on hard queries with unique terminology).
- **Cohen's *d* for effect size.** `d = (mean_challenger - mean_baseline) / pooled_std`. Publish alongside `p` — a highly significant tiny-`d` effect (0.01) is usually not worth shipping.
- **Query-count sensitivity curve.** For 10 random subsets of your query set at n = 20, 30, 50, 75, 100, plot Wilcoxon `p` and CI width. Shows how many queries you actually needed to reach `p < 0.05`; often striking that you were over/under-labeling.
- **Multiple challengers vs. one baseline.** Add a 3rd retriever from exercise 02. Run Wilcoxon for every (challenger, baseline) pair with FDR correction. Rank the challengers by mean delta and by CI-lower-bound; note when the two orderings differ.
- **Wire into the retriever CI stub.** Add a `--fail-on-regression` flag to `significance.py` that exits non-zero if the regression-alert check triggers. This is the primitive mod-111's CI wraps.

## Notes

- **The Wilcoxon `zero_method`.** Chapter 4's caveat: `wilcoxon(..., zero_method="wilcox")` drops zero-delta queries (the classical convention). `zero_method="pratt"` includes them and is the recommended default for IR eval — dropping too many zero-deltas biases the test.
- **Use BCa, not percentile.** BCa's bias and skewness corrections matter when the bootstrap distribution is asymmetric — which is the default for IR metric deltas.
- **A significant `p` on a tiny CI is not a ship condition.** Report both. A retriever change with `p < 0.001` on a delta whose CI is `[+0.001, +0.005]` is significantly non-zero and operationally irrelevant.
- **A non-significant `p` on a wide CI is not a "no ship" condition.** It means your judgment set is too small to resolve the question. Grow it (exercise 01 protocol B) before you decide.
- **Do not report only `p`.** The IR community's failure mode (Sakai 2016) is under-reporting effect size. Always publish the CI.
- **The regression alert is a *gate*, not a *verdict*.** It stops obvious regressions from shipping without review; it does not decide when to ship. Chapter 7's ship recommendation still requires the peer's application-layer eval.
