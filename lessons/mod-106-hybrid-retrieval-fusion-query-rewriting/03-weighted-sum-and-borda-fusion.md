# Chapter 3 — Weighted-Sum and Borda-Count Fusion: When They Beat RRF

## Why this chapter exists

RRF (chapter 2) is the well-tested default, but it is not the only fusion primitive worth knowing. Two others come up often enough that you should be able to reach for them when they fit:

- **Weighted-sum (linear) fusion** — a score-based combiner that expresses per-arm confidence directly, at the cost of the score-scale mismatch problem chapter 1 named.
- **Borda-count fusion** — the rank-based cousin of RRF that gives linear rather than reciprocal weight to ranks. Older, simpler, and occasionally the right pick.

This chapter covers the mechanics of both, the eval-set conditions under which each beats RRF, and — critically — the score-normalization strategies you must use to make weighted-sum usable. Chapter 4 turns to query rewriting; this chapter is the final piece of the fusion toolkit at the retrieval-output stage.

## Weighted-sum fusion

### The formula

Given retrievers `r ∈ R`, each producing a score `s_r(d)` for a document `d`:

```
score_hybrid(d) = Σ_{r ∈ R}   w_r · normalize(s_r(d))
```

Where:

- **`normalize(...)`** is a per-arm score-normalization function that maps the arm's raw score into a shared, per-query-comparable range. Chapter 1 named the three common normalizers: min-max, z-score, and CDF/rank. See "Normalization strategies" below.
- **`w_r`** is a per-arm weight in [0, 1], typically constrained so `Σ_r w_r = 1`. Set by grid search on a labeled eval set.
- **Documents absent from an arm** get `s_r(d) = 0` after normalization (or, in some implementations, the mean of the retrieved scores — pick a convention and stick with it).

When it works, weighted-sum can express things RRF can't: "trust the dense arm 70%, trust BM25 30%." When it doesn't, it fails silently — a poorly-normalized weighted-sum can rank docs worse than either single arm.

### Normalization strategies

Every honest weighted-sum implementation is at least half normalization code. The three normalizers you will see:

**Min-max normalization.** Rescale each arm's top-N scores to [0, 1]:

```
normalize_min_max(s_r(d)) = (s_r(d) - min_r) / (max_r - min_r)
```

Where `min_r, max_r` are the min and max scores in retriever `r`'s top-N output for *this query*. Pros: bounded, easy. Cons: fully sensitive to outliers (one very-high top-1 score compresses the rest of the distribution toward 0), and destroys "how much stronger is top-1 than top-2" information because the rescaling is per-query.

**Z-score normalization.** Subtract mean, divide by standard deviation:

```
normalize_z(s_r(d)) = (s_r(d) - μ_r) / σ_r
```

Where `μ_r, σ_r` are the mean and standard deviation of retriever `r`'s top-N scores for this query. Pros: robust to single outliers, well-defined statistically. Cons: assumes roughly Gaussian score distributions. BM25 is not remotely Gaussian (long tail); cosine at the top of a strong retriever is not Gaussian either (near-ceiling). Z-score misbehaves on both.

**Sum normalization.** Divide each score by the sum of the arm's top-N scores:

```
normalize_sum(s_r(d)) = s_r(d) / Σ_{d' in top-N_r} s_r(d')
```

Turns each arm's output into a probability distribution over the top-N. Related to the "convex combination" methods in the fusion literature. Pros: no distributional assumptions. Cons: still per-query-dependent; comparing across queries needs care.

**Reciprocal normalization / rank normalization.** Replace the score with `1 / (k + rank)` or with `1 - rank/N`. At which point you have reinvented RRF or Borda-count and should just use those directly.

The pattern in serious weighted-sum implementations: **min-max normalization per query per arm, or z-score if you have strong evidence the score distribution is well-behaved.** Neither is beautiful; both work if you tune them.

### The eval loop for weighted-sum

Weighted-sum has two knobs to tune (normalizer choice + per-arm weights) versus RRF's one (`k`). That is more knobs, more expressive power, more overfitting risk. The right eval loop:

1. **Baseline against RRF.** Compute recall@100 for RRF with `k = 60` on your labeled eval set.
2. **Fix a normalization strategy.** Start with min-max per-query.
3. **Grid-sweep per-arm weights.** For two arms, try `w_dense ∈ {0.0, 0.1, 0.2, ..., 1.0}` with `w_lex = 1 - w_dense`. That is 11 configurations. Report recall@100 for each.
4. **Compare to RRF.** Best weighted-sum vs. RRF baseline, on a held-out portion of the eval set (not the tuning portion). If weighted-sum wins by a real margin (paired significance), ship it. If it does not, ship RRF and save the tuning cost.
5. **Try a different normalizer if step 4 lost.** Switch to z-score or sum normalization; re-sweep the weights. If none beat RRF, keep RRF.
6. **Beware of eval-set overfit.** Weighted-sum's extra knobs make it easy to tune to eval-set quirks. Use a hold-out (mod-109 covers this rigorously); expect the shipped-config number to be 1–3 points below the tuning-set best.

Weighted-sum tends to win RRF when:

- The two arms have **very different quality** (dense recall@100 is 0.75, BM25 recall@100 is 0.45). RRF weights them equally and drags the strong arm down; a well-tuned weighted-sum expresses "trust dense more."
- The **corpus is stable** and the eval set is representative. Weight overfitting is a real risk if either of these fails.
- You have a **large labeled eval set** (500+ queries, ideally). Small eval sets can't reliably distinguish weighted-sum from RRF, and the tuning wins are usually noise.

Weighted-sum tends to lose RRF when:

- The corpus or query mix shifts frequently. Weights tuned on last quarter's eval set may not fit this quarter's traffic.
- The eval set is small (≤ 100 queries). Weight sweep on 100 queries usually finds a noise-driven winner that doesn't generalise.
- The retrievers are near-equally strong. There is nothing for the weights to express beyond what RRF already captures.

### Weighted-RRF as a middle ground

Some implementations expose **weighted-RRF**:

```
weighted_RRF(d) = Σ_r   w_r / (k + rank_r(d))
```

This gets you per-arm weighting without the score-normalization headache. OpenSearch's hybrid-search pipeline supports it (via the `arithmetic_mean` or `harmonic_mean` normalization processors with per-subquery weights); Elasticsearch's `rrf` retriever does not (as of 8.16). LangChain's [EnsembleRetriever](https://python.langchain.com/docs/integrations/retrievers/ensemble/) exposes `weights` for a weighted-RRF combiner.

The eval loop for weighted-RRF is simpler than weighted-sum: skip the normalizer choice, sweep `w_r` and possibly `k`. This is the natural next step after RRF-with-defaults if you find one arm is clearly stronger than the other.

## Borda-count fusion

Named after Jean-Charles de Borda's 18th-century voting system ([Borda count](https://en.wikipedia.org/wiki/Borda_count)), applied to IR as early as Aslam and Montague 2001 (*Models for Metasearch*, SIGIR 2001). The formula:

```
Borda(d) = Σ_r   (N_r - rank_r(d) + 1)
```

Where `N_r` is the length of retriever `r`'s ranked output (usually `top_n`). Documents absent from an arm get 0.

The intuition: rank 1 in a top-100 list contributes 100 points, rank 2 contributes 99, ..., rank 100 contributes 1. Linear decay by rank.

### Borda vs. RRF

The two are structurally similar (both rank-only, both sum-of-per-arm) and differ only in the shape of the rank-weighting function:

- **RRF** — `1 / (k + rank)`. Steep near the top, flat past rank 20.
- **Borda** — `N - rank + 1`. Linear all the way through.

The consequence: **Borda gives more weight to the tail than RRF.** A document at rank 100 in a top-100 list still contributes 1 point in Borda; in RRF with `k = 60`, it contributes `1 / 160 ≈ 0.006` — small but nonzero. The relative weight of top-1 vs. rank-50 is much smaller in Borda than in RRF.

Borda tends to win RRF when:

- **The retrievers are noisy at the top and reliable through the tail.** If your top-5 is a mess but ranks 20–100 are consistently OK, Borda gives you the averaging benefit RRF underweights.
- **You care about set inclusion more than order.** Borda's flatter shape produces a more "the union of the top-100s" behavior; RRF is more "the top-5s dominate the top." If your downstream is a reranker that will re-order the top-100 anyway, Borda's flatter shape is fine.

Borda tends to lose RRF when:

- **The retrievers are reliable at the top.** RRF's top-heavy shape captures the "top-1 vs. top-2 matters most" intuition. Borda flattens that away.
- **You have very different `N_r` per arm.** Borda's per-arm score depends on `N_r`; if one arm returns top-1000 and another returns top-100, the tail contributions from the long-arm dominate. Normalize `N_r` to a common value across arms before combining.

The Cormack 2009 paper compares RRF explicitly against Borda-count and Condorcet-style fusions and reports RRF as the winner on TREC collections. Later work (Bruch et al. 2023) confirms RRF as the general default, with Borda occasionally competitive on specific tasks. Treat Borda as "worth trying once on your eval set," not as a default.

### Reference implementation

Same shape as the RRF reference, one line different:

```python
def borda_count_fusion(
    ranked_lists: list[list[str]],
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with Borda count. Assumes all lists are the same length N.
    Rank 1 gets N points, rank 2 gets N-1, ..., rank N gets 1.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        n = len(ranked)
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += (n - rank + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n is not None else fused
```

Note the "assumes all lists are the same length" assumption; if not, truncate them before calling, or you will unfairly reward the longer list's participants.

## The one-page decision table

Which fusion to use, at a glance:

| Situation | Fusion pick | Why |
|---|---|---|
| First hybrid stack, no eval set yet | RRF, `k = 60` | Cormack default, minimal tuning risk. |
| Small eval set (≤ 100 queries) | RRF, `k = 60` | Weighted-sum will noise-fit; not worth the risk. |
| Arms are very different quality | Weighted-RRF or weighted-sum | Express per-arm trust directly. |
| Large eval set (500+ queries), stable corpus | Try RRF baseline, then weighted-sum sweep | The extra tuning knobs can be defended with signal. |
| Retrievers are noisy at the top, reliable through the tail | Try Borda-count | Its flatter weight shape averages the tail more. |
| Corpus / query mix shifts weekly | RRF, `k = 60` | Weight-tuning payoff decays; RRF is stable across drift. |
| Downstream reranker will re-order top-100 anyway | RRF or Borda; both work | Set-inclusion matters more than fused ordering. |

## The three anti-patterns to avoid

1. **Weighted-sum with no per-query normalization.** You are back in the score-scale-mismatch pit chapter 1 described. Every real weighted-sum has some normalization; treat "raw score weighted-sum" as broken by construction.
2. **Grid-searching weights on the training / retrieval-labeled set only.** Weighted-sum's extra knobs make eval-set overfit easy. Always hold out at least 30% of your labeled queries for a final comparison to the RRF baseline. If the hold-out shows no lift, ship RRF.
3. **Trusting an off-the-shelf "hybrid" that doesn't say which fusion it uses.** Frameworks silently pick fusion algorithms and often silently change them across versions. Ask, read, or measure — do not trust "hybrid: true" in a config file to be doing what you think.

## What this fusion pipeline looks like end-to-end

A concrete production shape for a two-arm hybrid with RRF-baseline-then-optional-weighted-sum:

```
Query
  │
  ├─► BM25 retriever (top-100)   ─┐
  │                                ├─► Fusion (RRF or weighted-sum)
  ├─► Dense retriever (top-100)  ─┘        │
  │                                          ▼
  ▼                                  Top-100 fused candidates
Optional query rewrites (ch 4)                │
  │                                            ▼
  └─► Optional routing (ch 5)          Reranker (mod-107)
                                                 │
                                                 ▼
                                        Top-3 to top-10 for LLM
```

That is the pipeline mod-106 delivers. Chapter 4 adds the *rewriting* step; chapter 5 adds *routing*; chapter 6 layers *filters* under the retrievers; chapter 7 hands the whole thing off to `applied-ai-engineer` via a fixed interface.

## What is NOT in this chapter (and where it goes)

- **Query rewriting patterns** — chapter 4.
- **Semantic routing** — chapter 5.
- **Filter-heavy composition** — chapter 6.
- **The retriever interface contract** — chapter 7.
- **Neural rerankers** — mod-107.
- **The right paired-significance test to defend a fusion delta** — mod-109.

## What you should be able to do after this chapter

- Write the weighted-sum formula from memory: `Σ_r w_r · normalize(s_r(d))`, and name at least two normalization strategies (min-max, z-score, sum).
- Name the two conditions under which weighted-sum tends to beat RRF (very different arm quality, large stable eval set) and the two under which it loses (small eval set, drifting corpus).
- Distinguish RRF's `1 / (k + rank)` from Borda's `N - rank + 1` and know which is more top-heavy.
- Recognise weighted-RRF as a middle ground: per-arm weighting without score-normalization headaches.
- Defend a fusion-algorithm choice on a real corpus using the one-page decision table.

## Further reading

- Aslam, J. A., and Montague, M. (2001). *Models for metasearch.* SIGIR 2001. [ACM DL](https://dl.acm.org/doi/10.1145/383952.384007) — the classical treatment of Borda, Condorcet, and rank-combination methods in IR.
- Fox, E. A., and Shaw, J. A. (1994). *Combination of Multiple Searches.* TREC-2 proceedings. [NIST TREC](https://trec.nist.gov/pubs/trec2/papers/txt/22.txt) — the earliest CombSUM / CombMNZ paper; the weighted-sum family origin.
- Manmatha, R., Rath, T., and Feng, F. (2001). *Modeling Score Distributions for Combining the Outputs of Search Engines.* SIGIR 2001. [ACM DL](https://dl.acm.org/doi/10.1145/383952.384005) — the score-distribution modelling paper that justifies careful weighted-sum normalization.
- Bruch, S., Nardini, F. M., Nardo, C., Perego, R., and Tonellotto, N. (2023). *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM TOIS. [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) — a 2023 empirical study across BEIR and open-domain benchmarks; the modern rigorous view on when each fusion function wins.
- Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009. [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the RRF paper, cited from chapter 2 and re-cited here because it compares RRF against the alternatives this chapter covers.
