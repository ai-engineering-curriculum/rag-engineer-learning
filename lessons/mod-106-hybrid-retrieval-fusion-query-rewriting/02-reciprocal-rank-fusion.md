# Chapter 2 — Reciprocal Rank Fusion: The Default Fusion Algorithm

## Why this chapter exists

Chapter 1 named the score-scale mismatch problem: BM25 and dense-cosine scores live on scales that shift per query, and a naïve weighted sum fights that problem badly. **Reciprocal-rank fusion (RRF)**, introduced by Cormack, Clarke, and Büttcher in a two-page 2009 SIGIR short paper ([*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)), sidesteps the entire problem by throwing away the scores and combining only the *ranks*. It has been the well-tested default across the retrieval literature ever since and is now the built-in fusion primitive in Elasticsearch ([`rrf` retriever](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)), OpenSearch (via the [hybrid query](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) and the [neural search + normalization processor](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) pipeline), Vespa (as a [rank profile phase](https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile)), and every serious retrieval framework (LangChain, LlamaIndex, Haystack).

This chapter covers the RRF formula, the intuition behind it, the one parameter (`k`) and how to set it, the caveats worth naming, and a reference implementation. Chapter 3 covers the alternatives (weighted-sum, Borda) and when they beat RRF.

## The formula

For a document `d` and a set of retrievers `R` where each retriever `r ∈ R` produces a ranked list, the RRF score is:

```
RRF(d) = Σ_{r ∈ R}   1 / (k + rank_r(d))
```

Where:

- **`rank_r(d)`** — the rank of document `d` in retriever `r`'s output. Ranks are **1-indexed** in the original paper: the top result is rank 1, second is rank 2, etc. Documents not returned by retriever `r` contribute nothing to the sum (equivalently, they contribute `1 / (k + ∞) = 0`).
- **`k`** — a constant, empirically set. The Cormack paper used **`k = 60`** and demonstrated that value works across many collections; it remains the industry default. See "The `k` parameter" section below for the intuition and when to move it.

That is the entire algorithm. Every implementation, from the two-line Python function in chapter 2 of the LangChain docs to the built-in Elasticsearch retriever, is this formula with edge-case handling.

### Worked example

Take two retrievers, each returning a top-5 list for the query "how to reset admin password":

| Rank | Dense arm | Lexical arm |
|---|---|---|
| 1 | doc_A | doc_C |
| 2 | doc_B | doc_A |
| 3 | doc_C | doc_D |
| 4 | doc_D | doc_E |
| 5 | doc_E | doc_F |

With `k = 60`:

- **doc_A**: `1 / (60 + 1) + 1 / (60 + 2) = 0.01639 + 0.01613 = 0.03252`
- **doc_B**: `1 / (60 + 2) + 0 = 0.01613`
- **doc_C**: `1 / (60 + 3) + 1 / (60 + 1) = 0.01587 + 0.01639 = 0.03226`
- **doc_D**: `1 / (60 + 4) + 1 / (60 + 3) = 0.01563 + 0.01587 = 0.03150`
- **doc_E**: `1 / (60 + 5) + 1 / (60 + 4) = 0.01538 + 0.01563 = 0.03101`
- **doc_F**: `0 + 1 / (60 + 5) = 0.01538`

Fused ranking: **doc_A → doc_C → doc_D → doc_E → doc_B → doc_F**. The document that appears in both arms near the top (doc_A) wins. A document that appears only in one arm high up (doc_B, dense only at rank 2) loses to a document that appears in both arms further down (doc_D, both arms around rank 3–4). That is exactly the property you want: presence in multiple retrievers is evidence beyond presence in any single retriever.

## Why RRF works — three properties worth naming

Two pages of algebra in the Cormack paper are worth reading once. The three properties that make the algorithm robust:

### 1. It ignores the scores entirely

The dense-cosine 0.82 vs. 0.75 gap and the BM25 22 vs. 8 gap are both invisible. Only the ranks feed the fused score. This is the property that makes RRF immune to the score-scale mismatch chapter 1 spent a section on.

The obvious objection: "aren't we throwing away information?" You are — but the information you throw away (raw scores) is noisier than you think. Real-world BM25 top-5 gaps are dominated by IDF fluctuations across queries and by document-length quirks, not by relevance signal. Real-world dense-cosine top-5 gaps are compressed near the top (the model is confident about the top result and less confident about ranks 4–5, so the score gap is small even though the relevance gap is large). Ranks are a coarser but more robust signal. Rank-based fusion consistently matches or beats score-based fusion in the Bruch et al. 2023 study ([arXiv:2210.11934](https://arxiv.org/abs/2210.11934)) precisely for this reason.

### 2. The `1 / (k + rank)` shape produces the right tail behavior

The reciprocal-of-rank function has two properties:

- **Steep near rank 1, flat by rank 100.** The gap between rank 1 and rank 2 is bigger than the gap between rank 50 and rank 51. This matches your intuition that "top-1 vs. top-2 matters more than top-50 vs. top-51."
- **Never negative, never zero for present docs, exactly zero for absent docs.** The absence-vs-presence gap dominates the presence-in-position gap once you go past rank ~10. That is the right prior: a doc that appears in one arm's top-5 but not at all in the other arm's top-100 is meaningfully less strong than a doc that appears in both arms' top-20.

### 3. It behaves well with more than two arms

The sum formulation extends trivially to three, four, or *n* retrievers. If you add a SPLADE arm and a rewrite-fanout arm, you sum over all of them; each retriever contributes to any doc it returned. There is no "how do I combine three normalized-score distributions?" problem because there is no distribution to normalize.

The main *thing to watch* with many arms is diminishing returns: past 3–4 sufficiently-diverse arms, the recall gain flattens fast and the compute cost keeps growing linearly. Bruch et al. 2023 quantifies this on BEIR.

## The `k` parameter

`k` is the only knob. The Cormack paper set `k = 60` based on grid search across TREC collections and argued the algorithm is not very sensitive around that value. Sixteen years of production use has borne that out — most implementations ship `k = 60` and most eval-set sweeps land between `k = 20` and `k = 100`.

The intuition:

- **Small `k` (e.g., `k = 5`) — top-ranked docs dominate.** With `k = 5`, `1/(5+1) - 1/(5+2) = 0.024` and `1/(5+10) - 1/(5+11) = 0.004`. The top few ranks matter a lot more than mid-ranks. Use when your retrievers are highly reliable at the top and you want a document that appears at rank 1 in one arm to be very hard for a document appearing at ranks 5+10 in two arms to beat.
- **Large `k` (e.g., `k = 200`) — the whole tail matters.** The score gap between rank 1 and rank 100 shrinks. A document appearing at rank 50 in two arms is competitive with a document appearing at rank 1 in one arm. Use when the retrievers are noisy at the top (unreliable top-1) and you want fusion to average over more of the tail.
- **`k = 60` (Cormack default)** — a well-tested middle. On BEIR-shaped heterogeneous collections and enterprise-search-shaped corpora, it is a hard number to beat without a lot of tuning.

The right way to set `k` if defaults don't work: **grid-sweep on a labeled eval set** with recall@100 as the metric. Try `k ∈ {10, 20, 40, 60, 100, 200}`; pick the best. Do not tune `k` by eyeballing top-3 on ad-hoc queries — you will overfit to your intuition, which does not generalise.

**One thing to watch**: `k` should be *large enough to keep the reciprocal function well-behaved for low ranks* (`k = 0` would blow up division on rank 1 in some implementations if you count from 0; more importantly, `k` around 1 makes the top-1 gap so dominant that the algorithm degenerates into "whichever arm's rank-1 document wins"). Anything above `k ≈ 10` is safe; below that, verify explicitly.

## Implementation — reference Python

The full algorithm, less edge cases, in ~15 lines:

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists into one ranked list using RRF.

    Args:
        ranked_lists: one list of doc IDs per retriever, in rank order (rank 1 first).
        k: RRF k constant. Cormack default is 60.
        top_n: return only the top-N fused docs. None returns all.

    Returns:
        List of (doc_id, rrf_score) tuples, sorted descending by score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n is not None else fused
```

The edges to handle in a production version:

- **Duplicate doc IDs within a single retriever's output.** Should not happen with well-formed retrievers; if it does, treat the *first* occurrence's rank as the rank and ignore subsequent occurrences.
- **Very different top-*n* per arm.** If dense returns top-1000 and BM25 returns top-100, docs appearing only in the dense long-tail will get small RRF scores from just one arm. Usually you truncate all arms to the same `top_n` before fusion — RRF over `top_n = 200` per arm is a common shape.
- **Tie-breaking.** If two docs have identical RRF scores (e.g., both appear only in one arm at the same rank in a rare collision), break the tie by lexical doc_id or by the more-trusted arm's rank. Ties are extremely rare with real-valued scores but do occur, so pick a deterministic policy.

## Implementation — Elasticsearch native

Elasticsearch 8.8+ ships RRF as a first-class [`rrf` retriever](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html):

```json
POST /support_articles/_search
{
  "size": 20,
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "multi_match": {
                "query": "reset admin password",
                "fields": ["title^3", "body"]
              }
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector_builder": {
              "text_embedding": {
                "model_id": "sentence-transformers__all-minilm-l12-v2",
                "model_text": "reset admin password"
              }
            },
            "k": 100,
            "num_candidates": 500
          }
        }
      ],
      "rank_constant": 60,
      "rank_window_size": 100
    }
  }
}
```

Two knobs matter:

- **`rank_constant`** — the `k` in the formula. Elasticsearch defaults to 60.
- **`rank_window_size`** — the top-N per arm that participates in fusion. Larger windows give more recall at more cost; 100–200 is a common band.

The result is a single ranked list; the underlying arms' scores are gone. Retrieve `size` docs from the fused ranking.

## Implementation — OpenSearch native

OpenSearch exposes RRF via the [hybrid query](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) combined with a [normalization processor](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) in the search pipeline. You define a pipeline that specifies the fusion technique:

```json
PUT /_search/pipeline/rrf-hybrid-pipeline
{
  "description": "RRF hybrid retrieval pipeline",
  "phase_results_processors": [
    {
      "score-ranker-processor": {
        "combination": { "technique": "rrf", "parameters": { "rank_constant": 60 } }
      }
    }
  ]
}
```

Then query with a `hybrid` compound query that runs a BM25 sub-query and a neural sub-query. Details in the OpenSearch [hybrid search tutorial](https://opensearch.org/docs/latest/search-plugins/hybrid-search/); the RRF-specific processor lands in 2.19+.

## RRF's caveats worth naming

Three real limitations you should know so you can defend the default:

### 1. It has no way to express "one arm is more trustworthy than the other"

RRF weights all arms equally. If your dense arm is genuinely twice as good as your lexical arm on your corpus, RRF cannot express that. You have three options:

- Live with it — RRF is usually good enough that per-arm weighting is not necessary.
- Use weighted-RRF, which some implementations offer: `RRF(d) = Σ_r w_r / (k + rank_r(d))`. Elasticsearch's `rrf` retriever does *not* support per-retriever weights (as of 8.16); OpenSearch's hybrid query does support per-subquery weights via the arithmetic-mean normalization processor. Chapter 3 discusses this pattern under weighted-sum.
- Switch to weighted-sum fusion with score normalization (chapter 3). Only worth doing if you have eval-set labels to tune the weights against.

### 2. It hides the arm-arm agreement signal

A document that appears at rank 1 in both arms and a document that appears at rank 1 in one arm and rank 20 in the other can produce similar RRF scores at typical `k`. The fusion loses the "both arms strongly agreed" signal. If you have a downstream reranker (mod-107), this rarely matters — the reranker's per-doc score will separate them. If you don't have a reranker, weighted-sum with a bonus for high-agreement docs may serve you better.

### 3. It assumes each arm's ranks are independently meaningful

If two of your arms are near-clones (say, two BM25 arms with slightly different analyzers), RRF will double-count their agreement. This is the "hidden ensemble" problem: adding a similar arm inflates the RRF weight on any doc both arms return. In practice this is why you should hybridize *different signals* (dense + lexical + optionally learned-sparse), not two BM25s or two dense models.

## The eval loop for RRF

The right eval loop when you introduce RRF into an existing single-arm stack:

1. **Baseline.** Compute recall@100, MRR@10, nDCG@10 on your labeled eval set for each arm individually (dense-only and BM25-only).
2. **RRF with `k = 60`.** Fuse the two arms. Report the same metrics.
3. **Check the delta.** RRF should beat both arms on recall@100 by a real margin (2–15 percentage points on heterogeneous corpora is the ballpark I've seen; verify on yours). If RRF loses to either arm, something is wrong — most commonly, one arm is producing degenerate rankings and dragging RRF down.
4. **Sweep `k`.** Try `k ∈ {10, 20, 40, 60, 100, 200}`. Report the best. Usually the improvement over `k = 60` is small; sometimes you find `k = 30` or `k = 100` gives a couple more points.
5. **Statistical significance.** Use paired significance (paired *t*-test on per-query metric differences, or the more-robust randomization test). Mod-109 covers this rigorously. A 1-point recall@100 delta on 20 queries is noise; a 4-point delta on 200 queries usually isn't.

The right eval loop when adding an *additional* arm (say, SPLADE) to a two-arm RRF:

1. Measure the two-arm RRF baseline.
2. Add the third arm. Measure the three-arm RRF.
3. If the three-arm number is meaningfully better and the third arm's serving cost is acceptable, ship it. If not, keep two arms.
4. Do NOT keep adding arms without measuring. The recall curve flattens fast past ~3 arms; each new arm doubles some infrastructure cost.

## The three anti-patterns to avoid

1. **Using RRF on the same arm twice.** "I have BM25 with analyzer A and BM25 with analyzer B; I'll RRF them and call it hybrid." That is not hybrid — you get some diversification from analyzer differences but the signal is fundamentally one lexical arm. Real hybrid needs signal-diverse retrievers (dense + lexical + learned-sparse at most).
2. **Setting `k = 1` because "the top result matters most."** Small `k` makes RRF degenerate — a single arm's top-1 dominates the fused ranking, and you lose most of the fusion benefit. Stay above `k ≈ 10`; the Cormack default of 60 is a fine place to sit.
3. **Skipping the eval loop and shipping RRF because "the paper says it works."** RRF *usually* wins, but it doesn't always. On some corpora BM25-only wins; on some, weighted-sum with careful normalization wins. Measure it on your corpus before you ship the change — that measurement is what mod-109 exists for.

## What is NOT in this chapter (and where it goes)

- **Weighted-sum fusion mechanics, when it beats RRF, and Borda-count fusion** — chapter 3.
- **Query rewriting patterns** — chapter 4.
- **Semantic routing** — chapter 5.
- **Filter-heavy composition** — chapter 6.
- **Recall@k / MRR / nDCG methodology and paired significance testing** — mod-109.

## What you should be able to do after this chapter

- Write the RRF formula from memory: `Σ_r 1 / (k + rank_r(d))`.
- Explain the score-mismatch problem RRF sidesteps, and defend the "throw away scores, keep ranks" design decision.
- Set `k` on a new corpus: start at 60, sweep `k ∈ {10, 20, 40, 60, 100, 200}` on an eval set.
- Implement RRF in ~15 lines of Python for a two-arm setup.
- Configure the Elasticsearch `rrf` retriever with two retrievers, `rank_constant`, and `rank_window_size`.
- Name the three RRF caveats (no per-arm weighting, agreement signal hidden, clone-arm double-counting) and know which chapter or module covers the workarounds.

## Further reading

- Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009. [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the two-page origin paper.
- Bruch, S., Nardini, F. M., Nardo, C., Perego, R., and Tonellotto, N. (2023). *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM TOIS. [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) — a decade-plus follow-up study; useful for the empirical "why rank-based" defense.
- Elasticsearch: [*Reciprocal rank fusion*](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — the native `rrf` retriever docs.
- OpenSearch: [*Hybrid search with score normalization*](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) — RRF via search pipeline processor.
- Kamphuis, C., de Vries, A. P., Boytsov, L., and Lin, J. (2020). *Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants.* ECIR. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4) — for the reader who wants evidence that BM25 raw scores are less comparable across implementations than they look; motivates the rank-based fusion default.
