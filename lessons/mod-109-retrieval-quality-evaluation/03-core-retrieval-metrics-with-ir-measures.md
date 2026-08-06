# Chapter 3 — Core Retrieval Metrics with the `ir_measures` Toolchain

## Why this chapter exists

The judgment set from chapter 2 is inert until a metric harness turns it into numbers. Every serious IR toolchain — `trec_eval` (C), `pytrec_eval` (Python bindings), `ir_measures` (Python) — computes the same standardized set of metrics on the same qrels + run inputs. Once you know which metric answers your product's real question and which library implements it, the mechanics are straightforward.

This chapter is that mechanics: the four metric families, the code for each in `ir_measures`, and the mental model for picking one.

## The four metric families

Chapter 1 named the four operational questions retrieval metrics answer. Here they are again with the metrics that answer each.

| Question | Metric family | Canonical formula |
|---|---|---|
| Did the relevant docs enter the top-k window? | **recall@k**, **hit-rate**, **success@k** | `|relevant ∩ top-k| / |relevant|` (recall), `1 if |relevant ∩ top-k| > 0 else 0` (hit-rate) |
| Is the top-1 (or first few) correct? | **precision@1**, **MRR**, **RR** | `1 / rank_of_first_relevant` (RR), mean of RR over queries (MRR) |
| Is the top-k ranked well? | **nDCG@k** | discounted-gain over the ranking, normalized by ideal ranking |
| Average precision across the ranked list? | **MAP**, **AP** | mean over queries of `mean(precision@i for i where doc-i is relevant)` |

There is a fifth family — **rank-biased precision** (RBP), **expected reciprocal rank** (ERR) — that shows up in TREC papers but is used more rarely in production RAG work. They are in `ir_measures`; use them if you want a user-behavior-model that decays hits deeper in the list (RBP is Moffat & Zobel 2008, ERR is Chapelle et al. 2009).

### Which to report

Every retrieval eval this module produces should report **at least three** metrics:

1. **A recall metric** at the reranker's window: **recall@k** where `k` is your top-N (from mod-107 chapter 5's funnel). This tells you if the retriever gave the reranker a fair chance.
2. **A ranked-order metric**: **nDCG@10** by default (the BEIR / MTEB primary). Answers "is the top-10 order good?"
3. **A top-1 metric**: **MRR@10**. Answers "if we shipped this to a chatbot that shows one answer, how often is the first one right?"

Reporting all three protects you from the failure mode of a metric that looks good but hides the failure — a retriever that has great top-1 (MRR up) but drops recall (recall@k down) is *worse* for a reranker-based pipeline, better for a single-answer chatbot.

## The metrics, one at a time

Formulas here match Manning-Raghavan-Schütze (MRS) chapter 8 and BEIR's Appendix C. Where symbols could ambiguate, this chapter uses BEIR's.

### recall@k

**What it means.** Of all the relevant documents for a query, what fraction landed in the top `k` of the ranked list?

$$
\text{recall@k} = \frac{|\text{relevant} \cap \text{retrieved}_{1..k}|}{|\text{relevant}|}
$$

**Range.** [0, 1]. 1.0 = every relevant doc is in the top `k`.

**When it answers your question.** When your pipeline has a downstream stage (reranker in mod-107, generator in mod-108) that filters and re-orders. The retriever's job is to *not lose* the relevant docs. If a relevant doc is at rank 8 of the retriever's top-20 window, the reranker can rescue it; if it's at rank 200, it cannot.

**Picking k.** Set `k` to your **retriever-to-reranker window width** (mod-107's `N`). If your funnel is retrieve-top-50 → rerank-to-10, measure **recall@50**. If it is retrieve-top-200 → rerank-to-8, measure **recall@200**. Don't measure recall at the *rerank* output (`k=10`) — that is a joint retriever-reranker metric and hides the retriever's contribution.

**Watch out for.** Recall is undefined for queries with zero relevant documents in the qrels. Filter those out or the metric averages badly. Also: if your qrels file has *partial* relevance judgments (protocol A's pooling deliberately does not label every doc), recall is a lower bound — it counts only *judged-and-relevant* hits. This is the standard TREC assumption; it holds if the pool depth was ≥ your `k`.

### precision@k

**What it means.** Of the top `k` results, what fraction were relevant?

$$
\text{precision@k} = \frac{|\text{relevant} \cap \text{retrieved}_{1..k}|}{k}
$$

**Range.** [0, 1]. 1.0 = all `k` are relevant.

**When it answers your question.** Rarely, in isolation, for RAG. Precision@k is the classical IR metric for search-result pages where the user judges each result independently. In RAG, once relevant docs go into the reranker or LLM, the *ranked order* usually matters more than the count — so nDCG@k is usually the better ranked-metric choice. Precision@1 is the exception: it is the "top-hit is right" number for one-answer surfaces (a chatbot).

**Watch out for.** Precision@k penalizes retrieval that surfaces *more* relevant docs than the qrels have judged. If a doc is relevant but not in the qrels (an *unjudged* relevant), precision counts it against you. This is why chapter 2 emphasized pooled labeling — an unpooled retriever's precision looks worse than it deserves.

### MRR — Mean Reciprocal Rank

**What it means.** For each query, take the reciprocal of the rank of the **first relevant** document. Average across queries. Ignores everything after the first relevant hit.

$$
\text{RR}(q) = \frac{1}{\text{rank of first relevant doc for } q}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)
$$

If no relevant doc appears in the ranked list, RR = 0.

**Range.** [0, 1]. Higher is better. An MRR of 0.5 means "on average the first relevant doc is at rank 2." MRR of 1.0 means "the top hit is always relevant."

**When it answers your question.** When the product surface is "the first result is what the user sees" — a chatbot answer, a "did you mean X" suggestion, an auto-complete pick. Also the primary metric in most Q&A benchmarks (MS MARCO passage reports MRR@10).

**Watch out for.** MRR is defined against **binary** relevance. If your qrels are graded (0-3), any label > 0 is treated as relevant. If you want to reward "very relevant" more than "somewhat relevant," reach for nDCG.

**MRR@k vs. MRR.** In practice always use MRR@k (usually MRR@10) — the cutoff at `k` says "we only look at the first `k` hits; anything after doesn't count." This matches how the downstream stage will actually behave.

### MAP — Mean Average Precision

**What it means.** For a single query, compute the precision *at each rank where a relevant doc appears*, average those, and you have AP for that query. Mean across queries → MAP.

$$
\text{AP}(q) = \frac{1}{|\text{relevant}_q|} \sum_{k=1}^{n} \text{precision@k} \cdot \mathbb{1}[\text{doc at rank } k \text{ is relevant}]
$$

$$
\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q)
$$

**Range.** [0, 1]. Higher is better.

**When it answers your question.** When the user consumes many results per query — e.g., an evidence-gathering flow that expects to review 5-20 hits, or a compliance-search flow ("find every relevant regulation"). MAP is the classical TREC-era default; retrieval papers often report it alongside nDCG for historical continuity.

**Watch out for.** MAP is defined against **binary** relevance (like MRR). Graded relevance information is lost. Also, MAP is more sensitive to labeling completeness than nDCG — unjudged-but-relevant docs pull it down more.

### nDCG@k — Normalized Discounted Cumulative Gain

**What it means.** Reward each relevant doc according to its **grade**, discount its reward by its **rank position**, sum across the top `k`, and normalize by the ideal ranking's score. Introduced by Järvelin & Kekäläinen (2002); the field's default aggregate metric.

Two pieces:

1. **DCG@k** — discounted cumulative gain:

    $$
    \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
    $$

    Where `rel_i` is the relevance grade of the doc at rank `i`. The `2^rel - 1` term makes highly-relevant docs count exponentially more than merely-relevant ones. The `log_2(i+1)` denominator is the rank discount — a doc at rank 1 contributes fully, rank 2 contributes `log(2)/log(3) ≈ 0.63`, and so on.

2. **nDCG@k** — normalized:

    $$
    \text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
    $$

    Where `IDCG@k` is the DCG of the *ideal* ranking (all relevant docs sorted by grade, most-relevant first). Normalization puts every query on [0, 1] so cross-query averaging is fair.

**Range.** [0, 1]. Higher is better.

**When it answers your question.** When ranked order in the top-k matters *and* your relevance is graded (0-3 or 0-4). This is the BEIR / MTEB primary metric and the field default for retrieval leaderboards. If you have binary labels, nDCG still works (it collapses to a rank-discounted precision) but MAP or MRR may be more informative.

**Watch out for.** Two log-base conventions (base-2 in the original 2002 paper; base-e in some implementations) and two definitions of the gain numerator (`2^rel - 1` in the "exponential gain" variant, plain `rel` in the "linear gain" variant). The BEIR / TREC / `ir_measures` default is base-2, exponential-gain — the formula above. Always name the convention when reporting `nDCG@10 = X` across sources.

### Hit-rate (a.k.a. success@k)

**What it means.** For each query, `1` if any relevant doc is in the top `k`, else `0`. Mean across queries.

$$
\text{hit-rate@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}[|\text{relevant}_q \cap \text{retrieved}_{1..k}| > 0]
$$

**Range.** [0, 1]. Fraction of queries with at least one relevant hit in top `k`.

**When it answers your question.** Coverage screening — "for what fraction of queries did we get *anything* useful?" More sensitive than precision@k for single-relevant-doc corpora (where recall@k = hit-rate@k because each query has only one relevant doc). Useful as an early-alert metric in mod-111's drift monitor.

**Watch out for.** Insensitive to *how many* relevant docs land or *where in the ranking* they land — hit-rate@10 = 1 for both "one relevant hit at rank 10" and "five relevant hits at ranks 1-5." Use alongside nDCG, not instead of.

## The Python toolchain: `ir_measures` (recommended default)

`ir_measures` (MacAvaney et al., [github.com/terrierteam/ir_measures](https://github.com/terrierteam/ir_measures)) is a modern Python library that wraps `pytrec_eval` (the Python bindings to the venerable NIST `trec_eval`) with a cleaner API and support for measures `trec_eval` doesn't cover. It is this module's default; use it for exercises 02–05.

**Installation.**

```bash
pip install ir_measures
```

**Reading qrels and runs.** `ir_measures` accepts TREC qrels files, TREC run files, JSONL, dicts, and iterables of dataclasses. The cleanest programmatic form:

```python
import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG, RR, R, P, MAP

qrels = [
    Qrel("q1", "d17", 1),
    Qrel("q1", "d5",  2),
    Qrel("q2", "d43", 1),
]

run = [
    ScoredDoc("q1", "d5",  0.9),
    ScoredDoc("q1", "d17", 0.8),
    ScoredDoc("q2", "d43", 0.7),
    ScoredDoc("q2", "d99", 0.6),
]
```

Or, from files:

```python
qrels = ir_measures.read_trec_qrels("eval/qrels.tsv")
run   = ir_measures.read_trec_run("runs/bge-large.run")
```

**Computing aggregate metrics.**

```python
result = ir_measures.calc_aggregate(
    [nDCG@10, RR@10, R@50, P@10, MAP],
    qrels, run
)
# → {nDCG@10: 0.612, RR@10: 0.734, R@50: 0.821, P@10: 0.203, MAP: 0.451}
```

**Computing per-query metrics** (which chapter 4 requires — significance tests operate on per-query arrays, not aggregates):

```python
per_query = list(ir_measures.iter_calc(
    [nDCG@10, RR@10, R@50],
    qrels, run
))
# each element is a Metric(query_id, measure, value)

# Group by measure for chapter 4's paired-delta arrays:
from collections import defaultdict
by_measure = defaultdict(dict)
for m in per_query:
    by_measure[m.measure][m.query_id] = m.value
# by_measure[nDCG@10] == {"q1": 0.62, "q2": 0.71, ...}
```

**Measure syntax.** `ir_measures` uses `MeasureFamily @ cutoff` syntax:

- `nDCG @ 10` → nDCG@10
- `R @ 50` → recall@50
- `P @ 10` → precision@10
- `RR @ 10` → reciprocal rank at cutoff 10 (compute aggregate to get MRR@10)
- `MAP` → MAP over the full ranked list
- `Success @ k` → hit-rate at k (also alias `ir_measures.SetR @ k`)

You can also parse from strings — `ir_measures.parse_measure("nDCG@10")` — which makes CI configuration straightforward.

**Under the hood.** `ir_measures` uses `pytrec_eval` (which wraps NIST `trec_eval`) for the metrics they both implement (nDCG, MAP, MRR, precision, recall) and falls back to its own implementations for the newer measures (RBP, ERR, INST). Numbers agree with `trec_eval` to at least 4 decimals; the two implementations of nDCG converge on the base-2 exponential-gain convention.

## The other libraries you will see

- **`pytrec_eval`** — the Python binding to NIST `trec_eval`. Same metrics, slightly noisier API (all values are strings in the returned dict). Use if you need bit-exact reproducibility with a TREC-era paper.
- **`trec_eval`** (C) — the original NIST binary. Command-line only: `trec_eval qrels.txt run.txt`. Used in the TREC community for decades. If you inherit an old evaluation script, this is what it likely calls.
- **`ranx`** ([github.com/AmenRa/ranx](https://github.com/AmenRa/ranx)) — a pure-Python evaluation library with a nice fusion API and built-in significance tests. Slower than `ir_measures` for large runs but ergonomic; some teams prefer it. Metric numbers agree with `ir_measures` / `trec_eval` on the shared measures.
- **`pyterrier.evaluate`** — Terrier's Python front-end has its own evaluation module that also wraps `pytrec_eval`. Use when you are already inside PyTerrier's pipeline abstractions.

For this module: pick `ir_measures` for exercises 02–05. It's the modern default, has the cleanest per-query API (essential for chapter 4), and is used by a majority of recent IR / RAG papers.

## Concrete: a full comparison of two retrievers

Given the artifacts from chapter 2 (`eval/qrels.tsv`, `runs/bm25.run`, `runs/bge-large.run`), the minimal compare-two-retrievers script:

```python
import ir_measures
from ir_measures import nDCG, RR, R, MAP

qrels = ir_measures.read_trec_qrels("eval/qrels.tsv")

def eval_run(run_path, name):
    run = ir_measures.read_trec_run(run_path)
    agg = ir_measures.calc_aggregate(
        [nDCG@10, RR@10, R@50, R@100, MAP],
        qrels, run,
    )
    print(f"{name:15s}  nDCG@10={agg[nDCG@10]:.3f}  RR@10={agg[RR@10]:.3f}  "
          f"R@50={agg[R@50]:.3f}  R@100={agg[R@100]:.3f}  MAP={agg[MAP]:.3f}")

eval_run("runs/bm25.run",      "bm25")
eval_run("runs/bge-large.run", "bge-large")
```

Output shape:

```
bm25             nDCG@10=0.412  RR@10=0.523  R@50=0.741  R@100=0.812  MAP=0.318
bge-large        nDCG@10=0.487  RR@10=0.612  R@50=0.803  R@100=0.867  MAP=0.371
```

The bge-large run appears to win across all metrics. Whether the delta is statistically significant is chapter 4's question — do *not* declare the winner from the aggregate table alone. Chapters 4 and 5 chain onto this exact output.

## When each metric answers what — a decision table

| Situation | Primary metric | Also report |
|---|---|---|
| Reranker-based pipeline; want to know retriever gave reranker a chance | **recall@N** (N = reranker window) | nDCG@10, MRR@10 |
| Chatbot with single-answer output surface | **MRR@10** | recall@10, nDCG@10 |
| Search-results page with a ranked list users scroll | **nDCG@10** | MRR@10, recall@10 |
| Compliance / evidence gathering; expects to review many hits | **MAP** | nDCG@20, recall@50 |
| Drift monitor / early-warning canary in production | **hit-rate@10** | nDCG@10 (as regression detector) |
| Comparing to a BEIR / MTEB leaderboard baseline | **nDCG@10** (their default) | recall@100 (BEIR secondary), MRR@10 |

If in doubt, report **nDCG@10 + recall@N + MRR@10** as a triple and pick the primary based on which product question is closest to your use.

## What NOT to report (and why)

- **Just the mean, no per-query record.** You have thrown away the input to chapter 4's significance test. Always persist the per-query values.
- **A single metric.** A retriever that ships with only nDCG@10 in its release notes has hidden the recall regression under a top-heavy improvement. Report the triple.
- **Cross-collection aggregate averages.** Averaging nDCG@10 across BEIR-14 to compare to a paper that averaged over BEIR-11 is a category error — the sets differ. Report per-dataset numbers when comparing across collections.
- **Numbers without the qrels version.** A metric result is meaningless without the specific qrels file (and its version tag from chapter 2) it was computed against. Always cite the qrels version.

## What you should be able to do after this chapter

- Match a product question to the right metric from the four families (top-of-window → recall@k, top-1 → MRR, ranked-order → nDCG@k, coverage-along-list → MAP).
- Use `ir_measures` to compute aggregate metrics and per-query metric arrays from a qrels + run pair.
- Explain each metric's formula in one paragraph and its two most common gotchas.
- Report a retriever comparison as a triple (recall@N + nDCG@10 + MRR@10) with the qrels version and know why reporting only one is insufficient.
- Recognize the two nDCG conventions (base-2 exponential-gain vs. others) and name the one `ir_measures` / BEIR / TREC use.

## Further reading

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval,* [Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — the textbook derivation of every metric in this chapter.
- Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM Transactions on Information Systems, 20(4). [DOI:10.1145/582415.582418](https://doi.org/10.1145/582415.582418) — the paper that introduced nDCG.
- Moffat, A., & Zobel, J. (2008). *Rank-Biased Precision for Measurement of Retrieval Effectiveness.* ACM TOIS 27(1). [DOI:10.1145/1416950.1416952](https://doi.org/10.1145/1416950.1416952) — RBP, an alternative to nDCG with a cleaner user-model interpretation.
- Chapelle, O., Metlzer, D., Zhang, Y., & Grinspan, P. (2009). *Expected Reciprocal Rank for Graded Relevance.* CIKM. [DOI:10.1145/1645953.1646033](https://doi.org/10.1145/1645953.1646033) — ERR.
- MacAvaney, S., Macdonald, C., & Ounis, I. (2022). *Streamlining Evaluation with `ir_measures`.* ECIR. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-99739-7_38) — the `ir_measures` paper.
- Van Gysel, C., & de Rijke, M. (2018). *`pytrec_eval`: An Extremely Fast Python Interface to `trec_eval`.* SIGIR. [DOI:10.1145/3209978.3210065](https://doi.org/10.1145/3209978.3210065) — the Python binding underneath `ir_measures`.
- NIST, [`trec_eval` repository](https://github.com/usnistgov/trec_eval) — the reference C implementation; the source of truth for TREC-canonical metric definitions.
- BEIR repository, [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — the reference RAG-adjacent evaluation harness; uses nDCG@10 as its primary and reports recall@100 secondarily.
- Sakai, T. (2020). *On the Significance of the Non-Significance of the Top-Ranked Systems.* SIGIR (short paper). [DOI:10.1145/3397271.3401266](https://doi.org/10.1145/3397271.3401266) — why an aggregate metric alone is not enough; the bridge to chapter 4.
