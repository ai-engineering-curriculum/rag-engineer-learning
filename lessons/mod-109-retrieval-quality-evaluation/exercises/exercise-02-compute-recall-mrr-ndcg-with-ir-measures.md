# Exercise 02 — Compute recall@k, MRR, and nDCG with `ir_measures`

**Estimated effort:** ~2 hours. Python, the judgment set from exercise 01 (or a BEIR stand-in), `pip install ir_measures`. No LLM API and no GPU required.

## Objective

Turn the judgment set from exercise 01 into a metric report. Compute the four canonical retrieval metrics (recall@k, precision@k, MRR@k, nDCG@k, MAP, hit-rate) with `ir_measures`, report the "triple" (top-of-window + ranked-order + top-1) for two retrievers on your corpus, and persist per-query metric arrays as CSV for exercise 03.

At the end, you have `report/aggregate.md` (the numbers you'd share), `report/per-query.csv` (the input to exercise 03's significance tests), and a runnable evaluation script that mod-111's CI will later wrap.

## Prerequisites

- Chapter 3 read.
- Exercise 01 complete — you have `eval/qrels.tsv` and `eval/queries.jsonl`.
- Two retrievers you can run over your corpus. Any two of:
  - BM25 (`rank_bm25`, `pyserini`, an Elasticsearch instance, or any Lucene-family index).
  - Dense retriever (`bge-base-en-v1.5`, `e5-small-v2`, or a hosted embedder).
  - Hybrid (BM25 + dense fused with RRF; mod-106 chapter 2).
- Python 3.10+, `pip install ir_measures pandas` (and whatever your retriever stack needs).
- (Optional) `pip install ranx pytrec_eval trec-eval-cli` for the cross-library sanity check in step 5.

## Steps

**Step 1 — Wire your retrievers to produce TREC-shape run files (30 minutes).** In `runs/`:

For each of your two retrievers, produce a run file at `runs/{retriever-name}.run` in TREC format:

```
q1  Q0  d17  1  0.912  bge-large
q1  Q0  d5   2  0.876  bge-large
...
```

Six whitespace-separated columns: `query_id`, literal `Q0`, `doc_id`, `rank` (1-indexed), `score`, `run_name`. Retrieve top-100 for each query (chapter 3's default; enough headroom for recall@100).

Sanity check: `wc -l runs/*.run` should show approximately (num_queries × 100) lines each.

**Step 2 — Compute aggregate metrics with `ir_measures` (20 minutes).** In `evaluate.py`:

```python
import ir_measures
from ir_measures import nDCG, RR, R, P, MAP, Success

qrels = ir_measures.read_trec_qrels("eval/qrels.tsv")

MEASURES = [
    R@10, R@50, R@100,           # recall — top-of-window
    P@10,                         # precision — sanity
    RR@10,                        # reciprocal rank — top-1 family
    nDCG@10,                      # ranked-order — the primary
    MAP,                          # coverage-along-list
    Success@10,                   # hit-rate
]

for run_name in ["bm25", "bge-large"]:
    run = ir_measures.read_trec_run(f"runs/{run_name}.run")
    agg = ir_measures.calc_aggregate(MEASURES, qrels, run)
    print(f"\n=== {run_name} ===")
    for m in MEASURES:
        print(f"  {str(m):15s}  {agg[m]:.4f}")
```

**Step 3 — Compute per-query metrics and persist as CSV (20 minutes).** In `evaluate.py`:

```python
import pandas as pd
from collections import defaultdict

def per_query(qrels, run, measures):
    by = defaultdict(dict)
    for m in ir_measures.iter_calc(measures, qrels, run):
        by[str(m.measure)][m.query_id] = m.value
    return by

rows = []
for run_name in ["bm25", "bge-large"]:
    run = ir_measures.read_trec_run(f"runs/{run_name}.run")
    pq = per_query(qrels, run, MEASURES)
    for measure_name, per_q in pq.items():
        for q, v in per_q.items():
            rows.append({"run": run_name, "measure": measure_name,
                         "query_id": q, "value": v})

df = pd.DataFrame(rows)
df.to_csv("report/per-query.csv", index=False)
```

This CSV is the input to exercise 03's paired significance tests. It must include every query in `qrels`, not just the queries that returned a non-null score — `ir_measures` returns 0 for queries where the retriever missed, which is what you want for a fair average.

**Step 4 — Write the aggregate report (15 minutes).** In `report/aggregate.md`:

- Table of aggregates by (run, metric).
- One paragraph naming which metric is the primary for your product (per chapter 3's decision table) and why.
- The "triple" (recall@N + nDCG@10 + MRR@10) called out explicitly.
- The qrels version (from `eval/VERSION`).

Do NOT publish a "winner" here. Chapter 4 (exercise 03) is where you find out whether the delta is real.

**Step 5 — Cross-library sanity check (15 minutes).** In `sanity/`:

Run the same qrels + run through **one** other library to catch bugs. Pick one:

- **`pytrec_eval`** (the C `trec_eval` wrapper):

    ```python
    import pytrec_eval
    with open("eval/qrels.tsv") as f: qrels_p = pytrec_eval.parse_qrel(f)
    with open("runs/bge-large.run") as f: run_p  = pytrec_eval.parse_run(f)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_p, {"ndcg_cut_10", "recall_50", "recip_rank"})
    results = evaluator.evaluate(run_p)
    # results[query_id][measure_name] → float
    ```

- **`ranx`**:

    ```python
    from ranx import Qrels, Run, evaluate
    q = Qrels.from_file("eval/qrels.tsv", kind="trec")
    r = Run.from_file("runs/bge-large.run",  kind="trec")
    print(evaluate(q, r, ["ndcg@10", "recall@50", "mrr@10"]))
    ```

- **NIST `trec_eval` binary** (if installed): `trec_eval -m ndcg_cut.10 -m recall.50 -m recip_rank eval/qrels.tsv runs/bge-large.run`.

Compare against your `ir_measures` numbers. They should agree to at least 4 decimals. If they don't, debug — nDCG's base-2 exponential-gain convention is the most common divergence source (chapter 3's caveat).

**Step 6 — Interpret the numbers (20 minutes).** In `report/interpretation.md`:

Answer these questions in one paragraph each:

1. Which retriever wins on the **primary** metric (the one you chose in step 4)?
2. Which retriever wins on the **top-of-window** metric (recall@N)? Does it match #1?
3. Which retriever wins on the **top-1** metric (MRR@10)? Does it match #1?
4. If any of these disagree, what is the trade — and which one matters more for your product?
5. What is the *effect size* of the winning delta? Small (< 0.02 nDCG@10 points), medium (0.02-0.05), large (> 0.05)?

Do NOT conclude "we ship X." Chapter 4 will resolve whether the win is real.

## Acceptance criteria

- `runs/{retriever}.run` files for ≥ 2 retrievers, in TREC format, top-100 per query.
- `evaluate.py` produces aggregate and per-query metrics via `ir_measures`.
- `report/aggregate.md` contains: aggregate table, the retrieval triple called out, the primary-metric choice justified, the qrels version.
- `report/per-query.csv` contains per-query metric values for all queries × all measures × all runs (long-format).
- A cross-library sanity check (against `pytrec_eval`, `ranx`, or `trec_eval`) agrees with the `ir_measures` numbers to at least 4 decimals.
- `report/interpretation.md` names the primary-metric winner, the recall winner, and the MRR winner and interprets any disagreements.

## Stretch goals

- **Metric matrix.** Add three more metrics your product might care about: `nDCG@20`, `R@200`, `RBP(0.8)` (rank-biased precision with persistence 0.8; Moffat & Zobel 2008). Show whether they change the winner.
- **Per-stratum reporting.** If exercise 01's stretch goal produced difficulty strata (easy / medium / hard queries), compute per-stratum aggregates. The winning retriever may flip inside a stratum.
- **Run-file diff.** Write a small script that shows, for the 10 queries with the largest per-query delta between the two runs, which docs each retriever got right and which it got wrong. Useful ammunition for exercise 04's ablation attribution.
- **Metric-metric correlation.** Compute Pearson / Spearman correlation between nDCG@10 and MRR@10 across your queries. If it's > 0.9, they are essentially the same metric on your corpus (common); if it's < 0.7, they measure different things and the choice of primary metric is more consequential.
- **`ir_measures` iterator vs. batch.** Compare `ir_measures.calc_aggregate` (loads all-at-once) with `ir_measures.iter_calc` (streams per-query) on a large run file (10k+ queries). Note memory and wall-clock differences. Mod-111's CI cares about this at production scale.

## Notes

- **Include zero-score queries.** If your retriever returns nothing for a query, `ir_measures` returns 0 for that query on every metric. That is the correct behavior — dropping missed queries makes the average look better than it should. Keep them.
- **Do not mix `pytrec_eval`'s `ndcg_cut_10` with `ir_measures.nDCG@10` in the same report without checking.** They agree on the exponential-gain, base-2 convention that TREC / BEIR use — but a version mismatch or a `linear`-gain override would silently produce different numbers.
- **Print the qrels version.** Every metric result belongs to a specific qrels version. Copy the tag from `eval/VERSION` into the report header. A metric result with no qrels tag is not reproducible.
- **The aggregate table is not the ship decision.** Do not skip to a conclusion in this exercise. The Wilcoxon and CI in exercise 03 are what resolve the question of "is the delta real."
- **`Success@k` is `ir_measures`'s name for hit-rate.** Chapter 3 uses both names.
