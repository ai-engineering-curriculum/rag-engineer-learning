# Exercise 04 — Orthogonal Ablation Grid and Attribution

**Estimated effort:** ~3 hours. Python, exercise 01's judgment set, exercise 02's evaluation script, exercise 03's significance-testing script. At least one embedding model beyond the exercise-02 baseline. One reranker (`bge-reranker-base` is enough) or a hosted rerank API. No GPU strictly required, though re-embedding a corpus is much faster on one.

## Objective

Build a one-axis-at-a-time ablation grid — chunker × embedding × retriever × reranker — over a shared judgment set, evaluate every row, run chapter 4's paired significance tests with Benjamini-Hochberg FDR correction, and publish a decision memo that attributes each metric delta to the single axis that moved it.

At the end you have `grid/configs.jsonl` (the row spec), `grid/runs/*.run` (one TREC-shape run file per row), `report/ablation-grid.md` (the attributed results table), and `report/decision-memo.md` (what you would ship and defer, with the honest confounder list). This is the artifact mod-111 chapter 2's retriever CI pins to as its "baseline" reference.

## Prerequisites

- Chapter 5 read; chapters 3 and 4 as background.
- Exercise 01 complete — `eval/qrels.tsv`, `eval/queries.jsonl`, `eval/VERSION` exist and are frozen for the duration of this exercise (chapter 5's "different qrels per row" confounder).
- Exercise 02 complete — `evaluate.py` produces per-query metric arrays from a run file + qrels.
- Exercise 03 complete — `significance.py` computes paired Wilcoxon and BCa bootstrap CI on per-query deltas.
- The ability to run **at least two options on each of the four axes**. If your stack lacks one option on an axis, either add a plausible option (see step 1) or explicitly narrow the sweep and document it.
- Python 3.10+, `pip install ir_measures scipy statsmodels pandas` (already installed if you did exercises 02 and 03).

## Steps

**Step 1 — Enumerate the axes and options (20 minutes).** In `grid/axes.md`:

Fill in a table of the four canonical axes with **the specific options you can actually run**, not aspirational ones. Chapter 5's default shortlist:

| Axis | Options you will sweep | Notes |
|---|---|---|
| **Chunker** (mod-102) | e.g. `fixed-512`, `structural`, `parent-child` | Each option requires re-ingesting + re-embedding the corpus. Note the on-disk cost. |
| **Embedder** (mod-103) | e.g. `bge-base-en-v1.5`, `bge-large-en-v1.5`, `e5-small-v2` | Each option requires re-embedding. If any needs an instruction prefix (E5, GTE-Qwen, e5-mistral), record it here. |
| **Retriever** (mod-104/-105/-106) | e.g. `dense-only`, `bm25-only`, `hybrid-rrf` | Reuses existing indexes; no re-embed. |
| **Reranker** (mod-107) | e.g. `none`, `bge-reranker-base`, `bge-reranker-large` (or hosted) | Runs on top-N candidates from the retriever; no re-embed. |

Rules for this exercise:

- **At least 2 options per axis** across the whole grid. If you honestly only have 1 option for an axis, pick a second one from the shortlist above and add it.
- **Name a reference row.** One row is the "current shipping pipeline" or a well-known baseline. Every other row moves *exactly one* axis from the reference. Chapter 5's example uses `structural / bge-base / hybrid-rrf / none`; use whatever matches your corpus.
- **Do not attempt the full 3^4 cross product.** One-at-a-time with 2-3 options per axis is 8-12 rows total. That is the target.

**Step 2 — Emit the grid as a config file (20 minutes).** In `grid/configs.jsonl`:

One JSON object per row, in the reverse-cost order chapter 5 recommends (reference first; reranker sweep next; retriever sweep; embedder sweep; chunker sweep last):

```json
{"row_id": "ref",  "chunker": "structural",   "embedder": "bge-base-en-v1.5", "retriever": "hybrid-rrf", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "k-1",  "chunker": "structural",   "embedder": "bge-base-en-v1.5", "retriever": "hybrid-rrf", "reranker": "bge-reranker-base", "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "k-2",  "chunker": "structural",   "embedder": "bge-base-en-v1.5", "retriever": "hybrid-rrf", "reranker": "bge-reranker-large","retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "r-1",  "chunker": "structural",   "embedder": "bge-base-en-v1.5", "retriever": "dense-only", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "r-2",  "chunker": "structural",   "embedder": "bge-base-en-v1.5", "retriever": "bm25-only",  "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "e-1",  "chunker": "structural",   "embedder": "bge-large-en-v1.5","retriever": "hybrid-rrf", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "e-2",  "chunker": "structural",   "embedder": "e5-small-v2",      "retriever": "hybrid-rrf", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "c-1",  "chunker": "fixed-512",    "embedder": "bge-base-en-v1.5", "retriever": "hybrid-rrf", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
{"row_id": "c-2",  "chunker": "parent-child", "embedder": "bge-base-en-v1.5", "retriever": "hybrid-rrf", "reranker": "none",              "retrieve_top_n": 50, "rerank_to_k": 10}
```

Machine-check the file: for every non-reference row, exactly one of `chunker | embedder | retriever | reranker` must differ from `ref`. Write a two-line assertion in `grid/validate_configs.py`; run it before doing any expensive work.

**Step 3 — Build a minimal grid runner (45 minutes).** In `grid/run_grid.py`:

Wrap exercise 02's evaluation loop with a runner that iterates over `grid/configs.jsonl`, produces a TREC-shape run file at `grid/runs/{row_id}.run`, and caches expensive per-`(chunker, embedder)` intermediates.

Skeleton (from chapter 5's *Concrete: a minimal grid runner*):

```python
# grid/run_grid.py — skeleton
import json
from dataclasses import dataclass, replace
from pathlib import Path

@dataclass(frozen=True)
class Config:
    row_id: str
    chunker: str
    embedder: str
    retriever: str
    reranker: str
    retrieve_top_n: int = 50
    rerank_to_k: int = 10

def load_configs(path="grid/configs.jsonl") -> list[Config]:
    with open(path) as f:
        return [Config(**json.loads(line)) for line in f if line.strip()]

def index_cache_key(cfg: Config) -> str:
    return f"{cfg.chunker}__{cfg.embedder}"

def ensure_index(cfg: Config):
    """Re-ingest with cfg.chunker + re-embed with cfg.embedder if not yet cached.
    Cache under `grid/indexes/{index_cache_key(cfg)}/`."""
    ...

def run_row(cfg: Config, queries):
    ensure_index(cfg)                          # cached per (chunker, embedder)
    candidates = retrieve(cfg, queries, top_n=cfg.retrieve_top_n)  # retriever family
    ranked     = rerank(cfg, candidates, k=cfg.rerank_to_k)        # cfg.reranker
    write_trec_run(f"grid/runs/{cfg.row_id}.run", ranked, run_name=cfg.row_id)
```

Two disciplines chapter 5 requires:

- **Cache by `(chunker, embedder)`.** Re-embedding the corpus is the dominant cost; every row with the same `(chunker, embedder)` reuses the same index. Do not re-embed per row.
- **Use each model's recommended prompt / prefix.** E5-family models need `"query: ..."` and `"passage: ..."`; some models need task-specific instructions. Chapter 5's per-embedder-prefix confounder wipes out 5-20 nDCG points if skipped.

Run the whole grid end-to-end; time it. Expect ~2-10 minutes per re-embedded row on CPU for a small corpus; more on a large one.

**Step 4 — Evaluate every row against the frozen qrels (20 minutes).** In `grid/evaluate_grid.py`:

Reuse exercise 02's per-query evaluation to produce `grid/per-query.csv` with columns `row_id, measure, query_id, value`. Include at minimum:

- **Primary metric.** `nDCG@10` (chapter 3's default).
- **Top-of-window.** `R@50` (or match the top-N you use to feed the reranker).
- **Top-1 family.** `RR@10`.

Then produce an aggregate table `grid/aggregate.md`:

| row_id | chunker | embedder | retriever | reranker | nDCG@10 | R@50 | RR@10 |
|---|---|---|---|---|---|---|---|
| ref | structural | bge-base | hybrid-rrf | none | … | … | … |
| k-1 | structural | bge-base | hybrid-rrf | bge-reranker-base | … | … | … |
| … |

**Step 5 — Paired significance vs. reference, per row per metric (30 minutes).** In `grid/attribute.py`:

Reuse exercise 03's significance functions. For every non-reference row and every metric, compute:

- **Mean delta** (row − reference), per-query.
- **BCa 95% CI** on the mean delta.
- **Paired Wilcoxon `p`** with `zero_method="pratt"`.

Then apply **Benjamini-Hochberg FDR** across the whole grid's significance tests (chapter 4 + chapter 5):

```python
from statsmodels.stats.multitest import multipletests

# entries: list of (row_id, measure, delta, ci_lo, ci_hi, p_wilcoxon)
raw_p = [e[-1] for e in entries]
_, p_bh, _, _ = multipletests(raw_p, method="fdr_bh")
```

`p_bh` is the number to report; the raw `p_wilcoxon` is kept in a "diagnostics" column but does not drive decisions. On a grid of 8-12 rows × 3 metrics, correction typically shifts p-values by a small but non-trivial amount; on larger grids it dominates.

**Step 6 — Attribute each delta to its single moved axis (15 minutes).** In `report/ablation-grid.md`:

Chapter 5's canonical table shape:

```
row_id  moved-axis  chunker  embedder  retriever  reranker  nDCG@10   Δ           95% CI            p_BH   verdict
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
ref     —           structural  bge-base   hybrid-rrf  none     0.412  —           —                 —      reference
k-1     reranker    structural  bge-base   hybrid-rrf  bge-b-r  0.487  +0.075     [+0.041, +0.108]  0.001  better (real)
k-2     reranker    structural  bge-base   hybrid-rrf  bge-l-r  0.503  +0.091     [+0.052, +0.129]  <.001  better (real)
r-1     retriever   structural  bge-base   dense-only  none     0.398  -0.014     [-0.040, +0.011]  0.34   tie
r-2     retriever   structural  bge-base   bm25-only   none     0.371  -0.041     [-0.071, -0.010]  0.02   worse (real)
e-1     embedder    structural  bge-large  hybrid-rrf  none     0.451  +0.039     [+0.011, +0.068]  0.04   better (real)
e-2     embedder    structural  e5-small   hybrid-rrf  none     0.427  +0.015     [-0.011, +0.041]  0.29   tie
c-1     chunker     fixed-512   bge-base   hybrid-rrf  none     0.389  -0.023     [-0.048, +0.002]  0.11   tie
c-2     chunker     parent-child bge-base  hybrid-rrf  none     0.428  +0.016     [-0.010, +0.041]  0.28   tie
```

Two invariants chapter 5's *Reading the grid honestly* requires:

- **`moved-axis` is populated for every non-reference row** and is the single axis that changed from the reference. If a row moves two axes, either it should not be in a one-at-a-time grid or you have a bug.
- **`verdict` is a short label**: `better (real)`, `worse (real)`, `tie`. `real` == `p_BH < 0.05` **and** `0 ∉ CI`. Otherwise `tie`.

Add three sub-tables chapter 5 asks you to read:

- **Which axis matters most on this corpus?** Sum `|delta|` over the rows that swap each axis. Publish the ranking.
- **Which axes are noise on this corpus?** List axes whose best row is `p_BH > 0.10`.
- **Which axes' wins are largest per dollar?** For each `real` win, add a rough cost column (embedder GPU-hours + reranker latency + storage delta) and a `Δ per $` (or `Δ per ms`) column. `bge-reranker-large` almost always wins per-metric; whether it wins per-dollar depends on your latency budget (chapter 5).

**Step 7 — Write the decision memo (25 minutes).** In `report/decision-memo.md`:

Chapter 5 is explicit that the output of the grid is a **decision memo**, not a table. Structure:

1. **What you would ship.** The specific configuration (axis by axis). Justify with a citation of specific rows: "adopt bge-reranker-large from row k-2: `+0.091 nDCG@10`, CI `[+0.052, +0.129]`, `p_BH < 0.001`, adds ~150ms P50 rerank latency at rerank-to-k = 10, within our 500ms budget."
2. **What you would defer.** The rows that landed as `tie`. Why deferring: "chunker sweep landed in noise on n = 100 queries; would need ~n = 250 to detect a 0.02 delta at 80% power (chapter 4 sample-size math); revisit after growing judgment set."
3. **What you would refuse.** The rows that regressed with `real`. E.g. "do not swap to bm25-only — row r-2 is worse with `p_BH = 0.02`."
4. **The confounder audit.** Chapter 5's confounder list: same qrels version across all rows (cite `eval/VERSION`), same queries across all rows, same funnel widths (`retrieve_top_n`, `rerank_to_k`), embedder prefixes named per model, single seed vs. multi-seed for ANN retrievers noted. Any row you re-ran to reduce variance is called out.
5. **The hand-off pointer.** Chapter 7's discipline: the decision is on **retrieval quality alone**. Reference the eval peer's application-layer report (or note that one has been requested); do not conclude "ship" without it.

## Acceptance criteria

- `grid/axes.md` names the four axes and the specific options you actually ran.
- `grid/configs.jsonl` contains a reference row plus non-reference rows, each moving exactly one axis. A validator script asserts this.
- `grid/runs/{row_id}.run` for every row in `configs.jsonl`, in TREC format, top-K deep (K ≥ 10; K ≥ 50 preferred for headroom on `R@50`).
- `grid/per-query.csv` and `grid/aggregate.md` cover ≥ 3 metrics (nDCG@10, R@50, RR@10 at minimum).
- `report/ablation-grid.md` contains:
  - the attributed table with `moved-axis`, `Δ`, `95% CI`, `p_BH`, `verdict`;
  - the "which axis matters most" sum of `|delta|`;
  - the "which axes are noise" list;
  - a `Δ per $` (or `Δ per ms`) column for `real` wins.
- `report/decision-memo.md` lists ship / defer / refuse, includes the confounder audit, and defers final approval to the peer per chapter 7.
- Every row runs against the **same** qrels version (`eval/VERSION` matches across the whole grid). If you changed qrels mid-way, either revert or re-run every row.

## Stretch goals

- **Interaction ablation (2 × 2).** Pick the two axes chapter 5 flagged as most likely to interact on your corpus (chunker × embedder is the usual pair). Run the full 2 × 2 grid at the reference options plus one alternative each. Compute the additive expectation vs. observed and report the interaction term. Interpret its sign: does the stronger embedder subsume the parent-child chunker's help, or complement it?
- **Multi-seed for ANN retrievers.** If any row uses HNSW / IVF-PQ (mod-104), re-run 3 seeds per row and average per-query scores. Chapter 5's "one noisy run per row" confounder — reporting the variance of the mean delta across seeds tells you whether a `p_BH ≈ 0.04` finding is stable or flapping.
- **Cost-normalized ranking.** Extend the `Δ per $` column into a Pareto plot (mean-delta on y-axis, log-cost on x-axis). Rows on the Pareto frontier are the shippable ones; rows below it are strictly dominated. Publish the frontier.
- **Per-difficulty-stratum attribution.** If exercise 01's stretch goal produced difficulty strata (easy / medium / hard queries), re-run the paired significance per-stratum. Most rerankers concentrate their wins on the "medium" stratum; embedder swaps often help most on "hard". The single-axis attribution reads much more clearly when broken down.
- **Regression-alert integration.** Wire `report/ablation-grid.md`'s largest `real worse` row into exercise 03's regression-alert script — a change that would have shipped without the grid is now caught. Note the check as the primitive mod-111's CI uses.
- **Fractional factorial extension.** If you have budget for `~2N` runs instead of `~N`, add a Plackett-Burman or 2^(k-p) fractional factorial across three axes at two levels each. Chapter 5's *Interaction ablations* + Box, Hunter, & Hunter's *Statistics for Experimenters* is the reference.

## Notes

- **Freeze the qrels before starting the grid.** Chapter 5's "different qrels per row" confounder is the most-common and least-visible failure mode. If you find yourself wanting to add or re-label pairs mid-grid, stop, do it, then re-run every row — never mix qrels versions inside one grid.
- **Cache indexes by `(chunker, embedder)`, not by row.** Two rows that only differ in retriever or reranker share an index. Not caching costs hours of re-embedding for no reason.
- **Use each embedder's model card exactly.** Prefix, pooling, score function, max-tokens — chapter 6's checklist for BEIR debugging applies just as strongly inside a local grid. A mis-prefixed E5 row is not a real ablation row; it is a bug row disguised as data.
- **Do not compare across two grids' aggregates directly.** Aggregates from grid v1 (qrels v2024-11-a) and grid v2 (qrels v2024-12-a) are not comparable; the reference row moved. When you rev the qrels, rev the whole grid.
- **The purpose is attribution, not the highest score.** The row with the best `nDCG@10` is not automatically the ship row; it is a candidate. The decision memo (step 7) trades quality against cost, latency, storage, and confounder audit. The grid narrows the space; humans still decide.
- **The reference row is not "fair" — it is the anchor.** Do not spend time re-tuning the reference. Its job is to be the fixed point every non-reference row is measured against. Tune inside the sweep, not inside the anchor.
- **`bge-reranker-large` will almost certainly win nDCG@10 on your grid.** That is the field consensus (Nogueira & Cho 2019; mod-107). What the grid tells you is *how much* it wins on **your** corpus — which is what feeds the cost / latency trade in the decision memo.
