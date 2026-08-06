# Chapter 5 — Orthogonal Ablation Grids and Attribution

## Why this chapter exists

Every RAG pipeline is a stack of choices: **chunker × embedding × retriever × reranker** are the four canonical axes. Each has several plausible options. Swap any two at once and the metric can move — but you cannot tell *which* swap moved it. Ship the pair anyway and you have taken on a debt: when a future regression shows up, you have no isolated baseline to blame it against.

The discipline is the **orthogonal ablation grid**: a matrix of runs where every row moves *exactly one* axis from a fixed reference. Combined with chapter 4's paired significance test, it produces the artifact this module's downstream mod-111 CI gate depends on — a defensible attribution of "which axis actually moved the number."

## The four axes for a RAG retriever

Every module in this track corresponds to an axis of the grid:

| Axis | Options per axis (typical shortlist) | Owning module |
|---|---|---|
| **Chunker** | fixed-512, structural, semantic-similarity, parent-child hierarchical | mod-102 |
| **Embedding** | `bge-base-en`, `bge-large-en`, `e5-mistral-7b`, `text-embedding-3-large` (or the shortlist from mod-103 chapter 7) | mod-103 |
| **Retriever** | dense-only, BM25-only, hybrid (BM25 + dense fused with RRF) | mod-104 + mod-105 + mod-106 |
| **Reranker** | none, `bge-reranker-base`, `bge-reranker-large`, hosted cross-encoder | mod-107 |

Two derived axes worth naming:

- **Funnel widths** — retrieve-top-N and rerank-to-K from mod-107 chapter 5. Technically knobs on the retriever and reranker; often broken out as their own axis when they dominate the compute cost.
- **Query rewriting** — HyDE, multi-query, decomposition from mod-106 chapter 4. Treat as a separate axis if it's on the table.

## Reference point + one-axis-at-a-time

The grid is anchored on a **reference configuration** — the current shipping pipeline, or a well-known baseline. Every other row moves exactly one axis:

```
row  chunker         embedding         retriever      reranker              nDCG@10   Wilcoxon vs. ref
────────────────────────────────────────────────────────────────────────────────────────
ref  structural      bge-base-en       hybrid-rrf     none                  0.412     —
c-1  fixed-512       bge-base-en       hybrid-rrf     none                  0.389     p=0.02 (worse)
c-2  semantic-sim    bge-base-en       hybrid-rrf     none                  0.421     p=0.31 (tie)
c-3  parent-child    bge-base-en       hybrid-rrf     none                  0.428     p=0.04 (better)
e-1  structural      bge-large-en      hybrid-rrf     none                  0.451     p<0.001 (better)
e-2  structural      e5-mistral-7b     hybrid-rrf     none                  0.483     p<0.001 (better)
r-1  structural      bge-base-en       dense-only     none                  0.398     p=0.09 (tie)
r-2  structural      bge-base-en       bm25-only      none                  0.371     p<0.001 (worse)
k-1  structural      bge-base-en       hybrid-rrf     bge-reranker-large    0.487     p<0.001 (better)
```

Every row's *interpretation* is unambiguous: `c-3`'s +0.016 is attributable to the chunker swap alone. `k-1`'s +0.075 is attributable to adding the reranker alone. There is no confounding — because no row moves two axes at once, no delta requires disentangling.

Anti-pattern: rows like `"parent-child + e5-mistral-7b + hybrid + bge-reranker-large"` compared to reference. The delta may be +0.15 nDCG@10, gorgeous — but you cannot tell whether it was the chunker (small), the embedder (large), the reranker (largest), or additive effects across two of them. When the model gets deprecated 6 months later and you have to re-choose, the number is useless.

## Ordering matters: from cheap to expensive

Not every ablation row is equally expensive to run. A chunker swap forces re-embedding the entire corpus (mod-103's dominant cost). An embedding swap does the same. A retriever swap (going from hybrid to dense-only) reuses the existing indexes. A reranker swap is essentially free (it runs on the top-N candidates from an existing retriever).

The economical ordering is **reverse cost**: pin the expensive axes first, then sweep the cheap ones on top.

1. **Freeze the reference pipeline.** Everything is the current shipping configuration.
2. **Sweep the reranker (cheap, K rows).** No re-indexing, only re-scoring.
3. **Sweep the retriever composition (cheaper, K rows).** Reuses indexes.
4. **Sweep the embedder (expensive; each requires re-indexing).** Do this second-to-last so the reranker + retriever knobs are already tuned to the reference embedder.
5. **Sweep the chunker (expensive; each requires re-indexing).** Last, and often the smallest number of options.

If the budget is tight, the honest thing to publish is a partial grid with the cheap axes complete and the expensive ones with a top-1 or top-2 shortlist. Never fake a fuller grid than you actually ran.

## Two ablation designs beyond one-at-a-time

One-at-a-time is the default. Two extensions come up:

### Interaction ablations (Latin-square / fractional factorial)

Sometimes two axes interact — a specific chunker only pays off with a specific embedder. A pure one-at-a-time grid misses this. If you have budget for `~2N` runs instead of `~N`, do a small **fractional factorial** on the two axes you suspect interact: fix all others at reference, run the full 2×2 (or 3×3) grid on the pair.

Concrete example: chunker × embedder interaction on structural / parent-child × bge-base / e5-mistral:

```
row      chunker         embedder          nDCG@10
────────────────────────────────────────────────
ss-1     structural      bge-base-en       0.412 (reference)
ss-2     structural      e5-mistral-7b     0.483
ss-3     parent-child    bge-base-en       0.428
ss-4     parent-child    e5-mistral-7b     0.492
```

Additive expectation: `0.412 + (0.483 - 0.412) + (0.428 - 0.412) = 0.499`. Observed: `0.492`. Interaction: `-0.007` (i.e., parent-child helps *less* with the stronger embedder — the chunker's help is somewhat substitutable with the embedder's semantic capacity). Interesting; often small; occasionally decisive. Sakai's *Metrics, Statistics, Tests* chapter 5 discusses factorial designs for IR eval.

### Multi-baseline ablations

When you're comparing many configurations (chapter 5's grid can grow to 20+ rows), report each row against **both** the reference and the current top scorer. Otherwise a row that beats the reference by 0.05 but is 0.02 behind the current best gets mis-attributed as a win. Chapter 4's Benjamini-Hochberg FDR correction is what you apply across the resulting significance-test matrix.

## Attribution: reading the grid honestly

The purpose of the grid is not to pick the highest-scoring row. It is to attribute *how much* each swap contributed and *whether* the contribution is stable.

Three readings to do on every completed grid:

1. **Which axis matters most on this corpus?** Sum the |delta| across the rows that swap each axis. The axis whose sweep produced the largest total movement is the one the team should keep investing in. On chapter 5's example above: the reranker (`k-1`) contributed the largest single delta (+0.075); the embedder swaps (`e-1`, `e-2`) contributed the largest cumulative movement.
2. **Which axes are noise on this corpus?** Rows whose Wilcoxon `p > 0.10` after FDR correction are not moving the number reliably — either the change genuinely doesn't matter for your corpus, or your judgment set is too small to detect the effect (chapter 4's sample-size logic). Don't ship a change that lands in the noise floor.
3. **Which axes' wins are largest per-dollar?** Compute cost (embedder GPU-hours + reranker latency + storage) per row. A reranker swap that adds 200ms and 100ms of latency but +0.075 nDCG@10 may or may not clear the bar; an embedder swap that adds 10× the storage for +0.032 nDCG@10 usually does not.

The output of these three readings is a **decision memo**, not just a table. Ship what the grid supports, defer what the grid does not resolve, add queries and re-run when in doubt.

## Confounders: what makes a grid dishonest

The failure modes that turn a grid into a lie:

- **Different qrels per row.** Never re-label between rows. The judgment set is fixed for the duration of the grid; if you must expand it, re-run the reference too so all rows share the same qrels version.
- **Different queries per row.** Every row runs on the same query set. If you dropped queries from one row (e.g., because a retriever returned zero candidates), drop them from the reference too and re-compute.
- **Different funnel widths per row.** If your reference retrieves top-50 and a candidate retrieves top-200, you cannot compare their nDCG@10 fairly — you have swapped two knobs, not one. Fix the funnel widths and re-run.
- **Different tokenization / prompt per embedder.** Some embedders (E5, GTE-Qwen) require task-specific prefixes; skipping them costs 5-20 nDCG points (mod-103 chapter 2). If one row uses the recommended prefix and another does not, the comparison is meaningless. Always use each model's recommended prompt.
- **One noisy run per row.** ANN retrievers (mod-104) have run-to-run variance from HNSW's stochastic construction and `efSearch` sensitivity. For rows near the significance threshold, run the retriever 3-5 times with different seeds and average per-query scores — or state the variance explicitly.
- **Reranking with the same model that indexed.** Some reranker setups accidentally re-embed with the same model that indexed and never actually see the query — always verify the reranker is doing what you think.

Every one of these has produced a shipped-and-later-reverted decision at some point in some team's history. The grid's honesty is the module's contract.

## Concrete: a minimal grid runner

The infrastructure to run the grid is a small orchestrator that varies configs, produces run files, evaluates them, and writes a table:

```python
# grid_runner.py — skeleton
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    chunker: str
    embedder: str
    retriever: str    # "dense-only" | "bm25-only" | "hybrid-rrf"
    reranker: str     # "none" | "bge-reranker-base" | ...
    retrieve_top_n: int = 50
    rerank_to_k: int = 10

    def label(self) -> str:
        return f"{self.chunker}__{self.embedder}__{self.retriever}__{self.reranker}"

REFERENCE = Config("structural", "bge-base-en", "hybrid-rrf", "none")

def one_axis_grid(reference: Config) -> list[Config]:
    rows = [reference]
    for c in ("fixed-512", "semantic-sim", "parent-child"):
        rows.append(replace(reference, chunker=c))
    for e in ("bge-large-en", "e5-mistral-7b", "text-embedding-3-large"):
        rows.append(replace(reference, embedder=e))
    for r in ("dense-only", "bm25-only"):
        rows.append(replace(reference, retriever=r))
    for k in ("bge-reranker-base", "bge-reranker-large"):
        rows.append(replace(reference, reranker=k))
    return rows

def run_config(cfg: Config, queries) -> Path:
    """Produce a TREC-format .run file for this config; cache indexes by (chunker, embedder)."""
    ...  # ingest → index (if needed) → retrieve → rerank → write run

def evaluate_and_compare(rows, qrels_path, reference_label):
    import ir_measures
    from ir_measures import nDCG, RR, R
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    qrels = ir_measures.read_trec_qrels(qrels_path)
    per_query = {}
    for row in rows:
        runp = run_config(row, ...)
        by = {}
        for m in ir_measures.iter_calc([nDCG@10, RR@10, R@50], qrels,
                                       ir_measures.read_trec_run(runp)):
            by.setdefault(m.measure, {})[m.query_id] = m.value
        per_query[row.label()] = by

    ref = per_query[reference_label]
    p_values = []
    entries = []
    for row in rows:
        if row.label() == reference_label: continue
        cand = per_query[row.label()]
        for measure in [nDCG@10, RR@10, R@50]:
            common = sorted(set(ref[measure]) & set(cand[measure]))
            va = [ref[measure][q]  for q in common]
            vb = [cand[measure][q] for q in common]
            _, p = wilcoxon(vb, va, zero_method="pratt")
            entries.append((row.label(), measure, sum(vb)/len(vb) - sum(va)/len(va), p))
            p_values.append(p)

    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    for (label, measure, delta, _), p_bh in zip(entries, p_adj):
        print(f"{label:60s}  {str(measure):10s}  Δ={delta:+.4f}  p_BH={p_bh:.4f}")
```

That skeleton — plus caching so re-embedding is only done once per unique `(chunker, embedder)` pair — is what mod-111 chapter 2's retriever CI wraps in an automated pipeline. Chapter 5's exercise-04 walks it end-to-end.

## When to expand the grid, when to shrink it

- **Expand** when the wins are within noise. If your reranker sweep shows `p > 0.10` on the largest delta, either the reranker doesn't help on your corpus or your judgment set is too small. Grow the judgment set (chapter 2 protocol B) or add rerankers to the sweep before making a call.
- **Expand** when a stakeholder challenges an axis you didn't sweep. If someone asks "did you check `text-embedding-3-large`?", add the row rather than defending its absence.
- **Shrink** when the winner is stable across sub-samples. If you've established that reranker X wins reranker Y with `p < 0.001` on the full 200-query set, you don't need to re-run it in every subsequent grid — pin it and sweep the axes that are still open.
- **Shrink** when cost dominates value. A full 4-axis grid with 3 options per axis is `3^4 = 81` runs; almost never worth it. The one-at-a-time grid with the top 2-3 options per axis (12-16 rows) is the practical sweet spot.

## What you should be able to do after this chapter

- Draw the four-axis grid (chunker × embedding × retriever × reranker) and name the module that owns each axis.
- Author a one-axis-at-a-time grid where each row moves exactly one axis from the reference.
- Order rows from cheap to expensive to minimize re-indexing cost.
- Attribute a metric delta to the single axis that moved (and refuse to attribute when two moved).
- Enumerate the six confounders that make a grid dishonest and defend a grid against each.
- Extend a grid with a small factorial when two axes are suspected of interacting.
- Apply Benjamini-Hochberg FDR when the grid has > 5 significance tests.

## Further reading

- Sakai, T. (2014). *Metrics, Statistics, Tests.* [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — chapter 5 covers factorial designs for IR eval; chapter 4 of this module cites the same reference for paired testing.
- Voorhees, E. M., & Harman, D. K. (Eds.). (2005). *TREC: Experiment and Evaluation in Information Retrieval.* MIT Press. [DOI:10.7551/mitpress/6916.001.0001](https://doi.org/10.7551/mitpress/6916.001.0001) — the TREC playbook has decades of ablation-grid precedent.
- Lin, J., & Yang, P. (2019). *The Impact of Score Ties on Repeatability in Document Ranking.* SIGIR (short paper). [DOI:10.1145/3331184.3331339](https://doi.org/10.1145/3331184.3331339) — the underappreciated reproducibility trap that hits grid runs (score ties break differently across runs; affects nDCG at the boundary).
- Craswell, N., Mitra, B., Yilmaz, E., & Campos, D. (2020). *Overview of the TREC 2019 Deep Learning Track.* [arXiv:2003.07820](https://arxiv.org/abs/2003.07820) — a good example of a modern ablation-heavy IR report; the shape chapter 5's grid produces.
- Ma, X., Sun, K., Pradeep, R., & Lin, J. (2021). *A Replication Study of Dense Passage Retriever.* [arXiv:2104.05740](https://arxiv.org/abs/2104.05740) — a canonical example of a one-axis-at-a-time ablation on the DPR paper.
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley. — the general experimental-design reference underneath IR-specific work.
- MacAvaney, S., Yates, A., Feldman, S., et al. (2022). *Reproducing Personalised Search Systems: Do IR Papers Give Enough Info to Replicate Them?* SIGIR. [DOI:10.1145/3477495.3531662](https://doi.org/10.1145/3477495.3531662) — the discipline of reporting enough to enable someone else to re-run your grid.
