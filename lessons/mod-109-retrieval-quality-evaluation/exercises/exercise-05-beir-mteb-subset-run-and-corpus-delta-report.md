# Exercise 05 — BEIR / MTEB Subset Run and Corpus-Fit Delta Report

**Estimated effort:** ~3 hours (mostly download + embed wall-clock). Python, one embedding model you can also run on your own corpus (from exercise 02 or exercise 04's reference row), 5-15 GB of local disk for BEIR downloads, network to the public BEIR / MTEB hosts. GPU strongly recommended for embed-heavy datasets (NQ, HotpotQA); CPU is fine for NFCorpus / SciFact.

## Objective

Run 2-3 BEIR (or MTEB Retrieval) datasets locally with the same embedder you use on your own corpus, reproduce each dataset's leaderboard number to within ±0.01 nDCG@10 (or diagnose why it disagrees), and compute the **corpus-fit delta** — the difference between your corpus's nDCG@10 and the BEIR subset's — as chapter 6 defines it.

At the end you have `beir/runs/{dataset}.run`, `beir/aggregate.md` (leaderboard vs. local table), `report/corpus-fit-delta.md` (the interpreted delta and diagnostics), and a short note that gets appended to exercise 04's decision memo so a reviewer can trust the ablation grid's infrastructure.

## Prerequisites

- Chapter 6 read; chapter 3 as background.
- One embedder available in your stack, ideally the one used as the reference in exercise 02 / exercise 04. It must be listed on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) so you have a published number to compare against — `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, `intfloat/e5-small-v2`, `sentence-transformers/all-MiniLM-L6-v2`, or an OpenAI / Cohere / Voyage hosted embedder all qualify.
- Exercise 02 complete — you have an `nDCG@10` for the same embedder on **your** corpus.
- Python 3.10+, `pip install beir ir_measures sentence-transformers`. Optionally `pip install mteb` if you plan to use the MTEB harness (chapter 6's option 2).
- 5-15 GB of free disk under `datasets/` for BEIR corpora (NFCorpus is ~30 MB; NQ is several GB).
- Network access to `public.ukp.informatik.tu-darmstadt.de` (BEIR host) and `huggingface.co` (MTEB / model weights).

## Steps

**Step 1 — Pick 2-3 BEIR datasets that approximate your corpus (15 minutes).** In `beir/subset-choice.md`:

Chapter 6's picker table:

| Your corpus looks like… | Pick these BEIR datasets |
|---|---|
| Wikipedia / general knowledge | NQ, HotpotQA |
| Biomedical / scientific | NFCorpus, SciFact, TREC-COVID |
| Financial | FiQA |
| Community Q&A | CQADupStack, Quora |
| Argument / opinion | ArguAna, Touché-2020 |
| Short queries + long docs (default catch-all) | NQ, MS MARCO |

Choose **2-3 datasets**: two that resemble your corpus's domain / query shape, plus **one general-purpose baseline** (NQ, MS MARCO, or NFCorpus if the others are too large). Justify each pick in one sentence.

If none of the BEIR domains are close enough (chapter 6's *When to skip BEIR entirely*), pick two small datasets (NFCorpus, SciFact) purely as an **infrastructure smoke test** and label them as such in the report. The delta interpretation weakens in that case; the leaderboard sanity check still works.

**Step 2 — Download and load each dataset (10 minutes + download time).** In `beir/load.py`:

Chapter 6's canonical loop:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

DATASETS = ["nfcorpus", "scifact", "nq"]  # yours

for name in DATASETS:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"{name}: {len(corpus)} docs, {len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments")
```

Sanity check: **use the `test` split**. Chapter 6's checklist calls out `dev` vs `test` as one of the top-6 leaderboard-mismatch bugs. Loading `dev` will look like it works and will silently produce numbers that disagree with the leaderboard by 1-3 nDCG points.

**Step 3 — Run your embedder on each BEIR dataset (30-90 minutes + embed wall-clock).** In `beir/run.py`:

Reference implementation using the `beir` harness with a sentence-transformers model (chapter 6, option 1):

```python
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

MODEL = "BAAI/bge-base-en-v1.5"      # match your exercise-02 reference embedder

for name in DATASETS:
    _, _, _ = None, None, None
    corpus, queries, qrels = GenericDataLoader(data_folder=f"datasets/{name}").load(split="test")

    model = DRES(models.SentenceBERT(MODEL), batch_size=64)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")   # match model card
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [10, 100])
    print(f"{name}: nDCG@10 = {ndcg['NDCG@10']:.4f}")

    # Also persist a TREC-shape run file so ir_measures can re-score.
    with open(f"beir/runs/{name}.run", "w") as f:
        for qid, doc_scores in results.items():
            ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])[:100]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{MODEL}\n")
```

Two disciplines chapter 6 requires:

- **Use the model card's recommended prompt / prefix / score-function / pooling.** E5 needs `"query: ..."` / `"passage: ..."`, GTE-Qwen needs task-specific instructions, some models are cosine-scored and some are dot-product-scored. Skipping any of these is chapter 6's top-6 leaderboard-mismatch bug list.
- **Concatenate `title + text` (or whichever field combination the leaderboard used).** Some datasets have both; the leaderboard number was almost always produced with a specific concatenation. Match it.

**Step 4 — Compute nDCG@10 with `ir_measures` for a cross-check (15 minutes).** In `beir/evaluate.py`:

The `beir` library computes nDCG internally; re-score the same run file with `ir_measures` and confirm the numbers agree to at least 4 decimals:

```python
import ir_measures
from ir_measures import nDCG, R

for name in DATASETS:
    qrels_path = f"datasets/{name}/qrels/test.tsv"
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run   = ir_measures.read_trec_run(f"beir/runs/{name}.run")
    agg   = ir_measures.calc_aggregate([nDCG@10, R@100], qrels, run)
    print(f"{name}: nDCG@10 = {agg[nDCG@10]:.4f}   R@100 = {agg[R@100]:.4f}")
```

If `beir`'s number and `ir_measures`'s number disagree by more than 4 decimals, chapter 3's caveat kicks in: check the gain-function (exponential vs linear) and the logarithm base (base-2 is TREC default). This is exercise 02's cross-library sanity check applied on BEIR data.

**Step 5 — Compare to the leaderboard (20 minutes).** In `beir/aggregate.md`:

Look up the same `(model, dataset)` on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — filter by task type `Retrieval`, sort by nDCG@10, find your model's row. Record the published number. Publish a table:

| Dataset | Leaderboard nDCG@10 | Local nDCG@10 | Delta (local − leader) | Verdict |
|---|---|---|---|---|
| NFCorpus | 0.372 | 0.368 | -0.004 | within noise ✓ |
| SciFact | 0.741 | 0.736 | -0.005 | within noise ✓ |
| NQ | 0.508 | 0.412 | -0.096 | mis-configured ✗ |

Chapter 6's tolerance: **within ±0.01 is passing**. Larger disagreements need to be resolved before publishing anything else.

If any row is outside tolerance, work chapter 6's *Reading the delta against the leaderboard* six-item checklist:

1. Missing instruction prefix (E5, GTE-Qwen, e5-mistral)?
2. Wrong score function (`cos_sim` vs `dot`)?
3. Wrong pooling (mean vs CLS)?
4. Wrong tokenization / max-tokens (128 vs 512)?
5. Wrong split (`dev` vs `test`)?
6. Wrong corpus-field usage (`title + text` vs `text` only)?

Fix and re-run. Do NOT proceed to step 6 with an out-of-tolerance BEIR row unresolved — the corpus-fit delta is meaningless if the BEIR side is wrong.

**Step 6 — Compute the corpus-fit delta (15 minutes).** In `report/corpus-fit-delta.md`:

Chapter 6's definition:

$$
\Delta_{\text{corpus}} = \text{nDCG@10}_{\text{your corpus}} - \text{nDCG@10}_{\text{BEIR subset}}
$$

For each BEIR dataset you ran, publish a row:

| Comparison | Your corpus nDCG@10 | BEIR dataset nDCG@10 | Corpus-fit delta | Interpretation |
|---|---|---|---|---|
| Your corpus vs. NFCorpus | 0.412 | 0.368 | +0.044 | Easier than NFCorpus (smaller corpus / vocabulary overlap). |
| Your corpus vs. SciFact | 0.412 | 0.736 | -0.324 | Much harder than SciFact (SciFact is well-structured claims). |
| Your corpus vs. NQ | 0.412 | 0.508 | -0.096 | Harder than NQ (specialized terminology). |

Chapter 6's four interpretations:

- `Δ > 0, comfortably.` Your corpus is easier — vocabulary overlap, fewer distractors, or reranker helping more on your domain.
- `Δ ≈ 0.` Your corpus behaves like BEIR — the model transfers as advertised.
- `Δ < 0, comfortably.` Your corpus is harder — specialized terminology, atypical query shapes, missing negatives in your qrels.
- `Δ = large negative + BEIR score close to leaderboard.` Retriever is well-configured; corpus is genuinely hard. Ablation-grid wins have to come from that side.
- `Δ = large negative + BEIR score far below leaderboard.` **Do not conclude anything about your corpus** — the BEIR run is buggy. Return to step 5.

Add one paragraph naming the actionable takeaway: does your corpus need more relevance labels (the "harder than BEIR + BEIR looks right" reading), a stronger embedder (the "harder than BEIR + reranker isn't helping" reading), or is your ablation grid essentially the whole story (the "as-hard-as-BEIR" reading).

**Step 7 — Append the sanity note to the ablation grid decision memo (15 minutes).** In `report/decision-memo.md` (from exercise 04) or `report/corpus-fit-delta.md`:

Chapter 6 + chapter 7's boundary discipline: a reviewer of the ablation grid needs a one-paragraph sanity note confirming the infrastructure is correctly configured. Append to exercise 04's decision memo (or, if you did not do exercise 04, publish it as the top of `report/corpus-fit-delta.md`):

> **BEIR / MTEB infrastructure sanity.** With `BAAI/bge-base-en-v1.5` on the NFCorpus / SciFact / NQ test splits (loaded via `beir==2.0.0`), local nDCG@10 agrees with the MTEB leaderboard within ±0.005 across all three datasets → the retriever is correctly configured (correct prefix, score function, pooling, split, field concatenation). The corpus-fit delta vs. NFCorpus is +0.044 (our corpus is easier than NFCorpus for this embedder); vs. NQ is −0.096 (harder than NQ, specialized terminology). Ablation-grid wins in the decision memo should therefore be trusted for the retriever configuration and interpreted against the "harder than NQ" baseline.

That paragraph is the artifact chapter 6 says the grid needs. Without it, a reviewer cannot tell whether the grid's deltas are real signal or the noise of a mis-configured BEIR-adjacent run.

## Acceptance criteria

- `beir/subset-choice.md` names 2-3 BEIR / MTEB Retrieval datasets and justifies each pick (or explicitly labels them as an infrastructure smoke test).
- `beir/runs/{dataset}.run` in TREC format, one file per dataset, top-100 per query.
- `beir/aggregate.md` contains a leaderboard-vs-local table with published, local, delta, and verdict columns. Every row is within the ±0.01 tolerance (or the reason is documented).
- `ir_measures` cross-check agrees with `beir` / `mteb`'s nDCG@10 to at least 4 decimals per dataset.
- `report/corpus-fit-delta.md` contains one row per (your-corpus, BEIR-dataset) pair, with the delta computed and one of chapter 6's four interpretations selected.
- A one-paragraph BEIR sanity note is appended to exercise 04's `report/decision-memo.md` (or published at the top of `report/corpus-fit-delta.md` if exercise 04 is not done).
- The **exact** model, model version, prompt / prefix, score function, pooling, split, tokenizer max-tokens, and `beir` / `mteb` library version are recorded in `beir/aggregate.md` so a reviewer can reproduce.

## Stretch goals

- **Run the BEIR subset on your entire ablation grid.** Reuse exercise 04's grid runner: for each `(chunker × embedder × retriever × reranker)` row that uses `bge-base-en-v1.5`, run NFCorpus with the same downstream stack. Now each row of the ablation grid has a paired BEIR corpus-fit delta — which tells you whether an embedder swap that helped your corpus also helps BEIR (transfer) or only helps your corpus (over-fit / lucky).
- **BEIR-11 vs. BEIR-14 vs. BEIR-18 subset audit.** Chapter 6's *BEIR-11 vs. BEIR-14 vs. BEIR-18 trap*. If you plan to publish an "average BEIR" number, name which subset you ran, match the leaderboard's subset, and publish per-dataset numbers alongside the average.
- **MMTEB (multilingual) subset.** If your corpus is not English, replace BEIR with a subset of MMTEB tasks in the same language. Repeat the sanity-check + delta protocol. Same discipline, different benchmark artifact.
- **Contamination check hand-off.** For any embedder whose leaderboard number seems suspiciously high (much higher than similar-sized peers), draft a two-sentence question for the `model-evaluation-engineer` peer (chapter 7's boundary) asking whether the model's training data may overlap with the MTEB task. Do NOT try to answer this yourself.
- **Difficulty comparison.** Compute per-query nDCG@10 on your corpus and on the BEIR subset, and compare their distributions (5th, 25th, 50th, 75th, 95th percentiles). Two corpora with the same mean can have very different tails; if your corpus has a much longer "hard queries" tail than BEIR, exercise 01's difficulty-stratification stretch goal is where the next-round investment should go.
- **Re-run with a second embedder.** Pick a second model on the leaderboard (larger or a different family) and repeat steps 3-6. The corpus-fit delta as a function of embedder tells you whether corpus difficulty is model-specific or model-invariant on your corpus.

## Notes

- **Do not run all 18 BEIR datasets.** Chapter 6: 2-3 is the target. A full BEIR run is days of GPU and hundreds of GB of downloads; it is a benchmark-paper's project, not a sanity-check.
- **Match the split, exactly.** `test` is the leaderboard default for almost every BEIR dataset. If you load `dev`, your numbers will look self-consistent and will silently disagree with the leaderboard.
- **Do not publish a corpus-fit delta if the BEIR-vs-leaderboard delta is out of tolerance.** Chapter 6: fix the BEIR run first. A mis-configured BEIR run is worse than none.
- **The corpus-fit delta is directional, not a verdict.** `Δ < 0` (your corpus is harder) is neither good nor bad; it is diagnostic. It tells you where the ablation grid's next investment lives.
- **BEIR measures retrieval only.** Chapter 6 + chapter 7's boundary: BEIR / MTEB do not measure faithfulness, answer-relevance, latency, cost, storage, long-context behavior, or adversarial robustness. Do not read a good BEIR number as "our RAG system is good."
- **Cite the specific `beir` / `mteb` version.** Both harnesses have shipped bug-fixes that shift numbers by 0.001-0.005 nDCG; the leaderboard row is typically produced with a specific version. Record it.
- **Do not concatenate `title + text` if the leaderboard used only `text`** (or vice versa). Check each dataset's leaderboard column notes — this is chapter 6 bug #6 and is one of the sneakier disagreements to diagnose.
