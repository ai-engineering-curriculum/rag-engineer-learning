# Chapter 6 — BEIR / MTEB Subset Runs and the Corpus-Fit Delta

## Why this chapter exists

Chapters 2-5 gave you an eval on **your** corpus. Chapter 6 puts that eval in context: what does the same retriever score on a public benchmark? The **delta** between your corpus's score and the benchmark's is the actionable number — it tells you whether your corpus is harder, easier, or comparably difficult, and it lets you sanity-check your judgment set against a known baseline.

Two artifacts do this job:

- **BEIR** — 18 zero-shot retrieval datasets, primary metric `nDCG@10`. Introduced in Thakur et al. 2021.
- **MTEB Retrieval** — embedding-benchmark's retrieval subtasks, which are the BEIR datasets plus a few more (roughly 15 in the v1 spec).

Running a BEIR or MTEB subset alongside your own eval is a cheap discipline that catches four failure modes: mislabeled qrels, mis-configured retrievers, corpus-specific vs. general regressions, and stale leaderboard baselines.

## What "run a BEIR subset" actually means

You do **not** run all 18 BEIR datasets. That is a benchmark paper's project (days of GPU time, hundreds of gigabytes of corpus downloads). You run a **subset** — usually 2 or 3 datasets — chosen because they're close to your corpus in domain, corpus size, or query shape.

The typical shortlist:

| Your corpus looks like… | Pick these BEIR datasets |
|---|---|
| Wikipedia / general knowledge | NQ, HotpotQA |
| Biomedical / scientific | NFCorpus, SciFact, TREC-COVID |
| Financial | FiQA |
| News | Robust04 (if licensed), Signal-1M (if licensed) |
| Community Q&A | CQADupStack (pick the subforum), Quora |
| Argument / opinion | ArguAna, Touché-2020 |
| Short queries + long docs (default catch-all) | NQ, MS MARCO |
| Long queries + short passages | Robust04, TREC-COVID |

Pick two datasets whose domain matches yours (or is the closest available) plus **one general-purpose baseline** (NQ or MS MARCO) so you have a corpus-general anchor.

## The two ways to run BEIR

### Option 1 — the `beir` library directly

`pip install beir` gives you the reference retriever, corpus loaders, and evaluation script. The canonical loop from [the BEIR README](https://github.com/beir-cellar/beir):

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

dataset = "nfcorpus"          # small; good for smoke tests
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=64)
retriever = EvaluateRetrieval(model, score_function="cos_sim")
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [10, 100])
print(f"nDCG@10 = {ndcg['NDCG@10']:.4f}")
```

The BEIR library handles the corpus / queries / qrels download, the retriever loop, and the evaluation. Numbers agree with the Thakur et al. 2021 paper's published baselines to within noise.

### Option 2 — MTEB retrieval subtasks

`pip install mteb`, then:

```python
import mteb
tasks = mteb.get_tasks(tasks=["NFCorpus"])          # subtask names match BEIR
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")
```

`mteb` is a superset — it wraps BEIR's retrieval loop, publishes per-task JSON reports, and is what the [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) consumes. If you plan to ever submit a result or compare to a leaderboard row, use `mteb`; otherwise `beir` alone is fine.

## Running your **own** retriever on a BEIR subset

BEIR's harness accepts any callable that takes `queries + corpus` and returns a ranked list per query. To evaluate your production retriever on BEIR data (rather than only their reference retrievers):

```python
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class MyRetriever:
    """Wraps your production retriever (mod-104's index + mod-106 fusion + mod-107 reranker).
    BEIR expects encode_queries and encode_corpus, but you can subclass DRES and override.
    """
    def encode_queries(self, queries: list[str], batch_size, **kw):
        return self.embedder.encode(queries)   # your embedding stack
    def encode_corpus(self, corpus: list[dict], batch_size, **kw):
        texts = [c["title"] + " " + c["text"] for c in corpus]
        return self.embedder.encode(texts)
```

For non-dense retrievers (BM25, hybrid, learned-sparse), BEIR's [`beir.retrieval.search.lexical`](https://github.com/beir-cellar/beir/tree/main/beir/retrieval/search/lexical) and [`beir.retrieval.search.sparse`](https://github.com/beir-cellar/beir/tree/main/beir/retrieval/search/sparse) modules have BM25-via-Elasticsearch and SPLADE reference implementations you can adapt.

## The corpus-fit delta

Once you have (a) your retriever's score on your corpus and (b) the same retriever's score on a BEIR / MTEB subset, you can compute the **corpus-fit delta**:

$$
\Delta_{\text{corpus}} = \text{nDCG@10}_{\text{your corpus}} - \text{nDCG@10}_{\text{BEIR subset}}
$$

Interpretations of the delta:

- **Δ > 0, comfortably.** Your corpus is *easier* than the BEIR subset for this retriever — usually because your queries and documents share more vocabulary, or the corpus is smaller (fewer distractors), or the reranker is helping more on your domain.
- **Δ ≈ 0.** Your corpus behaves like BEIR. This is what you want if the reason you chose the BEIR subset was that it approximates your domain — the model transfers as advertised.
- **Δ < 0, comfortably.** Your corpus is *harder* than the BEIR subset. Usually one of: specialized terminology the model wasn't trained on, atypical query shapes (very short, very long, code snippets, entity IDs), or missing negatives in the judgment set inflating the BEIR score.
- **Δ = large negative + BEIR score close to leaderboard.** The retriever is well-configured; your corpus is genuinely hard. Your ablation grid (chapter 5) is where the wins have to come from.
- **Δ = large negative + BEIR score far below leaderboard.** The retriever is mis-configured or the BEIR-side run had a bug. Debug the BEIR side before you conclude anything about your corpus.

The last row is the reason to run BEIR at all — it is the sanity check on your retriever setup that catches "we forgot the E5 instruction prefix" or "the reranker was never actually being called."

## Reading the delta against the leaderboard

Every embedding model on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) publishes per-task numbers. When you run the same model on the same BEIR subset locally:

| | Model report (leaderboard) | Your local BEIR run | Delta |
|---|---|---|---|
| bge-base-en-v1.5 on NFCorpus | 0.372 nDCG@10 | 0.368 | -0.004 (within noise) |
| bge-base-en-v1.5 on your corpus | — | 0.412 | +0.040 (easier corpus) |

If your **local BEIR run** disagrees with the **leaderboard number** by more than ~0.01, one of six things is wrong:

1. **Missing instruction prefix.** Some models require `"query: ..."` / `"passage: ..."` (E5) or task-typed instructions (E5-mistral). Skipping them costs 5-20 nDCG points. Check the model card.
2. **Wrong score function.** Some models are trained for cosine similarity; some for dot product. `EvaluateRetrieval(model, score_function="dot")` vs. `"cos_sim"`.
3. **Wrong pooling.** For sentence-transformers models, default is mean-pooling; some models want CLS-token pooling. Match the model card.
4. **Wrong tokenization / max-tokens.** If your corpus loader truncates at 128 tokens but the model was benchmarked at 512, scores drop.
5. **Wrong BEIR split.** BEIR datasets have `test` / `dev` splits — most reported scores are on `test`. Loading `dev` accidentally is a common bug.
6. **`corpus.jsonl` field usage.** Some datasets have `title` and `text`; if you concatenate them for indexing but the leaderboard number came from `text` only (or vice versa), the numbers diverge.

Fix these before publishing a delta. The whole point of running BEIR is the sanity check; a mis-configured BEIR run is worse than none.

## The BEIR-11 vs. BEIR-14 vs. BEIR-18 trap

Different papers and leaderboards report BEIR "averages" over different subsets. Mod-103 chapter 2 named this trap; it lives here too.

- **BEIR-18** — the full 18 datasets from Thakur et al. 2021. Some (Robust04, TREC-News, Signal-1M) require licensed access.
- **BEIR-14** — dropping the licensed / harder-to-download datasets. Common in academic papers written after 2022.
- **BEIR-11** — a smaller "core" subset some vendor blogs use.

If you're reporting your BEIR run *as a proxy for the leaderboard*, name which subset you ran and match the leaderboard's subset. If you're using BEIR datasets *as a corpus-fit signal only* (this chapter's default), report per-dataset numbers and skip the average.

## What BEIR / MTEB do NOT tell you

- **Application-layer quality.** BEIR and MTEB measure retrieval only. Whether the generator will produce a good grounded answer is `ai-eval-engineer`'s question, not this module's.
- **Latency, cost, storage.** Quality benchmarks are silent on the systems axis. A model that wins on BEIR by 0.03 nDCG@10 but has 10× the embedding dimension is *not* a straightforward win in production. Mod-104 chapter 6's index-sizing worksheet is where that trade lives.
- **Long-context behavior.** BEIR passages are typically 60-500 tokens. If your corpus is 8k-token chunks, BEIR's per-dataset numbers do not translate directly.
- **Adversarial robustness.** BEIR queries are natural-language questions or descriptive topics. If your users type single-word queries, code snippets, or entity IDs, BEIR is silent on how the model handles those.
- **Multi-lingual corpora.** MMTEB (multilingual MTEB) covers this; plain BEIR is English-only.

## Concrete: the minimal BEIR + your-corpus report

The artifact this chapter produces sits alongside chapter 5's ablation grid:

```
BEIR / MTEB baseline runs (for corpus-fit sanity check)

reference model: BAAI/bge-base-en-v1.5

Dataset          Leaderboard nDCG@10   Local run nDCG@10   Delta (local - leader)
────────────────────────────────────────────────────────────────────────────────
NFCorpus         0.372                 0.368               -0.004  (within noise ✓)
NQ               0.508                 0.506               -0.002  (within noise ✓)
SciFact          0.741                 0.736               -0.005  (within noise ✓)

Local run on YOUR corpus (from chapter 5's reference row):
Your corpus      —                     0.412

Corpus-fit deltas:
Your corpus vs. NFCorpus:  +0.044   (easier than NFCorpus — expected; smaller corpus)
Your corpus vs. NQ:        -0.096   (harder than NQ — specialized terminology)
Your corpus vs. SciFact:   -0.329   (much harder than SciFact — expected; SciFact is well-structured claims)

Notes:
- Instruction prefix used: none (bge-base does not require one per model card).
- Score function: cos_sim (bge default).
- BEIR subset: NFCorpus / NQ / SciFact test splits, loaded via `beir==2.0.0`.
- Local BEIR runs agree with leaderboard within ±0.005 → retriever is correctly configured.
```

That table is the sanity check. It tells the reviewer of chapter 5's ablation grid two things: (1) your infrastructure is correctly configured (local BEIR matches leaderboard), and (2) your corpus's difficulty is calibrated against known baselines. The reviewer can now trust chapter 5's deltas.

## When to skip BEIR entirely

- Your corpus is so specialized (internal legal, customs classifications, trading tickers) that no BEIR dataset is a proxy — the delta is uninterpretable.
- You are on a language BEIR does not cover; use MMTEB instead.
- Your queries are entirely non-natural-language (SQL fragments, code identifiers) — BEIR's queries won't tell you what will happen.

In those cases, publish your corpus-specific chapter-5 grid without a BEIR corpus-fit column and note the reason. Do *not* skip the sanity check on your infrastructure — use a smaller BEIR dataset (NFCorpus, SciFact) purely as an "is my retriever correctly configured?" smoke test, and label it as such.

## What you should be able to do after this chapter

- Pick 2-3 BEIR datasets that best approximate your corpus's domain and query shape.
- Run BEIR (via `beir` or `mteb`) on a chosen model and reproduce a published leaderboard number to within ±0.01 nDCG@10.
- Diagnose a > 0.01 disagreement between your local BEIR run and the leaderboard (instruction prefix, score function, pooling, split, tokenization, field concatenation).
- Compute the corpus-fit delta and interpret it (easier / equivalent / harder / mis-configured).
- Enumerate what BEIR does not measure (application-layer quality, latency, storage, long-context, adversarial, multilingual).
- Recognize the BEIR-11 / BEIR-14 / BEIR-18 subset trap when comparing to a paper's average.

## Further reading

- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the source of truth for the 18 datasets, splits, and evaluation protocol.
- BEIR repository, [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — the canonical Python harness; corpus / qrels download scripts.
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — MTEB and its retrieval subtasks; superset of BEIR.
- Muennighoff, N., et al. (2024). *MMTEB: Massive Multilingual Text Embedding Benchmark.* [arXiv:2502.13595](https://arxiv.org/abs/2502.13595) — the multilingual expansion.
- MTEB repository, [`embeddings-benchmark/mteb`](https://github.com/embeddings-benchmark/mteb) — the canonical Python harness for MTEB.
- Hugging Face, [*MTEB Leaderboard*](https://huggingface.co/spaces/mteb/leaderboard) — the interactive leaderboard; sort by Retrieval nDCG@10 for this track.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the reference for MS MARCO passage reranking numbers commonly reported on BEIR-adjacent leaderboards.
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — the DPR paper; its BEIR follow-up in Thakur et al. 2021 is the "BM25 beats DPR out-of-domain" result that motivates chapter 6's discipline.
- Lin, J., Nogueira, R., & Yates, A. (2021). *Pretrained Transformers for Text Ranking: BERT and Beyond.* Synthesis Lectures on Human Language Technologies. [DOI:10.2200/S01123ED1V01Y202108HLT053](https://doi.org/10.2200/S01123ED1V01Y202108HLT053) — the standard neural-IR reference; chapter 6 of that book covers benchmark methodology at greater depth.
