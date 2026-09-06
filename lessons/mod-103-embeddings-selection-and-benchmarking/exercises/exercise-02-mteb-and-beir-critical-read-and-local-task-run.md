# Exercise 02 — MTEB / BEIR Critical Read + Local Task Run

**Estimated effort:** ~3 hours.

## Objective

Move from "I have heard of MTEB" to "I have run one, read the JSON it writes, and know how to add my own task." The exercise has two halves. The first is a **critical read** of the public MTEB leaderboard: capture six candidate models into a shortlist table with the fields chapter 2 argued you should actually be looking at (retrieval nDCG@10 specifically, params, dim, max tokens, license), not the leaderboard's default sort. The second is a **local run**: install `mteb`, run a small built-in BEIR retrieval task against two open embedders, aggregate the output JSON into a comparison table, and register one custom `AbsTaskRetrieval` subclass over a tiny hand-labeled corpus so the harness treats your data as a first-class task.

The output is muscle memory for chapters 6 and 7: you will have already touched the runner, the JSON format, and the wrapper contract before those chapters ask you to sweep six models across your real corpus.

## Prerequisites

- Chapters 2 (*Reading MTEB and BEIR Critically*) and 3 (*Running MTEB Locally on a Custom Retrieval Task*) read once.
- Python env with `mteb>=1.14`, `sentence-transformers`, `pandas`, `datasets`. A GPU is helpful but not required — SciFact and NFCorpus (the two tiny BEIR tasks used below) run on CPU in a few minutes.
- The mod-101 toolchain is up. No vector DB is required for this exercise — MTEB does retrieval in-process against the task's own corpus.
- A Hugging Face account is not required to pull the two small BEIR datasets, but rate-limits are friendlier if you `huggingface-cli login`.

## Part A — Leaderboard critical-read (30 min)

Open the [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Do **not** sort by the default "Average" column.

1. Sort by **Retrieval Average (nDCG@10)** — English tab.
2. Pick **six** models spanning size and provenance:
   - one small open bi-encoder (~30–100M params),
   - one mid-size open bi-encoder (~300–500M params),
   - one large open bi-encoder or LLM-backbone embedder (≥1B params),
   - one hosted model (OpenAI / Cohere / Voyage / Google),
   - one model that requires an instruction prefix (E5, GTE-Qwen, Nomic, Snowflake Arctic Embed v2, Jina v3),
   - one model you have never heard of that is Top-20 on retrieval.
3. For each, capture the fields chapter 2 called out. Save as `docs/model-shortlist.md`:

```markdown
| Model | Provenance | Params | Native dim | MRL? | Max tokens | MTEB retrieval nDCG@10 | Note on training corpus / prompt requirement | License | Card URL |
|---|---|---|---|---|---|---|---|---|---|
| ... | HF repo or vendor page | ... | ... | Y/N | ... | ... | ... | ... | ... |
```

Rules:

- "Params" is not on the leaderboard for hosted APIs — write `n/a` and note it.
- "MTEB retrieval nDCG@10" is the value in the *Retrieval* column, not "Average". If the leaderboard has drifted, capture the current value and the leaderboard revision timestamp.
- "Note on training corpus / prompt requirement" is the axis that will bite you when you run the model — capture E5's `query: /passage: ` prefix, GTE-Qwen's task instruction, Cohere's `input_type`, etc.
- **Do not skip the license check.** Jina v3 is CC-BY-NC. Some Chinese-origin models have non-commercial or use-restriction clauses. Read the model card, not the leaderboard.

In two closing sentences at the bottom of the file, name one pair of models that are within 1 nDCG point of each other on retrieval and explain (per chapter 2) why the leaderboard cannot break that tie for your corpus.

## Part B — Run a built-in BEIR retrieval task locally (45 min)

Install and smoke-test:

```bash
pip install "mteb>=1.14" sentence-transformers pandas
```

Write `scripts/run_scifact.py` that runs the SciFact retrieval task (small, ~5k docs, downloads in seconds) against **two** open embedders:

- `BAAI/bge-small-en-v1.5` (baseline).
- One model from your Part A shortlist that requires an instruction prefix. **Use MTEB's registered model** for that model (`mteb.get_model(...)`) so the harness handles the prefix correctly — do not write a raw `SentenceTransformer.encode()` call for prefix-sensitive models.

Requirements:

- Write results to `results/scifact/<model_name_slug>/` — MTEB's default output shape.
- Print each model's `main_score` (which is `ndcg_at_10` on retrieval tasks).
- Cross-check both values against the model card's published SciFact score. If your local number is ≥5 points below the model card's, you almost certainly forgot the prompt. Fix it. Chapter 3 covered this.
- Run NFCorpus (medical, ~3.6k docs) as a second task in the same script so you have a 2 model × 2 task cell to aggregate in Part C.

Reproducibility hygiene per chapter 3:

- Pin `sentence-transformers` and `mteb` versions in a `requirements.txt`.
- Pass an explicit `revision=` when constructing the `SentenceTransformer` — you want a stable snapshot, not "whatever HF cached today."
- Set `torch.manual_seed(0)` and call `model.eval()` explicitly.
- Note the `mteb.__version__` in the header of your results doc.

## Part C — Aggregate the JSON MTEB wrote (30 min)

Write `scripts/aggregate_results.py` that globs `results/scifact/*/*.json` and `results/nfcorpus/*/*.json` and produces `docs/beir-mini-comparison.md`:

```markdown
# BEIR mini comparison — {date}, mteb=={version}

| Model | Revision | SciFact nDCG@10 | SciFact Recall@100 | NFCorpus nDCG@10 | NFCorpus Recall@100 |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |
```

The aggregator should:

- Load each JSON at the `scores.test[0]` (or `scores.<split>[0]`) path chapter 3 named.
- Pull `ndcg_at_10`, `recall_at_100`, `mrr_at_10`.
- Sort the final table by SciFact nDCG@10 descending.
- Include a small **BM25 row** you compute by hand (values from the BEIR paper, or from Pyserini's [reproduction docs](https://castorini.github.io/pyserini/2cr/beir.html) — cite the source in the table's caption). Chapter 2 argued BM25 is a mandatory baseline; putting it in the table cements the habit.

## Part D — Register your own retrieval task (45 min)

Build a tiny domain-specific eval set (10 queries is enough for a shape check — you are not defending a model choice here, you are proving you can plug custom data into MTEB). Any small corpus works: your team's internal wiki dump, 50 chunks from mod-102's exercise-03, or ten Wikipedia articles on a niche topic. Whatever you pick:

- Save the corpus as `data/tiny_task/corpus.jsonl` (BEIR shape: `{"_id": "...", "title": "...", "text": "..."}` per line).
- Save `data/tiny_task/queries.jsonl` (BEIR shape: `{"_id": "...", "text": "..."}` per line, ~10 entries).
- Hand-label `data/tiny_task/qrels/test.tsv` in TREC format:
  ```
  query-id  corpus-id  score
  q1        d17        1
  q1        d42        1
  q2        d05        1
  ...
  ```
  Each query needs at least one relevant doc; ideally 1–3 relevant docs.

Then subclass `AbsTaskRetrieval` as `TinyDomainRetrieval` in `src/tiny_task.py`. Follow the shape chapter 3 laid out: a `TaskMetadata` with `type="Retrieval"`, `main_score="ndcg_at_10"`, and a `load_data` method that populates `self.corpus`, `self.queries`, and `self.relevant_docs` as the three-dict contract.

Register it and run **the same two models** from Part B against it. Extend `docs/beir-mini-comparison.md` with a third dataset column (`TinyDomain nDCG@10`).

Rules to follow (chapter 3's reproducibility discussion):

- Give your task a `dataset["revision"]` string (e.g., `"v1"`). If you touch `qrels/test.tsv` after running, bump the revision.
- Keep the doc-ID convention stable between the JSONL corpus and the qrels file — a typo here shows up as an artificially bad score, not an error.

## Part E — The two-paragraph read (15 min)

Close `docs/beir-mini-comparison.md` with two paragraphs:

1. **What the numbers tell you.** Which model won on SciFact? On NFCorpus? On your tiny domain? Do the rankings agree? If they do not, name the datasets where the ordering flipped — that is chapter 2's "leaderboard averages hide task-level variance" showing up in your own runs.
2. **What is missing before you would ship any of these.** Chapter 2 named five things (real query distribution, corpus terminology, latency budget, storage cost, license) that a leaderboard score does not tell you. Which one of them is your biggest gap for making an actual choice? This paragraph is the setup for exercise-04 (domain-fit benchmark on *your* real corpus).

## Deliverables

- `docs/model-shortlist.md` — Part A's six-row table with the closing tie-breaker note.
- `scripts/run_scifact.py`, `scripts/aggregate_results.py` — Parts B and C.
- `results/scifact/<model>/…json`, `results/nfcorpus/<model>/…json`, `results/tiny_task/<model>/…json` — raw MTEB output, checked in (they are small).
- `data/tiny_task/{corpus.jsonl,queries.jsonl,qrels/test.tsv}` — Part D's corpus.
- `src/tiny_task.py` — the `AbsTaskRetrieval` subclass.
- `docs/beir-mini-comparison.md` — Part C's table with the tiny-domain column and the Part E closing paragraphs.
- `requirements.txt` with pinned `mteb`, `sentence-transformers`, `torch`.

## Acceptance criteria

- `docs/model-shortlist.md` has six rows, all fields populated, license and prompt-requirement column non-empty for every row.
- `python scripts/run_scifact.py` runs end-to-end and writes MTEB JSON for both models on both tasks; the values are within ~3 nDCG points of the corresponding model card / BEIR paper numbers (larger gaps indicate a prompt-handling bug — investigate before continuing).
- `docs/beir-mini-comparison.md` contains a table with rows for both models plus a BM25 baseline row citing its source.
- The custom `TinyDomainRetrieval` task registers and runs; `mteb.MTEB(tasks=[TinyDomainRetrieval()]).run(model, ...)` writes a JSON with a populated `main_score`.
- The Part E closing paragraphs are specific ("model A won SciFact, model B won NFCorpus, they tied on TinyDomain") — not vague ("results were interesting").

## Stretch goals

- Add a hosted-API model (OpenAI `text-embedding-3-small`, Cohere `embed-english-v3.0`, Voyage `voyage-3`, or Google `text-embedding-005`) to the sweep by writing the ~30-line adapter chapter 3 sketched. Confirm the JSON is comparable to the open-model rows.
- Register your `TinyDomainRetrieval` task as an `MTEB` task with a `ModelMeta`-style contribution to a local fork of the `embeddings-benchmark/mteb` repo. Look at any of the tasks in `mteb/tasks/Retrieval/eng/` for the concrete pattern.
- Replace SciFact with a larger BEIR task (FiQA, TREC-COVID) if you have a GPU. The pipeline is unchanged; you get a stronger read on model differences.
- Run the same models on the **MTEB reranking** family (`MindSmallReranking`, `StackOverflowDupQuestions`) alongside retrieval. Chapter 2 argued the reranking column is what you sort on when picking a *reranker*; this stretch goal makes that concrete.
