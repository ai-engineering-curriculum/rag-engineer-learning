# Chapter 3 — Running MTEB Locally on a Custom Retrieval Task

## Why this chapter exists

Chapter 2 argued that the number that decides your embedding choice is a *domain-fit* score on *your* corpus and *your* queries. The MTEB Python package makes that surprisingly easy: it is a runner with a task registry, a model wrapper, and a JSON-schema output format. If you shape your own eval set into a task class, you get MTEB's harness — batched embedding, standard metrics, deterministic seeds, per-model result files — for the same effort as a hand-rolled recall@k script, and you get results that plug into the same reporting conventions the leaderboards use.

This chapter is the mechanics. Chapter 7 is the *design* of a domain-fit benchmark (choose the eval set, choose the models, run the sweep). Here you learn how to run a single MTEB task locally, how to run one of the built-in BEIR datasets, and how to register a custom `AbsTaskRetrieval` for your own qrels — so chapter 7's design becomes ~50 lines of code, not a bespoke harness.

## Install and first sanity check

```bash
pip install "mteb>=1.14" sentence-transformers
```

The one-liner smoke test that proves your install works — this runs the SciFact retrieval task (small, ~5k docs, downloads in seconds) against a small BGE model:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
tasks = mteb.get_tasks(tasks=["SciFact"])
evaluator = mteb.MTEB(tasks=tasks)
results = evaluator.run(model, output_folder="results/bge-small-scifact")

for result in results:
    print(result.task_name, result.scores)
```

Two things this establishes:

- MTEB wraps `SentenceTransformer` (and other model classes; see below) with its own encode-and-score loop. You do not have to write the encode-batch code.
- Results land in `output_folder` as JSON files — one per task, with the model name, revision, and metric values. This is what chapter 7's sweep will aggregate.

Expected wall clock: seconds to a couple of minutes on CPU for SciFact; longer for the bigger BEIR tasks (MS MARCO, NQ, HotpotQA, DBPedia) — those need a GPU or you will be waiting hours.

## The three things MTEB wants from you

To run *any* eval — built-in or custom — MTEB needs three inputs:

1. **A model wrapper.** A callable that turns a list of strings into a NumPy or torch array of embeddings. `SentenceTransformer` is the reference implementation and most open embedders load through it directly; hosted APIs (OpenAI, Cohere, Voyage) need a small adapter. The MTEB repo ships adapters under [`mteb/models/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models) — check whether your model is already registered before writing one.
2. **A task.** Either a built-in (via `mteb.get_tasks(tasks=[...])`) or a subclass of one of the `AbsTask*` classes (`AbsTaskRetrieval`, `AbsTaskClassification`, `AbsTaskSTS`, ...).
3. **An output folder.** MTEB writes one JSON per (model, task) run. Same folder across runs = comparable results across models.

## Running a built-in BEIR retrieval task

MTEB includes every BEIR dataset as a retrieval task. To run the whole BEIR-14 English subset in one call:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# BEIR tasks as MTEB knows them (names match the BEIR paper's dataset list):
BEIR_ENGLISH = [
    "ArguAna", "ClimateFEVER", "CQADupstackAndroidRetrieval",
    "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO",
    "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS",
    "SciFact", "Touche2020", "TRECCOVID",
]

tasks = mteb.get_tasks(tasks=BEIR_ENGLISH)
mteb.MTEB(tasks=tasks).run(model, output_folder="results/bge-small-beir")
```

That call downloads (via `datasets`), embeds corpora, embeds queries, computes nDCG@10 / Recall@100 / MRR@10, and writes one JSON per task. Do not do this on a laptop for the big ones (MS MARCO, HotpotQA, FEVER, DBPedia) unless you have GPU or patience.

To run just the retrieval *task type* across all currently registered retrieval tasks:

```python
tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
```

Discover what is registered:

```python
for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"]):
    print(t.metadata.name, t.metadata.eval_splits, t.metadata.n_samples)
```

## Wrapping a hosted API as an MTEB-compatible model

Not every model you want to compare is a `SentenceTransformer`. Hosted APIs (OpenAI, Cohere, Voyage, Google) need a thin adapter. MTEB's contract for a model object:

```python
class EmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Return shape (len(sentences), dim)."""
        ...
```

The **kwargs are important — MTEB passes `task_name` (and sometimes `prompt_name`) so the wrapper can route to per-task instructions for models that expect them (E5, GTE-Qwen, Snowflake Arctic Embed, Voyage input_type). A minimal OpenAI adapter:

```python
import numpy as np
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int | None = None):
        self.client = OpenAI()
        self.model = model
        self.dim = dim   # None → default dim for the model

    def encode(self, sentences, *, batch_size=64, show_progress_bar=False, **kwargs):
        out = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            kw = {"model": self.model, "input": batch}
            if self.dim is not None:
                kw["dimensions"] = self.dim
            resp = self.client.embeddings.create(**kw)
            out.extend([e.embedding for e in resp.data])
        return np.asarray(out, dtype=np.float32)
```

Same idea for Cohere / Voyage / Google. For Cohere and Voyage the wrapper must additionally pass an `input_type` (`"search_query"` for queries, `"search_document"` for the corpus) — that is what `**kwargs` and MTEB's `task_name` / role hints exist for. Check the MTEB [`mteb/models/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models) directory before you write one from scratch; several providers already have contributed adapters you can copy.

The stronger practice is to register your adapter through MTEB's `ModelMeta` system so it appears in the model registry and the JSON output records the model correctly:

```python
from mteb.model_meta import ModelMeta

model_meta = ModelMeta(
    name="openai/text-embedding-3-small",
    revision="1",
    release_date="2024-01-25",
    languages=["eng"],
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    embed_dim=1536,
    license=None,
    max_tokens=8191,
    reference="https://platform.openai.com/docs/guides/embeddings",
    loader=lambda **kwargs: OpenAIEmbedder(model="text-embedding-3-small"),
)
```

This is what the MTEB repo does for every registered model; browsing that folder is the fastest way to see the pattern in practice.

## Registering a custom retrieval task — the payoff

The `AbsTaskRetrieval` class is what turns "my corpus and my qrels" into "a task any MTEB-compatible model can be evaluated on."

The data contract:

- **Corpus**: `dict[doc_id, {"title": str, "text": str}]` — the searchable documents / chunks.
- **Queries**: `dict[qid, str]` — one query string per qid.
- **Qrels**: `dict[qid, dict[doc_id, int]]` — relevance judgments. Value ≥ 1 means relevant; 0 or missing means non-relevant.

Everything else — batching, scoring, metrics, JSON output — MTEB provides.

The subclass shape (current MTEB API; check the [`AbsTaskRetrieval` source](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/AbsTaskRetrieval.py) for the exact fields your version expects):

```python
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

class InsuranceHandbookRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InsuranceHandbookRetrieval",
        description="30 hand-labeled queries against a 1,200-chunk insurance handbook.",
        reference=None,                       # your internal doc, if any
        dataset={"path": "local://data/insurance_handbook", "revision": "v1"},
        type="Retrieval",
        category="s2p",                        # short-to-passage
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-08-01", "2026-08-01"),
        domains=["Legal"],
        task_subtypes=None,
        license="proprietary",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": {"test": 30},
            "avg_character_length": {"test": {"query": 55, "document": 620}},
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # Populate exactly the three dicts MTEB expects.
        self.corpus  = {"test": self._load_corpus()}   # dict[doc_id, {"title","text"}]
        self.queries = {"test": self._load_queries()}  # dict[qid, str]
        self.relevant_docs = {"test": self._load_qrels()}  # dict[qid, dict[doc_id, int]]
        self.data_loaded = True

    def _load_corpus(self): ...
    def _load_queries(self): ...
    def _load_qrels(self): ...
```

Once registered, this task is a first-class citizen of the harness:

```python
task = InsuranceHandbookRetrieval()
mteb.MTEB(tasks=[task]).run(model, output_folder="results/bge-small-insurance")
```

You get nDCG@10, Recall@10, Recall@100, MRR@10, MAP@10, precision@10 out of the box. To sweep six models across this one task, the outer loop is:

```python
for model_name in [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
    "Alibaba-NLP/gte-large-en-v1.5",
    "Snowflake/snowflake-arctic-embed-l",
    "openai/text-embedding-3-small",
    "cohere/embed-english-v3.0",
]:
    model = load_model(model_name)   # SentenceTransformer for open, adapter for hosted
    mteb.MTEB(tasks=[InsuranceHandbookRetrieval()]).run(
        model,
        output_folder=f"results/{model_name.replace('/', '__')}",
    )
```

Aggregating six per-model JSON files into one comparison table is a `glob` + `pandas` script. Chapter 7 makes it a deliverable.

## Loading your data into the corpus / queries / qrels shape

The three-dict contract is the same as BEIR's on-disk format. If your judgment set is already in BEIR shape (three TSV / JSONL files: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), the `beir` library helper is the cleanest loader:

```python
from beir.datasets.data_loader import GenericDataLoader

corpus, queries, qrels = GenericDataLoader(data_folder="data/insurance_handbook").load(split="test")
# corpus: dict[str, dict], queries: dict[str, str], qrels: dict[str, dict[str, int]]
```

If your judgment set came out of `exercise-03` in mod-102 (TREC-style `qrels.tsv`), a ~30-line converter to BEIR shape gets you into the same format. Keep the doc/chunk-ID convention stable between mod-102 chunker output and mod-103 eval — you will suffer if `chunk_id`s change between the ingest run and the eval run.

## What MTEB writes out — the reporting shape

For each `(model, task)` pair, MTEB emits a JSON like `results/bge-small-en-v1.5/InsuranceHandbookRetrieval.json`:

```json
{
  "dataset_revision": "v1",
  "task_name": "InsuranceHandbookRetrieval",
  "mteb_version": "1.14.7",
  "scores": {
    "test": [{
      "hf_subset": "default",
      "languages": ["eng-Latn"],
      "main_score": 0.612,
      "ndcg_at_1": 0.567, "ndcg_at_5": 0.598, "ndcg_at_10": 0.612,
      "recall_at_10": 0.733, "recall_at_100": 0.867,
      "mrr_at_10": 0.591, "map_at_10": 0.578, "precision_at_10": 0.147
    }]
  }
}
```

Everything you need for the comparison table in chapter 7 is in `scores.test[0]`. Aggregating:

```python
import json, glob, pandas as pd

rows = []
for path in glob.glob("results/*/InsuranceHandbookRetrieval.json"):
    payload = json.loads(open(path).read())
    model = path.split("/")[1]
    score = payload["scores"]["test"][0]
    rows.append({
        "model": model,
        "ndcg@10": score["ndcg_at_10"],
        "recall@10": score["recall_at_10"],
        "recall@100": score["recall_at_100"],
        "mrr@10": score["mrr_at_10"],
    })

df = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
print(df.to_markdown(index=False))
```

That is 20 lines from raw MTEB output to a defensible ranked table.

## Prompts and instructions — the one thing that will bite you

Some models (E5, Instructor, GTE-Qwen, Snowflake Arctic Embed, Nomic v1.5) expect a task-typed prefix on both queries and documents. If you skip the prefix, you can lose 5–20 nDCG points relative to the model card's number and mistakenly conclude the model is bad.

MTEB's model registry encodes the expected prompt per model per task. If you use a model MTEB already knows, it handles this. If you wrap a model manually, you must handle it. Two patterns:

```python
# Pattern 1: model-side prompt injection (wrapper knows about it)
class E5Wrapper:
    def __init__(self, name="intfloat/e5-large-v2"):
        self.st = SentenceTransformer(name)

    def encode(self, sentences, *, batch_size=32, show_progress_bar=False,
               task_name=None, prompt_name=None, **kwargs):
        # E5 wants "query: ..." for queries and "passage: ..." for corpus.
        prefix = "query: " if prompt_name == "query" else "passage: "
        return self.st.encode([prefix + s for s in sentences],
                              batch_size=batch_size,
                              show_progress_bar=show_progress_bar,
                              convert_to_numpy=True)
```

```python
# Pattern 2: rely on MTEB's built-in prompt handling for a registered model
# (Preferred — check mteb.get_model("intfloat/e5-large-v2") first before writing a wrapper.)
```

Always cross-check the numbers you get from your MTEB run against the model card's published MTEB retrieval score on a task the card reports. If your local number is 8+ points below the card's on the same dataset, you are almost certainly running without the right prompt.

## Comparing against BM25 (the mandatory baseline)

Chapter 2 said "always run BM25 as your baseline." MTEB does not ship a BM25 model, but you can register one in a few lines using `rank_bm25` or Pyserini and score it on the same tasks. Alternatively, run BM25 through the `beir` library's built-in BM25 (Elasticsearch- or Pyserini-backed) and pull the numbers into your comparison table by hand. The mod-105 material owns the Lucene / BM25 side of this in detail; for chapter 3 you only need to *include* a BM25 row in your ranked table, so a hand-computed row is fine.

## Reproducibility hygiene

- **Pin the model revision.** MTEB records `revision` in the output JSON. When your `sentence-transformers` cache picks up a new snapshot of a model, the revision changes and your old numbers stop being comparable. Pass `revision=` explicitly.
- **Pin the MTEB version.** Task definitions and metric implementations evolve. `pip freeze | grep mteb` and record it alongside your results.
- **Pin the dataset revision.** For the built-in tasks MTEB tracks a `dataset_revision`; for your custom task, put a version string in `metadata.dataset["revision"]` and bump it whenever you edit `qrels.tsv` or `corpus.jsonl`.
- **Set random seeds** — MTEB is largely deterministic given the same batch order, but some model wrappers (dropout in encoders you accidentally left in `train()` mode) are not. Set `torch.manual_seed(0)` and `model.eval()` explicitly.

## What you should be able to do after this chapter

- Install MTEB and run a built-in BEIR retrieval task against a `SentenceTransformer`-loadable open model in under 30 minutes.
- Write a `~30-line` adapter that wraps a hosted embedding API (OpenAI, Cohere, Voyage, Google) in the shape MTEB's `encode` contract expects, with correct `input_type` / `prompt_name` handling.
- Subclass `AbsTaskRetrieval` to register your own corpus + queries + qrels as a first-class MTEB task and run it.
- Read the JSON output MTEB writes and aggregate results across a sweep of models into a ranked comparison table.
- Diagnose the "10-point score gap" that comes from forgetting an instruction prefix (E5-style / GTE-Qwen-style).
- Name the reproducibility knobs (model revision, MTEB version, dataset revision) that make a benchmark run comparable across days.

## Further reading

- MTEB repo, [`embeddings-benchmark/mteb`](https://github.com/embeddings-benchmark/mteb) — the runner, task registry, and per-model adapters. Browse [`mteb/models/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models) and [`mteb/tasks/Retrieval/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks/Retrieval) as living documentation.
- Muennighoff et al., *MTEB paper* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — sections 3 and 4 for the metric definitions and task-family framing.
- BEIR repo, [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — `GenericDataLoader` and the BEIR dataset format that MTEB's retrieval tasks inherit.
- Reimers & Gurevych, `sentence-transformers`, [*Applications: Semantic Search*](https://www.sbert.net/examples/applications/semantic-search/README.html) and [*Information Retrieval*](https://www.sbert.net/examples/applications/information-retrieval/README.html) — reference framing for the retrieval eval loop MTEB automates.
- ir_measures, [*Docs*](https://ir-measur.es/en/latest/) — an alternative to MTEB's built-in metrics when you want to compute nDCG / MAP / MRR / Recall from run files directly (used in mod-102 exercise-03 and mod-109).
