# Exercise 05 — Bootstrap the local toolchain and pass five smoke tests

**Estimated effort:** ~2 hours. Mostly setup + a short Python script per smoke test.

## Objective

You will stand up the local retrieval-engineering toolchain from chapter 6 and pass five end-to-end smoke tests: imports, embedding round-trip, pgvector round-trip, BM25 round-trip, and an `ir_measures` round-trip. When this exercise is green, every subsequent module (starting with mod-102) assumes this stack is working; a flaky bootstrap now becomes a lost afternoon in mod-104.

## Prerequisites

- Chapter 6 (*The Local Retrieval-Engineering Toolchain*).
- A machine with:
  - Python 3.11+ available.
  - Docker (for the Postgres + Elasticsearch/OpenSearch containers).
  - ~15 GB of free disk (Hugging Face model cache, container images).
  - A stable internet connection for the first-time model download.
- (Optional) A Hugging Face account if you plan to test gated models later. Not required for this exercise.

## Part A — Project scaffold (20 min)

1. Create a fresh directory `mod-101-toolchain/`.
2. Initialize a Python project with `uv init` (or `poetry init`, or the tool you know).
3. Add the dependencies from chapter 6:
   - `llama-index`
   - `sentence-transformers`
   - `torch` (installed transitively; pin explicitly if you need a specific CUDA / CPU variant — [PyTorch install matrix](https://pytorch.org/get-started/locally/))
   - `psycopg[binary]`
   - `pgvector`
   - `elasticsearch` *or* `opensearch-py` (pick one and stick with it)
   - `ir-measures`
   - `pytest`
4. Run `uv sync` (or equivalent) so the lockfile is written. Commit the lockfile.

**Deliverable A:** `pyproject.toml` + lockfile in the project directory.

## Part B — Local services with Docker Compose (20 min)

Write a `docker-compose.yml` using chapter 6's shape. Run `docker compose up -d`. Verify:

- `psql -h localhost -U rag -d rag -c "SELECT 1;"` returns `1`.
- `curl -s http://localhost:9200 | head -20` returns the engine's welcome JSON (cluster name, version).

If you cannot use Docker, install Postgres + `pgvector` and Elasticsearch/OpenSearch natively per the [pgvector install docs](https://github.com/pgvector/pgvector#installation) and the [Elasticsearch install docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) or [OpenSearch install docs](https://opensearch.org/docs/latest/install-and-configure/). Note the versions you install.

**Deliverable B:** `docker-compose.yml` (or an install-notes file) plus the output of both connectivity checks.

## Part C — Write and pass the five smoke tests (60 min)

Create `tests/test_smoke.py`. Each test is a real `pytest` test that fails loudly if the stack is not correctly wired.

### Smoke test 1 — imports

```python
def test_imports():
    import llama_index                # noqa: F401
    import sentence_transformers      # noqa: F401
    import torch                      # noqa: F401
    import psycopg                    # noqa: F401
    import pgvector                   # noqa: F401
    # pick the one you installed:
    import elasticsearch              # noqa: F401
    import ir_measures                # noqa: F401
```

Every top-level module must import without error.

### Smoke test 2 — embedding round-trip

Load `BAAI/bge-small-en-v1.5` (a small, permissively licensed, 384-dim model that fits comfortably on CPU) with `sentence_transformers.SentenceTransformer`. Embed the two-string batch `["hello world", "retrieval is fun"]`. Assert:

- The output tensor's shape is `(2, 384)`.
- The two embeddings are non-equal.
- Cosine similarity of the query `"greeting"` against the two candidates ranks `"hello world"` above `"retrieval is fun"`.

### Smoke test 3 — pgvector round-trip

Connect to the local Postgres. In a fresh schema:

- `CREATE EXTENSION IF NOT EXISTS vector;`
- Create a table `chunks(id serial primary key, text text, embedding vector(384))`.
- Insert three rows using embeddings from test 2 (embed three short strings of your choice).
- Create an HNSW index on `embedding` with `vector_cosine_ops`.
- Query for the nearest neighbor of a fourth embedded query and assert the returned row matches the semantically-closest inserted string.

Refer to [pgvector-python examples](https://github.com/pgvector/pgvector-python#examples) for the exact SQL and binding shape.

### Smoke test 4 — BM25 round-trip (Elasticsearch / OpenSearch)

Using the engine you chose:

- Create an index `smoke-bm25` with a single `text` field.
- Index three short documents.
- Run a `match` query for one word that appears in exactly one document.
- Assert the returned top-1 is the correct document and that scores are non-zero.

If you picked OpenSearch, swap the client import (`opensearchpy.OpenSearch` instead of `elasticsearch.Elasticsearch`); the API shape is nearly identical for basic operations.

### Smoke test 5 — `ir_measures` round-trip

Author, in a temp directory or in-memory, a minimal TREC-format qrels and run for two queries and three documents. Compute `nDCG@10` and `Recall@10` with `ir_measures`. Assert that the returned metric objects have plausible values (`0.0 ≤ ndcg ≤ 1.0`) and that a run that ranks the ground-truth document first scores strictly higher than a run that ranks it second.

The API shape lives in the [`ir_measures` quickstart](https://ir-measur.es/en/latest/getting-started.html). Read it and translate to code.

Run `pytest -q`. All five tests must pass.

**Deliverable C:** `tests/test_smoke.py` and a green `pytest` output.

## Part D — Reproducibility note (20 min)

Write a short `TOOLCHAIN.md` in the project root that records:

- Python version.
- OS and hardware (CPU / GPU).
- Container image tags (if using Docker).
- Model IDs and revisions (`BAAI/bge-small-en-v1.5` at revision `<sha>`).
- Any deviations from chapter 6 (e.g. you installed pgvector natively rather than via Docker).

The point is not paperwork. Every downstream module will occasionally have you swap one component (a new embedding model, a new pgvector version); a written baseline is the diff-target.

**Deliverable D:** `TOOLCHAIN.md`.

## Acceptance criteria

- `pytest -q` in the project directory returns `5 passed`.
- `docker compose ps` (or the native-install equivalent) shows Postgres and the Lucene-family engine healthy and reachable.
- `TOOLCHAIN.md` records enough that a colleague on a similar machine could reproduce your setup without asking a follow-up question.

## Stretch goals

- Add a sixth smoke test: an end-to-end **naïve RAG** run. Load ten short documents, embed them, insert into pgvector, retrieve top-3 for one query, and print the results. This is the shape that mod-102 will build on. Do not chunk yet — one document = one chunk = one row. (Optional: pair this with the paired lab `lab-01`, if your track offers it.)
- Swap the Elasticsearch container for OpenSearch (or vice versa) and re-run test 4. Note any client-API differences you had to work around; this is exactly the friction mod-105 will explore in more depth.
- Add a GitHub Actions workflow that runs `pytest tests/test_smoke.py` on push, using service containers for Postgres and Elasticsearch/OpenSearch. This is the CI shape mod-111 will build on. Confirm it goes green.
