# Chapter 6 — The Local Retrieval-Engineering Toolchain

## Why this chapter exists

Every downstream module in this track assumes a working local stack: a Python environment with the retrieval libraries pinned, a Postgres with `pgvector`, a Lucene-family engine (Elasticsearch or OpenSearch), and `ir_measures` for evaluation. Standing this up once, deliberately, saves hours of "why does the notebook not run" debugging in later modules.

This chapter names the tools, explains why each is here, and points at the official install docs. The paired exercise (`exercise-05`) walks through the actual smoke test. Version pins move; always confirm current supported versions against the primary docs linked below rather than hard-coding what a chapter says.

## The tools and what each is for

### Python 3.11+ and a real project manager

You need a modern Python (3.11 or newer at time of writing; check the ecosystem — sentence-transformers, LlamaIndex, and `pgvector-python` support move together). Use a project manager that pins reproducibly: `uv` ([docs.astral.sh/uv](https://docs.astral.sh/uv/)) or `poetry` ([python-poetry.org](https://python-poetry.org/docs/)). `pyproject.toml` is the source of truth for what your project depends on. Don't rely on `pip install <thing>` at a global site-packages — indexes and eval runs downstream will drift.

### LlamaIndex

[LlamaIndex](https://docs.llamaindex.ai/) is one of the two major retrieval-orchestration frameworks (LangChain / LangGraph is the other; both are covered in mod-110). We use LlamaIndex from mod-101 forward as a *reference implementation* — chunker abstractions (`NodeParser`), retriever composition (`QueryEngine`, `RouterQueryEngine`, `SubQuestionQueryEngine`), and a large collection of connectors. Installing it now surfaces its transitive dependency surface (which is real) so nothing is a surprise in mod-102 or mod-110.

### sentence-transformers

[sentence-transformers](https://www.sbert.net/) wraps Hugging Face models with the bi-encoder / cross-encoder interfaces you will use for candidate embedding models (BGE, E5, Nomic, GTE, Snowflake Arctic Embed, Alibaba GTE-Qwen, Jina) and cross-encoder rerankers (`bge-reranker`, `ms-marco-MiniLM-*`). It is *the* interface — even when you swap to a hosted API for production, the local benchmark harness in mod-103 runs through this library. Model downloads from the [Hugging Face Hub](https://huggingface.co/models) are cached under `~/.cache/huggingface/`; set `HF_HOME` if you want the cache elsewhere.

### pgvector (Postgres extension)

[`pgvector`](https://github.com/pgvector/pgvector) is a Postgres extension that adds a `vector` column type and HNSW / IVFFlat ANN indexes. It is the vector store we use in mod-101 through mod-104 because (a) most production systems can run Postgres and (b) the ANN parameter names match what you will see in FAISS, Weaviate, and Qdrant, so the concepts transfer. Two ways to run it locally:

- **Docker** — the [`pgvector/pgvector:pg16`](https://hub.docker.com/r/pgvector/pgvector) image (or newer PG version) is the fastest path.
- **Native Postgres + extension** — install Postgres, then `git clone` and `make install` the extension per [pgvector README](https://github.com/pgvector/pgvector#installation).

The Python client is [`pgvector-python`](https://github.com/pgvector/pgvector-python), which integrates cleanly with `psycopg` and with SQLAlchemy.

### Elasticsearch or OpenSearch

You need a Lucene-family engine for the lexical (BM25) baseline every hybrid stack fuses against (mod-105, mod-106) and — later — for the dense-vector fields and LTR plugin (mod-105).

- **Elasticsearch** — Elastic Stack, [`docker.elastic.co/elasticsearch/elasticsearch`](https://www.docker.elastic.co/), Elastic License 2.0 / SSPL. Follows the [official Elasticsearch install docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
- **OpenSearch** — Apache-2.0 fork of Elasticsearch 7.10, `opensearchproject/opensearch` on Docker Hub, [OpenSearch docs](https://opensearch.org/docs/latest/).

Either is fine for the coursework. Pick one and stay with it through mod-105 — the analyzer, mapping, and LTR-plugin details are engine-specific.

### ir_measures

[`ir_measures`](https://ir-measur.es/en/latest/) is the retrieval evaluation library we use in mod-109 (and referenced from mod-101 chapter 2 and the exercise here). It reads TREC-format run files and qrels, computes `recall@k`, `precision@k`, `MRR`, `MAP`, `nDCG@k`, and much more, and integrates with `trec_eval` and `pytrec_eval` for the same numbers. Installing and running a one-liner smoke computation now saves rework later.

## The minimal `pyproject.toml`

The exercise walks through the actual bootstrap, but a reasonable starting `pyproject.toml` looks like:

```toml
[project]
name = "rag-engineer-lab"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
  "llama-index",              # core + defaults; sub-packages installed as needed
  "sentence-transformers",    # local embeddings + cross-encoder rerankers
  "torch",                    # Hugging Face / sentence-transformers backend
  "psycopg[binary]",          # Postgres client
  "pgvector",                 # pgvector-python bindings
  "elasticsearch",            # OR "opensearch-py" — pick one and stick with it
  "ir-measures",              # retrieval-only evaluation
  "pytest",                   # for the smoke tests
]
```

Do not paste this straight in without checking the current published versions — every one of these libraries has had breaking changes in the last twelve months. Use `uv add <pkg>` or `poetry add <pkg>` so the lockfile records the resolved version.

## A minimal `docker-compose.yml` for the local stack

Two services — Postgres+pgvector and a Lucene-family engine — is enough to start:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports: ["5432:5432"]

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.14.3
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports: ["9200:9200"]
```

For OpenSearch instead of Elasticsearch, use `opensearchproject/opensearch:2.15.0` with `plugins.security.disabled=true` and `discovery.type=single-node`. Consult the [Elasticsearch install docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) or [OpenSearch install docs](https://opensearch.org/docs/latest/install-and-configure/) for current supported image tags and the license notes for the version you pick.

`xpack.security.enabled=false` and `plugins.security.disabled=true` are **local-only** conveniences. Do not carry either into any deployment.

## The smoke tests you want to have run

If, at the end of the paired exercise, you can execute all five of these end-to-end from a single Python session, your toolchain is ready for mod-102:

1. **Python import check** — every top-level module (`llama_index`, `sentence_transformers`, `psycopg`, `pgvector`, `elasticsearch` or `opensearchpy`, `ir_measures`) imports without error.
2. **Embedding round-trip** — `SentenceTransformer("BAAI/bge-small-en-v1.5")` loads, embeds `["hello", "world"]`, and returns a `(2, 384)`-shaped tensor.
3. **pgvector round-trip** — `CREATE EXTENSION vector;` succeeds, a `CREATE TABLE ... (embedding vector(384))` accepts inserts, and a `<->` ANN query returns rows.
4. **BM25 round-trip** — Elasticsearch / OpenSearch is reachable on `localhost:9200`, an index is created with a text field, a document is indexed, and a `match` query returns it.
5. **`ir_measures` round-trip** — a hand-authored two-query, three-document qrels + run file computes `nDCG@10` without error.

The exercise gives you the exact shape of each smoke test; do not skip any of them. Every one has caught a real environment problem for one cohort or another.

## Governance and cost notes for hosted APIs

Several downstream modules will call hosted embedding APIs (OpenAI, Cohere, Voyage, Google) and hosted rerankers (Cohere Rerank, Voyage Rerank, Jina Reranker). You do *not* need any of these keys to complete mod-101, but be aware:

- **Never commit an API key.** Use environment variables (`OPENAI_API_KEY`, `COHERE_API_KEY`, etc.) or a `.env` file that is git-ignored.
- **Every hosted API call costs money.** For a benchmark that embeds tens of thousands of chunks, cost adds up fast. Prefer a small local model (`bge-small-en-v1.5`, `all-MiniLM-L6-v2`) for exploration and only reach for hosted APIs for the actual selection benchmark in mod-103.
- **Rate limits vary widely.** A benchmark loop with no backoff will trip provider rate limits within minutes. Use the vendor SDK's built-in retry with exponential backoff.

## What you should be able to do after this chapter

- Name each tool in the local stack and one sentence on why it is there.
- Explain what the pgvector extension adds to Postgres.
- Explain what `ir_measures` computes and why it is a retrieval-*only* evaluation library.
- List the five smoke tests the exercise walks you through.

Once the exercise is green, you are ready for mod-102 (Document Ingestion and Chunking), which is the first module that assumes a working local stack throughout.

## Further reading

- Python: [Python docs](https://docs.python.org/3/), [`uv` docs](https://docs.astral.sh/uv/), [Poetry docs](https://python-poetry.org/docs/).
- LlamaIndex: [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- sentence-transformers: [sbert.net](https://www.sbert.net/) and the [Hugging Face Hub](https://huggingface.co/).
- pgvector: [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) and [pgvector-python](https://github.com/pgvector/pgvector-python).
- Elasticsearch: [Install docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
- OpenSearch: [Install docs](https://opensearch.org/docs/latest/install-and-configure/).
- `ir_measures`: [ir-measur.es](https://ir-measur.es/en/latest/).
