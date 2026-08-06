# mod-101 — Retrieval-Engineering Foundations and Role Position

**Track:** RAG & Retrieval Engineer (role level 30, AI Engineering family)
**Estimated effort:** 12 hours
**Position in track:** first module. Every downstream module drills into one stage of the pipeline this module maps.

## What this module does

This module gives you the picture in your head that the rest of the track drills into. By the end of it you can (a) diagram the seven-stage production RAG pipeline and name the engineering decision each stage forces, (b) refresh the classical-IR mental model — inverted indexes, tf-idf, BM25, the Probability Ranking Principle, precision / recall / F1 — that every retrieval decision hangs off, (c) place the retrieval-engineering role on the level ladder and write a one-page ownership contract with peer and higher-level tracks, (d) name the difference between "retrieval as a feature" and "retrieval as an engineered system", (e) read Lewis et al. (2020) and Gao et al. (2024) as the canonical framing papers, and (f) stand up the local toolchain (Python, LlamaIndex, sentence-transformers, pgvector, Elasticsearch/OpenSearch, `ir_measures`) that every downstream module assumes is working.

## Learning objectives

- Diagram the end-to-end RAG pipeline (ingest → chunk → embed → retrieve → rerank → assemble → generate) and name the engineering decision made at each stage.
- Refresh classical IR fundamentals — inverted indexes, tf-idf, BM25, the Probability Ranking Principle, precision / recall / F1 — drawing on Manning-Raghavan-Schütze *Introduction to Information Retrieval*.
- Position the role on the level ladder — what this track owns vs. what `llm-application-developer` (level 20) already covers and what `applied-ai-engineer` (peer, level 30), `ai-eval-engineer` (peer, level 30), `agentic-ai-engineer` (level 40), and `ai-infra-senior-engineer` (level 40) inherit or extend.
- Distinguish 'retrieval as a feature' (built by an application developer) from 'retrieval as an engineered system' (built by this role).
- Read Lewis et al. 'Retrieval-Augmented Generation for Knowledge-Intensive NLP' (Meta, 2020) and Gao et al. 'Retrieval-Augmented Generation for Large Language Models: A Survey' (2024) as the canonical framing papers.
- Set up the local retrieval-engineering toolchain (Python, LlamaIndex, sentence-transformers, pgvector, Elasticsearch/OpenSearch, `ir_measures`) used by every downstream module.

## Prerequisites

See the track root [`PREREQUISITES.md`](../../PREREQUISITES.md). In brief: comfortable production Python (FastAPI, asyncio, packaging), Postgres and SQL, Docker + Docker Compose, PyTorch + Hugging Face at the "load a model and run inference" level, tf-idf / BM25 intuition at the level `ml-engineer` (level 20) leaves you with, and the retrieval basics `llm-application-developer` (level 20, mod-005) covers.

## Chapters

Read the chapters in order. Each is short and focused; run the paired exercise before starting the next chapter.

| # | Chapter | Paired exercise |
|---|---|---|
| 01 | [The End-to-End RAG Pipeline and Decisions](01-end-to-end-rag-pipeline-and-decisions.md) | [exercise-01 — Map the RAG pipeline and decision points](exercises/exercise-01-map-the-rag-pipeline-and-decision-points.md) |
| 02 | [Classical IR Fundamentals](02-classical-ir-fundamentals.md) | [exercise-02 — Classical IR fundamentals refresher](exercises/exercise-02-classical-ir-fundamentals-refresher-bm25-tfidf.md) |
| 03 | [Retrieval as a Feature vs. an Engineered System](03-retrieval-as-feature-vs-engineered-system.md) | (paired with chapter 4's exercise) |
| 04 | [Role Position on the Ladder](04-role-position-on-the-ladder.md) | [exercise-03 — Role scope and ownership contract](exercises/exercise-03-role-scope-and-ownership-contract-with-peers.md) |
| 05 | [Canonical Framing Papers: Lewis 2020, Gao 2024](05-canonical-framing-papers.md) | [exercise-04 — Read Lewis 2020 and Gao 2024, annotate](exercises/exercise-04-read-lewis-2020-and-gao-2024-and-annotate.md) |
| 06 | [The Local Retrieval-Engineering Toolchain](06-local-toolchain-setup.md) | [exercise-05 — Local toolchain bootstrap and smoke test](exercises/exercise-05-local-toolchain-bootstrap-and-smoke-test.md) |

## How to work through it

1. Read chapters 1–2 back-to-back; they build the map and the vocabulary. Do exercises 01 and 02 before moving on.
2. Read chapters 3–4 together; they are two views of the same shape (what the role owns vs. where it sits). Do exercise 03 after both.
3. Read chapter 5 and do exercise 04 — this is where you actually read the two papers rather than reading *about* them.
4. Read chapter 6 and do exercise 05 to get the local stack green. The green pytest is your gate into mod-102.

## Structure

- `01-…md` … `06-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs (see `labs/README.md`).
- `quizzes/`: knowledge checks.
- `resources.md`: external references.

## Where to go next

Once exercise 05's smoke tests pass, move to [`mod-102-document-ingestion-and-chunking`](../mod-102-document-ingestion-and-chunking/README.md) — the first stage-depth drill-down.
