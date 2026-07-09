# mod-001 · RAG Foundations

First module in the RAG & Retrieval Engineer track. Establishes the shared vocabulary the rest of the curriculum builds on: what retrieval-augmented generation is, why it exists, and the moving parts of a minimal pipeline.

## Learning objectives

By the end of this module a learner can:

- Define retrieval-augmented generation (RAG) in the terms the original paper introduced it.
- Distinguish parametric knowledge (weights) from non-parametric knowledge (an external corpus).
- Name the four stages of a minimal RAG pipeline — **ingest → chunk → embed → retrieve → generate** — and explain what each stage owns.
- Identify the failure modes that motivate later modules (hallucination on out-of-distribution queries, stale facts, unattributable answers).
- Read a RAG system diagram and point to where each component would live.

## Prerequisites

See the repo-level [`PREREQUISITES.md`](../../PREREQUISITES.md). No prior retrieval or vector-store experience is assumed for this module specifically.

## Contents

| Path | What it is |
|---|---|
| [`lectures/01-what-is-rag.md`](lectures/01-what-is-rag.md) | Concept walkthrough — origin, motivation, anatomy of a pipeline. |
| [`exercises/01-classify-retrieval-failures.md`](exercises/01-classify-retrieval-failures.md) | Short paper-and-pen exercise: given failing Q&A pairs, name the stage at fault. |
| [`labs/01-first-rag-pipeline.md`](labs/01-first-rag-pipeline.md) | Hands-on: build a five-file naïve RAG pipeline over a small corpus. |
| [`quizzes/01-foundations-check.md`](quizzes/01-foundations-check.md) | Ten-item self-check on the vocabulary. |

## How to work through it

1. Read the lecture end-to-end once. Do not stop to run code.
2. Do the exercise on paper. It takes ~15 minutes and surfaces which stage of the pipeline you still find fuzzy.
3. Do the lab. Expect ~60–90 minutes if you are new to embedding APIs.
4. Take the quiz. Anything you miss, re-read the corresponding lecture section — do not memorise the answer.

## Primary sources

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
