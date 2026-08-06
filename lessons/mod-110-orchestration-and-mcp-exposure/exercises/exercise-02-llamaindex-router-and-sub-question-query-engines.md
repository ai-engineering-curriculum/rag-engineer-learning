# Exercise 2 — LlamaIndex `RouterQueryEngine` and `SubQuestionQueryEngine`

**Time:** ~3 hours. Hands-on Python. Requires an OpenAI or equivalent LLM API key.

## What you are practising

Building composed LlamaIndex `QueryEngine`s that route across multiple sub-corpora and decompose multi-part questions — reference implementations of the routing and sub-question patterns from `mod-106`.

## Problem statement

Build a two-tier retrieval graph over **three heterogeneous sub-corpora**. A top-level `RouterQueryEngine` picks the right sub-engine per query. One of the sub-engines is itself a `SubQuestionQueryEngine` that decomposes multi-part questions across two of the sub-corpora. Measure and report routing accuracy and answer quality on a small hand-authored query set.

## Corpora

Assemble three small sub-corpora yourself. Use public content so you can share your work; keep each corpus small (~10–20 documents, plain text preferred).

- **`policy/`** — formal policy-style text. Suggested source: 5–10 pages from a public HR handbook, ISO 27001 controls, or GDPR articles.
- **`faq/`** — short, question-shaped content. Suggested source: any public FAQ page (a hosting provider's, a language's official FAQ).
- **`changelog/`** — dated release-note-style entries. Suggested source: an open-source project's changelog (Django, PostgreSQL, React).

Each sub-corpus should be distinct enough that the router has a real signal to route on.

## Requirements

Build the following in Python, using LlamaIndex 0.10+.

1. **A per-corpus `QueryEngine`.** For each of the three corpora, build a `VectorStoreIndex` and expose it as a `QueryEngine` with a reasonable `similarity_top_k`, a similarity cutoff post-processor, and a compact response mode. Any embedding model is fine; document your choice.

2. **A `SubQuestionQueryEngine` over two of the three.** Compose a sub-question engine that decomposes questions across `policy` and `changelog` (the "have you enforced the policy in a release?" shape). This engine is itself one of the leaves of the router.

3. **A `RouterQueryEngine` at the top.** Three tools:
    - `faq` — the FAQ sub-engine.
    - `policy_and_changelog` — the sub-question engine from step 2.
    - Optionally: a fourth catch-all fallback tool that returns "I don't know."

    Use an `LLMSingleSelector`. Write the tool descriptions carefully — the router only sees the descriptions.

4. **A small evaluation set.** Author 12–15 questions distributed across the three intended routes:
    - ~5 questions the router should send to `faq`.
    - ~5 questions the router should send to `policy_and_changelog`.
    - ~3 questions that are genuinely unanswerable from any corpus.

    For each question, record the *intended* route and the *ground-truth* answer or "unanswerable".

5. **Report.** Run every question through the router. Produce a short `results.md` reporting:
    - **Routing accuracy** — fraction of questions routed to the intended tool.
    - **Per-tool answer quality** — for questions correctly routed, whether the answer is faithful to the retrieved passages (spot-check by hand; a strict rubric is not required).
    - **Failure modes** — the top two routing mistakes and the top two answer-quality mistakes, with a one-sentence hypothesis for each.

6. **Code layout.** A clean, single-package layout:
    ```
    exercise-02/
    ├── corpora/                # your three sub-corpora
    ├── engines.py              # builds the three sub-engines + router
    ├── eval.py                 # loads the question set, runs, scores
    ├── questions.jsonl         # your 12–15 questions with intended route
    ├── results.md              # your written report
    └── README.md               # how to run it
    ```

## Starter guidance

- Refer to chapter 03 for the composition snippets. Copy the shape, not the corpora — invent your own three corpora so the descriptions you write are yours.
- Route descriptions are prompt engineering. Iterate on them if routing accuracy is low; log which description is currently active.
- Use `verbose=True` on both engines during development — the selector's per-query pick and the sub-question decomposer's sub-queries are printed to stdout.
- Do not hand-tune to your eval set. Author the eval set once; if you change descriptions or engines, re-run, do not re-label.

## Acceptance criteria

The exercise is complete when:

1. All four Python artefacts (`engines.py`, `eval.py`, `questions.jsonl`, `README.md`) run end-to-end from a clean checkout with a documented pip install and API-key setup.
2. `results.md` reports numeric routing accuracy across the 12–15 questions.
3. `results.md` names at least two concrete failure modes with hypotheses.
4. The three tool descriptions are visibly the primary lever you tuned — an inline comment or a line in `results.md` documents at least one description you changed and what changed the routing accuracy.
5. The sub-question engine actually decomposes at least one of your test questions into two or more sub-questions (verify with `verbose=True`).
6. Your `README.md` states which framework version and which embedding model you used.

## Stretch goals

- **Swap the selector.** Replace `LLMSingleSelector` with `PydanticSingleSelector` or a hand-rolled embedding-nearest-cluster selector. Report the routing-accuracy delta and the latency delta.
- **Add a fourth corpus.** Add `code/` (source code and READMEs from an open-source project) with its own sub-engine and its own route. Report whether routing accuracy holds at four routes.
- **Add rewriting to the FAQ engine.** Wrap the FAQ retriever in an LLM query-rewrite step and compare the retrieval-hit rate before and after on your FAQ-intended questions.
