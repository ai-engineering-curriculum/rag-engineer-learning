# Exercise 3 — LangGraph retrieve-and-verify loop

**Time:** ~2 hours. Hands-on Python. Requires an LLM API key.

## What you are practising

Modelling a cyclic retrieval flow as an explicit LangGraph state machine — the canonical case that motivates picking LangGraph over LCEL — and reasoning about the resulting cost and correctness trade-offs.

## Problem statement

Build a LangGraph state machine that implements `retrieve → verify → maybe-rewrite → retrieve → answer`, with a bounded retry budget and a clean give-up path. Run it on a small set of questions that are deliberately hard for the first-pass retriever (vocabulary mismatch, multi-hop, questions with no answer in the corpus). Measure how often the rewrite loop actually helps.

## Corpus

Use whatever small corpus you already have (from earlier exercises, `mod-101`, or a fresh one). 20–50 short documents is enough. The point of this exercise is the loop, not the corpus.

## Requirements

1. **Retriever.** Wrap a hybrid retriever (BM25 + dense, or dense-only if BM25 is cumbersome to wire up quickly) into a function `retrieve(query: str, k: int) -> list[Passage]`. This is your `retrieve` node.

2. **Verifier.** An LLM call that reads the query and the retrieved passages and returns one of `"ok"` / `"insufficient"`. Author the prompt carefully — a permissive verifier turns the loop off, a strict verifier makes it never stop.

3. **Rewriter.** An LLM call that takes the *original query* and the *failing passages* and returns a rewritten query aiming at different vocabulary or scope.

4. **Answerer.** An LLM call that produces a grounded, citation-carrying answer from the retrieved passages.

5. **The graph.** Nodes: `retrieve`, `verify`, `rewrite`, `answer`, `give_up`. Conditional edge from `verify` that goes to `rewrite` if `verdict == "insufficient"` and `attempts < 2`, to `answer` if `verdict == "ok"`, or to `give_up` if the retry budget is exhausted. Edge from `rewrite` back to `retrieve`. Edges from `answer` and `give_up` to `END`.

6. **State schema.** Explicit `TypedDict` or Pydantic `BaseModel`. Carry: `query`, `original_query`, `passages`, `verdict`, `attempts`, `answer`, and a `history` of previous queries (helpful for debugging and to prevent the rewriter from cycling).

7. **Checkpointing.** Attach a `MemorySaver` checkpointer and demonstrate — in a small script — running the graph, retrieving the state history, and inspecting each step. You do not need to persist across processes.

8. **Evaluation.** Author ~10 questions distributed as:
    - ~4 answerable in one retrieval (baseline).
    - ~4 that require a rewrite (vocabulary mismatch — deliberately phrase the query with different words from the corpus).
    - ~2 unanswerable from the corpus.

    Run each through the graph. Report:
    - **Answer correctness** (spot-checked by hand).
    - **Retry rate** — how often the loop rewrote.
    - **Give-up rate** — how often the graph terminated in `give_up`.
    - **Latency and LLM-call count per question**, in aggregate.

## Starter guidance

- Chapter 05 has the full graph skeleton. Copy the shape; write the four LLM prompts yourself.
- The verifier is the highest-leverage prompt. Give it a rubric: "return `ok` if any passage contains a direct answer to the question. Return `insufficient` otherwise." Test the verifier in isolation before wiring it into the graph.
- Feed the rewriter the *failing passages*, not just the query. Without them you often get near-synonyms of the original.
- Log the per-question retry count. If it is always 0 or always 2, one of the LLM prompts is miscalibrated.
- Use `app.stream(..., stream_mode="updates")` while developing so you see each node's output.

## Acceptance criteria

The exercise is complete when:

1. The graph is defined in a file (`graph.py`) with an explicit `State` schema.
2. The state machine has the required nodes and edges (verify by drawing the graph with `app.get_graph().draw_mermaid()` and committing the Mermaid source).
3. A runnable script (`run.py`) loads your 10 questions, runs each through the graph, and writes `results.md` with the four metrics from step 8.
4. `results.md` reports at least one question where the rewrite loop *changed* the outcome (better *or* worse — a negative result counts). Include the original query, the rewritten query, and the passages returned for each.
5. Give-up is reachable — verify by including at least one truly unanswerable question and confirming the graph terminates in the `give_up` node.
6. Attempts are bounded — verify by inspecting the state history that no run exceeded 2 rewrites.

## Stretch goals

- **Add a "hop" counter.** Distinguish "same query rewritten" from "multi-hop follow-up query". Split into two loops and compare.
- **Persist to SQLite.** Swap `MemorySaver` for `SqliteSaver`. Demonstrate resuming a paused graph across processes.
- **Adaptive `k`.** On the second retry, double `k`. Report whether that beats the rewrite loop for your vocabulary-mismatch questions.
- **Compare against a no-loop baseline.** Run the same 10 questions through a plain `retrieve → answer` chain (no verify, no rewrite). How much answer quality did the loop actually buy per extra LLM call?
