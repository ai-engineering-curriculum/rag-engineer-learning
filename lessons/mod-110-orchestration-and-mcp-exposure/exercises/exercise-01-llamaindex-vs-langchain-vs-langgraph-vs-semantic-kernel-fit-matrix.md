# Exercise 1 — LlamaIndex vs. LangChain vs. LangGraph vs. Semantic Kernel: fit matrix

**Time:** ~2 hours. Mostly reading and short prose. Some code sketching, no full implementations.

## What you are practising

Reasoning about *framework fit* against a set of realistic retrieval workloads, and defending your choices in writing. This is the write-up you would circulate before a team commits to a framework for a new retrieval service.

Do this exercise **last** in the module — you get more accurate answers after building with each framework in exercises 02–05.

## Problem statement

Your team is starting a new retrieval-backed application and has to pick one primary orchestration framework (with the option of composing others). Read chapters 01 and 02, then produce a written **fit matrix** and a **recommendation** based on the four scenarios below.

## Scenarios

**Scenario A — internal knowledge assistant.** A chat assistant over the company's Confluence, Google Drive, and a support-ticket export. Retrieval is hybrid (BM25 + dense) with a reranker. The user asks *one* question at a time; the assistant retrieves, cites, and answers. Latency budget p95 ~1.5 s. Team of two engineers; you are one of them.

**Scenario B — SQL-and-docs hybrid assistant.** Same audience as A, but the answerable questions span "look it up in a document" *and* "run a SQL query against our finance warehouse." Some questions need both — "what did we spend on AWS last quarter and where is the policy that governs it?" — and the answer has to combine the two.

**Scenario C — retrieve-and-verify pipeline.** A high-stakes internal QA app for legal. Every answer must cite; every cited passage must be verified to actually contain the claim; if verification fails, the retriever must retry with a rewritten query up to twice before giving up. Latency budget p95 ~4 s. Correctness beats speed.

**Scenario D — .NET-hosted internal Copilot.** The primary application is an ASP.NET service integrated with Microsoft 365. Retrieval is one of several capabilities behind the same service; the retriever must be exposed as a native plugin the .NET code calls directly, without an out-of-process hop.

## Requirements

Deliver a single markdown document with three sections.

### 1. Fit matrix (table)

Reproduce and complete a table with the four scenarios on one axis and the four frameworks (LlamaIndex, LangChain / LCEL, LangGraph, Semantic Kernel) on the other. Each cell contains a **fit rating** — one of `first choice`, `credible`, `possible`, `wrong tool` — and a one-line reason. Example row skeleton:

| Scenario | LlamaIndex | LangChain (LCEL) | LangGraph | Semantic Kernel |
|---|---|---|---|---|
| A — internal KB | ??? | ??? | ??? | ??? |
| ... | | | | |

### 2. Recommendation per scenario (prose)

For each of the four scenarios, in 3–6 sentences:

- State which framework you would choose as the primary and (if any) which you would compose alongside it.
- Name the *one* abstraction cost you are accepting by making that choice.
- Name the *one* alternative you would fall back to if the primary choice broke down, and what would trigger the fallback.

### 3. Decision heuristic (prose)

In under 200 words, state your own three-question test for picking a retrieval framework — the questions you would ask a peer choosing a framework for a new service. This is deliberately your own opinion; it should be defensible against chapter 01's framing but does not have to match it.

## Starter guidance

- The chapter 02 matrix is a template. Do not copy it — use it as scaffolding for evidence-based cells informed by exercises 02–05.
- For each scenario, sketch (do not build) a pseudocode outline of the retrieval pipeline you would ship. That outline is what makes the fit rating concrete.
- Cite framework docs by URL where you claim a specific capability exists. Do not invent APIs; if you are unsure a capability exists, mark it `?` and note what you would verify.
- If two frameworks are genuinely tied for a scenario, say so. `first choice` is not required to be unique.

## Acceptance criteria

Your submission is complete when:

1. All 16 cells of the fit matrix are filled in with a rating and a one-line reason.
2. Each of the four scenarios has a recommendation covering the primary choice, one accepted abstraction cost, and a documented fallback plan.
3. The decision heuristic is under 200 words and lists three concrete questions.
4. Every framework capability you claim is linked to an official doc URL (framework docs, blog posts from the maintainers, or the framework's GitHub README).
5. No invented APIs. If you are unsure, mark it and say what you would check.

## Stretch goals

- **Add a fifth "bare code" column.** For which scenarios is *no framework* actually the right answer? Defend it.
- **Add a cost dimension.** For each recommended framework choice, estimate the migration cost from a plain Python retriever (in engineer-days), and the upgrade cost per major version.
- **Add an on-call dimension.** Which of your choices would you be least willing to be paged for at 3 AM? Why? Does that change your recommendation?
