# Chapter 2 — LlamaIndex, LangChain, LangGraph, Semantic Kernel: a fit matrix

## Why compare them at all

The four frameworks in the title overlap heavily on paper — each has a "retriever", each ingests documents, each can send a prompt to an LLM. In practice they were built with different centres of gravity, and the *centre of gravity* is what determines which one fits your problem. The comparison below is not "which framework is best." It is "which framework was designed for the shape of work you have."

## The four in one paragraph each

### LlamaIndex

LlamaIndex started as `gpt_index` (2022) and was rebranded as LlamaIndex in early 2023. Its original and continuing centre of gravity is *indexing structures* over documents — `VectorStoreIndex`, `SummaryIndex`, `TreeIndex`, `KeywordTableIndex`, `KnowledgeGraphIndex`, and now `PropertyGraphIndex`. Its `QueryEngine` abstraction composes retrievers, node post-processors, and response synthesisers into a callable that returns an answer plus its source nodes. If you sketch a retrieval pipeline on a whiteboard, LlamaIndex is the framework where the whiteboard maps most directly to classes. See [LlamaIndex docs — Query engines](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/).

### LangChain

LangChain (Harrison Chase, late 2022) is a Python and TypeScript framework organised around *composable LLM components*. Its current composition primitive is LCEL (LangChain Expression Language) — every component is a `Runnable` and `|` composes them into pipelines. It has the largest adapter catalogue of the four (retrievers, chat models, vector stores, document loaders, tools). It is the framework you most commonly find already-imported in someone else's codebase, and the one whose scope has grown widest — well beyond retrieval. See [LangChain — conceptual guide](https://python.langchain.com/docs/concepts/) and [LCEL](https://python.langchain.com/docs/concepts/lcel/).

### LangGraph

LangGraph (LangChain Inc., 2024) is a *stateful, graph-based orchestration* library that sits alongside LangChain rather than replacing it. You define nodes (functions), edges (transitions), and a shared `State` (a typed dict). Conditional edges let you build loops, retries, and human-in-the-loop pauses. Checkpointers persist state so a long-running graph can be resumed. LangGraph is the answer when your retrieval flow has *cycles* — retrieve → verify → rewrite → retrieve — that LCEL pipelines model awkwardly. See [LangGraph — concepts](https://langchain-ai.github.io/langgraph/concepts/) and [LangGraph — API reference](https://langchain-ai.github.io/langgraph/reference/).

### Semantic Kernel

Semantic Kernel (Microsoft, 2023) is a *plugin-oriented SDK* available in C#, Python, and Java. Its `Kernel` object hosts *plugins* (collections of functions, either "prompt functions" defined by a template or "native functions" defined in code); the SDK provides planners (`FunctionCallingStepwisePlanner`, `HandlebarsPlanner`) that let an LLM pick which plugin function to call. Retrieval is one plugin among many, typically via a `Memory` abstraction backed by a vector store. Semantic Kernel is the framework you reach for when your target stack is .NET / Azure / Copilot or when you need C# parity with your Python service. See [Semantic Kernel — overview](https://learn.microsoft.com/en-us/semantic-kernel/overview/) and [Semantic Kernel — Python samples](https://github.com/microsoft/semantic-kernel/tree/main/python).

## The fit matrix

The table below is a compressed decision aid. The dimensions are the ones that matter for a retrieval-engineering role. Cells are qualitative — treat them as "where each framework's centre of gravity sits", not as absolutes.

| Dimension | LlamaIndex | LangChain (LCEL) | LangGraph | Semantic Kernel |
|---|---|---|---|---|
| **Primary metaphor** | Index + query engine | Pipeline of `Runnable`s | State machine over shared `State` | Kernel hosting typed plugins |
| **Sweet spot** | Complex retrieval graphs over heterogeneous corpora | Composable one-shot retrieval / generation pipelines | Cyclic flows (retrieve-verify, self-correct, agent scaffolds) | .NET / Azure / Copilot integration; polyglot plugins |
| **Retrieval primitives out of the box** | `VectorStoreIndex`, `RouterQueryEngine`, `SubQuestionQueryEngine`, `PropertyGraphIndex`, `NodePostprocessor` | `Retriever` interface + a large adapter set, RAG chains, contextual compression | Inherits LangChain's retrieval primitives; graph structure is the differentiator | `Memory` + `TextMemoryPlugin`; vector-DB connectors |
| **Graph vs. pipeline** | Composable engines; cycles awkward | Linear composition via `|`; branching via `RunnableBranch`, loops difficult | Cycles and conditionals are first-class | Pipelines via planners; explicit cycles less common |
| **Multi-language** | Python; TS is a separate community port | Python + TypeScript (first-party) | Python + TypeScript (first-party) | C#, Python, Java (first-party) |
| **License** | MIT | MIT | MIT | MIT |
| **Ecosystem centre of gravity** | RAG-shaped applications | Broad LLM applications | Agent-like workflows | Microsoft stack, enterprise .NET |
| **Observability story** | Native callbacks + adapters (Arize, Langfuse, W&B) | LangSmith is first-party; broad tracer support | LangSmith + built-in checkpoint stream | Application Insights / OpenTelemetry native |
| **Migration risk** | Rebroke API at 0.10; core is now stable-ish | Repeatedly refactored (`chains` → LCEL; package split); expect churn | Younger; API surface is smaller and less churny so far | Two major API iterations pre-1.0; steadier post-1.0 |
| **Best "when to drop it"** | Non-standard scoring/fusion where the `QueryEngine` shape fights you | Latency-critical hot paths; library code | Simple linear retrieval that doesn't loop | Non-Microsoft ecosystems where the kernel/plugin overhead is unpaid |

## How to read the matrix in practice

The matrix is a starting point, not a decision. Three heuristics that keep it honest:

- **Pick by the shape of your work, not by which framework is trending.** Retrieval-heavy application with routing and sub-question decomposition? LlamaIndex maps most cleanly. Retrieval + verify loop? LangGraph. Retriever exposed to a polyglot .NET client? Semantic Kernel. Straight retrieval-then-generate pipeline? LangChain LCEL, or bare code — either is fine.
- **You may use more than one.** A common pattern in the wild: LlamaIndex owns the retrieval graph (its advanced-RAG primitives are the strongest of the four), LangGraph wraps a retrieve-verify loop around it, and neither shows up at the network boundary — the whole thing lives behind an MCP server. Chapters 03, 05, and 07 walk this configuration end-to-end.
- **Framework choice is not "the retriever."** The retriever you tuned in mod-102–mod-107 remains yours. What differs is the *harness* the framework wraps around it. If you cannot cleanly plug your tuned retriever into a candidate framework, that is a signal about the framework, not about your retriever.

## When each is clearly wrong

Complementary to the "sweet spot" column, it is worth naming the mis-fits.

- **LlamaIndex is wrong** when the workload is not primarily retrieval-shaped — arbitrary tool use, long-running agents, workflows with dozens of non-retrieval steps. LlamaIndex ships `Workflows` for this, but it is not the framework's strongest surface.
- **LangChain is wrong** when latency budgets are tight, or when your code is a library published to peers. The `Runnable` stack shows up in tracebacks and the dependency footprint is heavy.
- **LangGraph is wrong** when your flow is genuinely acyclic. You are paying for a state machine you never advance around; LCEL or bare code is simpler.
- **Semantic Kernel is wrong** outside the Microsoft ecosystem for teams with no need for cross-language parity. Its plugin/planner concepts are useful but the ecosystem depth for open-source vector DBs, embedding providers, and retrieval-specific patterns lags LlamaIndex and LangChain today.

## Cost-of-abstraction, revisited

Chapter 01 named the abstraction cost. The matrix concretises where it hits hardest:

- LlamaIndex's cost lives in the *response synthesiser* layer — it is easy to hide two LLM calls behind one `query_engine.query(...)`.
- LangChain LCEL's cost lives in the *dispatch stack* — tracebacks and profiling flame graphs get deeper than the human-authored code suggests.
- LangGraph's cost lives in *state design* — a poorly typed `State` becomes a bag-of-anything that every node reads and mutates, and correctness lives in your head rather than in the graph.
- Semantic Kernel's cost lives in *planner opacity* — the planner is an LLM, so which plugin function ran and why is a prompt-engineering question, not a static-analysis question.

Knowing where the cost lives is what lets you decide to pay it — or route around it.

## Summary

- LlamaIndex: index-and-query-engine metaphor; the strongest RAG-shaped primitives.
- LangChain / LCEL: broadest adapter set; linear composition; largest ecosystem.
- LangGraph: state machine with cycles; the answer for retrieve-verify loops.
- Semantic Kernel: kernel + plugins; the answer for .NET / Azure / polyglot targets.
- Frameworks are per-stage decisions. Compose them, and expose the composed thing behind a tool contract (chapters 06–08) so downstream agents never see your framework choice.

Chapters 03–05 tour the retrieval-relevant surfaces of LlamaIndex, LangChain, and LangGraph in that order. Semantic Kernel appears again in the fit-matrix exercise; a deep tour is out of scope for this module because the retrieval-specific surface is a subset of what LlamaIndex and LangChain already teach.
