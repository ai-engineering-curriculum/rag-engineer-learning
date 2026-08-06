# mod-110 · Retrieval Orchestration and MCP Exposure

Tenth module in the RAG & Retrieval Engineer track. Takes the tuned retriever the previous modules produced and puts it inside — or in front of — the two things that will actually call it in production: an *orchestration framework* (LlamaIndex / LangChain / LangGraph / Semantic Kernel) and a *tool contract* (Model Context Protocol, OpenAI tool-call, Anthropic tool-use, Bedrock Converse). This is the module where a "retriever" stops being a Python function and becomes a component other systems consume.

**Estimated effort:** 12 hours.

## Learning objectives

By the end of this module a learner can:

- Compare LlamaIndex, LangChain, LangGraph, and Semantic Kernel for retrieval-orchestration use — where each shines, the abstraction cost of each, and when to drop the framework and write the retriever bare.
- Use LlamaIndex advanced-RAG patterns — `QueryEngine` composition, `SubQuestionQueryEngine`, `RouterQueryEngine`, `PropertyGraphIndex` — as reference implementations of the patterns from `mod-106`.
- Use LangChain / LangGraph retrieval chains, and the LangGraph state-machine model where a retrieval + verify loop is required.
- Wrap a tuned retriever as a Model Context Protocol (MCP) server — tool schema, streaming, resource-list vs. tool-call shape — using the official MCP Python SDK, and understand what the emerging MCP ecosystem lets an agent consume.
- Wrap the same retriever as an OpenAI tool-call / Anthropic tool-use / Bedrock Converse tool endpoint for direct-consumer patterns.
- Cite the boundary to `agentic-ai-engineer` (level 40), which owns agent orchestration *consuming* the MCP retriever; this module owns only the *exposure* side.

## Prerequisites

- All prior modules in this track. In particular:
  - `mod-106-hybrid-retrieval-fusion-query-rewriting` — the patterns LlamaIndex's advanced-RAG engines encode are the patterns from mod-106.
  - `mod-107-reranking-and-context-assembly` — the retriever you expose in this module should be one you built and tuned in mod-107, not a naïve top-*k*.
  - `mod-109-retrieval-quality-evaluation` — you need a judgment set to justify any orchestration-layer choice.
- Comfort with Python `async` and JSON-RPC. FastAPI or an equivalent async-HTTP framework helps.
- API access for one hosted LLM provider (OpenAI, Anthropic, or Bedrock) to complete the tool-endpoint exercises.

## Structure

| Path | What it is |
|---|---|
| [`01-framework-or-bare-retriever.md`](01-framework-or-bare-retriever.md) | Why an orchestration framework, what it costs, when to drop it. |
| [`02-framework-fit-matrix.md`](02-framework-fit-matrix.md) | LlamaIndex vs. LangChain vs. LangGraph vs. Semantic Kernel — a decision matrix. |
| [`03-llamaindex-advanced-rag-patterns.md`](03-llamaindex-advanced-rag-patterns.md) | `QueryEngine` composition, `RouterQueryEngine`, `SubQuestionQueryEngine`, `PropertyGraphIndex`. |
| [`04-langchain-retrieval-chains.md`](04-langchain-retrieval-chains.md) | LCEL and the `Runnable` interface for retrieval chains. |
| [`05-langgraph-retrieve-and-verify.md`](05-langgraph-retrieve-and-verify.md) | `StateGraph`, conditional edges, and the retrieve-verify loop pattern. |
| [`06-mcp-contract-primer.md`](06-mcp-contract-primer.md) | Model Context Protocol — the JSON-RPC contract an MCP-consuming agent sees. |
| [`07-mcp-server-python-sdk.md`](07-mcp-server-python-sdk.md) | Wrap a retriever as an MCP server with the official Python SDK. |
| [`08-direct-tool-endpoints.md`](08-direct-tool-endpoints.md) | OpenAI, Anthropic, and Bedrock Converse tool shapes for a retriever. |
| [`09-exposure-consumption-boundary.md`](09-exposure-consumption-boundary.md) | What this module owns and where `agentic-ai-engineer` (level 40) takes over. |
| [`exercises/`](exercises/) | Five hands-on prompts — one per major objective. |
| [`labs/`](labs/) | Long-form hands-on labs (scaffolded separately). |
| [`quizzes/`](quizzes/) | Knowledge check (scaffolded separately). |
| [`resources.md`](resources.md) | Primary-source references for the topics covered. |

## How to work through it

1. Read chapters 01–02 back-to-back. They set the frame; the rest of the module is a tour through the frameworks and protocols named there.
2. Read chapter 03 and complete `exercises/exercise-02-…`. LlamaIndex is the most opinionated of the four and the easiest place to see the patterns from mod-106 as running code.
3. Read chapters 04–05 and complete `exercises/exercise-03-…`. LangGraph is where an explicit state machine earns its keep — a retrieve-verify loop is the smallest example that motivates it.
4. Read chapters 06–07 and complete `exercises/exercise-04-…`. This is the module's centrepiece: your retriever behind the MCP contract.
5. Read chapter 08 and complete `exercises/exercise-05-…`. The same retriever, three vendor-native tool shapes. The point is to see how thin the shape actually is.
6. Read chapter 09 and complete `exercises/exercise-01-…` (the fit-matrix write-up) last — you now have the direct experience to fill it in.

## Ownership boundary

This module owns *exposure* of a retriever: how it is orchestrated inside a retrieval pipeline, and how it is presented as a tool to an outside caller. It does **not** own the agent that calls the tool. Anything about *tool-selection policy*, *multi-step planning*, *cross-tool memory*, or *guardrails on the calling agent* belongs to [`agentic-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/agentic-ai-engineer-learning) at level 40. Chapter 09 draws the line explicitly.

## Primary sources

- Anthropic. *Introducing the Model Context Protocol.* (2024-11-25). <https://www.anthropic.com/news/model-context-protocol>
- Model Context Protocol. *Specification.* <https://modelcontextprotocol.io/specification>
- Model Context Protocol. *Python SDK.* <https://github.com/modelcontextprotocol/python-sdk>
- LlamaIndex. *Advanced RAG modules and query engines.* <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>
- LangChain. *LCEL — LangChain Expression Language.* <https://python.langchain.com/docs/concepts/lcel/>
- LangGraph. *Concepts and API reference.* <https://langchain-ai.github.io/langgraph/>
- Microsoft. *Semantic Kernel documentation.* <https://learn.microsoft.com/en-us/semantic-kernel/overview/>

Further reading — chapter-by-chapter and grouped by topic — is in [`resources.md`](resources.md).
