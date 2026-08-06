# Resources — mod-110 · Retrieval Orchestration and MCP Exposure

Primary sources (specifications, official documentation, and maintainer-authored writing) for the topics in this module. Grouped by chapter. Third-party tutorials are omitted intentionally — the framework and protocol landscape moves fast and secondary sources rot quickly.

## Chapter 01 — Framework or bare retriever?

- LangChain. *Why LangChain?* — the maintainers' own framing of what LangChain adds and what it doesn't. <https://python.langchain.com/docs/concepts/why_langchain/>
- LlamaIndex. *High-level concepts.* — the framework's stated centre of gravity. <https://docs.llamaindex.ai/en/stable/getting_started/concepts/>

## Chapter 02 — Framework fit matrix

- LlamaIndex. *Documentation home.* <https://docs.llamaindex.ai/en/stable/>
- LlamaIndex on GitHub (source, changelogs). <https://github.com/run-llama/llama_index>
- LangChain. *Conceptual guide.* <https://python.langchain.com/docs/concepts/>
- LangChain on GitHub. <https://github.com/langchain-ai/langchain>
- LangGraph. *Concepts.* <https://langchain-ai.github.io/langgraph/concepts/>
- LangGraph on GitHub. <https://github.com/langchain-ai/langgraph>
- Microsoft. *Semantic Kernel overview.* <https://learn.microsoft.com/en-us/semantic-kernel/overview/>
- Microsoft. *Semantic Kernel on GitHub.* <https://github.com/microsoft/semantic-kernel>

## Chapter 03 — LlamaIndex advanced-RAG patterns

- LlamaIndex. *Query engine module guide.* <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>
- LlamaIndex. *`RouterQueryEngine` example.* <https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/>
- LlamaIndex. *`SubQuestionQueryEngine` example.* <https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/>
- LlamaIndex. *Property Graph Index guide.* <https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/>
- LlamaIndex. *Node postprocessors.* <https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/>
- LlamaIndex. *Response synthesisers.* <https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/>

## Chapter 04 — LangChain retrieval chains (LCEL)

- LangChain. *LCEL — LangChain Expression Language.* <https://python.langchain.com/docs/concepts/lcel/>
- LangChain. *Tutorial — build a retrieval-augmented generation app.* <https://python.langchain.com/docs/tutorials/rag/>
- LangChain. *Retriever concept.* <https://python.langchain.com/docs/concepts/retrievers/>
- LangChain. *`EnsembleRetriever` (RRF hybrid).* <https://python.langchain.com/docs/how_to/ensemble_retriever/>
- LangChain. *`MultiQueryRetriever`.* <https://python.langchain.com/docs/how_to/MultiQueryRetriever/>
- LangChain. *Contextual compression.* <https://python.langchain.com/docs/how_to/contextual_compression/>
- LangSmith. *Trace with LangChain.* <https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langchain>

## Chapter 05 — LangGraph retrieve-and-verify

- LangGraph. *Low-level API concepts.* <https://langchain-ai.github.io/langgraph/concepts/low_level/>
- LangGraph. *Streaming concepts.* <https://langchain-ai.github.io/langgraph/concepts/streaming/>
- LangGraph. *Persistence — checkpointers.* <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- LangGraph. *Human-in-the-loop.* <https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/>
- LangGraph. *API reference.* <https://langchain-ai.github.io/langgraph/reference/>

## Chapter 06 — MCP contract primer

- Anthropic. *Introducing the Model Context Protocol.* (2024-11-25). <https://www.anthropic.com/news/model-context-protocol>
- Model Context Protocol. *Specification (root).* <https://modelcontextprotocol.io/specification>
- MCP spec — lifecycle. <https://modelcontextprotocol.io/specification/basic/lifecycle/>
- MCP spec — transports. <https://modelcontextprotocol.io/specification/basic/transports/>
- MCP spec — server features (tools, resources, prompts). <https://modelcontextprotocol.io/specification/server/>
- MCP spec — authorization. <https://modelcontextprotocol.io/specification/basic/authorization/>
- MCP spec — progress notifications. <https://modelcontextprotocol.io/specification/basic/utilities/progress/>
- MCP spec — logging. <https://modelcontextprotocol.io/specification/server/utilities/logging/>

## Chapter 07 — MCP server with the Python SDK

- Model Context Protocol. *Python SDK — repository.* <https://github.com/modelcontextprotocol/python-sdk>
- Model Context Protocol. *Python SDK — README (installation, `FastMCP` quickstart, transports).* <https://github.com/modelcontextprotocol/python-sdk#readme>
- Model Context Protocol. *Example servers (reference implementations across capabilities).* <https://github.com/modelcontextprotocol/servers>
- MCP spec — tool results and content types. <https://modelcontextprotocol.io/specification/server/tools/#tool-results>
- MCP spec — resources. <https://modelcontextprotocol.io/specification/server/resources/>

## Chapter 08 — Direct tool endpoints

- OpenAI. *Function calling guide.* <https://platform.openai.com/docs/guides/function-calling>
- OpenAI. *API reference — chat completions (tools parameter).* <https://platform.openai.com/docs/api-reference/chat/create>
- Anthropic. *Tool use.* <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
- Anthropic. *Messages API reference.* <https://docs.anthropic.com/en/api/messages>
- AWS. *Bedrock — Converse tool use.* <https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html>
- AWS. *Bedrock — Converse API reference.* <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>

## Chapter 09 — Exposure vs. consumption boundary

- `agentic-ai-engineer-learning` (peer role, level 40) — the role that consumes the tools you build in this module. <https://github.com/ai-engineering-curriculum/agentic-ai-engineer-learning>
- This track's `CURRICULUM.md` — the ownership contract stated in prose. [`../../CURRICULUM.md`](../../CURRICULUM.md)

<!-- needs-research: refresh the MCP ecosystem inventory (clients, curated server registry, latest transport / auth revisions) on the next content pass — the ecosystem is young and the authoritative catalogue on modelcontextprotocol.io evolves. -->
