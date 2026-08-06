# Chapter 4 — LangChain retrieval chains (LCEL)

## Why this chapter is short

LangChain is broad — chat models, tools, agents, output parsers, memory. This chapter covers only the *retrieval-relevant* surface: the `Retriever` interface, LCEL composition, and the RAG-shaped chain that shows up in every LangChain tutorial. Everything else in LangChain is worth knowing eventually; only these bits are worth knowing to complete this module.

Snippets follow the LangChain 0.2+ API (packages split into `langchain-core`, `langchain-community`, and partner packages such as `langchain-openai`). Consult the [LangChain conceptual guide](https://python.langchain.com/docs/concepts/) for current signatures — the framework has iterated repeatedly and old blog posts often reference dead imports.

## The `Runnable` interface and LCEL

LangChain's composition primitive is the `Runnable`. Every LangChain component — retrievers, chat models, prompt templates, output parsers — implements `Runnable`, which gives you:

- `.invoke(input)` — single call.
- `.batch(inputs)` — parallel calls.
- `.stream(input)` — iterator of chunks.
- `.ainvoke`, `.abatch`, `.astream` — async variants.

LCEL — LangChain Expression Language — is the operator overload of the `|` pipe: `a | b` composes two `Runnable`s into a new one where `b` runs on the output of `a`. See [LCEL — concepts](https://python.langchain.com/docs/concepts/lcel/).

A minimal retrieval chain:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the passages below. If the passages do not contain
the answer, say so. Cite each claim with the source URL in square brackets.

Passages:
{context}

Question: {question}
""".strip())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("When is our office closed for winter break?")
```

Four things worth naming:

1. **The dict on the left** is itself a `Runnable`. It runs each value in parallel with the same input, producing a `{"context": ..., "question": ...}` that the prompt then fills.
2. **`RunnablePassthrough()`** is the identity — it passes the input straight through. It exists because the dict pattern above needs each key to be a `Runnable`.
3. **The retriever is any `Runnable`** with `.invoke(str) -> list[Document]`. LangChain's built-in retrievers (`VectorStoreRetriever`, `BM25Retriever`, `EnsembleRetriever`) already are; a custom one is a one-method subclass of `BaseRetriever`.
4. **The chain is a `Runnable` too.** `chain.batch(list_of_questions)` and `chain.astream(question)` work with no further code. Streaming in particular is one of LCEL's better-earned wins.

See [LangChain — build a retrieval-augmented generation app](https://python.langchain.com/docs/tutorials/rag/).

## The retriever interface

`BaseRetriever` defines two methods:

```python
class BaseRetriever(RunnableSerializable[str, list[Document]]):
    def _get_relevant_documents(self, query, *, run_manager): ...
    async def _aget_relevant_documents(self, query, *, run_manager): ...
```

Subclass and implement the sync variant (async can be default-generated from sync):

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class MyHybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        results = hybrid_search(query, k=20)
        return [
            Document(page_content=r.text, metadata=r.metadata | {"score": r.score})
            for r in results
        ]
```

Now `MyHybridRetriever()` is a `Runnable`, works in an LCEL chain, and inherits `.invoke`, `.batch`, `.astream`, plus the LangSmith trace hooks. This is the analogue of the LlamaIndex `BaseRetriever` wrap in chapter 03 — the moral is the same: **keep your tuned retriever, wear the framework's interface.**

## The retriever adapters worth knowing

LangChain's retriever catalogue is large; three are worth calling out because they encode retrieval patterns you already met.

- **`EnsembleRetriever`** — takes a list of retrievers and a list of weights, runs them, and fuses with reciprocal-rank fusion. This is the RRF hybrid from mod-106 with a two-line import.
- **`MultiQueryRetriever`** — asks an LLM to generate *n* rewordings of the query, retrieves for each, and unions the results. This is the multi-query fanout pattern from mod-106.
- **`ContextualCompressionRetriever`** — wraps another retriever and applies a `DocumentCompressor` (e.g., `LLMChainExtractor`, `EmbeddingsFilter`, `CohereRerank`) to shrink or filter the returned documents. This is the reranker/compressor slot from mod-107.

Each of these is a `Runnable`. You can chain them: `MultiQueryRetriever(retriever=EnsembleRetriever(retrievers=[...]))` is a hybrid-and-multi-query stack you get for the price of two constructors.

## A retrieve-generate chain end to end

Compose the pieces:

```python
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

bm25   = BM25Retriever.from_documents(docs, k=20)
vector = vector_store.as_retriever(search_kwargs={"k": 20})
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])

reranked = ContextualCompressionRetriever(
    base_retriever=hybrid,
    base_compressor=CohereRerank(model="rerank-english-v3.0", top_n=6),
)

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {d.page_content}  (source: {d.metadata.get('source_url', '?')})"
        for i, d in enumerate(docs)
    )

chain = (
    {"context": reranked | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Read the chain top-to-bottom: hybrid → rerank → format → prompt → LLM → text. Every stage is a `Runnable`; every stage is independently swappable. This is LCEL earning its keep.

## Streaming and the callback surface

Two features that come free with LCEL:

- **Streaming.** `chain.astream(question)` yields tokens as the LLM produces them. Because the retriever and rerank run first (sync work) and the LLM is the last node in the graph, the retrieval latency is a fixed prefix; only the generation is streamed. This is usually what you want in a chat UI.
- **Callbacks / tracing.** Attach a `LangChainTracer` or a `LangfuseCallbackHandler` at the top of the chain and every retriever call, every prompt, and every LLM call is logged with parent-child relationships. If you enabled LangSmith, no code change is needed — the tracer picks up `LANGCHAIN_TRACING_V2` from the environment. See [LangSmith — LangChain tracing](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langchain).

## Where LCEL fights you

LCEL is designed around a *linear pipeline with occasional branching*. Three cases where it stops being the right tool:

- **Loops.** Retrieve → verify → maybe-rewrite → retrieve is not naturally expressible in LCEL. `RunnableBranch` handles a branch, not a cycle. This is the case chapter 05 hands to LangGraph.
- **Long-running / resumable work.** LCEL runs to completion; no built-in checkpointing. Chapter 05 covers checkpointers.
- **Complex state.** If the chain needs to accumulate structured intermediate state across many steps (an agent's scratchpad, a set of hypothesised queries, per-node scores), passing it as dicts through LCEL becomes painful. The `RunnableParallel` / `RunnablePassthrough.assign` patterns work but read poorly. Reach for LangGraph.

## Debugging notes

- **`chain.get_graph().print_ascii()`** prints the runtime graph of a chain — the first thing to run when the chain is not doing what you think it is.
- **`chain.invoke(x, config={"callbacks": [ConsoleCallbackHandler()]})`** dumps every step to stdout without needing LangSmith.
- **`.with_retry(...)` and `.with_fallbacks(...)`** are `Runnable` methods that turn a failing chain into a resilient one without adding try/except around every call. Use for LLM providers with occasional 5xxs; do not use to paper over correctness bugs.
- **`get_openai_callback()` context manager** (in `langchain_community.callbacks`) totals up token usage across the chain. Handy for a first cost estimate.

## Summary

- `Runnable` + LCEL is LangChain's composition primitive. Every retriever, prompt, LLM, and parser is a `Runnable`; `|` composes them.
- A retrieval chain is a dict of `{"context": retriever, "question": passthrough}` piped into a prompt piped into an LLM.
- Subclass `BaseRetriever` to wear the interface without giving up your tuned retriever.
- `EnsembleRetriever`, `MultiQueryRetriever`, and `ContextualCompressionRetriever` are the reference implementations of the fusion, multi-query, and rerank patterns from mod-106 and mod-107.
- LCEL is a pipeline, not a state machine. When you need cycles or resumable state, chapter 05's LangGraph is the right tool.

Chapter 05 does exactly that with the retrieve-and-verify loop.
