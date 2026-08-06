# Chapter 1 — Framework or bare retriever?

## The choice this module opens with

By the end of mod-107 you own a tuned retriever: an ingest pipeline, an ANN index, a hybrid fusion pass, a reranker, a context assembler. In Python it is a function — call it `retrieve(query: str, k: int, filters: dict) -> list[Passage]`. It is *already* a coherent, testable component.

Yet almost every production RAG system you will read about is built on top of a *framework* — LlamaIndex, LangChain, LangGraph, Semantic Kernel — and almost every framework will happily hand you a "retriever" object of its own. Before you learn how to use those, you should be able to defend, in one sentence, why you would want one at all. The rest of this module is easier if you know what it is trying to earn.

## What a retrieval framework actually adds

A retrieval framework provides four things a hand-written retriever does not automatically get:

1. **A common interface for retrievers, LLMs, and post-processors.** Every framework defines something with a `.retrieve()` or `__call__` signature so a reranker, a router, or a query-rewriter can be composed with a retriever without either component knowing the other's implementation.
2. **Reference implementations of well-known retrieval patterns.** RRF fusion, HyDE, sub-question decomposition, multi-hop retrieval, contextual compression, hybrid vector-plus-lexical — the patterns from mod-106 and mod-107 exist as named classes you can wire up in an afternoon.
3. **Adapters to the vendor zoo.** Chroma, Pinecone, Weaviate, Qdrant, pgvector, Elasticsearch, OpenSearch, Vespa; OpenAI, Anthropic, Cohere, Voyage, Google, Bedrock, Azure OpenAI; PDF loaders, HTML loaders, S3, Confluence, Notion. A framework is often, in practice, a library of adapters with a modest amount of glue.
4. **An orchestration primitive.** LlamaIndex has `QueryEngine`; LangChain has LCEL (`Runnable`); LangGraph has `StateGraph`; Semantic Kernel has `Kernel` + `Plugin`. Each is a way to say "these components run in this order, with this state, under these error semantics."

## What a retrieval framework charges you for

Frameworks are not free. The bill shows up in five places:

1. **Abstraction leakage.** The framework's retriever interface flattens options your tuned retriever cared about — hybrid weights, per-field boosts, ACL context, per-tenant index selection — into a generic `search_kwargs` dict. What you fed as first-class parameters yesterday becomes a stringly-typed side channel.
2. **Latency you can't see.** A `QueryEngine` call can hide two LLM round-trips (query rewrite + answer generation) and three retriever calls (fusion) behind a single `.query()`. Trace-level observability is on you.
3. **Upgrade churn.** LangChain in particular has moved through several major refactors (`chains` → LCEL, `langchain` → `langchain-core` + `langchain-community` + partner packages). LlamaIndex renamed and repackaged extensively at the 0.10 boundary. Semantic Kernel's C# and Python SDKs are on different versions of the same idea. Any framework you adopt at scale is a *dependency you will be migrating* on a roughly-annual cadence.
4. **Debuggability tax.** A hand-written pipeline is a stack of function calls. A framework pipeline is a stack of `Runnable.invoke` frames, dispatch tables, and callback managers. When it breaks in production, the framework is the first place your on-call will point at, whether it deserves it or not.
5. **Vendor pull.** Each framework has "blessed" providers — retrievers, LLMs, vector DBs — where the paved path is smoother. That pull is usually gentle, but it is real, and it compounds if you also adopt a hosted observability tool (LangSmith, Langfuse) tied to the framework.

## The three-question test

For any given retrieval component, the choice between framework and bare code comes down to three questions.

**1. Does the framework's abstraction match my actual retrieval shape?**

If your retrieval is "call one vector store, rerank, return", any framework can do it and none of them earn their keep — a 40-line function does the same thing with less indirection. If your retrieval is "route by intent to one of five sub-corpora, run sub-question decomposition on the multi-hop bucket, verify with a second retrieval pass, and stream the citations as they arrive," you want an orchestration primitive so you are not reinventing the state machine.

**2. Am I willing to inherit the framework's upgrade cadence?**

If your team ships every two weeks and can absorb an occasional afternoon of migration, LangChain / LlamaIndex are fine. If your retriever runs inside an SLA-bound platform service that touches production data and you cannot upgrade on the maintainers' schedule, the "wrap the SDKs yourself" answer looks better than it did five minutes ago.

**3. Who consumes this retriever?**

If the retriever is called by *one* application you also own, framework choice is a build-time decision. If it is called by *many* applications you don't own — the case this module is really about — the interesting interface is *not* the Python object. It is the tool schema (MCP, OpenAI, Anthropic, Bedrock Converse) you expose at the network boundary. Chapters 06–08 make that concrete.

## When to drop the framework and write the retriever bare

There is a recognisable set of cases where the framework is a net loss:

- **Latency-critical hot path.** A p95 target below ~200 ms for the retrieval leg leaves no headroom for framework overhead, dispatch chains, or callback managers. The bare SDK call to your vector DB plus a hand-rolled fusion is faster and easier to profile.
- **Non-standard scoring or fusion.** If you have a learned-to-rank stage, per-tenant boosts, or a fusion function that does not fit RRF or weighted-sum, the framework's built-in classes will not model it. You will either subclass around them or write the loop yourself; the loop wins.
- **Regulatory / audit surface.** In domains where every retrieval decision has to be logged with a stable schema for compliance (finance, health, defence), a small, boring, well-typed retriever is easier to defend than a chain of `Runnable`s.
- **Library, not application.** If the retriever ships to callers as a Python package they import (i.e., the SDK your peers write against), you should not import a framework in your public API. Consumers should be free to compose you into their framework of choice, not required to adopt yours.
- **You already own it.** If your team has a bespoke retrieval library that has been production-hardened for two years and is well-tested, the framework migration cost is usually not repaid by the incremental abstractions the framework offers.

Note the framing: "when to drop the framework" is not "when to reject frameworks." Most retrieval systems will use *at least one* framework for *at least one* stage — LlamaIndex's `PropertyGraphIndex` alone can save weeks in mod-112. The point is to pick the smallest set of framework surfaces that pay for themselves.

## A worked contrast

Suppose you have a tuned hybrid retriever `hybrid_search(query, k)`. Compare the bare version against the LangChain LCEL version:

```python
# bare
def answer(question: str) -> str:
    passages = hybrid_search(question, k=4)
    prompt = build_prompt(question, passages)
    return llm.complete(prompt)
```

```python
# LCEL — LangChain's Runnable composition
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke(question)
```

The LCEL version buys you three things: (1) `chain.batch(...)` and `chain.astream(...)` for free, (2) automatic tracing when a LangSmith tracer is attached, (3) an interface that another LangChain component can slot into. The bare version is one function and can be read without the framework's docs open.

Neither is wrong. If you never plan to compose the chain with other framework components and you already have your own tracing, "bare" is the correct answer. If you plan to build ten retrieval variants and evaluate them in a matrix, LCEL is the correct answer because you get `batch` and `astream` without writing them.

## The module's mental model

For the rest of this module, hold these three ideas in your head at once:

1. **Framework choice is a per-stage decision, not a per-project decision.** You can use LlamaIndex for the retrieval graph, drop into bare code for the reranker, and expose the whole thing behind an MCP server that itself uses no framework at all.
2. **The framework is a build-time convenience; the tool contract is a run-time interface.** Downstream agents see the MCP or vendor tool shape. They do not see your `QueryEngine`.
3. **The abstraction cost is highest for teams whose retrieval is unusual.** If your retriever *is* the differentiator — LTR, per-tenant scoring, graph over property graphs, permission-aware fusion — the framework's generic interface will fight you. If your retriever is a textbook hybrid, the framework will save you a week.

## Summary

- A retrieval framework earns its keep with a composition interface, reference patterns, adapter breadth, and an orchestration primitive.
- It bills you in abstraction leakage, hidden latency, upgrade churn, debuggability, and vendor pull.
- Use the three-question test — does the abstraction match, can you absorb the cadence, who consumes the retriever — to decide per-stage.
- Drop the framework when the hot path is latency-critical, your scoring is non-standard, your audit surface is high, or your code is a library rather than an application.

Chapter 02 puts each of the four frameworks in this module on the same axes and calls the differences.
