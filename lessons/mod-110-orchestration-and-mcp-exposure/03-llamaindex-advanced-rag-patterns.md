# Chapter 3 — LlamaIndex advanced-RAG patterns

## Why LlamaIndex first

The advanced-retrieval patterns you met in mod-106 (routing, sub-question decomposition, multi-hop, graph traversal) are all *pipelines of retrievers and post-processors*. LlamaIndex is the framework whose object model most directly names these building blocks — you can point at a Python class and say "this is the router" or "this is the sub-question engine." Reading LlamaIndex is therefore a fast way to read *the patterns themselves* as running code.

This chapter walks four LlamaIndex primitives you will use again in the exercise: `QueryEngine` composition, `RouterQueryEngine`, `SubQuestionQueryEngine`, and `PropertyGraphIndex`. Snippets follow the LlamaIndex 0.10+ API. Consult the [official docs](https://docs.llamaindex.ai/en/stable/) for exact imports and current signatures — the framework moves.

## The core layer cake

Every LlamaIndex retrieval pipeline is a stack of four layers:

```
    ┌──────────────────────────────┐
    │   ResponseSynthesizer        │  ← turns nodes + query into an answer
    ├──────────────────────────────┤
    │   NodePostprocessors         │  ← rerank, filter, deduplicate
    ├──────────────────────────────┤
    │   Retriever(s)               │  ← vector, keyword, hybrid, custom
    ├──────────────────────────────┤
    │   Index(es) over Nodes       │  ← VectorStoreIndex, SummaryIndex, …
    └──────────────────────────────┘
```

- **Nodes** are LlamaIndex's chunk type; every node carries text plus metadata.
- **Indexes** wrap a store of nodes (e.g., a vector DB) and expose a `.as_retriever(...)` factory.
- **Retrievers** implement `retrieve(query) -> list[NodeWithScore]`.
- **NodePostprocessors** implement `postprocess_nodes(nodes, query) -> list[NodeWithScore]`. Rerankers, MMR filters, recency boosters live here.
- **ResponseSynthesizers** implement `synthesize(query, nodes) -> Response`. The default (`compact`) stuffs nodes into a prompt; `refine` iterates; `tree_summarize` builds a hierarchical summary.

A `QueryEngine` is the *composition* — one retriever, a chain of post-processors, one response synthesiser, wrapped with a `.query(str)` entry point.

See [LlamaIndex — Query engine module guide](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/).

## QueryEngine composition

Start with the two-line minimum:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("corpus/").load_data()
index = VectorStoreIndex.from_documents(docs)
qe = index.as_query_engine(similarity_top_k=8)
response = qe.query("What is our refund policy for enterprise contracts?")
print(response)
for node in response.source_nodes:
    print(node.score, node.node.metadata["file_name"])
```

Two lessons live in this snippet:

1. `as_query_engine` is a *builder* — `similarity_top_k`, `response_mode`, `node_postprocessors`, and `llm` are keyword arguments. The class it returns is `RetrieverQueryEngine`, and you can construct it explicitly when you want more control:

    ```python
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    retriever = index.as_retriever(similarity_top_k=20)
    postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.55),
        CohereRerank(top_n=6, model="rerank-english-v3.0"),
    ]
    qe = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_mode="compact",
    )
    ```

2. Every response carries its source nodes with scores and metadata. That is the retrieval-quality debugging surface — inspect `response.source_nodes` before you argue about the answer.

### Custom retrievers and post-processors

Your tuned retriever from mod-107 does not need to be discarded to work with LlamaIndex. Wrap it:

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

class MyHybridRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        results = hybrid_search(query_bundle.query_str, k=20)
        return [
            NodeWithScore(node=TextNode(text=r.text, metadata=r.metadata), score=r.score)
            for r in results
        ]
```

Now `MyHybridRetriever` can slot into any `QueryEngine`, `RouterQueryEngine`, or `SubQuestionQueryEngine` in the rest of the chapter. This is the earn-your-keep case for the framework — you keep your retriever *and* get the composition surface for free.

## RouterQueryEngine

`RouterQueryEngine` picks one (or more) child engines per query. It is the "semantic routing" pattern from mod-106 as a class you can instantiate.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

policy_qe = policy_index.as_query_engine(similarity_top_k=6)
faq_qe    = faq_index.as_query_engine(similarity_top_k=6)
code_qe   = code_index.as_query_engine(similarity_top_k=4)

tools = [
    QueryEngineTool.from_defaults(policy_qe, name="policy",
        description="HR, security, and compliance policies. Formal, corporate documents."),
    QueryEngineTool.from_defaults(faq_qe, name="faq",
        description="Customer-facing FAQ entries. Short, question-shaped, informal."),
    QueryEngineTool.from_defaults(code_qe, name="code",
        description="Source code and code comments from internal repositories."),
]

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),  # pick one; use LLMMultiSelector for many
    query_engine_tools=tools,
    verbose=True,
)
```

Three things to notice:

- **The description is the routing signal.** The selector sees only the tool `name` and `description`; the corpus content is invisible until the selector picks a tool. If routing accuracy is low, the fix almost always lives in the descriptions, not in the selector.
- **Selectors are pluggable.** `LLMSingleSelector` (one) and `LLMMultiSelector` (many) are LLM-backed. `PydanticSingleSelector` uses structured outputs. You can subclass `BaseSelector` for an embedding-nearest-cluster or a classifier-backed selector — the semantic-router pattern from mod-106 fits here.
- **Verbose mode is the trace.** Set `verbose=True` while you develop; capture the chosen tool per query so you can build a routing-accuracy metric later.

See [LlamaIndex — Router query engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/).

## SubQuestionQueryEngine

`SubQuestionQueryEngine` is the *sub-question decomposition* pattern (mod-106) as a class. Given a multi-part question and a set of sub-tools, it asks an LLM to decompose the query into sub-questions, routes each to the most fitting sub-tool, runs them, and synthesises the results.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

engines = [
    QueryEngineTool.from_defaults(finance_qe, name="finance_10k",
        description="Financial statements and MD&A from 10-K filings."),
    QueryEngineTool.from_defaults(product_qe, name="product_docs",
        description="Product documentation, release notes, and roadmap."),
    QueryEngineTool.from_defaults(support_qe, name="support_tickets",
        description="Support tickets and resolutions from the customer help desk."),
]

sq = SubQuestionQueryEngine.from_defaults(query_engine_tools=engines, verbose=True)
resp = sq.query("Which product line drove the most support ticket growth in Q3, "
                "and did revenue for that line grow in the same quarter?")
```

The right shape of query for `SubQuestionQueryEngine` is *decomposable across sources*: the answer requires facts from more than one sub-corpus, and no single retriever call would return all of them. Two failure modes to plan for:

- **Over-decomposition.** For a simple single-source question, the engine may still generate sub-questions, spending LLM calls it does not need. Set a fast-path (e.g., a router that only escalates to the sub-question engine when a classifier says the question is multi-part).
- **Aggregation drift.** The final synthesis step is a second LLM call over the sub-answers. If the sub-answers already had citations, propagate them into the synthesised answer or you lose per-claim provenance.

See [LlamaIndex — Sub-question query engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/).

## PropertyGraphIndex

`PropertyGraphIndex` is LlamaIndex's property-graph retriever — the framework's reference implementation of Graph RAG. It stores nodes and typed relationships in a graph backend (Neo4j, Kùzu, Nebula, or an in-memory `SimplePropertyGraphStore`) and lets you retrieve by graph traversal *and* by vector similarity in the same call.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor, ImplicitPathExtractor
)

graph_index = PropertyGraphIndex.from_documents(
    documents=docs,
    kg_extractors=[SimpleLLMPathExtractor(), ImplicitPathExtractor()],
    show_progress=True,
)

graph_qe = graph_index.as_query_engine(
    include_text=True,          # return the source chunk text alongside triples
    similarity_top_k=6,         # vector-based node retrieval
)
resp = graph_qe.query("Which of our engineering leaders previously worked at "
                      "companies that were acquired by Microsoft?")
```

The engine supports several retriever modes — `LLMSynonymRetriever` (LLM expands the query into candidate entity synonyms), `VectorContextRetriever` (top-*k* by vector, then include their neighbourhood), and `CypherTemplateRetriever` (parametrise a Cypher template with LLM-extracted entities). Multi-hop questions benefit most; single-hop questions rarely justify the extraction cost.

`PropertyGraphIndex` is covered in more depth in `mod-112-enterprise-retrieval-acl-structured-graph-multimodal`. Here, note two things:

- It is one of the strongest advertising points for LlamaIndex — no other framework in this module has an equivalent primitive at the same level of polish.
- The extraction step (turning documents into triples) is an LLM cost you pay at ingest time, not at query time. Budget accordingly.

See [LlamaIndex — Property Graph Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/).

## Composing the four

The four primitives compose. A production LlamaIndex retrieval graph often looks like:

```
   query ─▶ RouterQueryEngine ─▶┬─▶ VectorStoreIndex query engine
                                 ├─▶ SubQuestionQueryEngine
                                 └─▶ PropertyGraphIndex query engine
```

The router picks one; the sub-question engine may itself route into the other two. This is exactly the pattern the mod-110 exercise on router + sub-question engines asks you to build.

## Debugging notes

- **Use callbacks.** `Settings.callback_manager = CallbackManager([LlamaDebugHandler()])` prints every LLM and retriever call. Turn it on the first time a `QueryEngine` misbehaves.
- **Inspect `source_nodes`.** Every `Response` has `.source_nodes`. If they are not what you expected, the failure is upstream of the synthesiser.
- **`response_mode="no_text"`.** Skip the synthesis LLM call and just return retrieved nodes. Fast way to isolate a retrieval failure from a generation failure.
- **Log the router pick.** Wrap the selector or set `verbose=True` and store which tool was chosen per query. Routing accuracy is a first-class metric.

## Summary

- LlamaIndex's four layers — index → retriever → node postprocessors → response synthesiser — factor a retrieval pipeline into pluggable classes.
- `QueryEngine` composition is the base primitive; wrap your tuned retriever as a `BaseRetriever` subclass to keep it and get the composition surface.
- `RouterQueryEngine` is the semantic-routing pattern from mod-106 as a class; the routing signal is the tool description.
- `SubQuestionQueryEngine` is sub-question decomposition; best on decomposable multi-source questions, wasteful on single-source ones.
- `PropertyGraphIndex` is the framework's reference Graph RAG implementation; used properly it lifts recall on multi-hop questions.
- Compose all four — router at the top, sub-question and property-graph engines as children — for realistic advanced-RAG pipelines.

Chapter 04 does the equivalent tour for LangChain LCEL retrieval chains.
