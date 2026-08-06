# Chapter 5 — Graph RAG on a Property Graph

## Why this chapter exists

Some questions cannot be answered by top-k of independent chunks, no matter how well tuned the embedding model or reranker is. "Which of our vendors also supply our top-3 competitors, and through which intermediate distributors?" is a multi-hop question — the answer requires connecting *vendor → distributor → competitor* through relationships that don't appear as text in any single chunk. A vector search over the vendor contracts, the distributor invoices, and the competitor filings will return relevant chunks but will not *compose* them into the traversal the question asks for.

Graph RAG is the retrieval shape that makes those traversals first-class. The corpus is (or becomes) a **property graph** — nodes with typed properties and typed relationships — and the retriever is a graph query (Cypher, Gremlin, or an LLM-generated equivalent), possibly followed by a vector search *inside* the traversed neighborhood.

This chapter covers:

- **The property graph model** and the three graph databases you're most likely to meet — Neo4j, Kùzu, Memgraph.
- **Two ways to build the graph** — deterministic construction from structured sources (safer) vs. LLM-driven extraction (Microsoft GraphRAG's approach; more flexible, more error-prone).
- **LlamaIndex's Property Graph Index** as the retrieval-orchestration reference implementation.
- **Microsoft GraphRAG** as the community-summary variant.
- **When graph structure actually lifts recall** — and the honest answer that most enterprise questions are one-hop and don't need it.

Ontology design, entity resolution at scale, schema.org modeling — orthogonal disciplines. This chapter is the retrieval-engineer's slice: assuming a graph exists (or is being extracted), how do you query it as a retriever?

## The property graph model

A property graph is:

- **Nodes** with a *label* (type) and a set of *properties* (key-value pairs).
- **Relationships** with a *type* and a *direction* and a set of *properties*.
- No fixed schema (mostly) — you can add new labels, new relationships, new properties without a migration.

Example (Cypher notation, Neo4j-style):

```
(vendor:Company {name: "Acme Semiconductors", country: "Taiwan"})
  -[:SUPPLIES {volume: 12000, contract_id: "C-42"}]->
(customer:Company {name: "Our Corp", tier: "internal"})
```

Contrast with:

- **RDF triples** (subject, predicate, object) — more strictly typed, more academic, less common in enterprise RAG. Query language: SPARQL.
- **Relational tables** — everything as rows and foreign keys. You *can* express a graph this way (adjacency table), but recursive joins are painful; recursive CTEs work but scale poorly beyond a few hops.

The three property-graph engines this module covers:

- **[Neo4j](https://neo4j.com/docs/)** — the reference implementation. Cypher is its query language. Broad ecosystem (drivers, viz, integrations). Community edition is free; Enterprise adds clustering.
- **[Kùzu](https://docs.kuzudb.com/)** — embedded (in-process, no server), fast on OLAP-style graph queries, Cypher-compatible, Apache 2.0. A good default for local development and single-node deployments.
- **[Memgraph](https://memgraph.com/docs)** — in-memory graph DB, Cypher-compatible, focused on real-time streaming ingest. BSL license (open source with production usage caveats).

Cypher is the lingua franca — all three accept it (with minor dialect differences). Learn Cypher; you can then evaluate the engines on price / performance / operational shape rather than language.

## Two ways to build the property graph

### Deterministic construction from structured sources

If your corpus already has structure — a CRM (customer → account → opportunity → contact), an ERP (vendor → PO → invoice → line item), an IAM system (user → role → group → permission) — the graph is a mechanical projection from the source tables. This is the *safer* path: every node and every relationship has a source-system origin you can point back to.

```python
# Pseudocode — populate Neo4j from a customers + orders join
def load_customer_order_graph(driver, cursor):
    cursor.execute("SELECT id, name, tier FROM customers")
    with driver.session() as sess:
        for row in cursor.fetchall():
            sess.run("MERGE (c:Customer {id: $id}) SET c.name = $name, c.tier = $tier", **row)

    cursor.execute("SELECT id, customer_id, amount, product_id, created_at FROM orders")
    with driver.session() as sess:
        for row in cursor.fetchall():
            sess.run("""
                MATCH (c:Customer {id: $customer_id})
                MATCH (p:Product {id: $product_id})
                MERGE (o:Order {id: $id})
                SET o.amount = $amount, o.created_at = $created_at
                MERGE (c)-[:PLACED]->(o)
                MERGE (o)-[:CONTAINS]->(p)
            """, **row)
```

Trade-offs:

- **Pro:** every edge is verifiable against the source; provenance is trivial; the graph never invents relationships.
- **Con:** it only captures relationships that are *already* structured. The relationship between two documents that share a topic but no foreign key is invisible.

Use this pattern for structured slices of the corpus (CRM data, org charts, product hierarchies).

### LLM-driven extraction (Microsoft GraphRAG's approach)

For unstructured corpora, the graph is *extracted* from text: chunk the corpus, ask an LLM to identify entities and relationships per chunk, deduplicate entities across chunks, and load the resulting graph. This is Microsoft GraphRAG's shape (paper: [*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130); repo: [microsoft/graphrag](https://github.com/microsoft/graphrag)).

The GraphRAG pipeline:

1. **Chunk the corpus** (mod-102 shapes).
2. **Extract entities + relationships per chunk** using an LLM with a prompt that specifies entity types (`Person`, `Organization`, `Location`, `Event`) and relationship types.
3. **Resolve co-referent entities** across chunks (two mentions of "Anthropic" become the same node).
4. **Detect communities** in the resulting graph — Leiden algorithm is the default — and generate LLM summaries per community.
5. **At query time**, use *either* local search (traverse the graph around entities the query mentions) *or* global search (retrieve relevant community summaries and answer from them).

Trade-offs:

- **Pro:** captures relationships buried in prose; scales to any unstructured corpus.
- **Con:** the extracted graph is only as good as the extractor. Entity resolution is error-prone (`ACME`, `Acme Corp`, `Acme Corporation`). Extraction costs are high (an LLM call per chunk × N chunks). The graph structure has no verifiable ground truth.

GraphRAG's community-summary trick is what makes it interesting at scale — you don't traverse the whole graph at query time; you retrieve community summaries. But community-summary quality depends on the extractor's quality, and evaluating it is hard (see below).

**Practical middle-ground:** use deterministic construction for the structured slice you own, and use LLM extraction for the unstructured slice, and stitch them at a shared entity level (e.g., the extracted "Acme Corp" resolves to the CRM's `Account { id: "A-42" }`). Neo4j supports this cleanly; LlamaIndex's Property Graph Index makes the stitching explicit.

## LlamaIndex's Property Graph Index

LlamaIndex's [`PropertyGraphIndex`](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/) is the orchestration layer that fits both patterns above into a RAG-ready retriever. The reference shape:

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,      # LLM-driven extraction with a schema constraint
    ImplicitPathExtractor,        # extract from document metadata (structured)
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

docs = SimpleDirectoryReader("./corpus").load_data()
graph_store = Neo4jPropertyGraphStore(username="neo4j", password="...", url="bolt://localhost:7687")

kg_extractor = SchemaLLMPathExtractor(
    llm=OpenAI(model="gpt-4o-mini"),
    possible_entities=["Person", "Organization", "Product", "Location"],
    possible_relations=["works_at", "based_in", "makes", "acquired"],
    strict=True,
)

index = PropertyGraphIndex.from_documents(
    docs,
    kg_extractors=[kg_extractor, ImplicitPathExtractor()],
    property_graph_store=graph_store,
    embed_model=OpenAIEmbedding(),
)

# Retrieval: vector on nodes + Cypher traversal + optional keyword search
retriever = index.as_retriever(
    include_text=True,
    similarity_top_k=5,
    path_depth=2,
)
results = retriever.retrieve("Who at Acme Corp reports to the CEO?")
```

Key building blocks:

- **KG extractors** — pluggable extractors (`SchemaLLMPathExtractor`, `SimpleLLMPathExtractor`, `DynamicLLMPathExtractor`, `ImplicitPathExtractor`) that turn documents into `(entity, relation, entity)` triples with backing text.
- **Property graph store** — the persistence layer (Neo4j, Kùzu, Memgraph, in-memory `SimplePropertyGraphStore`).
- **Retrievers** — `LLMSynonymRetriever`, `VectorContextRetriever`, `CypherTemplateRetriever`, `TextToCypherRetriever`, and combinations. `VectorContextRetriever` does the vector-search-then-graph-expand pattern; `TextToCypherRetriever` asks the LLM to write a Cypher query directly (analog of text-to-SQL from chapter 4).

The combined retriever is what makes this useful — a query first vector-searches for the seed entities, then traverses the graph outward by `path_depth` hops, then returns the neighborhood's text as the "chunks" the generator sees. That combination is exactly the pattern that makes multi-hop questions tractable.

## Microsoft GraphRAG

The [Microsoft GraphRAG](https://github.com/microsoft/graphrag) project is a batteries-included implementation of the community-summary variant. The pipeline is heavier and more prescriptive than LlamaIndex's — it's an indexing pipeline that produces a set of parquet files (nodes, relationships, community reports, embeddings), plus a query interface that supports "local" and "global" search modes.

The paper's key claim: for **query-focused summarization** questions — questions that require synthesizing information across many documents ("what are the major themes across these 200 press releases") — global search over LLM-generated community summaries beats naïve top-k vector retrieval on comprehensiveness and diversity. The paper shows this on the [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) and podcast-transcript benchmarks with an LLM-as-judge evaluation.

Practical implications:

- **GraphRAG is expensive to build.** LLM extraction on every chunk plus community-summary generation is a fixed cost proportional to corpus size. Budget accordingly (the docs show cost-per-document estimates for standard corpora).
- **GraphRAG is best when questions are broad and synthesizing.** For narrow questions ("what is the price of product X"), it is over-engineered — a lookup or a vector search does the same job cheaper.
- **Provenance is preserved.** Each community report cites the source chunks; you can answer with citations back to text.
- **The graph is worth keeping even if you disable community summaries.** GraphRAG's extracted graph is often better than nothing for downstream analytics.

Read the paper and the [`docs/`](https://microsoft.github.io/graphrag/) directory before adopting. Don't run it on a 10K-chunk corpus expecting speed; do run it on a corpus of thousands of docs where users ask broad synthesis questions.

## When graph structure actually lifts recall — and when it doesn't

The honest assessment:

**Graph structure lifts recall on:**

- **Multi-hop questions** where the answer requires composing 2+ relationships. "Who does X, who used to work at Y, currently work with?" is unanswerable from top-k independent chunks and answerable from a two-hop graph traversal.
- **Structure-heavy questions** over structured corpora — org charts, product hierarchies, permission chains, molecular pathways, citation networks. The graph *is* the answer shape.
- **Query-focused summarization at scale** — synthesizing themes across many documents where community-summary retrieval beats naive vector top-k.

**Graph structure does not help (or hurts) on:**

- **Single-fact lookups.** "What is the price of X?" The graph adds latency without adding recall — a vector or SQL retriever wins.
- **Semantic-similarity questions** where the answer is a piece of text similar to the query and no traversal is required. Vector search wins.
- **Corpora where extraction quality is low.** If the LLM extractor gets entity names wrong 20% of the time, your graph has 20% wrong edges, and traversals compound the error. In this regime, a graph is worse than vector search because it *looks* structured while actually being noisy.
- **Very large graphs at query time** if you traverse too many hops. Latency scales badly with depth × branching factor. Limit `path_depth` (2–3 is common) and prune by vector similarity to the query at each hop.

The pragmatic default: **start with vector + lexical + SQL. Add a graph retriever only when your eval reveals a class of multi-hop questions that consistently under-perform.** Do not build a graph "just in case." Graph RAG has an infrastructure cost (a graph DB, a pipeline, ontology decisions) that has to be paid for by a measurable improvement.

## Evaluating a graph retriever

The retrieval-quality eval loop from mod-109 (recall@k, MRR, nDCG on labeled judgment sets) applies, with two additions specific to graphs:

- **Multi-hop question suite.** Author or find a judgment set of questions whose gold answers span 2+ documents linked by a relationship. Measure recall@k with the graph retriever vs. a vector-only baseline. The delta is the graph's contribution.
  - Reference sets: [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG), [MuSiQue](https://github.com/StonyBrookNLP/musique), [HotpotQA](https://hotpotqa.github.io/) — public multi-hop QA benchmarks you can adapt.
- **Extraction quality.** If your graph is LLM-extracted, sample a few hundred extracted triples and hand-label as `correct / wrong-entity / wrong-relation / wrong-both`. This is your extraction precision. If it's below ~80%, your downstream retrievals are noisy at the source; fix the extractor before evaluating the retriever.

For GraphRAG-style community-summary retrievers, the eval is harder (the "answer" is a summary, not a document set). LLM-as-judge is the pragmatic approach — pair the community-summary-based answer with a top-k vector-based answer for the same question and ask a judge model to rate comprehensiveness and diversity. mod-109's caveats about LLM judges (calibration, position bias) apply — validate the judge against a small human-labeled set.

## Practical recipe — the retriever this chapter recommends

For a first Graph RAG in your track:

1. **Pick Neo4j Desktop** for local dev (Cypher, viz, drivers all in one). Move to Kùzu if you want embedded and single-node; move to Neo4j AuraDB if you want managed.
2. **Deterministically construct** the parts of the graph you have structured data for (CRM, org chart, product hierarchy). Load with `MERGE` statements in a Python or Cypher script.
3. **LLM-extract** entities and relationships from the unstructured slice using LlamaIndex's `SchemaLLMPathExtractor` with a *tight* entity/relation schema. Do not let the extractor invent new types.
4. **Stitch** — resolve extracted entities to their structured-source counterparts by exact-match on name / ID / URL / email.
5. **Retriever** — LlamaIndex's `PropertyGraphIndex` with `VectorContextRetriever` + `TextToCypherRetriever` composed. Fall back to vector-only when the query has no clear entity anchor.
6. **Eval** — build a multi-hop question set (30–100 questions) whose gold answers span multiple docs. Compare vector-only vs. vector+graph. If the graph doesn't lift recall by a meaningful margin, keep it in the extraction layer for analytics but do not put it on the retrieval path.

## What you should be able to do after this chapter

- Explain the property graph model, distinguish it from RDF triples and relational tables, and name three engines that support it (Neo4j, Kùzu, Memgraph).
- Distinguish deterministic graph construction (from structured sources) from LLM-driven extraction (à la GraphRAG), and argue when each is right.
- Use LlamaIndex's `PropertyGraphIndex` with a KG extractor, a property graph store, and a composed retriever.
- Explain Microsoft GraphRAG's community-summary variant and when its comprehensiveness / diversity advantage justifies its indexing cost.
- Name three question shapes where graph structure lifts recall (multi-hop, structure-heavy, query-focused summarization) and three where it doesn't (single-fact, pure semantic similarity, low-quality extracted graphs).
- Design a graph-retriever eval: multi-hop judgment set + extraction-precision spot-check, wired into mod-111's retriever CI.

## Further reading

- Robinson, I., Webber, J., & Eifrem, E. (2015). *Graph Databases* (2nd ed.). O'Reilly. [Book page](https://neo4j.com/graph-databases-book/) — the property graph model, Cypher, use cases; the standard reference.
- Neo4j: [*Cypher Manual — Introduction*](https://neo4j.com/docs/cypher-manual/current/introduction/) — start here for Cypher.
- Kùzu docs: [*Getting Started*](https://docs.kuzudb.com/get-started/) — embedded property graph, single-node, Cypher-compatible; a good default for local development.
- Memgraph docs: [*Memgraph documentation*](https://memgraph.com/docs) — in-memory, real-time-focused, Cypher-compatible alternative.
- Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130). Implementation: [microsoft/graphrag](https://github.com/microsoft/graphrag) + [docs](https://microsoft.github.io/graphrag/).
- LlamaIndex: [*Property Graph Index Guide*](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/) — the reference for extractors, stores, and retrievers.
- Yixuan Tang & Yi Yang. (2024). *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries.* [arXiv:2401.15391](https://arxiv.org/abs/2401.15391); repo: [yixuantt/MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) — a multi-hop RAG benchmark you can adapt for graph-retriever eval.
- Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). *MuSiQue: Multi-hop Questions via Single-hop Question Composition.* TACL. [arXiv:2108.00573](https://arxiv.org/abs/2108.00573) — the compositional multi-hop QA set with attribution to the linking hops.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP. [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) — the older but still-canonical multi-hop QA benchmark.
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities.* Nature Scientific Reports. [arXiv:1810.08473](https://arxiv.org/abs/1810.08473) — the community-detection algorithm GraphRAG uses; useful for understanding why "communities" is the aggregation unit.
