# Exercise 03 — Graph RAG on a Property Graph with Neo4j

**Estimated effort:** ~3 hours. Python, Docker (for Neo4j Community 5.x), a small mixed corpus (a structured slice you can `MERGE` from CSV plus an unstructured slice of a few dozen documents), an LLM for extraction + generation, and a multi-hop question set.

## Objective

Stand up a small property graph in Neo4j, populate it with **both** a deterministic slice (from structured data) and an LLM-extracted slice (from prose), stitch the two on shared entities, and expose it as a retriever with LlamaIndex's `PropertyGraphIndex`. Then honestly measure whether the graph lifts recall on multi-hop questions relative to a vector-only baseline — and identify the question shape it helps on versus the shape it doesn't.

By the end you have:

- `docker/neo4j.yml` — a docker-compose that runs Neo4j Community with APOC enabled.
- `graph/deterministic_load.py` — `MERGE`-based ingest of a structured slice (CRM-shaped: `Customer`, `Order`, `Product`, `Employee`, `PLACED`, `CONTAINS`, `WORKS_AT`, `MANAGES`).
- `graph/extract.py` — LLM-driven extraction from ~30–80 unstructured documents into typed entities + relationships, stitched to the deterministic nodes by exact-match on name / ID / email.
- `retriever/vector_only.py` — the baseline: plain vector search over document chunks.
- `retriever/graph_rag.py` — LlamaIndex `PropertyGraphIndex` with a `VectorContextRetriever` + `TextToCypherRetriever` composed.
- `evals/multihop.py` — a 30–60 question set, hand-labeled with gold-answer document IDs, measuring recall@10 for both retrievers.
- `RESULTS.md` — the recall comparison, the extraction-precision spot-check, and one paragraph on the question shape the graph helped on and the shape it didn't.

## Prerequisites

- Chapter 5 read (property graph model, deterministic vs. extracted construction, `PropertyGraphIndex`, when graph structure actually lifts recall).
- Chapter 2 (belt-and-braces post-check) as background — the graph retriever must honor the same policy contract as any other retriever, even for this exercise where the ACL is trivial.
- Docker and `docker compose`. Neo4j Community 5.x image works; the [Neo4j Docker docs](https://neo4j.com/docs/operations-manual/current/docker/) cover the setup.
- Python 3.10+; `pip install neo4j llama-index llama-index-graph-stores-neo4j llama-index-embeddings-huggingface llama-index-llms-openai sentence-transformers`.
- Access to an LLM for extraction and text-to-Cypher. GPT-4o-mini / Claude Haiku / a local coder model via Ollama are all viable; extraction quality varies substantially — document your choice in `RESULTS.md`.

## Steps

### Step 1 — Bring up Neo4j (15 minutes)

Author `docker/neo4j.yml` (or `docker-compose.yml`):

```yaml
services:
  neo4j:
    image: neo4j:5-community
    environment:
      - NEO4J_AUTH=neo4j/testpassword
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./neo4j-data:/data
```

Bring up (`docker compose up -d`), open <http://localhost:7474>, log in, run `RETURN 1` to confirm.

Also verify APOC loaded (`RETURN apoc.version()`) — the LLM extractor uses APOC procedures for bulk loading.

### Step 2 — Author the deterministic slice (25 minutes)

Build or synthesize a small CRM-shaped dataset — three CSVs is enough:

- `customers.csv` — 50–100 rows: `id, name, tier, region, email`.
- `orders.csv` — 200–500 rows: `id, customer_id, product_id, employee_id, amount, created_at`.
- `products.csv` — 20–40 rows: `id, name, category, list_price`.
- `employees.csv` — 15–30 rows: `id, name, email, manager_id, department`.

`graph/deterministic_load.py`:

```python
from neo4j import GraphDatabase
import csv

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword"))

def load(sess, path, cypher):
    with open(path) as f:
        for row in csv.DictReader(f):
            sess.run(cypher, **row)

with driver.session() as sess:
    load(sess, "customers.csv",
         "MERGE (c:Customer {id: $id}) SET c.name = $name, c.tier = $tier, c.region = $region, c.email = $email")
    load(sess, "products.csv",
         "MERGE (p:Product {id: $id}) SET p.name = $name, p.category = $category, p.list_price = toFloat($list_price)")
    load(sess, "employees.csv",
         "MERGE (e:Employee {id: $id}) SET e.name = $name, e.email = $email, e.department = $department")
    load(sess, "employees.csv",
         "MATCH (e:Employee {id: $id}), (m:Employee {id: $manager_id}) MERGE (e)-[:REPORTS_TO]->(m)")
    load(sess, "orders.csv", """
        MATCH (c:Customer {id: $customer_id}), (p:Product {id: $product_id}), (e:Employee {id: $employee_id})
        MERGE (o:Order {id: $id}) SET o.amount = toFloat($amount), o.created_at = date($created_at)
        MERGE (c)-[:PLACED]->(o) MERGE (o)-[:CONTAINS]->(p) MERGE (e)-[:HANDLED]->(o)
    """)
```

Verify with a spot-check Cypher query in Neo4j Browser — e.g., `MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS]->(p:Product) RETURN c.name, o.amount, p.name LIMIT 10`.

### Step 3 — Author the unstructured slice (15 minutes)

Compile 30–80 short documents (a few paragraphs each) that reference entities from the CRM by name or email. Example shapes:

- Meeting notes: "Sync with **Contoso Ltd** on Q3 renewal. Attended by **Sam Rivera** (their VP Ops) and **Jamie Chen** on our side."
- Support tickets: "Ticket 4521: **Acme Foundry** reported latency on the **RapidWidget-3000** product; escalated to **Priya Patel**."
- Internal wiki: "**Priya Patel** owns the RapidWidget product line; she reports to **Marcus Kim**."

You want ~2–5 named entities per document with a mix of relationships (some already in the CRM, some new — e.g., "Sam Rivera works at Contoso Ltd" is a new fact).

Save as plain text or Markdown under `corpus/`. Also generate the embeddings (any small text embedder) and store them for the vector baseline.

### Step 4 — LLM extraction into the graph (30 minutes)

`graph/extract.py`: use LlamaIndex's `SchemaLLMPathExtractor` with a *tight* entity/relation schema so the extractor doesn't invent types.

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

docs = SimpleDirectoryReader("corpus/").load_data()

graph_store = Neo4jPropertyGraphStore(
    username="neo4j", password="testpassword", url="bolt://localhost:7687"
)

extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=["Customer", "Employee", "Product", "Organization", "Person"],
    possible_relations=["WORKS_AT", "REPORTS_TO", "ATTENDED", "OWNS", "MENTIONED_IN"],
    strict=True,
)

index = PropertyGraphIndex.from_documents(
    docs,
    kg_extractors=[extractor],
    property_graph_store=graph_store,
)
```

Reference: LlamaIndex [`PropertyGraphIndex` guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/).

### Step 5 — Stitch extracted entities to the CRM (20 minutes)

The extractor will produce entities like `(name: "Priya Patel")` that need to resolve to the deterministic `Employee {id: "e-042", email: "priya.patel@..."}`. Write a stitcher Cypher pass:

```cypher
// Merge extracted Person nodes into Employee nodes by exact name match
MATCH (p:Person), (e:Employee)
WHERE toLower(p.name) = toLower(e.name)
CALL apoc.refactor.mergeNodes([e, p], {properties: "combine", mergeRels: true})
YIELD node
RETURN count(node);
```

Repeat for `Organization` → `Customer` (by name), `Product` → `Product` (by name). Log how many nodes were merged and how many extracted entities had no match (those are new — new customers, new people; keep them, but note the count as a signal for the resolution precision below).

### Step 6 — Vector-only baseline retriever (15 minutes)

`retriever/vector_only.py`: plain vector search over the document chunks from step 3. No graph, no re-ranker — the point is to be an honest baseline.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("corpus/").load_data()
vindex = VectorStoreIndex.from_documents(docs)
vretriever = vindex.as_retriever(similarity_top_k=10)

def vector_only(question):
    return [n.node.metadata.get("source", n.node.node_id) for n in vretriever.retrieve(question)]
```

### Step 7 — Graph-RAG retriever (25 minutes)

`retriever/graph_rag.py`: LlamaIndex `PropertyGraphIndex.as_retriever` composed with `VectorContextRetriever` (vector-seed then graph-expand) and `TextToCypherRetriever` (LLM-generated Cypher). Both are documented in the [Property Graph Index guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/).

```python
from llama_index.core.indices.property_graph import (
    VectorContextRetriever, TextToCypherRetriever
)

retriever = index.as_retriever(
    sub_retrievers=[
        VectorContextRetriever(index.property_graph_store, embed_model=Settings.embed_model,
                               similarity_top_k=5, path_depth=2),
        TextToCypherRetriever(index.property_graph_store, llm=Settings.llm),
    ],
    include_text=True,
)

def graph_rag(question):
    return [n.node.metadata.get("source", n.node.node_id) for n in retriever.retrieve(question)]
```

Log the Cypher the `TextToCypherRetriever` emits so you can inspect it — many "why did the graph not help here" investigations end at "the LLM wrote a Cypher that returned nothing."

### Step 8 — The multi-hop question set (25 minutes)

Hand-author 30–60 questions with gold-answer document IDs, drawn from three buckets so you can see per-bucket behavior:

- **Single-hop / lookup** (~10 questions). "What tier is Contoso Ltd?" — the answer is in a single doc or a single node's property. Vector-only should nail these; graph should not lose them.
- **Multi-hop / traversal** (~15 questions). "Which employees have handled orders from tier-1 customers in the west region?" or "Who does the owner of RapidWidget report to?" — the answer requires composing 2+ relationships. This is where the graph should lift recall.
- **Aggregation / structure** (~10 questions). "Which product category has the most customers?" — arguably text-to-SQL territory (chapter 4); worth including a few to see the honest failure of both retrievers here.

Save as JSONL:

```json
{"id": "q001", "question": "...", "gold_doc_ids": ["doc-004", "doc-017"], "bucket": "multi-hop"}
```

### Step 9 — Extraction-precision spot-check (15 minutes)

Sample 50 extracted `(entity, relation, entity)` triples uniformly at random. Hand-label each as `correct / wrong-entity / wrong-relation / wrong-both / hallucinated`. Report per-class rate.

If extraction precision is below ~80%, downstream Graph-RAG will be noisy at the source; note this loud in `RESULTS.md` — a graph built on noisy extraction can look *worse* than vector-only, and knowing which is which matters.

### Step 10 — Evaluate and write up (25 minutes)

`evals/multihop.py`: for each question, retrieve top-10 from both retrievers; compute `recall@10 = |gold ∩ retrieved| / |gold|`. Aggregate per bucket.

Publish `RESULTS.md` with:

- Recall@10 for vector-only and graph-RAG, aggregated **and split by bucket**.
- Extraction precision (from step 9).
- One paragraph on the multi-hop bucket — did the graph lift recall meaningfully? (If not, is it the extractor, the Cypher, or the question set?)
- One paragraph on the single-hop bucket — did the graph lose any lookups? (If yes, why?)
- One paragraph on the aggregation bucket — neither retriever should nail these; both under-performing is the honest baseline that motivates chapter 4's text-to-SQL as the right tool.
- A recommendation: keep the graph on the retrieval path, drop it, or use extraction-only-for-analytics.

## Acceptance criteria

- Neo4j is up with both the deterministic slice loaded and the LLM-extracted slice loaded and stitched.
- `retriever/vector_only.py` and `retriever/graph_rag.py` both callable against the same question set.
- The multi-hop question set is at least 30 questions, hand-labeled with gold document IDs, and split into buckets.
- Extraction-precision spot-check on ≥ 50 triples with per-class breakdown.
- `RESULTS.md` publishes per-bucket recall@10 for both retrievers and states the "keep / drop / analytics-only" recommendation with a one-paragraph justification per bucket.
- The `TextToCypherRetriever`'s generated Cypher is logged for at least the multi-hop bucket (so a reader can see when the LLM did or didn't compose a valid traversal).

## Stretch goals

- **Add Microsoft GraphRAG as a third retriever.** Follow the [`microsoft/graphrag`](https://github.com/microsoft/graphrag) quickstart on the same unstructured corpus; compare its "global search" (community-summary) answers to your `TextToCypherRetriever` on the multi-hop bucket. Note the indexing cost.
- **Swap the store for Kùzu.** LlamaIndex has a [Kùzu graph store integration](https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_kuzu/); repeat the retrieval and note the operational differences (embedded vs. server, install footprint).
- **Improve extraction with a two-pass extractor.** First pass extracts entities; second pass extracts relationships given the entities. Measure the delta in extraction precision.
- **Add a `path_depth = 3` variant** and see whether the extra hop lifts recall on the deepest multi-hop questions or just adds latency.
- **Wire the ACL contract.** Add a `tenant_id` property to every node and a query-time filter (Cypher `WHERE`) restricting traversal to nodes in the caller's tenant. Fuzz-test with two tenants; assert zero cross-tenant leakage. This is chapter 7's policy contract instantiated for a graph backend.
- **Author a Cypher-template retriever** for one hot query shape (`RETRIEVE_EMPLOYEE_ORDER_HISTORY`); compare its accuracy and latency to the free-form `TextToCypherRetriever` for that shape.

## Notes

- **Extraction quality is the ceiling.** If your extractor is bad, your graph is bad, and no amount of retrieval cleverness will make Graph-RAG win. Do the step-9 spot-check before drawing conclusions from step-10 numbers.
- **Log the Cypher.** Every `TextToCypherRetriever` call produces a Cypher string; save it alongside the question. This is your debugging surface when the graph "should have helped" and didn't.
- **`SchemaLLMPathExtractor` with `strict=True` is the safer default.** A permissive extractor invents entity types and blows up your label space; a strict one at least keeps the graph queryable.
- **Neo4j Community caps at one database.** If you want per-tenant physical isolation *inside* Neo4j, that's Enterprise. For this exercise a `tenant_id` property + query-time filter is fine; the point is to see the shape, not to build the platform.
- **APOC is not part of core Neo4j.** The docker-compose above installs it via `NEO4J_PLUGINS`; if you install Neo4j another way, you'll need to add APOC manually. LlamaIndex uses APOC for bulk operations under the hood in `Neo4jPropertyGraphStore`.
- **Chapter 5's "most enterprise questions are one-hop" is the point of the bucketed eval.** If your multi-hop bucket is small because you couldn't think of many honest multi-hop questions in your domain, that itself is a signal — the graph may not earn its keep in that domain.
