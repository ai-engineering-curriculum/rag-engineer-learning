# Chapter 6 — Hierarchical, Parent-Child, and Small-to-Big Retrieval

## Why this chapter exists

Chapter 5 named the trade-off: small chunks embed precisely but shred context; large chunks preserve context but embed loosely. This chapter is the escape hatch. Instead of picking one chunk size, index at *one* granularity and return at *another*.

The three patterns this chapter walks are the ones LlamaIndex and LangChain converge on: **parent-child** (a.k.a. "small-to-big"), **hierarchical / auto-merging**, and **sentence-window**. All three are widely deployed in production and all three come with pre-built abstractions in both major frameworks. `exercise-04` will have you implement one end-to-end and A/B it against the single-size baseline from chapter 5.

## The shared shape

All three patterns share the same architecture:

```
    write path                                     read path
    ───────────                                    ────────
    1. Parse document                              1. Query embedding
    2. Split into "small" units (indexing units)   2. Search small-unit index
    3. Split into "big" units (context units)      3. For each hit, resolve to its "big" parent
    4. Record parent-of relationship               4. Return the big units to reranker / LLM
    5. Embed the small units
    6. Index (small_embedding, small_id, parent_id)
```

- The **small units** are what the retriever searches over. They are tight, so cosine similarity to a specific query is high.
- The **big units** are what leaves the retriever. They are large, so the LLM has coherent context.
- The **parent-of** link is the piece that makes this work as a system. It can be a foreign key in Postgres, a metadata field in a vector index, or a lookup dict alongside a FAISS index — the mechanism is a detail; the shape is what matters.

## Pattern 1 — Parent-child (small-to-big)

The simplest of the three. Split each document into big chunks (say, 2000 tokens); split each big chunk into small chunks (say, 400 tokens); index the small; store the big keyed by their ID.

### LlamaIndex

```python
from llama_index.core.node_parser import SentenceSplitter

parent_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter  = SentenceSplitter(chunk_size=400,  chunk_overlap=40)

# Parent nodes
parent_nodes = parent_splitter.get_nodes_from_documents(docs)

# Child nodes, each carrying a ref back to its parent
child_nodes = []
for parent in parent_nodes:
    children = child_splitter.get_nodes_from_documents([parent.to_document()])
    for child in children:
        child.metadata["parent_id"] = parent.node_id
        child_nodes.append(child)

# Index children in the vector store; store parents in a docstore
vector_index = VectorStoreIndex(child_nodes, storage_context=storage)
storage.docstore.add_documents([p.to_document() for p in parent_nodes])
```

On retrieval, look up each child hit's `parent_id` and return the parent text.

### LangChain

LangChain ships the pattern as a first-class retriever:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      # e.g. Chroma / PGVector — stores children
    docstore=InMemoryStore(),     # stores parents keyed by ID
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
```

See the [ParentDocumentRetriever guide](https://python.langchain.com/docs/how_to/parent_document_retriever/) for the full pattern.

### When it wins

- Q&A over structured documentation where a specific sentence answers the question but the LLM benefits from the surrounding paragraph for context.
- Any corpus where a single-size ablation (chapter 5) shows a genuine trade-off — small chunks helped recall on some queries and big chunks helped on others.

### When it does not

- Small corpora where the whole document fits in one big chunk. A parent-of-one indirection buys nothing.
- Corpora with truly heterogeneous document lengths (a 10-page policy vs. a 400-page manual) — a fixed "big" size does not fit all. Use pattern 2.

## Pattern 2 — Hierarchical / auto-merging

Generalizes parent-child to a *tree* of sizes. Split into large, medium, and small chunks; store all of them; on retrieval, look at which small chunks matched and *promote* to a parent when enough of its children hit.

### LlamaIndex

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(docs)
leaves = get_leaf_nodes(nodes)

# Index the leaves; keep the parents in the docstore
storage.docstore.add_documents(nodes)
leaf_index = VectorStoreIndex(leaves, storage_context=storage)

base_retriever = leaf_index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(base_retriever, storage_context=storage, verbose=True)
```

The `AutoMergingRetriever`'s promotion rule: if a threshold fraction (default 0.5) of a parent's children appear in the retrieved set, return the parent instead of the individual children. This packs the top-*k* with fewer, richer chunks when the query hits a concentrated region of a document.

See LlamaIndex's [Auto Merging Retriever guide](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/) for the full walk-through.

### When it wins

- Corpora with heterogeneous documents where the "right" context size depends on the query. Multi-hop questions concentrate hits on one region and promote to a parent; broad questions get shallow hits from many regions and stay at the leaf granularity.

### When it does not

- Small corpora — again, the tree of sizes buys nothing.
- Corpora with weak sectional structure — hierarchical parsers assume a meaningful nesting. If everything is one flat blob, you get a two-level tree and might as well use parent-child.

## Pattern 3 — Sentence-window

Index each *sentence*; on retrieval, expand each hit to a `±k`-sentence window from the original document. This gives you the tightest possible embedding while still handing the LLM enough surrounding context to make sense of the sentence.

### LlamaIndex

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                         # ±3 sentences of context on retrieval
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)
nodes = node_parser.get_nodes_from_documents(docs)
# Index nodes; on retrieval, use MetadataReplacementPostProcessor to swap in "window"
```

The [Sentence Window Retrieval example](https://docs.llamaindex.ai/en/stable/examples/node_parsers/MetadataReplacementDemo/) shows the accompanying `MetadataReplacementPostProcessor` that swaps the retrieved node's text for its stored window before returning it.

### When it wins

- Extractive fact-lookup over long factual documents (a specific number, name, or clause in a long contract).
- Legal / medical corpora where the sentence is *the* answer but a reader needs surrounding sentences to trust it.

### When it does not

- Question types whose answer spans paragraphs (policy summaries, methodology explanations). A three-sentence window will be too little.
- Chatty corpora where sentence boundaries are fuzzy.

## The one gotcha all three share: de-duplication

If several small hits share a parent, or several sentence windows overlap, the top-*k* you hand the LLM can duplicate the same text. The assembler (mod-107) is the primary line of defense — but the chunker can help:

- **Stable parent / window IDs.** So the assembler can dedup by ID, not by string comparison.
- **Score aggregation policy.** When multiple children match one parent, decide up-front whether the parent's "score" is the max of its children's scores, the mean, or the sum. Different rules produce different orderings; document the choice.
- **Cap children-per-parent in retrieval.** If you retrieve top-50 children and 40 of them belong to one parent, you have over-fetched from one region. Cap at, say, 3 children per parent per query.

## Cost, storage, complexity

The patterns are not free:

- **Storage.** Sentence-window indexes one embedding per sentence — often 5–10× more chunks than a paragraph baseline. Parent-child indexes small chunks but stores parents alongside them; expect ~2× storage. Hierarchical stores every level; ~3× is typical.
- **Ingest cost.** More chunks embedded = more embedding calls on ingest. Semantic and hierarchical patterns compound with chapter 4's semantic chunker.
- **Pipeline surface area.** Every extra layer is another place a schema change, a doc-store bug, or a parent-ID mismatch can silently corrupt retrieval. Add integration tests: "the returned text is the parent's text, not the child's" is a real test worth writing.

The decision to reach for hierarchical / small-to-big is a decision to accept these costs in exchange for measurable recall @ k gains on your judgment set. Measure. `exercise-04` provides the harness.

## Interaction with reranking

Small-to-big and reranking compose well. Retrieve many small chunks; the reranker (mod-107) scores them precisely because they are small and topically tight; the top-*K* rerank survivors get expanded to their parents before being handed to the LLM. This shape lets you push `N` (retrieved) high without paying `f(chunk_size) * N` cost at rerank time, because the reranker sees the small units.

The obverse: if you rerank *after* expanding to parents, you undo the precision advantage of the small embeddings, and you make the reranker's job harder (bigger inputs cost more). Rerank small; expand at assembly time.

## What you should be able to do after this chapter

- Draw the shared architecture (write path + read path) for parent-child, hierarchical, and sentence-window patterns.
- Configure and call LlamaIndex's `SentenceSplitter` in a parent-child shape, and LangChain's `ParentDocumentRetriever`, on the same corpus.
- Configure and call `HierarchicalNodeParser` + `AutoMergingRetriever` and name what the "auto merge" step does.
- Configure and call `SentenceWindowNodeParser` and its `MetadataReplacementPostProcessor` companion.
- Reason about which pattern fits which corpus / query mix, and about their storage + ingest cost overhead.
- Order operations so the reranker sees the *small* chunks and the LLM sees the *big* parents.

Chapter 7 turns to the retrieval-critical *metadata* every chunk needs to carry: the section path, page number, publish date, author, permission scope — the fields the retriever filters on and the generator cites.

## Further reading

- LlamaIndex, [*Advanced Retrieval Strategies*](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/), [*Auto Merging Retriever*](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/), [*Sentence Window Retrieval*](https://docs.llamaindex.ai/en/stable/examples/node_parsers/MetadataReplacementDemo/).
- LangChain, [*Parent Document Retriever*](https://python.langchain.com/docs/how_to/parent_document_retriever/), [*Multi-Vector Retriever*](https://python.langchain.com/docs/how_to/multi_vector/) — the generalization to "one big doc, many indexed representations."
- Anthropic engineering, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — related pattern where each small chunk is *augmented* with a synthesized parent-context prefix before embedding.
