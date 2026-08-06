# Exercise 04 — Hierarchical, parent-child, and small-to-big chunk design

**Estimated effort:** ~2 hours.

## Objective

Implement one hierarchical retrieval pattern end-to-end (parent-child with LlamaIndex or LangChain, *or* auto-merging with LlamaIndex `HierarchicalNodeParser`, *or* sentence-window with `SentenceWindowNodeParser`) and A/B it against the best single-size chunker from `exercise-03`. Report where the pattern helps, where it hurts, and where it costs more than the recall gain justifies.

The point is to feel the "index small, return big" mechanic — including the assembly-time surprises (dedup, score aggregation, over-fetching from one document) — with real numbers on real queries.

## Prerequisites

- `exercise-03` completed and its winning single-size chunker is your baseline.
- Chapter 6 (*Hierarchical, Parent-Child, and Small-to-Big*) read once.
- Same judgment set and evaluation harness from `exercise-03`.
- The same pgvector setup, embedding model held constant.

## Part A — Pick and implement one hierarchical pattern (60 min)

Pick **one** of the three patterns from chapter 6 and implement it end-to-end. All three are acceptable; the choice depends on your corpus and query mix:

### Option 1 — Parent-child (small-to-big)

```python
# LlamaIndex flavor
parent_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter  = SentenceSplitter(chunk_size=400,  chunk_overlap=40)

parent_nodes = parent_splitter.get_nodes_from_documents(docs)
children = []
for parent in parent_nodes:
    for child in child_splitter.get_nodes_from_documents([parent.to_document()]):
        child.metadata["parent_id"] = parent.node_id
        children.append(child)

storage.docstore.add_documents(parent_nodes)
index = VectorStoreIndex(children, storage_context=storage)
```

On retrieve: search `children`, resolve each hit to its parent via `parent_id`, dedup parents, return parent text.

Or the LangChain shape:

```python
from langchain.retrievers import ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=..., docstore=InMemoryStore(),
    child_splitter=..., parent_splitter=...,
)
retriever.add_documents(docs)
```

### Option 2 — Auto-merging hierarchical

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever

parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = parser.get_nodes_from_documents(docs)
leaves = get_leaf_nodes(nodes)
storage.docstore.add_documents(nodes)
leaf_index = VectorStoreIndex(leaves, storage_context=storage)

base = leaf_index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(base, storage_context=storage)
```

### Option 3 — Sentence-window

```python
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

parser = SentenceWindowNodeParser.from_defaults(
    window_size=3, window_metadata_key="window", original_text_metadata_key="orig",
)
nodes = parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=10)
# post-process: swap each returned node.text with node.metadata["window"]
```

Whichever you pick, document your choice in `docs/hierarchical-choice.md` with two sentences on why (usually: expected query granularity).

## Part B — Instrument the pattern (30 min)

Add three instrumentation hooks:

1. **`n_children_hit`** — for each query, how many child chunks matched? Log.
2. **`n_unique_parents_returned`** — for parent-child / auto-merging, after dedup by `parent_id`, how many distinct parents did the retriever surface for one query?
3. **`children_per_parent_max`** — the max number of children hitting one parent for one query. If this is consistently 10+, you are over-fetching from one document — cap it.

Chapter 6's cost / storage warnings become concrete when you can see the numbers.

## Part C — A/B against the exercise-03 winner (30 min)

Run both retrievers on the same 30+ query judgment set from `exercise-03`. Compute:

| Metric | exercise-03 winner | hierarchical variant | Δ |
|---|---|---|---|
| Recall@10 |  |  |  |
| nDCG@10 |  |  |  |
| MRR@10 |  |  |  |
| avg. tokens returned per query | | | |
| avg. n_chunks_returned per query | | | |
| index size (MB) | | | |
| ingest wall-clock (s) | | | |

Note that "Recall@10 for parent-child" needs a definition: it is `Recall@10` computed on *parents* returned, not children hit. If your judgment set is document-level, this is straightforward; if span-level, you may need to remap the qrels to parent granularity.

## Part D — Per-query error analysis (20 min)

- **Where the hierarchical variant wins by ≥ 0.3 recall@10** — what do those queries have in common? (Usually: the answer needs surrounding sentences that a small chunk cut off.)
- **Where the hierarchical variant loses by ≥ 0.3** — what do those queries have in common? (Usually: broad queries that got over-concentrated on one parent's children, missing another relevant document entirely.)
- **How often did the assembler have to deduplicate children of the same parent?** Report the mean and max across queries.

## Part E — Recommendation (10 min)

Close `docs/hierarchical-ab.md` with a decision:

- Deploy the hierarchical variant for this corpus? Yes / No / Only-for-certain-queries.
- If "only for certain queries," name the router condition: "route to hierarchical when the query is longer than 15 tokens" or "route to hierarchical when the query is a definition-style question."
- Storage / ingest cost overhead vs. recall gain — is the trade worth it?

## Deliverables

- `src/retrievers/hierarchical.py` — one of the three patterns implemented.
- `scripts/hierarchical_ab.py` — the A/B harness.
- `docs/hierarchical-choice.md` — two sentences on which pattern and why.
- `docs/hierarchical-ab.md` — the A/B table, per-query analysis paragraphs, and recommendation.
- Tests: at least one that asserts "hits are children, but returned text is the parent's" for whichever pattern you chose.

## Acceptance criteria

- The pattern is implemented as one of parent-child, auto-merging, or sentence-window (not a hand-rolled sketch).
- The A/B numbers are computed on the same 30+ query judgment set as `exercise-03`.
- The `avg. tokens returned per query` column is filled in and interpreted — hierarchical typically returns more tokens per query, which has downstream cost implications the recommendation must address.
- The dedup / children-per-parent instrumentation shows real numbers, not zeros.
- The recommendation is a decision, not a hedge.

## Stretch goals

- Combine two patterns: hierarchical parent-child *and* sentence-window (children are sentences; parents are 400-token windows; grandparents are 2000-token sections). Three levels. Measure whether it beats the two-level variant or whether the extra complexity is not paying its way.
- Add a rerank stage before parent resolution. Retrieve top-30 children, rerank with `bge-reranker-large` to top-10, *then* resolve to parents. Compare against reranking parents directly. Chapter 6 argued the small-first-then-expand order is stronger; check whether your data agrees.
- Take one hierarchical variant into a real assembly step: feed the resolved parents through a `LLMChainExtractor` (LangChain) or `ContextualCompressor` (LlamaIndex) to compress each parent to ≤400 tokens while preserving citation anchors, and measure how compression affects generation quality on a handful of end-to-end questions. This bridges into mod-107 material.
