# Chapter 6 — Contextual Compression

## Why this chapter exists

A cross-encoder-reranked top-10 is a much better set than an un-reranked one, but each of those 10 chunks may be 500–1500 tokens long, and only a small fraction of each chunk is actually about the query. When the generator sees the whole chunk it wastes tokens on irrelevant surrounding text and — thanks to lost-in-the-middle (chapter 8) — sometimes attends less reliably to the small on-topic span buried inside.

**Contextual compression** is the middle stage between reranking and assembly that extracts, per chunk, the sentences or spans that actually address the query. Two well-known frameworks — LlamaIndex's `ContextualCompressor` and LangChain's `LLMChainExtractor` — expose the pattern. This chapter is why it works, when it is worth the latency and cost, and the two things it will break if you are careless (citation anchors and answer completeness).

## The two problems compression solves

1. **Chunks are too long for the budget.** After the funnel (chapter 5) tells you K=10 and you have a 4K-token context budget, you may be given 10 × 800-token chunks and asked to fit them in 4000 tokens. Truncating them arbitrarily loses answer-supporting content unpredictably. Compression takes only the on-topic content per chunk, keeping the total inside the budget without cutting mid-sentence.

2. **The relevant content is diluted inside a longer chunk.** Even when everything fits, a 200-word passage about your query surrounded by 1000 words about a neighbouring topic weakens the generator's grounding. The generator produces vaguer, more paraphrased answers because it is *summarising a page*, not *citing a sentence*. Compression sharpens the grounding.

The trade-off: compression is either an LLM call per chunk (extractive-summarisation shape) or a strong-embedder computation per chunk (embedding-filter shape). Both add latency and cost that the rest of the pipeline doesn't have. Chapter 5's latency budget worksheet is the argument for or against compression on your stack.

## The four compression strategies

### 1. Sentence-level extractive filtering with an embedder

The simplest and cheapest shape: within each chunk, embed each sentence, embed the query, keep only the sentences with cosine similarity above a threshold (or the top-*m* sentences).

Pros: fast; no LLM call; the same embedder you already run for retrieval works. Cheap enough to run over every retrieved chunk without moving the latency budget meaningfully.

Cons: sensitive to sentence-splitting quality (mod-102 chapter on chunking is the reference); loses cross-sentence context (a sentence that only makes sense in the context of the previous one gets stripped out). Best for chunks that are already well-structured (bulletpoint documentation, FAQ passages).

LangChain names this shape [`EmbeddingsFilter`](https://python.langchain.com/docs/how_to/contextual_compression/#embeddings-filter). LlamaIndex has similar utilities in its `postprocessor` set.

### 2. LLM-based extractive compression

Ask an LLM to extract from each chunk *only the sentences relevant to the query*, verbatim. Prompt shape:

```
You are extracting relevant content from a document to help answer a user's query.
Return only the sentences from the document that directly help answer the query.
Return them verbatim — do NOT paraphrase, summarise, or add commentary.
If nothing in the document is relevant, return the string "NOT_RELEVANT".

Query: {query}

Document:
{chunk_text}

Relevant sentences (verbatim, one per line):
```

Pros: higher fidelity than embedder-based filtering; handles paraphrase mismatches between query and document that embeddings can miss.

Cons: an LLM call per chunk. K=10 chunks means 10 LLM calls per query — at cheap-model prices this is 20–100 ms if you parallelise, 200–1000 ms if you serialise; at expensive-model prices this is meaningful money.

This is LangChain's [`LLMChainExtractor`](https://python.langchain.com/docs/how_to/contextual_compression/#adding-contextual-compression-with-an-llmchainextractor). The equivalent in LlamaIndex is a [`Response Synthesizer`](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes/) in "compact" mode, or the [`SentenceEmbeddingOptimizer`](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/sentence_optimizer/) for embedding-based filtering.

### 3. LLM-based summarisation compression

Ask an LLM to summarise the chunk *with respect to the query*. Prompt shape:

```
Summarise the following document as it relates to the user's query.
Preserve any factual claims that answer the query. Preserve dates, numbers, and quoted phrases exactly.

Query: {query}

Document:
{chunk_text}

Summary:
```

Pros: shortest possible output for a given chunk; captures cross-sentence context that extractive can lose.

Cons: **the generator now grounds on a summary, not on the source text.** Citations will point at the summary's phrasing, not at anything in the actual document. This is often unacceptable — it converts a retrieval-augmented answer into a retrieval-augmented paraphrase, which is what RAG is supposed to prevent. Use summarisation only if your product does *not* need verbatim citations and your generator prompt (mod-108) is designed around summarised sources.

### 4. Cross-encoder-based sentence reranking

A specialised shape: use the same cross-encoder from chapter 2 not on chunks but on *sentences within chunks*, keeping only the top-scoring sentences per chunk.

Pros: reuses the reranker infrastructure; no separate LLM call; sentence-level precision.

Cons: many more forward passes than chunk-level rerank (10 chunks × 20 sentences per chunk = 200 forward passes vs. 100 in the chunk-level rerank), which can double or triple the rerank latency.

Only worth it if the sentence-vs-chunk granularity mismatch is genuinely hurting your quality — this is often the case for very long chunks (mod-102 chunk sizes at the upper end, 800+ tokens) but not for short ones (200-token chunks).

## The two things compression will silently break

If you turn on any of the LLM-based strategies (2, 3) and stop checking for it, two things degrade:

### 1. Citation anchor loss

Every compressor takes chunk text and produces different text. The *citation anchor* (`doc_id`, `chunk_id`, `page`, `section`) attached to the chunk must survive the transformation, or your downstream "cite the source" prompt (mod-108) has nothing to point at.

The correct shape:

```python
@dataclass
class CompressedChunk:
    doc_id: str
    chunk_id: str
    source: SourceRef        # UNCHANGED from the retriever output
    original_text: str        # kept for eval and for debugging
    compressed_text: str      # what the generator sees
    compression_ratio: float  # len(compressed_text) / len(original_text) — useful diagnostic
```

The `source` field must be preserved verbatim through compression. Both LlamaIndex and LangChain do this by default when you use their compressor pattern properly (the compressor operates on a `Document`/`NodeWithScore` object that carries metadata, not on a raw string), but common integration bugs strip metadata by:

- Storing chunks as bare strings, compressing them, then reattaching metadata by index (breaks if the compressor drops chunks — some produce fewer output chunks than input, e.g., when a chunk is entirely irrelevant and returned as "NOT_RELEVANT").
- Concatenating multiple chunks before compression and then not knowing which spans of the output came from which sources.
- Passing the compressed text to a downstream LLM that hallucinates citations because it never got a canonical anchor to cite.

Test citations *after* compression in every integration test. Exercise 03 covers this.

### 2. Answer-completeness loss

An overly-aggressive compressor throws out sentences the generator needed. The typical failure mode: the compressor uses a similarity threshold tuned for "obviously relevant sentences" and drops sentences that are only tangentially relevant — but the generator needed one of those tangential sentences to complete the answer ("the tax rate is 5% except in California, where it is 7%" — the compressor keeps the first clause and drops the second).

The mitigations:

- Prefer looser thresholds. Compression is meant to reduce, not to filter. Losing a 10-word sentence to compression is cheap if it was irrelevant; recovering an answer from a stripped sentence is expensive.
- Include a "return NOT_RELEVANT" escape hatch for entirely-off-topic chunks (safer than getting a hallucinated summary of an irrelevant chunk).
- Log compression ratios (per-chunk `compressed_len / original_len`). A ratio below 10% on a large chunk is a signal that the compressor may have thrown out too much.
- Eval end-to-end (not just at the retriever) after enabling compression. Faithfulness and answer relevance (mod-109) are the metrics that catch answer-completeness regressions.

## The LlamaIndex `ContextualCompressor` pattern

[LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/) organises compression around **node postprocessors** — objects that take a list of retrieved `NodeWithScore` and return a filtered / transformed list before the response synthesizer runs.

```python
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.postprocessor.node import LLMRerank

# Sentence-level embedding filter — strategy 1 above.
sentence_optimizer = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,   # keep sentences with top-50% similarity to query
    threshold_cutoff=None,   # or use an absolute threshold
)

# LLM-based rerank + filter — a hybrid of chapter 2's rerank and strategy 2's compression.
llm_rerank = LLMRerank(top_n=10)

query_engine = index.as_query_engine(
    similarity_top_k=100,
    node_postprocessors=[llm_rerank, sentence_optimizer],
)
```

Postprocessors compose left-to-right — the first one runs first. The pattern above reranks with an LLM, then sentence-filters within the survivors. That is a real compression pipeline expressed in idiomatic LlamaIndex.

## The LangChain `LLMChainExtractor` pattern

[LangChain documentation](https://python.langchain.com/docs/how_to/contextual_compression/) organises compression around a **`ContextualCompressionRetriever`** that wraps a base retriever.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=your_hybrid_retriever,
)

compressed_docs = compression_retriever.get_relevant_documents(query)
```

The `LLMChainExtractor` uses a prompt very similar to strategy 2 above. LangChain also offers:

- `EmbeddingsFilter` (strategy 1) — cheap similarity filtering.
- `LLMChainFilter` — an LLM binary yes/no filter (keep the whole chunk or drop it; strategy 2 without the sentence-extraction step).
- `DocumentCompressorPipeline` — chain multiple compressors together (e.g., cheap embeddings-filter first, expensive LLM-extractor on survivors).

The pipeline shape mirrors the LlamaIndex postprocessor chain: several compressors composed to trade quality against latency.

## When compression is worth the budget

Compression is worth the latency and cost when at least one of:

- **Your chunks are large** (500+ tokens on average) and K is large (10+). Total tokens without compression exceed budget; compression is the only alternative to shrinking K or shrinking chunks (mod-102 chunk-size decision).
- **Your corpus is noisy** — chunks with heterogeneous content, boilerplate mixed with substance, ads mixed with articles. Compression strips the boilerplate.
- **Your generator is expensive** — the per-token cost of the LLM is high enough that the compressor's overhead pays for itself in generator token savings.

Compression is *not* worth it when:

- **Your chunks are already tight** (200-token chunks of clean documentation, product records, or code). Compressing a 200-token chunk to 150 tokens is not worth the LLM call.
- **Your K is small** (K=3 or K=5) and everything fits with room to spare. Just pass the chunks through.
- **You cannot afford the citation-anchor engineering** to preserve provenance across compression. If your team is new to the framework, ship without compression first, add it once the citation-preserving path is proven.

## A defensive default: skip compression, revisit after eval

For a first RAG deployment: **do not enable compression**. Ship retrieve + rerank + naive assembly (chapter 7 without compression). Instrument for token counts, faithfulness, citation-verifiability. If your eval numbers show a specific weakness that compression could address — chunks too long for budget, boilerplate diluting answers — enable it then, with an A/B measurement.

This is the same discipline mod-101 argues for at the retriever level: build the simple end-to-end thing first, then instrument, then improve based on evidence. Compression is a common well-intentioned pre-optimisation.

## What you should be able to do after this chapter

- Name the four compression strategies (embedding filter, LLM extractive, LLM summarisation, sentence-level cross-encoder) and know which one preserves citation anchors most cleanly.
- Reason about when compression is worth the latency and cost and when it is a well-intentioned pre-optimisation.
- Deploy the LangChain `LLMChainExtractor` or LlamaIndex `SentenceEmbeddingOptimizer` on a retriever without dropping metadata / citation anchors.
- Recognise the two silent failure modes (citation anchor loss, answer-completeness loss) and design regression tests for each.
- Compose several compressors in a pipeline (cheap first, expensive on survivors) using either framework's abstractions.

## Further reading

- LangChain: [*Contextual Compression*](https://python.langchain.com/docs/how_to/contextual_compression/) — the framework's own guide, including `LLMChainExtractor`, `EmbeddingsFilter`, `LLMChainFilter`, and pipeline composition.
- LlamaIndex: [*Node Postprocessors*](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/) and [*Query Pipelines*](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/) — the framework's own guides for the equivalent patterns.
- Mod-102 of this track (Document Ingestion and Chunking) — the chunk-size decisions that determine whether compression is needed at all.
- Mod-109 of this track (Retrieval-Quality Evaluation) — faithfulness and answer-completeness metrics that catch compression regressions.
- Xu, Y., Lai, V., and Callison-Burch, C. (2023). *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.* [arXiv:2310.04408](https://arxiv.org/abs/2310.04408) — a research reference on compression for RAG, including an extractive-vs-abstractive comparison.
- Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., and Qiu, L. (2023). *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP. [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) — a specific compression technique for LLM prompts that is applicable to retrieved-context compression.
