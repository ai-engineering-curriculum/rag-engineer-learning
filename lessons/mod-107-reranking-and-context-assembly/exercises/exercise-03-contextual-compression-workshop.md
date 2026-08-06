# Exercise 03 — Contextual Compression Workshop

**Estimated effort:** ~3 hours. Python, an LLM API (OpenAI, Anthropic, or a local model with an OpenAI-compatible endpoint), a reranked top-K set from the same eval-set queries you used in exercises 01–02, and an eval harness that measures faithfulness and citation-verifiability.

## Objective

Wire up two of the three compression strategies from chapter 6 (embedding-filter + LLM extractive) on your reranked top-K set, measure the trade-off (tokens saved vs. quality lost vs. citation-anchor preservation), and produce a defended verdict: **for this corpus and this K, is compression worth it?**

The exercise centers on chapter 6's two silent failure modes — citation anchor loss and answer-completeness loss. You will explicitly test for both, not just look at aggregate quality.

## Prerequisites

- Chapter 6 read (compression strategies and failure modes).
- Chapter 7 read (context assembly and the `AssembledContext` shape — you need it to preserve anchors through compression).
- Chapter 8 skimmed (why long context does not obviate compression when it's needed).
- A first-stage retriever + reranker producing top-K (K=10) per eval query (from exercise 01 or 02).
- An LLM API for the extractive compressor and (optionally) for the generator.
- An eval harness with faithfulness and citation-verifiability metrics. Options: [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval](https://docs.confident-ai.com/), or a custom LLM-judge you author for this exercise.
- Optional: [LangChain](https://python.langchain.com/) or [LlamaIndex](https://docs.llamaindex.ai/) — the two frameworks that ship the compressor abstractions chapter 6 named.

## Steps

**Step 1 — Baseline: no compression (30 minutes).** Take your reranked top-K=10 per query. For each:

- Measure `total_tokens` (with the target LLM's tokenizer — `tiktoken`, `AutoTokenizer`, or the Anthropic count-tokens API).
- Feed to a generator LLM with a citation-per-claim prompt (a placeholder mod-108 pattern is fine — "cite each factual claim with the number of the source it came from").
- Measure faithfulness (does the answer stick to the retrieved context?), answer relevance, and citation-verifiability (do the citation numbers in the answer actually correspond to sources that support the cited claim?).

You will have three numbers per query: token count, faithfulness score, citation-verifiability. Aggregate over the eval set.

**Step 2 — Strategy 1: embedding-filter compression (30 minutes).** Implement or wire up an embedding-filter compressor:

- Split each chunk into sentences (or clauses if your chunks are shorter than ~200 tokens).
- Embed each sentence with the same embedder you use for retrieval.
- Score each sentence's cosine similarity to the query embedding.
- Keep sentences with similarity above a threshold (start at 0.5; adjust in stretch).
- Preserve the chunk's `SourceRef` and `anchor` — the compressed text is a subset of sentences from the same chunk.

If you use LangChain, `EmbeddingsFilter` gives you this shape out of the box. If you use LlamaIndex, `SentenceEmbeddingOptimizer` is the equivalent.

Rerun the eval leg from step 1. Compare token count and quality.

**Step 3 — Strategy 2: LLM extractive compression (60 minutes).** Wire up LangChain's `LLMChainExtractor` or write an equivalent from scratch:

```
Prompt:
You are extracting relevant content from a document to help answer a user's query.
Return only the sentences from the document that directly help answer the query, verbatim.
Do NOT paraphrase, summarise, or add commentary.
If nothing in the document is relevant, return the string "NOT_RELEVANT".

Query: {query}
Document: {chunk_text}

Relevant sentences (verbatim, one per line):
```

For each of the top-K chunks per query, run this compressor. Preserve the chunk's `SourceRef` and `anchor`. Handle the `NOT_RELEVANT` case — drop the chunk from the assembly, do not include an empty anchor.

Rerun the eval. Compare token count and quality.

**Step 4 — The citation-anchor regression test (30 minutes).** This is the failure mode chapter 6 warned about. Build a regression test that:

- Takes 10 sample queries.
- For each, records the `AssembledContext` produced by (a) baseline, (b) embedding-filter, (c) LLM extractive.
- Verifies: for every anchor in the final context, the `SourceRef` is present and identical to what the reranker produced.
- Runs the generator on each and checks that every citation number in the answer maps to a real source in the assembled context.

This test should pass on all three legs. If it fails on a compressed leg, your integration has the anchor-loss bug — fix it before proceeding.

**Step 5 — The answer-completeness regression test (30 minutes).** Take 5 queries where the *baseline* (no compression) produced a correct, well-cited answer. Rerun those through the compressed legs. Are the answers still correct? Are any citations dropped that were previously present?

Manual eyeball is fine at this scale. Look for the classic failure: "5% tax rate except in California, where it is 7%" — compressor keeps the first clause, drops the second, answer becomes wrong.

Report any completeness regressions with the specific query, the dropped sentence, and the compression strategy that caused it.

**Step 6 — Compose the trade-off table (20 minutes).**

| Strategy | Avg. tokens per query | Faithfulness | Citation-verifiability | Compression LLM cost per 1000 q | Completeness regressions |
|---|---|---|---|---|---|
| Baseline (no compression) | | | | $0 | 0 |
| Embedding-filter (threshold 0.5) | | | | ~$0 (compute-only) | ? |
| LLM extractive | | | | | ? |

Add a column for latency (the compression stage's p95).

**Step 7 — The verdict (20 minutes).** In a `compression-verdict.md`:

- Decide: for your corpus, K, and budget, do you enable compression in production?
- If yes: which strategy, and what is the trade-off you accepted (some citation drift for a token-cost win, or vice versa)?
- If no: what would have to change for you to reconsider (larger K, larger chunks, larger corpus, higher LLM cost)?
- Chapter 6 recommended "skip compression, revisit after eval" as a defensive default. Did your data support or contradict that guidance?

## Acceptance criteria

- Baseline + at least two compression strategies measured on the same eval set.
- Per-strategy metrics: token count, faithfulness, citation-verifiability, latency, cost.
- Citation-anchor regression test authored and passing on all legs (or a documented bug + fix).
- Answer-completeness regression test done with at least 5 manually-checked queries.
- Written verdict with an explicit "would-reconsider" clause.
- Framework use — LangChain, LlamaIndex, or hand-rolled — is fine, but the citation-preservation shape from chapter 6 (SourceRef preserved verbatim, NOT_RELEVANT handled) must be respected.

## Stretch goals

- **Strategy 3: LLM summarisation.** Implement the summarisation compressor (chapter 6 strategy 3). Measure the citation-verifiability trade-off explicitly — this strategy is expected to hurt verifiability. Confirm.
- **Compressor pipeline.** Cheap embedding-filter first (drop irrelevant chunks entirely), expensive LLM extractive on survivors. Measure whether the pipeline recovers most of the quality of LLM-only at meaningfully lower cost.
- **Per-chunk-size analysis.** Split your eval into queries whose retrieved chunks are short (< 200 tokens) vs. long (> 500 tokens). Is compression worth it uniformly, or only for the long-chunk bucket?
- **Compression threshold sensitivity.** For the embedding-filter, sweep the threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}. Plot faithfulness and token-count against threshold. Where is the elbow?
- **Diagnostic on the compression ratio.** Log the compression ratio per chunk (compressed_len / original_len). Are there specific chunks where the ratio is < 5% (i.e., compressor threw out most of the content)? Are those chunks systematically low-relevance in your reranker, or is the compressor over-aggressive?

## Notes

- Test the citation-anchor path on the very first working query, before any large batch runs. This is the single most common bug at this layer — catching it early avoids re-running 200 queries with broken anchors.
- Compression is not just about cost — chapter 6 argued it can also *hurt* by dropping answer-supporting sentences. Do not evaluate compression on token count alone.
- Frameworks (LangChain, LlamaIndex) will do the right thing if you pass `Document` / `NodeWithScore` objects through the compressor. They will do the wrong thing (drop metadata) if you compress bare strings and reattach metadata later. Use the framework's abstractions properly.
- If your generator is a small open-weight model, compression can help disproportionately — small models handle short focused contexts much better than long ones. Consider re-running the K sweep from exercise 02 with compression enabled, especially if your generator is not a top-tier long-context model.
- Chapter 6's "defensive default: skip compression" is a real recommendation. If your data supports it, do not enable compression just because you built it in this exercise. Ship what your numbers justify.
