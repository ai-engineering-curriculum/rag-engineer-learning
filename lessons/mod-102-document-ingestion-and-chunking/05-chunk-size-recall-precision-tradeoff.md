# Chapter 5 — Chunk Size: The Recall / Precision Trade-Off and Its Interactions

## Why this chapter exists

"What chunk size should I use?" is the question this module gets asked most. The tempting answer is a number. The honest answer is a *trade-off* that depends on your corpus, your embedding model's usable context, your reranker's cost, and how much of the generator's window you can afford to spend on one passage.

This chapter names the trade-off, walks its four interacting constraints, gives you sensible starting ranges to ablate around, and hands you a small evaluation harness you will use in `exercise-03`. It replaces "use 512, that is what the tutorial says" with "start at 512 for prose with an ~8K embedding model, then measure — and here is how."

## The core trade-off in one paragraph

Small chunks → each chunk is topically tight, so cosine similarity against a specific query is high (**precision-ish** goes up), and a chunk that matches leaves lots of budget for other chunks in the top-*k*. But small chunks fragment: a fact whose answer needs three neighbouring sentences to make sense is now split across three chunks, and the retriever's top-*k* may not include all three (**recall goes down** on multi-sentence answers). Large chunks → each chunk contains more context, so the retriever is more likely to include the answer *somewhere* (**recall goes up**), but its embedding is diluted by the surrounding paragraph so similarity to a specific query is weaker (**precision-ish goes down**), and each retrieved chunk spends more of the generator's context window on non-answer text (**lost-in-the-middle** risk rises; Liu et al. 2024 [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)).

Neither end wins outright. The right size is corpus-dependent, and the "right" question is often *not* a scalar chunk size but a hierarchical strategy (chapter 6) that indexes small and returns big.

## The four constraints

### 1. Embedding-model context length — the hard ceiling

Every embedding model has a maximum input length in tokens. Overrun it and the input is truncated silently (or the API errors, if you are lucky). Concrete anchors — check the model card for the current number:

- `BAAI/bge-small-en-v1.5`, `bge-large-en-v1.5`: 512 tokens.
- OpenAI `text-embedding-3-small`, `text-embedding-3-large`: 8,191 tokens ([OpenAI docs](https://platform.openai.com/docs/guides/embeddings)).
- Cohere Embed v3: 512 tokens per input ([Cohere docs](https://docs.cohere.com/docs/embeddings)).
- Voyage `voyage-3`, `voyage-large-2`: 16,000–32,000 tokens depending on model ([Voyage docs](https://docs.voyageai.com/docs/embeddings)).

The context length is a **ceiling**, not a target. A model rated for 8K tokens does not necessarily produce a *good* embedding for an 8K input — the effective information density collapses as the input gets longer, because a single fixed-size vector has to summarize more. Empirically, most bi-encoder models produce noticeably weaker embeddings past ~1K–2K tokens, even when their nominal ceiling is higher. Stay comfortably below the ceiling unless you have measured that your model handles the long inputs.

### 2. Query specificity — the granularity target

Ask "what is our parental leave policy?" and the answer wants a section-sized chunk (multiple paragraphs of policy). Ask "what is the notice period in weeks for maternity leave in Ontario?" and the answer wants a sentence.

Two implications:

- **A single corpus often has a mix of query granularities.** No single chunk size wins across all of them; this is what motivates hierarchical strategies in chapter 6.
- **The distribution of query specificity should shape the default.** If most queries are policy-summarization ("what does X say about Y"), start bigger (~800 tokens). If most are fact-lookup ("what is the number for X"), start smaller (~200–400 tokens).

### 3. Reranker cost — the retrieve-then-rerank funnel

If you have a reranker (mod-107), it re-scores the retriever's top-*N* to a top-*K*. Cross-encoders like `bge-reranker-large` pay a per-`(query, chunk)`-pair inference cost. Total cost is roughly `N * f(chunk_size)` where `f` grows more than linearly for a transformer processing longer inputs.

Concretely: doubling chunk size roughly doubles rerank latency at fixed `N`; halving chunk size lets you double `N` at the same latency budget. Bigger `N` gives the reranker more raw material to work with, which usually helps recall@K. This is the trade-off `exercise-03` will make you measure.

### 4. Generator context budget and lost-in-the-middle

The reranked top-*K* becomes the LLM's context. Two effects push toward smaller chunks:

- **Direct token cost.** Fewer tokens per prompt = lower per-request generation cost. On hosted APIs at typical prices, the difference between 4×500-token chunks and 4×2000-token chunks is real money at scale.
- **Lost in the middle** (Liu et al. 2024). Long-context LLMs give demonstrably less attention to information in the middle of the context than at the ends. Assembling four small chunks around the answer produces better generation quality than one large chunk containing the answer buried in a wall of surrounding text.

Chapter 6's small-to-big pattern is the direct response to this pair of effects: retrieve small (precision, lost-in-the-middle-friendly), return big (recall, coherence).

## Starting ranges — the numbers to ablate around

The right chunk size is corpus-dependent, but every ablation needs a starting range. These ranges are what practitioners typically test on ordinary English prose with a modern bi-encoder embedding model; they are a starting scaffold, not a recommendation.

| Corpus type | Sensible starting range (tokens) | Overlap | Notes |
|---|---|---|---|
| Handbook / policy prose | 400–1000 | 10–15% | Structural chunker at section boundary, then recursive-split large sections |
| Chat / support transcripts | 100–400 | 0–10% | Turns are already natural boundaries; small overlap unless replies span multiple turns |
| Academic / long-form articles | 500–1500 | 10–20% | Semantic segmenter or paragraph-level; larger chunks tolerate an ~8K-context embedding model |
| Reference (glossary, FAQ, KB) | 100–300 | 0–5% | Each Q/A pair is already a coherent unit — do not force a bigger window |
| Source code | 20–60 lines per function | 0–15% overlap in lines | Split at function / class; overlap in lines, not tokens |
| Tables | one chunk per table (or per row-group) + summary | 0 | Chapter 8 |

Do not pick a number off this table and ship. Do use it as the *center* of an ablation grid.

## The ablation harness — how to actually pick a size

`exercise-03` will have you build this end-to-end. The shape:

1. **Fix everything else.** Same corpus, same parser, same embedding model, same retriever (dense-only is fine at this stage), same top-*k*. If you vary chunker *and* embedding at once, you cannot attribute the delta.
2. **Build a small judgment set.** 50–100 queries. For each, list the doc(s) or chunk-id-range(s) that actually answer it. mod-109 will teach you how to build a proper judgment set; for now, hand-label from your corpus.
3. **Ablate on the axis under test.** Chunk sizes `{200, 400, 800, 1600}` tokens, overlaps `{0%, 10%, 20%}`. Twelve grid cells.
4. **Compute recall@k and nDCG@k per cell.** Use `ir_measures` from mod-101. Chart recall@10 as a function of chunk size, per overlap.
5. **Watch the shape of the curve, not one number.** A plateau tells you the trade-off is soft on your corpus; a peak tells you the sweet spot is real and worth pinning; a monotonic curve tells you your queries are all one granularity and you should redesign the judgment set.

The point of the harness is not to find "the answer." It is to make future arguments about chunk size *evidence-based on your corpus*, not tutorial-based on a demo corpus.

## Overlap: what it buys and what it costs

Overlap is the "insurance policy" against a fact landing exactly on a chunk boundary. A ~10–15% overlap is defensible on prose; 0% is defensible when boundaries are natural (chat turns, table rows, function definitions).

Costs:

- **Storage and embedding cost scale with (1 + overlap fraction).** 20% overlap means 20% more chunks embedded and stored.
- **De-duplication burden downstream.** Two adjacent chunks that share ~200 tokens of text will both score high for a query about that shared span, giving the top-*k* a duplicated view. The assembler (mod-107) needs to handle it, but the chunker should also avoid making the problem worse than necessary.

The single most common overlap pathology: `chunk_size=200, chunk_overlap=100`. That is 50% overlap. Every fact is stored twice. Do not do this by accident.

## When chunk size is the wrong knob

Sometimes changing chunk size does not help because chunk size is not the bottleneck. Two shapes to recognize:

- **The chunker itself is bad.** If the recall@k curve is flat across chunk sizes, your boundaries are probably wrong — you are splitting mid-table, mid-code, mid-sentence — and no size will save you. Fix the boundary policy (chapter 4) before tuning size.
- **The embedding model does not know your vocabulary.** A domain-specific corpus (medical, legal, quantitative finance) with a general-purpose embedding model produces embeddings where "chunk about drug interactions" and "chunk about restaurant reviews" are closer than either is to a specific query in either domain. No chunk size fixes this; fix the embedding model (mod-103).

The trade-off in this chapter is real *once the chunker and embedder are honest*. If the recall floor is ceiling-limited by upstream choices, adjusting chunk size is rearranging deck chairs.

## Design record: what to commit to source control

Whatever chunk configuration you land on, commit it as a config file, not a magic number in code:

```yaml
# chunker.yaml
chunker: llama_index.SentenceSplitter
version: 1.0.0
chunk_size: 512            # tokens, using the embedding-model tokenizer
chunk_overlap: 64
paragraph_separator: "\n\n\n"
respects_structural_boundaries: true
notes: |
  Chosen from exercise-03 ablation on 2026-08-15 sample of the HR corpus.
  Peak nDCG@10 at (size=512, overlap=64) on 63 judged queries.
```

This config is one of the inputs to the index-version key `(embedding_model, chunker_version, schema_version, corpus_snapshot)` mod-111 will build the CI/CD pipeline around.

## What you should be able to do after this chapter

- State the recall / precision trade-off of chunk size in one sentence and name the four constraints that interact with it (embedding context, query specificity, reranker cost, generator budget + lost-in-the-middle).
- Look up the max input token length of the embedding model you intend to use and set the target chunk size comfortably below it.
- Design an ablation grid (chunk sizes × overlaps × constant everything else) and know which metric (recall@k, nDCG@k) you will read off it.
- Recognize the two "chunk size is not the knob" pathologies (bad chunker; wrong embedding for the domain).
- Commit chunker configuration as a versioned artifact rather than magic numbers.

Chapter 6 shows the strategy that lets you escape the single-size trade-off entirely: index at one granularity, return at another.

## Further reading

- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
- LlamaIndex, [*Chunk Size Optimization*](https://docs.llamaindex.ai/en/stable/examples/param_optimizer/param_optimizer/) — the framework's own ablation walk-through and utilities.
- LangChain, [*Text splitting concepts*](https://python.langchain.com/docs/concepts/text_splitters/) — brief note on size trade-offs.
- OpenAI, [*Embeddings guide*](https://platform.openai.com/docs/guides/embeddings) — includes the token-length constraint for `text-embedding-3-*`.
- Cohere, [*Embed v3 docs*](https://docs.cohere.com/docs/embeddings) — token limits and length behavior for the Embed v3 family.
- Voyage AI, [*Embeddings docs*](https://docs.voyageai.com/docs/embeddings) — token limits for `voyage-3` and long-context variants.
- Anthropic engineering, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — one public engineering case study of the size-vs-context trade-off.
