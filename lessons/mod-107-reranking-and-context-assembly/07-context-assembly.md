# Chapter 7 — Context Assembly for the Generator

## Why this chapter exists

The reranker gave you a top-K ordered list. The compressor (optionally) shrank each chunk to its on-topic sentences. This chapter is the last step before the LLM: **assembly** — how you order the chunks, how you deduplicate them, how you attach citation anchors the generator can quote back, and how you fit the whole thing inside a token budget with room for the answer.

Chapter 1 named this as the fourth lever of the module. Chapter 6 warned that compression can silently break citations if you are not careful; this chapter is where citation-preservation actually lands, alongside three other assembly jobs.

## The four jobs of assembly

1. **Deduplicate** — collapse chunks that carry near-identical content.
2. **Order** — decide the sequence in which the LLM sees the chunks.
3. **Attach citation anchors** — stable, unambiguous handles the generator can quote back.
4. **Budget** — fit inside (context window) − (system prompt + question + reserved answer).

Each is a small standalone problem with a defensible default. Getting all four right is what separates a level-30 assembler from a "just concat the reranked list" first pass.

## Job 1: deduplication

The retrieved-then-reranked set routinely contains near-duplicates. Common sources:

- **Adjacent chunks of the same document** — chunking with overlap (mod-102) means two chunks may share 20–50% of their tokens.
- **The same passage indexed under different chunk IDs** — a rerun of ingestion with a different chunker, or a documentation page that duplicates a passage across sections.
- **The same fact stated in near-identical language across sources** — release notes cross-published to two different sites; Stack Overflow answers copy-pasting from documentation.

Each duplicate wastes context tokens and pushes a genuinely-different chunk out of the top-K.

The three dedup approaches:

### a. By `doc_id`

Keep the highest-scoring chunk per document. Simplest; safe when your goal is "cover many sources"; sometimes wrong when a single document has two different relevant sections (e.g., a long RFC where §4 and §7 both address the query).

Rule of thumb: cap chunks-per-document at 2 or 3, not 1. This gives you dedup benefit without losing multi-section coverage.

### b. By content hash

Normalise (lowercase, strip whitespace, remove punctuation), hash, drop duplicates. Catches exact-duplicate near-duplicates but is fragile — a single-character difference (a typo, a different date) survives.

### c. By near-duplicate similarity (MMR)

Maximal Marginal Relevance ([Carbonell & Goldstein 1998](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)) — iteratively pick the highest-scoring chunk whose embedding-similarity to the already-picked chunks is below a threshold.

```python
def mmr(candidates, query_embedding, lambda_=0.7, k=10):
    """
    candidates: list of (chunk, embedding, rerank_score)
    lambda_: trade-off between relevance and diversity (1.0 = pure relevance, 0.0 = pure diversity)
    """
    selected = []
    remaining = candidates.copy()
    while remaining and len(selected) < k:
        best_score = -float("inf")
        best_idx = None
        for i, (chunk, emb, rerank_score) in enumerate(remaining):
            if not selected:
                score = rerank_score
            else:
                max_sim_to_selected = max(cosine(emb, s[1]) for s in selected)
                score = lambda_ * rerank_score - (1 - lambda_) * max_sim_to_selected
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(remaining.pop(best_idx))
    return selected
```

MMR is the standard for diversifying a retrieved-then-reranked set when the corpus has known redundancy. LlamaIndex ships `MMRPostprocessor`; LangChain's `MaxMarginalRelevanceRetrievalMixin` provides it at the retriever level.

**When to use which:** by-`doc_id` cap for most stacks; MMR when the corpus has known near-duplicate redundancy (multi-site docs, forum reposts); content-hash when the sources publish exact copies (aggregated news).

## Job 2: ordering

The reranker's output is best-first. That is the right order for many stacks and the wrong order for a few. The three orderings that matter:

### a. Best-first (default)

The top-scoring chunk goes first. Straightforward; matches the reranker's intent. Vulnerable to lost-in-the-middle (chapter 8) if the context is long.

### b. First-and-last emphasis

Place the top chunk at the *end* of the context (or split: top chunk first *and* last). Motivated by the Liu et al. 2023 result that models attend more reliably to the beginning and end of long contexts. Practical shape: for K > 5, put the top-2 chunks first, then chunks 3..(K−1) in descending order, then repeat the top chunk at the end (with a marker so the generator knows it's a repeat).

Do not do this for K ≤ 5 — the context is short enough that ordering effects are small; repeating chunks wastes tokens.

### c. Document-structural ordering

For long-document RAG (single long source, retrieval across sections of it): order the chunks by their position in the document (Introduction → §2 → §4.2 → Conclusion), not by rerank score. Motivation: the generator can read the chunks *in the order the source presents them*, which matches the human reading experience and often yields more coherent citations.

Also worth it for multi-source assemblies with a natural temporal order (release notes across versions, timeline of an incident, chronological research literature): order by date.

The three orderings pair with different query shapes:

- Focused Q&A ("what is X?") → best-first, K small.
- Long-context synthesis ("summarise §4.2 through §7 of this paper") → structural.
- Multi-source comparison ("compare A and B") → best-first with grouping (all A chunks together, then all B chunks together).

## Job 3: citation anchors

Every chunk in the assembled context needs an anchor the generator can quote back. The two shapes in common use:

### a. Numbered inline markers

```
[1] Paris is the capital of France...
[2] The Eiffel Tower is located on the Champ de Mars...
[3] France's population is approximately 68 million...
```

The generator is prompted (mod-108 covers the prompt) to include `[1]`, `[2]` after the sentences it grounds on. Simple; works with almost every model; the anchor-to-source mapping is a separate lookup you keep in application state.

### b. Verbose per-source labels

```
[Source 1: Paris - Wikipedia, retrieved 2025-06-01, §Overview]
Paris is the capital of France...

[Source 2: France Statistical Yearbook 2024, page 42]
France's population is approximately 68 million...
```

Longer; more grounded (the generator sees the source's identity); more token cost. Worth it when the generator needs source-authority context to weigh conflicting chunks.

The interface expected by mod-108 (Retrieval-Specific Prompting) is:

```python
@dataclass
class AssembledContext:
    query: str
    chunks: list[AssembledChunk]
    total_tokens: int

@dataclass
class AssembledChunk:
    anchor: str                  # "[3]" or "[Source 3]" — the citation the LLM will quote back
    text: str                    # the chunk (compressed or not) that the LLM sees
    source: SourceRef            # the full provenance: doc URL, title, section, page, date
    rerank_score: float          # for logging; not shown to the LLM
```

The `SourceRef` is the pointer the *product surface* uses to render "Source: [title, link]" alongside the LLM's answer. That part is the peer track's job (chapter 9); the rag-engineer's job is to make sure the `SourceRef` is complete and correct.

The single most common bug at this layer: the LLM quotes `[3]` for a fact that is actually from chunk 5. Causes:

- The LLM's prompt (mod-108) doesn't tell it clearly to cite by number.
- Two adjacent chunks are near-duplicates and the LLM merges them mentally.
- The anchor numbers are assigned inconsistently across turns.

Test citation-verifiability as a first-class metric (mod-109); do not assume the LLM will use the anchor scheme just because you defined one.

## Job 4: token budgeting

The arithmetic:

```
    Available context tokens   =  model_context_window
    − System prompt                                       (typically 200–800 tokens)
    − Chat history (if applicable)                        (variable)
    − User query                                          (typically 5–100 tokens)
    − Reserved answer budget                              (typically 500–2000 tokens; 15–25% of window)
    ─────────────────────────────────────────
    Available context tokens for retrieved chunks
```

The **reserved answer budget** is the token count you leave for the LLM's generated output. If you don't reserve it, the generator's response gets truncated mid-sentence when the retrieved chunks fill the window. Practical numbers by task:

- Extractive Q&A (short answer + citation): 200–500 tokens reserved.
- Chat / conversational: 500–1000 tokens reserved.
- Synthesis / summarisation / long-form: 1500–3000 tokens reserved.

The token-counting reference tool:

- OpenAI: `tiktoken` ([openai/tiktoken](https://github.com/openai/tiktoken)) — the tokenizer used by OpenAI models. Different encodings for different model families (`cl100k_base` for GPT-3.5/4, `o200k_base` for GPT-4o).
- Anthropic Claude: use the model's tokenizer via the [Anthropic SDK's `count_tokens`](https://docs.claude.com/en/api/messages-count-tokens).
- Open models: their tokenizer via `transformers.AutoTokenizer`.

Always count with the actual tokenizer of the model you are targeting. A "1000 tokens" estimate using `tiktoken` will not match Claude's tokenizer.

### The greedy-pack algorithm

The default assembly loop:

```python
def assemble(chunks: list[RerankedChunk], budget_tokens: int, tokenizer) -> AssembledContext:
    assembled = []
    used_tokens = 0
    for i, chunk in enumerate(chunks):
        anchor = f"[{i+1}]"
        header = f"{anchor} "
        text = chunk.compressed_text or chunk.text
        chunk_tokens = len(tokenizer.encode(header + text))
        if used_tokens + chunk_tokens > budget_tokens:
            break
        assembled.append(AssembledChunk(anchor=anchor, text=text, source=chunk.source, rerank_score=chunk.score))
        used_tokens += chunk_tokens
    return AssembledContext(query=..., chunks=assembled, total_tokens=used_tokens)
```

This is the simplest correct version. Two extensions it does not include:

- **Per-chunk truncation** — if a single chunk is larger than the remaining budget, decide whether to truncate it and include the head, skip it, or route back to compression for a shrink pass. Compression + retry is often the right shape.
- **Diversity constraint** — enforce "no more than 2 chunks from the same `doc_id`" during the pack. Combines dedup-by-doc-id (job 1) with the pack loop.

## The assembled context object

A complete assembly produces this object for mod-108's prompt:

```python
AssembledContext(
    query="How do I cancel my subscription?",
    chunks=[
        AssembledChunk(
            anchor="[1]",
            text="You can cancel your subscription at any time from the Account Settings page. Cancellations take effect at the end of the current billing period.",
            source=SourceRef(
                doc_id="help-center/billing/cancel",
                chunk_id="help-center/billing/cancel#p2",
                url="https://help.example.com/billing/cancel",
                title="How to cancel your subscription",
                section="Cancelling your subscription",
                updated_at="2025-07-15",
            ),
            rerank_score=0.94,
        ),
        AssembledChunk(anchor="[2]", ...),
        AssembledChunk(anchor="[3]", ...),
    ],
    total_tokens=847,
)
```

Two properties of this object worth naming:

1. **The `anchor` and the `SourceRef` are both present.** The anchor is what the LLM sees and cites; the `SourceRef` is what the peer track's UI renders. They are decoupled — the anchor scheme can change (e.g., from `[1]` to `[Source A]`) without touching the source resolution downstream.

2. **`total_tokens` is measured, not estimated.** Assembling and reporting the wrong number will silently blow the model's context window when the system prompt or chat history is longer than a rough estimate assumed.

## Where assembly lands in the pipeline

```
    retriever (mod-106) → reranker (ch. 2–4) → compressor (ch. 6) → assembler (this chapter) → generator (mod-108)
                                                                          │
                                                                          ▼
                                                                    AssembledContext
                                                                          │
                                                                          ▼
                                                                    LLM prompt = system + question + assembled.chunks (rendered with anchors)
```

The `AssembledContext` is the payload that crosses the boundary to mod-108. Chapter 9 formalises this as the peer-track interface.

## Anti-patterns

1. **Assembling by concatenation without an anchor scheme.** The LLM sees `chunk1\n\nchunk2\n\nchunk3` and cannot cite. Add anchors.
2. **Setting the answer budget to zero.** The system happily produces truncated answers; no one notices for weeks. Reserve budget explicitly.
3. **Building the anchor from the reranker's absolute score.** `[Score 0.87]` is not a stable identifier and shifts with reranker version changes. Use sequential integers or per-source identifiers instead.
4. **Sending compressed text but citing against original text spans.** If the compressor took only sentences 3, 7, 9 of the chunk and you cite "sentence 5" you are citing something the LLM never saw. Anchor at chunk granularity, not sentence granularity, unless your compressor exposes per-sentence spans.

## What you should be able to do after this chapter

- Enumerate the four jobs of context assembly (dedup, order, citation anchors, token budget) and identify which is the fragile one (citation anchors, per chapter 6's compression discussion).
- Choose a deduplication strategy (by `doc_id`, by content hash, by MMR) for a given corpus.
- Choose an ordering (best-first, first-and-last, structural) for a given query shape.
- Compute the token-budget arithmetic for a given model and reserve an appropriate answer budget.
- Author the `AssembledContext` object shape and defend the anchor-vs-`SourceRef` separation to the peer track.
- Recognise the four assembly anti-patterns and correct each.

## Further reading

- Carbonell, J. and Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. [cmu.edu](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) — the reference for MMR.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — motivates the first-and-last ordering; chapter 8 works through it.
- OpenAI: [`tiktoken` on GitHub](https://github.com/openai/tiktoken) — the tokenizer for OpenAI models.
- Anthropic: [Count Tokens API](https://docs.claude.com/en/api/messages-count-tokens) — the tokenizer for Claude.
- LangChain: [*Retrieval + Answer Question with Sources*](https://python.langchain.com/docs/how_to/qa_sources/) — the framework's own guide on the citation-anchor problem.
- LlamaIndex: [*Citations in RAG*](https://docs.llamaindex.ai/en/stable/examples/query_engine/citation_query_engine/) — the framework's own citation-query-engine reference.
