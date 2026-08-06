# Chapter 9 — The Boundary to `applied-ai-engineer`: The AssembledContext Contract

## Why this chapter exists

Mod-106 chapter 7 drew a boundary at the `retrieve()` call: the peer track sees a `RetrieverResponse` and does not know how the reranker, fusion, or filtering inside was implemented. This chapter draws the analogous boundary one stage further down the pipeline, at the point where **the assembled context** crosses over to the generator.

The peer track — `applied-ai-engineer`, level 30 — owns the LLM prompt shape and the product surface. Their mod-103 (in the peer track's curriculum) covers *consuming a reranked, assembled context* at the application layer. This module (mod-107) is the other side: *producing* that context.

Between the two: the `AssembledContext` object from chapter 7, plus a small set of contract clauses. This chapter names both.

## Where this module sits in the level-30 division

Two peer tracks at level 30 work on RAG stacks:

- **rag-engineer (this track).** Everything from ingestion (mod-102) through context assembly (this module). The "how the context is produced" layer.
- **applied-ai-engineer (peer, level 30).** The LLM prompt, the generator, the user-facing experience, product-level eval (RAG triad at response granularity), fallbacks and retries at the product layer.

Mod-108 of *this* track (Retrieval-Specific Prompting) is a shared vocabulary layer — the prompt patterns that depend on the retriever's output shape (per-claim citation, refusal-when-no-context, contradiction handling). Everything else about the prompt (personality, style, memory, few-shot examples for non-retrieval tasks) is the peer track's.

The interface that separates the two:

```
                     rag-engineer (this track)                         applied-ai-engineer (peer)
                                                                      
   ingest → chunk → embed → index → retrieve → rerank →                LLM prompt → generate → 
   compress → assemble   ──────►    AssembledContext    ──────►        parse cites → render →
                                                                       telemetry → feedback loop
                                                                      
                                     (this chapter defines the interface)
```

Mod-106 chapter 7 had a version of the same picture at the `retrieve()` boundary. That version was for stacks where the peer track does the reranking themselves (a valid choice for very small products where the two roles are held by one person). At scale, in a mature team, the boundary moves down to `AssembledContext` — the rag-engineer owns rerank + compress + assemble.

## The `AssembledContext` interface

The chapter 7 shape, restated as the contract:

```python
@dataclass
class SourceRef:
    doc_id: str
    chunk_id: str
    url: str                  # canonical URL of the source
    title: str                # human-readable title for the peer track's UI
    section: str | None       # e.g., "§4.2 Deployment"
    page: int | None
    updated_at: str           # ISO-8601 date; freshness signal
    metadata: dict            # arbitrary structured fields the peer track can render or filter on

@dataclass
class AssembledChunk:
    anchor: str               # "[1]", "[Source A]" — what the LLM cites
    text: str                 # the (possibly compressed) chunk text the LLM sees
    source: SourceRef         # provenance; NOT shown to the LLM directly, used by the peer for UI/telemetry
    rerank_score: float       # for peer-side logging / debugging; not user-facing

@dataclass
class AssembledContext:
    query: str
    query_id: str             # UUID for traceability across the pipeline
    chunks: list[AssembledChunk]
    total_tokens: int         # measured with the target model's tokenizer
    diagnostics: dict         # reranker model + version, K, compression on/off, per-stage timing
```

Two design decisions in this shape worth defending to the peer track:

1. **`anchor` and `source` are decoupled.** The anchor is what the LLM sees and cites; the `SourceRef` is what the peer's UI renders. This lets the rag-engineer change the anchor scheme (e.g., from `[1]` to `[Source A]`) without touching the peer's citation-rendering code, and lets the peer render new source fields (add a "verified-source badge") without touching the rag-engineer's context-generation code.

2. **`total_tokens` is measured, not estimated.** The peer track uses this to compose the final LLM prompt (system + question + context) within the model's context window. A wrong number causes the peer's prompt to overflow silently. Measure with the target model's actual tokenizer.

## The four contract clauses

Analogous to mod-106 chapter 7's clauses, but for the reranker + assembler:

### Clause 1: precision SLO on a stable eval set

- **rag-engineer commits to:** `nDCG@K ≥ X` on the versioned eval judgment set, plus a citation-verifiability SLO (`≥ Y%` of chunks in `AssembledContext` are answer-supporting per human-verified labels). Concrete example: "nDCG@10 ≥ 0.75 and citation-verifiability ≥ 85% on eval-set v3."
- **applied-ai-engineer commits to:** using the anchors provided. The peer's prompt (mod-108) must instruct the LLM to cite by anchor; if the peer switches the LLM to a model that ignores the anchor scheme, that is not a rag-engineer regression.

Mod-109 owns the eval methodology; mod-111 owns the eval-set versioning.

### Clause 2: latency contract

- **rag-engineer commits to:** `p95(retrieve + rerank + compress + assemble) ≤ Z_p95` and `p99 ≤ Z_p99` at a stated QPS. Concrete example: "p95 ≤ 800 ms, p99 ≤ 1500 ms up to 50 QPS on a warm cache." The latency budget worksheet in chapter 5 is where these numbers come from.
- **applied-ai-engineer commits to:** timeout handling. If the assembly stage times out, the peer track handles it (fallback message, retry-on-simpler-K, or skip the retrieval and answer from parametric knowledge with a disclaimer).

### Clause 3: `total_tokens` accuracy contract

- **rag-engineer commits to:** `AssembledContext.total_tokens` is measured with the tokenizer of the model named in the request (or the peer specifies which tokenizer to use). The count is exact for the `chunks[].text` fields, not an estimate.
- **applied-ai-engineer commits to:** budgeting the LLM prompt around the reported `total_tokens`. The peer does not need to independently count the assembled context; if the rag-engineer says 4200, they use 4200.

This clause exists because a bug at this seam is invisible until the peer's LLM call fails with a context-length error in production. Measuring properly is worth the small extra cost.

### Clause 4: citation-anchor stability contract

- **rag-engineer commits to:** every chunk in `AssembledContext.chunks` has a unique `anchor` within the response, and the `anchor` → `SourceRef` mapping is stable within the response (i.e., the peer track can look up "which source is `[3]`?" by index into `chunks`).
- **applied-ai-engineer commits to:** treating the anchor as opaque. The peer's prompt tells the LLM to use whatever anchor was assigned; the peer's citation-parsing code maps back to the `SourceRef` by looking it up in `chunks`.

The reason this is a clause: the LLM will invent citations if the anchors are inconsistent across turns, ambiguous within a turn, or referenced by something other than what the prompt asks for. Making the anchor stable is the rag-engineer's job; making the LLM use it is the peer track's job.

## What the peer track owns downstream

Once `AssembledContext` crosses the interface, the peer track's level-30 deliverables include:

- **The full LLM prompt.** System prompt (personality, safety, constraints), the retrieval-specific prompt patterns (mod-108), few-shot examples for non-retrieval task types.
- **Citation parsing and rendering.** Extract `[1]`, `[2]` from the LLM's response, map back to `AssembledChunk.source`, render "Source: [Title, link]" in the UI.
- **Refusal and fallback logic.** "No relevant context" handling, low-confidence disclaimers, cross-checking against parametric knowledge.
- **RAG-triad eval at the response granularity.** Faithfulness (answer sticks to the retrieved context), answer relevance, context precision/recall on the *whole response*, not just the retriever's isolated nDCG. Mod-109 handles the retriever-side; the peer's eval framework (RAGAS, TruLens, DeepEval) handles the response-side. See the `ai-eval-engineer` peer track for eval infrastructure.
- **Product-facing observability.** Session-level telemetry, click-through-to-citation rate, user feedback ("was this answer helpful?"), and the human-in-the-loop labeling flow that generates data for both tracks to improve.
- **Application caching, retry, and fallback.** The rag-engineer publishes SLOs; how the app *reacts* to violations is the peer track's product decision.

## What this module owns upstream

Everything from mod-102 (chunking) through this module (rerank + compress + assemble). Concretely for this module's contribution:

- **Reranker choice, version, deployment, and monitoring.** Which model, at what precision, on what hardware, with what drift monitors (mod-111).
- **The N-K funnel calibration.** N, K, latency budgets, per-query overrides.
- **Compression on/off decisions and compressor tuning.**
- **Dedup strategy (`doc_id` cap vs. MMR vs. content-hash).**
- **Ordering strategy (best-first, first-and-last, structural).**
- **Anchor scheme.**
- **Token budgeting and reserved answer budget.**

Everything on that list is a rag-engineer decision. The peer track's job is to *use* the resulting `AssembledContext`, not to co-design it.

## Three anti-patterns at this boundary

### 1. Peer track re-ordering the chunks in the prompt

The rag-engineer chose first-and-last emphasis after measuring lost-in-the-middle on the eval set (chapter 7, 8). The peer track "cleans up" by re-sorting chunks best-first in the prompt because it "looks nicer" — and answer quality drops. Solution: the assembly order is the rag-engineer's. The peer's prompt inserts `chunks[]` in the given order, verbatim.

### 2. Peer track truncating the assembled context

The peer's prompt runs long; they truncate `chunks[-1]` to fit. This defeats the token-budget contract and silently drops the "second-strongest" chunk from the first-and-last ordering. Solution: if the peer's prompt overhead grew, either the rag-engineer reduces the assembly budget (renegotiate) or the peer trims the system prompt.

### 3. Peer track re-fetching sources based on anchors

The peer's UI, on click of `[3]`, calls the retriever again to "get the latest version of the source." The retrieved source at click time may differ from the source that grounded the LLM's answer. Solution: the `SourceRef` is a snapshot; the peer's UI resolves clicks against the snapshot's URL, not by re-retrieving.

## The one-page contract you put on a wiki

The template that ships alongside a mod-107-shaped assembler:

> **rag-engineer (level 30, this track) owns:** everything from ingestion through assembly — ingest and chunking; embedding selection; vector DB and ANN tuning; Lucene-family index authoring; hybrid retrieval, fusion, rewriting, routing, filter composition; **reranking family choice and tuning; contextual compression; dedup, ordering, citation-anchor assignment, and token budgeting**; retrieval-quality eval on judgment sets; MCP-exposure of retrievers as tools; retrieval MLOps.
>
> **applied-ai-engineer (peer, level 30) owns:** LLM prompt shape and generator configuration; citation parsing and rendering at the UI; refusal / fallback / disclaimer logic; RAG-triad eval at the response granularity; user-facing telemetry and feedback loops; product-side caching, retry, and fallback behavior; the "when to use retrieval at all" product decision.
>
> **The interface between them:** `AssembledContext` with the `chunks`, `total_tokens`, `diagnostics` fields as defined in this chapter; a precision SLO (nDCG@K plus citation-verifiability) on a shared eval set; a stated p95/p99 latency contract for the whole retrieve-through-assemble stage; a `total_tokens` accuracy contract; a citation-anchor stability contract.
>
> **Shared eval set** with per-query labels at chunk granularity (for the rag-engineer's precision + citation-verifiability metrics) and response granularity (for the peer's RAG-triad). Joint eval reviews on the same cadence as mod-106's boundary.

## The two hand-offs this module makes

- **To mod-108 (Retrieval-Specific Prompting).** The `AssembledContext` shape *is* mod-108's input. Mod-108 builds the prompt patterns (per-claim citation, refusal on empty context, contradiction handling) that consume it. Even though mod-108 is in this track (not the peer track), it lives on the *prompt* side of the boundary; the peer track integrates it into the full application prompt.
- **To mod-111 (Retrieval MLOps).** The reranker + compressor + assembler are versioned, deployed, monitored, and drift-detected under mod-111. Every clause 1–4 above is measured continuously; regressions page an oncall who knows this stack.

## What you should be able to do after this chapter

- State the `AssembledContext` interface (chunks with anchor + text + `SourceRef`, `total_tokens`, `diagnostics`) and defend the anchor-vs-`SourceRef` separation.
- Author the four contract clauses (precision SLO, latency SLO, `total_tokens` accuracy, anchor stability) with concrete numbers for a real project.
- Recognise the three boundary anti-patterns (peer re-ordering, peer truncation, peer re-fetching sources at click time) and correct each.
- Draft the one-page ownership contract naming rag-engineer vs. applied-ai-engineer deliverables with named owners.
- Point to where this module hands off — mod-108 for prompt patterns that consume `AssembledContext`, mod-111 for the MLOps that keep the SLOs green.

## Further reading

- The track's [`CURRICULUM.md`](../../CURRICULUM.md) — the module map and level-ladder positioning.
- The track's [`PREREQUISITES.md`](../../PREREQUISITES.md) — the peer / adjacent-track links.
- [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) — the peer track this chapter marks the boundary to.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the response-granularity eval framework the peer track uses.
- Trulens: [*RAG Triad evaluation guide*](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/) — the peer-track eval methodology reference.
- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — for the shape of the contract clauses.
