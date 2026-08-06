# Chapter 7 — The Boundary to `applied-ai-engineer`: The Retriever Interface Contract

## Why this chapter exists

`rag-engineer` and `applied-ai-engineer` are peer level-30 tracks in the AI Engineering family. They both work on RAG stacks. They both use the word "retrieval." A lot of ambiguity gets baked into that overlap, and every level-30 project I have seen suffers from at least one bug rooted in "who owns the retriever?" this chapter answers with a fixed contract.

The short version:

- **rag-engineer (this track)** — *builds and tunes* the retriever. Fusion algorithm, per-arm settings, rewriting patterns, routing, filter semantics, latency and cost tuning, retrieval-quality eval (mod-109), retrieval MLOps (mod-111). Chapters 1–6 of this module are the level-30 rag-engineer's deliverable inside a hybrid retrieval stack.
- **applied-ai-engineer (peer, level 30)** — *consumes* a retriever with a fixed interface and product-surfaces it. RAG integration into a product, prompt shape at the application layer, freshness / coverage / permission contracts with the retriever, product-facing eval (RAG triad — faithfulness, answer relevance, context precision/recall), user-facing observability.

The boundary is the **retriever interface** — the shape of the call the applied engineer makes and the shape of the response they receive. This chapter names that interface, so both sides can code and test against it.

Chapters 1–6 of this module built the pipeline behind the interface. This chapter defines the interface itself.

## The retriever interface: what applied-ai-engineer sees

The peer track sees the retriever as a single opaque call with a defined contract. They do not — and should not — see the fusion algorithm, the rewriting logic, the routing decisions, or the filter internals. The interface is:

```
retrieve(query: str, filters: dict, user_context: dict, top_n: int) -> RetrieverResponse
```

Where `RetrieverResponse` is a structured object:

```python
@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float          # A fused, normalized 0–1 relevance score.
    text: str             # The chunk text (already deduplicated across sources).
    source: SourceRef     # The provenance: doc URL, section path, page, source-timestamp.
    metadata: dict        # Arbitrary structured metadata the peer track might filter on downstream.

@dataclass
class RetrieverResponse:
    query: str            # Echo of the input query.
    query_id: str         # A UUID for logging / debugging / eval-set traceability.
    chunks: list[RetrievedChunk]      # Ranked, top-N.
    diagnostics: dict     # Non-user-facing: fusion algorithm, arms used, timing per arm, filter-count.
```

The five deliverables encoded in that shape:

1. **A ranked list of chunks (not documents).** Chunking has already happened (mod-102); the peer track does not re-chunk.
2. **A fused, normalized score.** Even if the fusion is RRF internally, expose a monotonic 0–1 score. The peer track's product logic may want to threshold ("if the top score is < 0.4, show a 'no results' message"); a stable score scale is the interface for that.
3. **Provenance for every chunk.** Doc URL, section path (§ 4.2.1), page number, source timestamp. The peer track's citation-in-the-response depends on this being complete and correct.
4. **Metadata.** Any structured fields the peer track might want to display or use for downstream filtering (product, category, tags, updated_at). *Not* the internal filter fields the retriever used to shape the candidate set.
5. **Diagnostics (not user-facing).** Fusion algorithm used, arms invoked, per-arm timing, filter tightness — enough for the peer track to file a good bug report if retrieval quality regresses.

## What crosses the boundary — the four contract clauses

Four things a rag-engineer promises the applied-ai-engineer, and vice versa. Put these in a `RETRIEVER_CONTRACT.md` in the repo where the retriever service lives.

### Clause 1: recall SLO on a stable eval set

- **rag-engineer commits to:** `recall@N ≥ X` and `nDCG@10 ≥ Y` on a versioned eval judgment set (mod-109 covers the eval methodology). Concrete numbers depend on the corpus. A working example: "recall@100 ≥ 0.85, nDCG@10 ≥ 0.72 on eval-set v3." A regression below either triggers a rollback.
- **applied-ai-engineer commits to:** consuming the top-N. Not "getting cute" (asking for top-5 and complaining about missed docs, or asking for top-500 and hitting a compute limit). Any tightening of the effective *k* the peer needs is a contract renegotiation.

### Clause 2: latency contract

- **rag-engineer commits to:** `p95(retrieve) ≤ Z_p95` and `p99(retrieve) ≤ Z_p99` at a stated QPS budget. Concrete example: "p95 ≤ 80 ms, p99 ≤ 250 ms up to 200 QPS on a warm cache." Rewriting-heavy stacks (HyDE, decomposition) will have very different numbers — negotiate them explicitly.
- **applied-ai-engineer commits to:** timeout behavior. If the retriever returns nothing within the latency budget, the peer track handles that (either shows a "still loading" state, retries once, or falls back to a canned response). The rag-engineer's SLO measures the successful path.

### Clause 3: freshness contract

- **rag-engineer commits to:** how quickly a new document (or an updated document) appears in retrieval. Concrete example: "T+5 min from ingest DAG completion" or "T+1 hour for scheduled reindex." Different pipeline shapes give different numbers; state the number.
- **applied-ai-engineer commits to:** communicating freshness to the user where it matters. If the user asked about the latest release and the retriever's freshness is T+1h, the peer track's job is to say "as of 1 hour ago…" not to pretend the retrieval is real-time.

### Clause 4: permissions and ACL contract

- **rag-engineer commits to:** never returning a doc the `user_context` does not permit. Full stop. Permission filtering is applied at the retrieval layer, not the application layer.
- **applied-ai-engineer commits to:** passing a well-formed `user_context` — auth token, tenant ID, user groups, whatever the retriever expects. If the peer track is showing docs to a signed-out user, they must pass an anonymous-user context that scopes retrieval accordingly.

## What the peer track owns downstream of retrieve()

Everything the retriever hands back is the peer track's raw material for **RAG at the product surface**. Their level-30 deliverables that the rag-engineer does not do:

- **Prompt shape at the application layer.** How chunks are formatted in the LLM prompt, how citations are structured, how the system prompt constrains the response. Mod-108 of this track (Retrieval-Specific Prompting) covers *retrieval-specific* prompt patterns — citation, refusal, contradiction handling — because those interact with the retriever's output shape. Everything else (chat memory, system-prompt personality, tone) is the peer track's.
- **Product-level eval (RAG triad).** Faithfulness (does the answer stick to the retrieved context?), answer relevance (does the answer address the question?), context precision/recall (do the retrieved chunks contain the answer?) at the *whole-response* granularity — not the retriever's isolated recall@k. See `ai-eval-engineer` peer track for the evaluation methodology (RAGAS, TruLens, DeepEval) and `applied-ai-engineer` for how the eval is wired into the product's release gate.
- **User-facing observability.** Session-level telemetry (which retrievals led to which user actions), the click-through / abandon feedback loop, the human-in-the-loop labeling flow. rag-engineer emits *retriever-internal* metrics (per-arm latency, fusion-window size, filter counts); applied-ai-engineer emits *product* metrics (session satisfaction, escalation-to-human rate).
- **The application's caching, retry, and fallback behavior.** The rag-engineer publishes a retrieve() SLO; how the application handles timeouts and consistency is the peer track's product decision.

## What this track owns upstream

Everything up to `retrieve()`. The rag-engineer's level-30 responsibilities (recap from mod-101):

- **Ingestion and chunking** (mod-102). Not "just PDFs" — the whole pipeline from source to chunk with the retrieval-critical metadata attached.
- **Embedding selection and benchmarking** (mod-103). Choosing an embedder, benchmarking on domain-relevant BEIR/MTEB subsets, deciding when to fine-tune vs. buy hosted.
- **Vector database and ANN tuning** (mod-104). pgvector / Pinecone / Weaviate / Qdrant / Milvus / FAISS / Chroma; HNSW / IVF-PQ / ScaNN parameters; the filter-and-ANN semantics chapter 6 relied on.
- **Lucene-family and LTR** (mod-105). Elasticsearch / OpenSearch / Solr / Vespa authoring; BM25 and analyzer authoring; classical LTR on the lexical arm.
- **Hybrid retrieval, fusion, rewriting, routing, filter composition** — mod-106 (this module).
- **Reranking and context assembly** (mod-107). Cross-encoders, ColBERT rerankers, dedup / ordering / citation-anchor packing before the LLM sees the context.
- **Retrieval-specific prompting** (mod-108). Per-claim citation, refusal-when-no-context, contradiction handling. The prompt patterns that *depend on* the retriever's output shape.
- **Retrieval-quality evaluation** (mod-109). Recall@k / MRR / nDCG / precision@k on labeled judgment sets; paired significance testing; eval-set versioning.
- **Retrieval orchestration and MCP-exposure** (mod-110). Where the retriever is composed with other retrievers, exposed as an MCP tool to downstream agents, and hardened for that surface.
- **Retrieval MLOps** (mod-111). Index versioning, retriever CI, drift monitoring, cost/latency tuning.
- **Enterprise retrieval** (mod-112). ACL enforcement (referenced in chapter 6), structured retrieval (text-to-SQL), graph RAG, multimodal retrieval.

Every one of those sits on this side of the boundary. The peer track does not need to know how you tune HNSW `M` or which BM25 `k1` you picked; the rag-engineer does not need to know which React component displays the retrieved snippet.

## Three anti-patterns at the boundary

1. **Leaking fusion algorithm into the interface.** "The retriever returns RRF scores; the app should threshold on RRF score." That ties the app to the fusion internals. If you switch RRF for weighted-sum later, the app breaks. **Normalize the score to a 0–1 scale before the interface**; keep the internal algorithm private.
2. **Applied-ai-engineer re-implementing filters at the app layer.** "The retriever returns too much; I'll filter to product=X in the app." This bypasses the retriever's own ACL and filter enforcement, may be incorrect (filters that need index-time knowledge), and definitely breaks the recall SLO because it changes the effective *k*. Filters go in the retriever interface, not in the app.
3. **rag-engineer's eval set and applied-ai-engineer's eval set diverging silently.** Retrieval recall@100 goes up 4 points on the rag-engineer's judgment set; product answer quality drops 2 points on the peer's eval set. Both teams are "right"; nobody caught the mismatch. Solution: a **joint eval set** with shared queries and matched labels at both granularities (chunk-level for rag-engineer, response-level for applied-ai-engineer). Mod-109 and the `ai-eval-engineer` track cover the joint-eval methodology.

## The one-page contract you put on a wiki

The template that ships with every mod-106-shaped retriever:

> **rag-engineer (level 30, this track) owns:** ingest and chunking; embedding selection and benchmarking; vector DB and ANN tuning; Lucene-family index authoring; hybrid retrieval fusion, query rewriting, semantic routing, filter composition; reranking and context assembly; retrieval-specific prompting patterns; retrieval-quality evaluation on judgment sets; MCP-exposure of retrievers as tools; retrieval MLOps (index versioning, CI, drift monitoring); enterprise retrieval concerns (ACL, structured, graph, multimodal).
>
> **applied-ai-engineer (peer, level 30) owns:** application-layer RAG integration; product-facing prompt shape (personality, tone, memory); RAG-triad evaluation at the response granularity; user-facing telemetry (session satisfaction, click-through, feedback loops); product-side caching, retry, and fallback behavior; the "when to use retrieval at all" product decision.
>
> **The interface between them:** `retrieve(query, filters, user_context, top_n) -> RetrieverResponse` with the fields defined in this chapter, a versioned recall SLO on a shared eval set, a stated p95/p99 latency contract, a freshness commitment, and a permissions contract that is fully enforced retriever-side.
>
> **Shared eval set** with per-query labels at chunk granularity (for rag-engineer's recall@k) and response granularity (for applied-ai-engineer's RAG-triad). A quarterly joint eval review compares both metrics; a movement in one without the other is a signal to investigate.

Every rag-engineer / applied-ai-engineer pairing at level 30 should have that on a wiki with named owners in each role before the first end-to-end user query flows.

## What this module has handed off, and what the rest of the track picks up

This module is one lever in a longer retrieval track. Explicit hand-offs:

- **To mod-107 (Reranking and context assembly).** The top-100 fused candidates from this module are the reranker's input. Chapter 1 of this module named recall@100 as the metric this module optimizes; mod-107 optimizes precision@10 on those top-100.
- **To mod-108 (Retrieval-specific prompting).** The retriever's `RetrieverResponse` shape (specifically, the provenance and metadata fields) is what mod-108's citation and refusal patterns depend on.
- **To mod-109 (Retrieval-quality evaluation).** All the "measure it on an eval set" hand-waves in this module resolve into rigorous methodology in mod-109.
- **To mod-110 (Orchestration and MCP-exposure).** The `retrieve()` interface this chapter names is the shape mod-110 wraps in an MCP tool for downstream agents.
- **To mod-111 (Retrieval MLOps).** The versioning, CI, and drift monitoring of the whole pipeline behind `retrieve()`.
- **To mod-112 (Enterprise retrieval).** ACL enforcement, structured retrieval, graph, and multimodal — the enterprise-scale versions of chapter 6's concerns.

Every one of those modules operates *behind* the interface this chapter defines. From the peer track's perspective, they are internal to the retriever.

## What is NOT in this chapter (and where it goes)

- **The eval methodology for the shared eval set (paired significance, held-out design, judgment authorship)** — mod-109.
- **Neural rerankers as the mod-107 counterpart to LTR** — mod-107.
- **Retrieval-specific prompt patterns (citation, refusal, contradiction)** — mod-108.
- **MCP-tool-shape and multi-tool composition** — mod-110.
- **The peer track's product-facing eval (RAG triad, RAGAS, TruLens)** — `applied-ai-engineer` and `ai-eval-engineer` tracks.

## What you should be able to do after this chapter

- Name the retriever interface: `retrieve(query, filters, user_context, top_n) -> RetrieverResponse` with a defined `RetrievedChunk` shape.
- Author the four contract clauses (recall SLO, latency SLO, freshness contract, permission contract) for a real project with concrete numbers.
- Recognise the three boundary anti-patterns (leaking fusion internals, app-layer re-filtering, diverging eval sets) and know how to correct each.
- Draft the one-page ownership contract with named owners in each role and a pointer to the shared eval set.
- Point to the specific downstream module in this track that picks up each thread this chapter deferred.

## Further reading

- The track's own [`CURRICULUM.md`](../../CURRICULUM.md) — the module map and level-ladder positioning of `rag-engineer` vs. `applied-ai-engineer` and the other peers.
- The track's own [`PREREQUISITES.md`](../../PREREQUISITES.md) — the explicit peer / adjacent-track links.
- [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) — the peer track this chapter marks the boundary to. Reading their level-30 module on retriever consumption is the fastest way to see the other side of the interface.
- Anthropic: [*Model Context Protocol (MCP) specification*](https://modelcontextprotocol.io/specification) — the retriever-as-tool shape mod-110 covers; useful for the reader who wants to see the interface's downstream extension.
- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — the reference on how to state and defend the recall / latency / freshness SLOs above.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — for the reader who wants to see the RAG-triad evaluation framework the peer track uses at the response granularity.
