# Chapter 5 — The Retrieve-Top-N → Rerank-to-K Funnel

## Why this chapter exists

Chapters 2–4 told you how a reranker works. This chapter is about the two numbers you set on the pipeline:

- **N — the size of the candidate pool the retriever hands to the reranker.**
- **K — the number of chunks the reranker forwards to the context assembler.**

These are the two knobs that trade off quality against latency and cost. There is a diminishing-returns curve for both. There is a p95 latency budget that constrains where you can sit on that curve. Chapter 1 named this as one of the four levers of the module; this chapter is the arithmetic and the measurement recipe.

The single most common mistake in a first RAG deployment is to pick N=100 and K=10 because they feel "reasonable" and never revisit them. The right N and K depend on the corpus, the reranker, the eval metric, and the latency budget — and they usually change when any of those change.

## The four constraints on N and K

Set N and K by trading four things off:

1. **Quality (recall at N, nDCG@K).** Bigger N means the reranker sees more candidates, which is only useful if the extra candidates contain the right chunks. Bigger K means the generator sees more chunks, which helps up to the point where lost-in-the-middle (chapter 8) or noisy dilution starts hurting.
2. **Reranker latency at the p95.** Cross-encoder latency is roughly linear in N; a MiniLM at N=100 is ~50 ms, at N=200 is ~100 ms, at N=500 is ~250 ms. The reranker is usually the dominant latency in the whole retrieve-and-rerank stage; N goes up, p95 goes up.
3. **Cost per query.** For hosted API rerankers, cost scales with N (or with total-input-tokens). For self-hosted rerankers, cost scales with GPU-hours, which scales with N times QPS. Either way, bigger N means bigger bill.
4. **Downstream context budget.** K feeds into the token budget for the generator's context window. K=20 chunks × 400 tokens = 8000 tokens; that plus the system prompt, the user's question, and the reserved answer budget must fit the model's context window. Chapter 7 walks through the arithmetic.

The chapter's central lesson: **N is a quality-latency-cost knob; K is a quality-context-budget knob.** They are related (K ≤ N, obviously) but tuned against different constraints.

## The diminishing-returns curve for N

Consider a fixed first-stage retriever (BM25 + dense with RRF, per mod-106) that produces a recall@N curve like this on your eval set:

```
    recall@N
      ▲
      │            ── plateau ──
  0.95│                     ●───●───●
      │              ●
  0.90│         ●
      │
  0.85│    ●
      │
  0.80│●
      │
      └───────────────────────────────►
        20   50  100  200  500  1000    N
```

The curve is *concave*. Recall rises fast up to ~100, more slowly up to ~200, and is essentially flat past ~500. That shape is corpus-dependent — very heterogeneous corpora need larger N; narrow corpora saturate earlier — but the concavity is universal.

The reranker's job is to lift precision@K over that candidate pool. If the retriever's recall@N=100 is 0.92 and recall@N=200 is 0.95, going from N=100 to N=200 adds 3 percentage points of *possible* recall but doubles the reranker's compute. Whether that trade is worth it depends on:

- How much the reranker converts extra recall into extra precision@K. A strong reranker with a good candidate at rank 150 in the retriever pool can pull it into the top-K; a weak reranker cannot.
- How steep your latency budget is. If p95 is already at the ceiling, N=200 is not on the table.

**The measurement recipe:** compute end-to-end nDCG@K (not just retriever recall) for N ∈ {20, 50, 100, 200, 500}. Plot nDCG@K vs. N. The elbow of that plot is the operating point.

## The diminishing-returns curve for K

K has a different curve — and, crucially, K can *decrease* quality past a point.

```
    end-to-end answer quality (RAG-triad or task-specific)
      ▲
      │       peak
  0.9 │         ●───●
      │      ●         ●
  0.8 │   ●               ●
      │                       ●
  0.7 │                          ● ← lost-in-the-middle degradation
      │
      └───────────────────────────────►
         1    3    5   10   20   50    K
```

Two forces:

- **Below the peak (K too small):** the answer is missing chunks that support it. Recall of *relevant material at the generator* is limited; the generator can't cite what it can't see.
- **Above the peak (K too large):** the generator sees more content but attends less reliably to each chunk. Lost-in-the-middle (chapter 8) is the direct name for this effect. Noisy dilution — filler chunks that are only marginally relevant — pushes the model to summarise vaguely rather than cite specifically.

The peak for K is usually in the 3–15 range for standard RAG. K=1 is right for very focused tasks where a single chunk is expected to answer (customer-support macros, product lookups). K=20+ is right for research-summary tasks where the generator is *expected* to synthesise across many sources.

Do not use the same K everywhere. K should be a per-task setting: shorter for focused Q&A, longer for synthesis. If your retriever is exposed as an MCP tool (mod-110), let the caller pass K.

## The latency budget worksheet

The p95 budget for the whole retrieve + rerank + assemble stage. A working example:

```
        Total user-visible latency budget for the RAG turn:     3000 ms  (p95)
    − Generator (LLM) latency at the target output length:    1800 ms
    − Network + application overhead:                          200 ms
    ────────────────────────────────────────────────────────────
        Retrieve + rerank + assemble budget:                  1000 ms
```

Divide the 1000 ms:

```
    Retrieval (BM25 + dense in parallel + RRF):                 100 ms
    Reranker at N=100, top_k=10:                                150 ms
    Contextual compression (chapter 6, if used):                200 ms
    Context assembly (chapter 7):                                10 ms  (cheap)
    Slack for retries, cold-cache, queueing:                    540 ms
    ────────────────────────────────────────────────────────────
    Sum:                                                       1000 ms
```

The slack matters. Individual step p95s add worse than in average; you want a full budget with ~20–40% headroom, or your end-to-end p95 will blow through the target on any bad hour.

If the numbers do not fit — a common outcome on a first pass — you have four levers, in order of typical usefulness:

1. **Shrink N.** Halve N and re-measure end-to-end quality. Usually the cheapest lever with the least quality cost, because of the diminishing-returns curve.
2. **Switch to a smaller reranker.** MiniLM-L-6 in place of BGE-large. Re-run the quality benchmark; if the delta is small, keep the small model.
3. **Skip contextual compression.** It is a latency and cost adder. Chapter 6 walks through when it's worth the budget; often it is not.
4. **Cache aggressively.** Rerank scores for (query-hash, chunk-hash) pairs are cacheable if you have a repeat-query pattern. Most consumer-facing RAG has enough long-tail queries that caching helps little; enterprise RAG with a moderate query vocabulary can cache 30–70% of rerank calls.

If none of that fits, you are in the wrong reranker family. LTR (chapter 4) at N=100 is single-digit-ms; it is the escape hatch for very tight latency budgets.

## The N-K joint tuning recipe

The end-to-end procedure that ships:

1. **Fix the retriever, the reranker family, and the metric.** Say: RRF over BM25 + dense at recall@200, BGE-reranker-v2-m3 at fp16, nDCG@10 on your eval set (mod-109 covers metric choice rigorously).
2. **Sweep N holding K fixed.** N ∈ {20, 50, 100, 200, 500}. Compute nDCG@10 and p95 latency for each. Plot both.
3. **Pick the N at the elbow of the quality plot that fits the latency budget.** If the quality curve is flat past N=100 and the budget allows 200, pick 100 — no point spending on a plateau.
4. **Sweep K holding N fixed.** K ∈ {1, 3, 5, 10, 15, 20}. Compute end-to-end answer quality (RAG-triad — faithfulness, answer relevance, context precision — or a task-specific eval) for each. Plot.
5. **Pick K at the peak of that plot.** If the peak is at K=5 and dropping past K=10, use K=5, not K=10.
6. **Re-check the N ∩ K interaction.** Occasionally the best K at N=100 is not the best K at N=200. If your first pass changed N, re-sweep K at the new N to confirm.

This is one full loop through mod-109's methodology; if you have not read mod-109 yet, do it in parallel with this exercise.

## Real-world numbers for calibration

Numbers you actually measure will beat any generic table. That said, ballparks for calibration:

- **Enterprise search on a corpus of 100K–1M docs, English text:** N=100, K=5–10 is a common default. Cross-encoder rerank; end-to-end p95 in the 300–800 ms range for retrieve+rerank+assemble.
- **Very heterogeneous corpus (web-crawl-scale RAG, blog+forum+docs):** N=200–500, K=10 is more typical because the recall curve doesn't saturate until larger N. Latency budgets are correspondingly larger.
- **Narrow, well-scoped corpus (product catalog, single-team codebase):** N=50, K=3–5 often suffices. Retrieval saturates fast; rerank latency is cheap.
- **Multi-hop / research summarisation:** N=100, K=15–20 with structural ordering (chapter 7) beats a smaller K with best-first ordering.

Do not treat these as defaults to copy. Measure on your data.

## What NOT to do at the funnel

Three anti-patterns worth naming:

1. **Setting N to the vendor's or library's default without measuring.** LlamaIndex's default is often 10, LangChain's often 20, cross-encoder examples often 100. The default is not calibrated to your corpus.
2. **Coupling N and K in code.** "Retrieve 100, take the top 10" hard-coded across the codebase is fine until you want to test K=5. Make them configurable, log both to every retrieve-response diagnostic (mod-106 chapter 7 defined the diagnostics field), and version-tie them to a specific reranker + eval set.
3. **Setting K to the context window's absolute maximum.** "The model has 128K tokens, so K=100" is a K-driven-by-context-space decision, not a K-driven-by-answer-quality decision. Almost always the wrong shape — chapter 8 details why.

## Instrumentation: what to log for a funnel

Every retrieve-rerank call should log:

- `retrieval_top_n_size` — the actual N returned (in case the retriever short-circuited).
- `rerank_input_size` — N passed to reranker.
- `rerank_output_size` — K returned.
- `rerank_score_top_k` — the reranker's top-K scores (for calibration and for detecting reranker drift over time).
- `p95_retrieval_latency_ms`, `p95_rerank_latency_ms` — per-stage timing.

Mod-111 covers what to do with those logs — the reranker's score distribution shifting over time is a leading indicator of drift; the p95 crossing a threshold is a paging-worthy incident.

## What you should be able to do after this chapter

- State the four constraints on N and K (quality, latency, cost, context budget) and identify which knob answers to which constraint.
- Recognise the concave diminishing-returns curve for N (recall@N vs. N) and the peaked-then-degrading curve for K (answer quality vs. K).
- Author a latency budget worksheet with the four stages (retrieval, rerank, compression, assembly) plus slack, and know the four levers when the budget doesn't fit.
- Run the N-K joint tuning recipe (sweep N, fix best; sweep K, fix peak) on an eval set.
- Name the three anti-patterns (default N, hard-coded N-K coupling, K-at-context-window-max) and correct each.

## Further reading

- Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. Chapter 8 (Evaluation in IR) and the discussion of recall-precision trade-offs remain the reference for what "diminishing returns on candidate pool" formally means. [Free online](https://nlp.stanford.edu/IR-book/).
- Rosa, G., Bonifacio, L., Jeronymo, V., Abonizio, H., Fadaee, M., Lotufo, R., and Nogueira, R. (2022). *In Defense of Cross-Encoders for Zero-Shot Retrieval.* [arXiv:2212.06121](https://arxiv.org/abs/2212.06121) — reports N sweep behavior on cross-encoder rerankers.
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the direct reference for why K has a peak, not a monotone-up curve. Chapter 8 works through it.
- Mod-109 (Retrieval-Quality Evaluation, this track) — the metric-selection methodology this chapter's "measure on an eval set" hand-waves defer to.
- Mod-111 (Retrieval MLOps, this track) — the operational plumbing (log-schema, drift monitors, reranker-version cutover) that makes ongoing N-K tuning safe in production.
