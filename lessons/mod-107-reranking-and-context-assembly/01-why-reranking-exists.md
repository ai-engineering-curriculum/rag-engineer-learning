# Chapter 1 — Why Reranking Exists and the Precision Job It Owns

## Why this chapter exists

Mod-106 handed you a fused ranked list — the top-100 candidates from a hybrid (BM25 + dense, sometimes with rewriting or routing on top) retriever tuned for **recall@100**. Ninety of those hundred are noise from the LLM's point of view; a well-tuned retriever guarantees the ~10 relevant chunks are *somewhere in the 100*, not that they are at the top.

The LLM never sees 100 chunks. Its context window can hold 100 chunks (a 128K-token model easily fits 100 × 800-token chunks), and cost-per-token stays modest at that size, but stuffing 100 candidates in front of the generator produces measurably worse answers than showing it 5–10 tightly ranked ones. The Liu et al. 2023 *"Lost in the Middle"* result — that models attend most reliably to the beginning and end of their context, and skip material in the middle — is the industry-standard reference for this failure ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)); chapter 8 walks through the paper and its consequences in detail. Cost-per-1M-input-tokens is the other reason: a 100-chunk context multiplies your inference bill by ~10× vs. a 10-chunk context, for a quality *loss*.

The two-stage pipeline that resolves this — cheap-and-wide retrieve, then expensive-and-narrow rerank — is the industry-standard shape for retrieval-augmented generation and the subject of this module.

## The precision job vs. the recall job

Mod-106 built for **recall@100**. This module builds for **precision@10** (or precision@5, or precision@3 — the exact *k* is tuned per stack; chapter 5 covers that funnel).

The two goals are complementary and neither can substitute for the other:

- **Retrieval** optimises for *"is the relevant chunk in the pool?"* — miss it here and no downstream stage can recover it. First-stage retrievers use cheap scoring (BM25 inverted-index lookups, HNSW ANN, RRF over both) that scales to millions of documents at ~10–50 ms per query.
- **Reranking** optimises for *"is the relevant chunk in the top few?"* — miss the *ordering* here and the LLM either grounds on the wrong chunk or ignores the good one. Rerankers use expensive scoring (a cross-encoder pass, an LTR model with dozens of features, or an LLM judge) that only runs on N = 20–200 candidates per query.

The economic argument for the split is simple: you cannot afford to run a cross-encoder over your whole 10-million-document corpus (10 M documents × 50 ms per doc = a week of compute per query). You *can* afford to run one over 100 candidates (100 × 50 ms = 5 s single-threaded, ~200 ms batched on a modern GPU, ~50 ms with a smaller model and a warm cache). The retrieval stage does the cheap filtering; the reranking stage does the expensive scoring on the filtered set.

The quality argument is what makes the split *pay*: a well-chosen reranker converts a mediocre first-stage top-100 into a strong top-10. Numbers reported in the reference literature — Nogueira & Cho 2019 ([arXiv:1901.04085](https://arxiv.org/abs/1901.04085)) on BERT rerankers over BM25, Khattab & Zaharia 2020 ([arXiv:2004.12832](https://arxiv.org/abs/2004.12832)) on ColBERT, the MS MARCO passage-ranking leaderboard ([microsoft/MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking)) — routinely show a reranker lifting nDCG@10 by 5–15 points over the first-stage list on the same top-*k*. That is a large enough gain to justify the second-stage compute in almost any RAG stack.

## The two-stage pipeline as a picture

```
                            [ Retrieve N ]              [ Rerank to K ]              [ Assemble ]
                            fast, wide                  slow, narrow                 pack, cite
                            recall@N                    precision@K                  token budget
    query ─────────►  BM25 + dense + fuse  ────►  cross-encoder / LTR / LLM  ────►  ordered chunks
                            top-100                     top-10                       ~4K–8K tokens
                            ~50 ms                      ~50–200 ms                   sub-ms
                            (mod-106)                   (chapters 2–5)               (chapters 6–7)
                                                                                          │
                                                                                          ▼
                                                                                     [ Generator ]
                                                                                     LLM produces
                                                                                     the answer
                                                                                     (mod-108, peer track)
```

The three columns after the retriever correspond to the three groups of chapters in this module: reranker choice and tuning (chapters 2–5), contextual compression (chapter 6), and context assembly (chapter 7). Chapter 8 revisits the "does long context obviate all of this?" question. Chapter 9 marks the boundary to `applied-ai-engineer` — the peer track that consumes the assembled context.

## The reranker's input shape and output shape

Every reranker in chapters 2–4 has the same interface:

```python
def rerank(
    query: str,
    candidates: list[Candidate],   # top-N from the first stage; each has doc_id, chunk_id, text, and optional metadata
    top_k: int,                    # how many to keep
) -> list[ScoredCandidate]:        # top-K, re-ordered, with a rerank score attached
    ...
```

What differs between reranker families is how the score is produced:

- **Cross-encoder (chapter 2):** the query and the candidate text are concatenated and fed to a transformer encoder that outputs a single relevance score. High quality; latency proportional to N × (query+chunk length).
- **Hosted API (chapter 3):** the same cross-encoder shape but served by a vendor (Cohere, Voyage, Jina). You pay per rerank call rather than per GPU-hour; you trade tunability and latency floor for zero ops.
- **LTR (chapter 4, first half):** a gradient-boosted tree (LightGBM / XGBoost) over hand-designed features — BM25 score, dense similarity, click signals, document freshness, source authority, whatever else your logs contain. Very fast; requires labeled training data and a feature pipeline.
- **LLM-as-reranker (chapter 4, second half):** ask an LLM to score or rank the candidates directly. Highest quality potential per candidate; slowest and most expensive; useful for the *last mile* of hard queries.

The score is only meaningful within a single query — cross-encoder scores are not calibrated across queries, LTR scores depend on the feature vector, LLM scores depend on the prompt. That means the reranker's output is a **relative ordering**, not an absolute relevance signal. Chapter 7 (context assembly) treats the rerank score as an ordering key, not a threshold.

## What "context assembly" adds after the reranker

The reranker gives you a top-K ordered list. Handing that list directly to the LLM is almost never what you want. Between the reranker and the generator sits **context assembly**, and it does four jobs:

1. **Deduplication.** Two chunks from adjacent paragraphs of the same document often surface together in the top-K. If both carry the same fact, the second is wasted context and pushes a genuinely-different chunk out of the window. Chapter 7 covers dedup by source, by content-hash on normalised text, and by MMR-style diversification.
2. **Ordering.** The reranker's ordering (best-first) is not always the right shape for the LLM. Lost-in-the-middle (chapter 8) suggests putting the top chunk *last* — or first-and-last — when the context is long. Long-document RAG often orders by document structure (Introduction → Section 4.2 → Conclusion) instead of by rerank score.
3. **Citation anchors.** Each chunk needs a stable, unambiguous handle the LLM can quote back — `[doc_id#chunk_id]`, `[Source 3, §4.2]`, a footnote marker. Preserving that anchor through compression (chapter 6) is the single most fragile part of the pipeline; a compressor that strips the anchor breaks every downstream citation-checking eval.
4. **Token budgeting.** You have (context window) − (system prompt) − (few-shot examples) − (reserved for answer) tokens for the retrieved chunks. Chapter 7 walks through the arithmetic and gives you a reserve-for-answer default (typically 15–25% of the window) that survives most model swaps.

Compression (chapter 6) is a middle stage: if the top-K chunks are too long for the budget from step 4, you extract only the sentences that actually address the query rather than truncating chunks arbitrarily. The two well-known frameworks are LlamaIndex's `ContextualCompressor` and LangChain's `LLMChainExtractor`; chapter 6 covers both.

## What this module does NOT cover (and where it goes)

- **The generator's prompt shape** — how the assembled context is inserted into the LLM prompt, the citation format the LLM is asked to produce, refusal patterns for "no relevant context." That is **mod-108 (Retrieval-Specific Prompting)** in this track.
- **Evaluating whether the reranker actually helped** — recall@k, nDCG@k, MRR, paired significance testing, eval-set versioning. That is **mod-109 (Retrieval-Quality Evaluation)**.
- **How the reranker service is versioned, deployed, monitored for drift** — that is **mod-111 (Retrieval MLOps)**.
- **How the whole retrieve + rerank + assemble pipeline is exposed as an MCP tool** — that is **mod-110 (Orchestration and MCP-Exposure)**.
- **Retrieving from graph, structured (SQL), or multimodal sources** — reranking those has domain-specific quirks; see **mod-112 (Enterprise Retrieval)**.
- **What the peer track does with the assembled context** — chapter 9 draws that boundary.

## The four levers this module gives you

Read the module as four levers you turn on a reranker + assembler stack:

1. **Reranker family choice** — cross-encoder vs. hosted API vs. LTR vs. LLM (chapters 2–4). The choice is a joint constraint of latency budget, cost budget, ops appetite, and quality ceiling.
2. **Funnel geometry** — the N-to-K ratio and the p95 latency it eats (chapter 5). Bigger N usually helps quality up to a plateau; you cannot afford unbounded N.
3. **Compression** — do you shrink chunks before packing, or take them whole (chapter 6). Compression buys tokens; it costs an LLM call per chunk (or per query, depending on the shape) and can shred citations if you are careless.
4. **Assembly order and dedup** — the shape of the context you hand the LLM (chapter 7). A good assembly is defensible against lost-in-the-middle and reserves budget for the answer.

## What you should be able to do after this chapter

- State the two-stage pipeline (retrieve for recall, rerank for precision) in one sentence and give the ballpark latency and quality delta each stage owns.
- Name the four reranker families this module covers and know which chapter owns each.
- Name the four context-assembly jobs (dedup, ordering, citation anchors, token budget) and state which one owns the citation-preservation risk during compression.
- Point to the specific downstream module that picks up each thread this module defers (prompt shape → mod-108, eval → mod-109, MLOps → mod-111, MCP → mod-110).
- Explain why long context does not remove the reranker (preview of chapter 8 — lost-in-the-middle plus cost-per-token).

## Further reading

- Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the paper that put cross-encoder reranking on the map for IR.
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the reference for why "just put everything in the context" underperforms; chapter 8 works through it.
- Khattab, O. and Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — the late-interaction reranker; a useful counterpoint to full cross-encoders that chapter 2 references.
- Microsoft: [*MS MARCO Passage Ranking Leaderboard*](https://microsoft.github.io/msmarco/) — the benchmark most public reranker numbers report on.
- Craswell, N., Mitra, B., Yilmaz, E., Campos, D., and Voorhees, E. M. (2020). *Overview of the TREC 2019 Deep Learning Track.* [arXiv:2003.07820](https://arxiv.org/abs/2003.07820) — the TREC-DL benchmark, the other place reranker numbers live.
