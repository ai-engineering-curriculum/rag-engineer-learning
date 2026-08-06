# Chapter 1 — Embedding Family Map: Bi-Encoder, Cross-Encoder, ColBERT, SPLADE

## Why this chapter exists

At level 20, "the embedding model" is a single dropdown: pick `text-embedding-3-small`, get a 1536-dim vector, put it in pgvector, top-k. That shape is enough to ship a demo — it is not enough to *reason* about why your retrieval quality is capped, why the reranker you added moved the number, or why your competitor's stack uses two neural models in the same query path.

At level 30 the interviewer expects you to know that there is not "one" neural retrieval model — there are **four architectural shapes**, they play **different roles in the same pipeline**, and picking the wrong shape for a role is the single most expensive misalignment a retrieval stack makes. This chapter is the map. Chapters 2–3 make the benchmark literacy concrete on top of it; chapters 4–5 survey the models that instantiate each shape; the exercises put a working example of each into your hands.

## The four families at a glance

| Family | Where it lives in a query | Latency at query time | Storage per chunk | Canonical citations |
|---|---|---|---|---|
| **Bi-encoder (dense)** | Retrieval — top-K over the whole corpus | One vector encode + ANN lookup (ms) | 1 dense vector (384–4096 dims) | Karpukhin et al. 2020 (DPR); Reimers & Gurevych 2019 (Sentence-BERT) |
| **Cross-encoder** | Reranking — rescore K candidates | K forward passes over `(query, doc)` pairs (10s–100s ms per K) | 0 (model is called per query) | Nogueira & Cho 2019 (BERT for MS MARCO passage reranking) |
| **ColBERT / late interaction** | Retrieval *or* reranking — MaxSim over per-token vectors | Per-token encode + per-token ANN or MaxSim over K (10s ms) | One vector *per token* (many small vectors) | Khattab & Zaharia 2020 (ColBERT); Santhanam et al. 2022 (ColBERTv2) |
| **SPLADE (sparse-neural)** | Retrieval on a Lucene inverted index | Sparse expansion + BM25-style lookup (ms) | Sparse weighted term dict (30–200 non-zeros) | Formal et al. 2021 (SPLADE); Formal et al. 2022 (SPLADEv2 / ++) |

Read the rows as **roles a stack plays**, not "models you rank against each other on MTEB." A production retrieval stack routinely combines two or three of them (bi-encoder for first-stage recall + cross-encoder rerank; or SPLADE + dense fused; or ColBERT for both stages) — the point of this chapter is knowing which shape earns its cost in which slot.

## Bi-encoder — the default first-stage retriever

**Shape.** Encode the query and every document independently into a single fixed-dimension vector; compare with cosine similarity (or dot product on L2-normalized vectors). Two separate forward passes at index time and at query time; no cross-attention between query and document.

```
   query ─► [encoder] ─► q ∈ ℝᵈ ─┐
                                 │  cosine(q, dᵢ) → top-K
   docᵢ ─► [encoder] ─► dᵢ ∈ ℝᵈ ─┘
```

**Why the shape wins for retrieval.** All document vectors are computed *once*, offline, and can be laid down in an ANN index (HNSW, IVF-PQ; mod-104 covers those). At query time you do one encode and one ANN lookup — milliseconds over millions of documents. The bi-encoder is the *only* one of the four families that scales to whole-corpus first-stage retrieval without either a Lucene backend or a per-token index.

**Why the shape has a quality ceiling.** The encoder never sees `(query, doc)` together — the model has to compress every possible query-relevance signal into one point in ℝᵈ. When the query has a rare entity the docs don't share, or when relevance depends on a specific quotation, the dense vector alone is often not enough. This is what a cross-encoder reranker (below) is for.

**Canonical papers.** Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA* ([arXiv:2004.04906](https://arxiv.org/abs/2004.04906)) established the shape for QA. Reimers & Gurevych, *Sentence-BERT* ([arXiv:1908.10084](https://arxiv.org/abs/1908.10084)) is the training-recipe canon and is what the `sentence-transformers` library codifies.

**Representative models** (chapters 4–5 go deep):

- **Hosted**: OpenAI `text-embedding-3-small`/`-large`, Cohere `embed-english-v3.0` / `embed-multilingual-v3.0`, Voyage `voyage-3` / `voyage-3-large`, Google `text-embedding-005`.
- **Open**: BAAI `bge-large-en-v1.5`, intfloat `e5-large-v2` / `multilingual-e5-large`, Alibaba `gte-large-en-v1.5`, Nomic `nomic-embed-text-v1.5`, Snowflake `snowflake-arctic-embed-l`, `mixedbread-ai/mxbai-embed-large-v1`, Jina `jina-embeddings-v3`.

## Cross-encoder — the second-stage reranker

**Shape.** One transformer forward pass over the *concatenation* `[CLS] query [SEP] doc [SEP]`, returning a scalar relevance score. Cross-attention is allowed between query and document at every layer.

```
   query ─┐
          ├─► [encoder w/ cross-attention] ─► score ∈ ℝ
   docⱼ ──┘
```

**Why the shape wins for reranking.** Because query and document are attended together, the model can pick up "this passage answers the specific question because …" signals that bi-encoders erase. On MS MARCO passage ranking, a cross-encoder reranker on the top-K from any first-stage retriever routinely lifts MRR@10 by several points — Nogueira & Cho's *Passage Re-ranking with BERT* ([arXiv:1901.04085](https://arxiv.org/abs/1901.04085)) is the canonical demonstration.

**Why it cannot be the first stage.** You would need to run one forward pass per `(query, every doc in the corpus)`. On a 10M-doc corpus that is 10M forward passes at query time — hours per query on a single GPU. The cross-encoder is only affordable as a *second-stage* rescoring over the top-K (typically K = 20 … 200) returned by a bi-encoder, SPLADE, or hybrid retriever.

**Sizing intuition.** If your first-stage retrieval takes ~10 ms and a cross-encoder inference on a small MiniLM checkpoint takes ~2 ms per pair on a modern GPU, reranking the top-50 costs ~100 ms. That is the shape that fits in an interactive p95 budget; K = 500 does not.

**Canonical papers.** Nogueira & Cho 2019 (above) is the shape. Nogueira, Jiang & Lin 2020, *Document Ranking with a Pretrained Sequence-to-Sequence Model* ([arXiv:2003.06713](https://arxiv.org/abs/2003.06713); "monoT5") extends the pattern to seq2seq. `sentence-transformers` ships the [`CrossEncoder`](https://sbert.net/docs/cross_encoder/pretrained_models.html) class with `ms-marco-MiniLM-L-*` checkpoints as the reference implementations.

**Representative models.** BAAI `bge-reranker-v2-m3` / `bge-reranker-large`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `mixedbread-ai/mxbai-rerank-large-v1`, Jina `jina-reranker-v2-base-multilingual`; hosted: Cohere Rerank v3, Voyage `rerank-2`, Jina Reranker. (mod-107 covers reranker choice end-to-end; this chapter is the shape-family map only.)

## ColBERT — late interaction, one vector per token

**Shape.** Encode the query into a *sequence* of per-token vectors; encode each document into a *sequence* of per-token vectors; score by **MaxSim**: for each query token, take the max cosine similarity across all document tokens; sum the maxes.

```
   query ─► [encoder] ─► [q₁, q₂, …, qₙ]                (n tokens ~ 32)
   doc   ─► [encoder] ─► [d₁, d₂, …, dₘ]                (m tokens ~ 128–512)

   score(q, d) = Σᵢ max_j cos(qᵢ, dⱼ)                   ("late" interaction)
```

**What "late interaction" means.** In a bi-encoder the query and document are combined *late*, at one point in ℝᵈ — the model has already committed to a single representation per side. In a cross-encoder the interaction is *early* — every layer of the model sees both together. ColBERT sits between them: the encoder runs independently on each side (so document vectors can be indexed offline), but the *scoring* function sums per-token max-similarities, giving the model a lexical-level match signal that a single-vector bi-encoder cannot express.

**Why the shape wins.** Empirically, ColBERT-family models close much of the gap between bi-encoders and cross-encoders on out-of-domain retrieval — including on the BEIR benchmark (chapter 2), where dense bi-encoders often underperform BM25 on tail domains. The MaxSim interaction gives explicit per-token attributability: "the model matched *because* query token `t` aligned with document token `d`," which is a useful debugging surface.

**Why the shape hurts.** Storage explodes. One vector per token, times the number of tokens per chunk — a 512-token chunk becomes 512 vectors of ~128 dims each = ~64 kB per chunk (vs. ~4 kB for a 1024-dim single vector). ColBERTv2 (Santhanam et al., [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)) introduced **residual compression** to bring this down (~2 bits per dimension after clustering), and PLAID (Santhanam et al., [arXiv:2205.09707](https://arxiv.org/abs/2205.09707)) is the low-latency indexer that pairs with it. You still pay a multiplier over dense; the trade-off is real.

**Representative models.** Stanford's original `colbert-ir/colbertv2.0`; Answer.AI's `answerdotai/answerai-colbert-small-v1` (a compact contemporary re-implementation); Jina's `jina-colbert-v2` (multilingual). The primary Python library is `RAGatouille` (a wrapper around ColBERT) or ColBERT directly from [`stanford-futuredata/ColBERT`](https://github.com/stanford-futuredata/ColBERT).

**Where it slots.** ColBERT can be a first-stage retriever *or* a second-stage reranker over the top-K of a bi-encoder. As a reranker it is cheaper than a cross-encoder at similar quality; as a first-stage retriever it needs its own indexer (PLAID or an emerging support in Vespa / Qdrant / Weaviate — check current status before committing). Sturua et al., *jina-embeddings-v3* ([arXiv:2409.10173](https://arxiv.org/abs/2409.10173)) is the current well-cited discussion of dense vs. late-interaction trade-offs from a vendor perspective.

## SPLADE — sparse-neural expansion over a Lucene inverted index

**Shape.** Encode the query and document into a **sparse vector over the vocabulary** (~30k dims for BERT; only 30–200 non-zeros per doc). Each non-zero is a `(term, learned weight)` pair. Score with dot product — which, because the vector is sparse, is exactly a BM25-shaped `sum(query_weight × doc_weight)` over shared terms. That means you can serve it on a **Lucene inverted index** with almost no infrastructure change.

```
   query ─► [MLM head] ─► sparse{"kubernetes": 0.9, "container": 0.4, "orchestration": 0.6, …}
   doc   ─► [MLM head] ─► sparse{"kubernetes": 1.1, "cluster": 0.7, "k8s": 0.5, "container": 0.8, …}

   score(q, d) = Σ_t q_t · d_t         (dot product over overlapping terms)
```

**What is "learned" that BM25 does not have.** SPLADE uses BERT's masked-language-model head to *expand* each token — a document containing "k8s" gets non-zero weight on "kubernetes" too, because the MLM head learned they co-occur. That expansion is what lifts SPLADE above raw BM25 on out-of-domain retrieval, while keeping the sparse-index shape that BM25 depends on.

**Why the shape wins.** You get **learned term expansion** and **learned term weighting** on infrastructure you already have (Elasticsearch, OpenSearch, Vespa, Lucene). No vector index; no ANN parameter tuning; queries look like normal `bool`-of-`term`-queries with weights. Formal et al., *SPLADE v2: Sparse Lexical and Expansion Model* ([arXiv:2109.10086](https://arxiv.org/abs/2109.10086)) is the canonical version paper; Formal et al., *From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective* ([arXiv:2205.04733](https://arxiv.org/abs/2205.04733)) is the SPLADE++ tuning follow-up.

**Why the shape hurts.** Query-time expansion is a full BERT forward pass (10s of ms on CPU), which is much slower than tokenize-and-lookup for plain BM25 — SPLADE moves the neural cost from index time to query time. Document expansion at index time inflates the number of postings, which affects storage. And the model has to be re-tuned or replaced when the vocabulary shifts (a specialized domain corpus with lots of OOV terms behaves differently than English news).

**Representative models.** `naver/splade-v3` (current canonical), `naver/splade-cocondenser-ensembledistil` (SPLADE++ family), `prithivida/Splade_PP_en_v2`. Serving: Vespa has native sparse-tensor support; Elasticsearch/OpenSearch expose it via the [text-expansion query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-text-expansion-query.html) (in Elastic's case, using their built-in ELSER model, which is a SPLADE-family sparse expansion model).

**Where it slots.** SPLADE is a **first-stage retriever** in the same slot as BM25, and its most common production use is as one leg of a hybrid stack: `SPLADE ⊕ dense bi-encoder → rerank`, fused with reciprocal-rank fusion (Cormack et al. 2009 — mod-106 covers fusion in detail). It is not a reranker.

## The role map — which family fits which slot

The four families are complements, not competitors. The way a mature stack looks:

```
   ┌──────────────── first-stage retriever(s) ────────────────┐
   │                                                          │
   │   BM25 (Lucene)   ─┐                                     │
   │   SPLADE          ─┼──►  reciprocal-rank fusion  ──► top-K
   │   Bi-encoder      ─┘                                     │
   │                                                          │
   └──────────────────────┬───────────────────────────────────┘
                          │
                          ▼
   ┌───────── second-stage reranker (choose one) ─────────┐
   │   Cross-encoder                                      │
   │   ColBERT (as reranker)                              │
   │   Hosted rerank API (Cohere / Voyage / Jina)         │
   │   LLM-as-reranker (mod-107)                          │
   └──────────────────────┬───────────────────────────────┘
                          │
                          ▼
                    context assembly (mod-107) → LLM
```

That diagram is descriptive, not prescriptive — plenty of production stacks skip the reranker, and plenty skip SPLADE. What it captures is the **type-level constraint**: cross-encoders never live at the top; bi-encoders never live alone when quality matters and the corpus is not the domain they were trained on; ColBERT and SPLADE are dual-mode but each has a natural home.

## Choosing an interaction shape by role

For each slot you own, the question is not "what's on top of the leaderboard" — it is:

**First-stage retriever over N documents:**

- Bi-encoder if your corpus is close to the training distribution and you have modest storage.
- BM25 or SPLADE if your corpus is jargon-heavy, low-resource, or has OOV terms your dense model has never seen.
- **Both, fused** if you can afford the query-time cost of two retrievers. (This is the empirical default for high-quality open-domain stacks — BEIR results in chapter 2 make it visible.)
- ColBERT if you want retrieval-quality closer to a cross-encoder without paying per-query cross-encoder cost, and you can budget the ~10× storage over a single-vector dense index.

**Reranker over top-K candidates:**

- Cross-encoder for the highest quality per candidate; latency is O(K) forward passes so K is bounded by your budget.
- ColBERT-as-reranker as a middle ground: cheaper than cross-encoder, better than none.
- Hosted rerank APIs (Cohere, Voyage, Jina) if you do not want to serve a model — mod-107 covers the cost model.
- LLM-as-reranker only when nothing else fits or you need pairwise/listwise reasoning; expensive and slow.

## Concrete: two runnable snippets

### Bi-encoder retrieval with `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
docs = [
    "Kubernetes schedules containers across a cluster of nodes.",
    "The Python `asyncio` event loop coordinates coroutines.",
    "HNSW is a graph-based approximate-nearest-neighbor index.",
]
doc_embs = model.encode(docs, normalize_embeddings=True)   # (n_docs, dim)

q = model.encode("how does the k8s scheduler pick a node?", normalize_embeddings=True)
scores = doc_embs @ q                                       # cosine on L2-normalized vectors
top = int(np.argmax(scores))
print(docs[top], scores[top])
```

### Cross-encoder rerank on the top-K

```python
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
query = "how does the k8s scheduler pick a node?"
pairs = [(query, d) for d in docs]
rerank_scores = ce.predict(pairs)                           # (n_docs,) scalar per pair
reranked = sorted(zip(docs, rerank_scores), key=lambda x: -x[1])
for d, s in reranked:
    print(f"{s:6.3f}  {d}")
```

Two models, two shapes, one query. That is the muscle-memory to walk into every subsequent chapter with.

## What is NOT in this chapter (and where it goes)

- **How you actually pick between five specific bi-encoders.** Chapters 2 (benchmark literacy) and 7 (domain-fit benchmark) do that.
- **How to serve a cross-encoder as a reranker in a production pipeline.** mod-107 owns the reranker slot end-to-end — latency budget, fan-out, hosted-vs-self-hosted for rerank.
- **How to fine-tune any of these models.** Explicitly the peer track (`fine-tuning-engineer`, chapter 9 boundary). This module *selects* and *evaluates*; it does not train.
- **Multimodal embedding families (CLIP / SigLIP / ColPali).** Enterprise multimodal is in mod-112.

## What you should be able to do after this chapter

- Name the four families and state which one lives in each slot of a hybrid retrieve-then-rerank pipeline.
- Explain in one sentence per family *why* it fits that slot (interaction shape, storage cost, query-time cost).
- Point to a representative model for each family — one hosted where applicable, one open.
- Recognize that "the embedding model" question is really at least *two* model choices in a serious stack (retriever + reranker) and possibly three (SPLADE / dense / cross-encoder).
- Read the rest of this module — MTEB (bi-encoders / SPLADE), BEIR (all four), quantization (mostly bi-encoders and ColBERT residuals) — and know which family each result applies to.

## Further reading

- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* [arXiv:2004.04906](https://arxiv.org/abs/2004.04906). — The canonical bi-encoder retrieval paper.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084). — The training-recipe canon; `sentence-transformers` implements this line.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085). — The cross-encoder rerank shape.
- Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* [arXiv:2004.12832](https://arxiv.org/abs/2004.12832). — The late-interaction paper.
- Santhanam, K., et al. (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* [arXiv:2112.01488](https://arxiv.org/abs/2112.01488). — Residual compression and quality gains.
- Formal, T., Piwowarski, B., & Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* [arXiv:2107.05720](https://arxiv.org/abs/2107.05720). Follow-ups: [SPLADE v2 (arXiv:2109.10086)](https://arxiv.org/abs/2109.10086); [SPLADE++ (arXiv:2205.04733)](https://arxiv.org/abs/2205.04733).
- Sentence-Transformers docs, [*Cross-Encoders*](https://sbert.net/docs/cross_encoder/pretrained_models.html) — the maintained list of publicly available cross-encoder checkpoints.
- Elastic, [*ELSER — Elastic Learned Sparse EncodeR*](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html) — production SPLADE-family sparse expansion model shipped inside Elasticsearch.
