# Chapter 7 — Designing a Domain-Fit Benchmark

## Why this chapter exists

Chapters 2 and 3 argued: MTEB and BEIR are priors, not answers. Chapter 6 argued: cost is scale-dependent and needs numbers. The synthesis is a **domain-fit benchmark** — a small, corpus-native, hand-labeled retrieval eval you can point at any embedding candidate and get an honest quality number for the model on *your* data. Everything else in the model-choice conversation is subordinate to this artifact.

This chapter is the *design* of that benchmark. What eval set to build, how large, what queries to write, how to label, which models to include, how to hold everything else constant, how to read the ranking table. The paired exercise (`exercise-04`) turns the design into a working sweep with a defended winner. mod-109 goes deeper on judgment sets, paired significance testing, and the metric families in general; this chapter is the shape of the specific benchmark you need to *choose* an embedder.

## The five design decisions

Every domain-fit benchmark comes down to five explicit choices:

1. **The eval set** — the (query, relevant-doc-IDs) pairs. Size, sourcing, labeling shape.
2. **The corpus** — the fixed set of documents / chunks the retriever searches over. Chunker held constant.
3. **The model shortlist** — ≥5 candidate embedders spanning shape, size, and hosted/self-hosted.
4. **The controlled axes** — chunker version, retriever, top-K, prompt/instruction handling, ANN parameters. All held constant.
5. **The metrics and significance test** — nDCG@10 and Recall@k as the primaries; paired significance test on per-query deltas for tie-breaking.

Get all five right and the sweep produces a defensible ranking. Get any one wrong and you are measuring the wrong thing.

## Decision 1 — the eval set

**Size.** 30–100 queries with per-query relevance labels is enough to *rank* models when the quality gaps are moderate. 200+ is more comfortable — smaller sets have wider confidence intervals per model, and near-ties become impossible to break. Below 30, you are measuring noise. mod-109 covers judgment-set sizing rigorously; for chapter 7 the rule of thumb is: aim for 50 as a floor.

**Query sourcing.** Ranked from best to worst:

1. **Real user queries** from your product's search logs (if you have any). Sample by time, not by frequency — top-frequency queries are dominated by one intent shape and will underrepresent the tail.
2. **Real user queries** from an analog product's public logs (MS MARCO, Natural Questions) — better than synthetic if you have nothing else, but be aware the distribution will not match your corpus.
3. **Domain-expert-written queries** — ask a subject-matter expert (a lawyer for a legal corpus, a doctor for a medical one) to write 50–100 questions they would want their retriever to answer. This is what most teams end up doing.
4. **LLM-generated queries** with human spot-check — feed the LLM a random sample of chunks and ask it to write plausible queries whose answer is in the chunk. Cheapest by far; and the queries are systematically biased toward the vocabulary of the chunks that generated them (which flatters embedders that are strong on paraphrase and hurts embedders that need lexical overlap). Use as a supplement, not the whole set.

**Query diversity.** Cover the axes you care about:

- **Granularity** — some fact-lookup, some section-summary, some full-doc-scope.
- **Vocabulary** — some using the corpus's own terminology, some using synonyms a user would type.
- **Query length** — a few very short, a few medium, a few long.
- **Query shape** — questions, keywords, phrases, entity names.
- **Difficulty** — some obvious hits, some intentionally tricky ones (near-synonyms, edge cases, out-of-scope).

Do not label 100 easy queries. If every model gets 0.9 nDCG, the benchmark cannot distinguish them — you have measured that they can all do the easy thing. Include hard cases so the ranking has signal.

**Labels — two shapes, both acceptable:**

- **Coarse (document-level).** For each query, list the doc_ids of documents whose content answers the query. `rel = 1` if relevant, `0` otherwise. Faster to label; adequate when the coarse question "does the retriever find the right document?" is what matters.
- **Fine (chunk-level or span-level).** For each query, list the chunk_ids (or the section_paths a chunk falls in) that contain the answer. Slower to label; the correct choice when you want to distinguish embedders' *within-document precision*.

Graded relevance (0/1/2) is stronger than binary and nDCG rewards it, but doubles labeling burden. mod-109 covers graded labeling in depth; for chapter 7 an honest binary set beats a rushed graded set.

**Label the qrels once, freeze them, version them.** If you re-label mid-sweep because a query "seems wrong," you are not comparing model A to model B, you are comparing model A to a moving target. Fix your labels *before* you look at any model's output. Store as `qrels.tsv` in TREC format under version control.

**LLM-assisted labeling — carefully.** For coarse doc-level labels on a large corpus, an LLM can prescreen candidates and a human confirms. This is common and defensible if you disclose it. What you do *not* do: let an LLM be the sole labeler, then evaluate an LLM-family embedder against those labels — you will get an inflated score because the labels and the embedder share a bias.

## Decision 2 — the corpus, held constant

The corpus is the fixed set of chunks the sweep will retrieve over. Pin:

- **The chunker.** Same `chunker_id`, same `chunker_version`, same chunk-size / overlap / boundary rules across all models. The mod-102 `Chunk` schema is what you use — every model reads the same chunk text.
- **The doc-to-chunk mapping.** Every model retrieves over the same chunk IDs. This is what makes per-query error analysis (Part D in the exercise) meaningful.
- **Any metadata filtering.** If you use pre-filter on `permission_scope` or `updated_at`, use the same filter across all runs.

The chunker itself is a variable that affects the number heavily (mod-102 chapter 5 quantifies this) — but it is a variable owned by mod-102's ablation, not this chapter's. Pick the mod-102-blessed chunker for your corpus and hold it constant across the embedding sweep. If you want to co-vary chunker and embedder, do it as a stretch goal *after* you have a defensible embedder ranking at the fixed chunker.

## Decision 3 — the model shortlist

At least **five** candidates. The shortlist should span:

- **Family shape.** At least one bi-encoder (chapter 1). If your stack will use SPLADE or ColBERT, include one of those too, even if the metric comparison is not apples-to-apples with the dense ones — you will get an early signal on whether the hybrid stack is worth building.
- **Provenance.** At least one hosted (chapter 4) and at least two self-hosted (chapter 5). Comparing only hosted APIs, or only open models, misses the two decisions the chapter-6 cost analysis wants you to have data on.
- **Size / cost tier.** Include a small model (`bge-small` or `text-embedding-3-small` at 256-dim MRL) and a larger one (`bge-large`, `voyage-3`, `text-embedding-3-large`, or `gte-Qwen2-7B-instruct`). If the small one is only 2 nDCG points behind the large one on your data, that changes the deployment cost story dramatically.
- **Language / domain.** If your corpus has a domain-specialized model available (`voyage-code-3` for code, `voyage-law-2` for legal), include it. If your corpus is multilingual, include `bge-m3` or `multilingual-e5-large` or Cohere / Voyage multilingual.

**Include BM25 as a mandatory baseline.** Chapter 2 argued this; chapter 5 of mod-105 owns the Lucene side. For chapter 7, run `rank_bm25` (Python-native, no service) or Pyserini as the "no dense model at all" row in the table. If your fanciest embedder is not measurably above BM25 on your corpus, that is the single most important finding of the sweep — and it will tell you to build a hybrid retriever (mod-106) rather than pick a better dense model.

**Include the current default (the model you would ship if this benchmark did not exist)** as a fixed row. That is your reference point. Every candidate is judged against "does it beat the default enough to justify the switch cost?"

A realistic shortlist for a general English enterprise corpus (adjust for your case):

| # | Model | Family | Provenance | Rationale |
|---|---|---|---|---|
| 1 | BM25 (`rank_bm25`) | Lexical baseline | Local | The mandatory floor |
| 2 | `openai/text-embedding-3-small` (dim=512 MRL) | Bi-encoder | Hosted | Common default; smallest reasonable hosted |
| 3 | `openai/text-embedding-3-large` (dim=1024 MRL) | Bi-encoder | Hosted | Common upgrade path |
| 4 | `cohere/embed-english-v3.0` | Bi-encoder | Hosted | Different vendor, role-typed inputs |
| 5 | `voyage/voyage-3` | Bi-encoder | Hosted | Long-context; often top of MTEB retrieval |
| 6 | `BAAI/bge-large-en-v1.5` | Bi-encoder | Self-hosted | Strong open baseline |
| 7 | `Snowflake/snowflake-arctic-embed-l` | Bi-encoder | Self-hosted | Different training recipe at same param count |
| 8 | (optional) `Alibaba-NLP/gte-Qwen2-7B-instruct` | Bi-encoder, LLM-backbone | Self-hosted (A100) | Max-quality tier; is the cost worth it? |

Eight candidates is manageable; ten is the practical ceiling before the sweep takes a full day.

## Decision 4 — the controlled axes

Anything that is not the embedding model must be identical across runs. The list:

- **Chunker + chunker version + chunks.** As above.
- **Retriever backend.** Same vector DB (pgvector or Qdrant or Chroma — the mod-101 toolchain), same distance metric (cosine, computed on L2-normalized vectors).
- **ANN parameters.** For an HNSW baseline, pin `M`, `efConstruction`, `efSearch`. mod-104 tunes these; chapter 7 holds them fixed.
- **top-K.** Same `top_k` across all runs — 10 is what most nDCG@10 metrics compare against, but capture Recall@100 too so you see the tail.
- **Prompt/instruction handling per model.** Non-negotiable: each model gets its recommended prefix / role type / task instruction. This is per-model, not per-run — E5 needs `"query: "` / `"passage: "`, Cohere / Voyage need `input_type`, GTE-Qwen needs an English instruction. Skipping the prefix is not "holding the axis constant" — it is running the model in a broken configuration.
- **Query preprocessing.** Same query text goes to every model (with per-model prefix as above). No per-model lowercasing / stopword-stripping / lemmatization. If you preprocess for BM25 (usually lowercase + stopwords), preprocess *only* for BM25 and note it.
- **Random seeds** for anything stochastic (embedding batches that dropout, sampling within `sentence-transformers` normalization).

Reproduce these controls in a `benchmark_config.yaml` that ships with the results:

```yaml
chunker: {id: llama_index.MarkdownNodeParser, version: 1.2.0}
retriever: {backend: pgvector, distance: cosine, top_k: 10}
ann: {index: hnsw, M: 16, efConstruction: 200, efSearch: 100}
qrels: {path: data/judgments/qrels.tsv, revision: v1, n_queries: 50, labeling: fine}
seed: 0
mteb_version: 1.14.7
```

## Decision 5 — the metrics and the tie-breaker

**Primary:** nDCG@10 — the same metric MTEB and BEIR report; it accepts graded relevance and rewards ranking the right doc high, not just recalling it somewhere.

**Secondary (also report):**

- **Recall@10** — how often the right doc is in the top-10 at all. Simple and interpretable.
- **Recall@100** — the "tail" — matters when a reranker (mod-107) will rescore top-100; a model with weaker top-10 but strong top-100 recall may win a hybrid stack.
- **MRR@10** — mean reciprocal rank; sensitive to the *position* of the first relevant hit; useful when there is typically one correct answer per query.

**Tie-breaker:** paired significance test on per-query nDCG deltas. If model A averages 0.60 and model B averages 0.61, the 0.01 difference is very likely within noise. Run a **paired t-test** (parametric) or **Wilcoxon signed-rank** (non-parametric) on the per-query pairs; a p > 0.05 means "these are tied." mod-109 covers this in depth; for chapter 7 the pattern is:

```python
from scipy.stats import wilcoxon
p_stat, p_val = wilcoxon(ndcg_by_query["model_A"], ndcg_by_query["model_B"])
# p_val > 0.05 → cannot reject the null "these two models are equivalent on this eval"
```

Report both the point estimate and the pairwise significance table.

## The workflow — three phases

### Phase 1 — corpus + eval set (one-time; hours to days)

1. Pick the corpus. Chunk it with your mod-102 chunker; freeze the chunk-set (write out `chunks.jsonl` with stable `chunk_id`s).
2. Write / collect 50+ queries (Decision 1).
3. Label — coarse or fine. Write `qrels.tsv` in TREC format. Freeze.
4. Commit `data/corpus/`, `data/chunks.jsonl`, `data/judgments/queries.tsv`, `data/judgments/qrels.tsv`, `benchmark_config.yaml` to version control.

This artifact is the reusable one — every future embedding decision on this corpus can rerun against it. Treat it as an asset.

### Phase 2 — the sweep (per model; automated)

For each model in the shortlist:

1. Embed the frozen chunk set (with the correct per-model prefix / role type).
2. Load into the retriever backend (pgvector table per model, or Qdrant collection per model).
3. Embed each query (correct per-model prefix).
4. Retrieve `top_k=100` per query.
5. Compute nDCG@10, Recall@10, Recall@100, MRR@10 against `qrels.tsv`.
6. Persist per-query scores in a JSON per model (chapter 3's MTEB shape works exactly).

Automate the loop — don't run it by hand for each model. The exercise makes this a script.

### Phase 3 — the ranking + decision (hours)

1. Aggregate per-model JSON into a comparison table. Sort by primary metric (nDCG@10).
2. Run pairwise significance tests. Draw a matrix of `p < 0.05` between each pair.
3. Compute the cost-latency-quality table from chapter 6.
4. Do per-query error analysis: find 5 queries where the top-1 model beats the runner-up by ≥0.5 and 5 where the runner-up wins. Read them. Write two paragraphs.
5. Write the recommendation: "We should ship model X because …; runner-up model Y is within noise but costs Z more per month; both dominate the current baseline B by ΔnDCG."

That doc is the artifact the architect reads before signing off on your embedding choice.

## The three failure modes to avoid

**Failure mode 1: the "MTEB winner" trap.** You pick the top-of-MTEB model, do not run a domain-fit benchmark, and are surprised when it does not beat your current default on your data. This chapter exists to stop this.

**Failure mode 2: the "unfair comparison" trap.** Model A got `nDCG=0.68`, model B got `nDCG=0.62`, but model A ran with its recommended instruction prefix and model B did not. Or model A retrieved from a fine-chunked index and model B from coarse chunks. Or the qrels were labeled after seeing model A's output (giving A better labels). All three inflate model A over B. This is where "held-constant axes" earns its keep.

**Failure mode 3: the "tied is tied" trap.** Model A at 0.611, model B at 0.610. The team argues for two weeks over which to ship. Both are tied by any reasonable significance test. Ship whichever is cheaper. Deploy the discipline of "if p > 0.05 on the paired test, pick on cost, not on the point estimate."

## What is NOT in this chapter (and where it goes)

- **Detailed judgment-set methodology, pooled qrels, inter-annotator agreement, graded relevance carefully.** mod-109.
- **Chunker choice.** mod-102, exercise-03.
- **Retriever / ANN tuning.** mod-104.
- **Reranker choice.** mod-107.
- **Application-layer RAG eval (RAGAS, faithfulness).** Peer track `ai-eval-engineer` (level 30).

## What you should be able to do after this chapter

- Explain the five design decisions of a domain-fit benchmark and the trade-offs in each.
- Justify a 50-query minimum and name the query diversity axes.
- Pick a five-to-eight-model shortlist that spans hosted/open, size, and family shape, with BM25 as a mandatory baseline.
- List the axes that must be held constant (chunker, retriever, ANN params, top-K, per-model prompt handling) and the *one* axis that varies (the embedding model).
- Name the primary metric (nDCG@10), the secondaries (Recall@k, MRR@10), and the tie-breaker (paired Wilcoxon signed-rank on per-query deltas).
- Write a `benchmark_config.yaml` and a ranking / recommendation doc in the shape architect review expects.

## Further reading

- Thakur, N., et al. (2021). *BEIR* [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the benchmark design your domain-fit benchmark mirrors at 1/50th scale.
- Muennighoff, N., et al. (2022). *MTEB* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — the task metadata and per-task scoring shape (chapter 3 shows how to reuse the harness).
- Sakai, T. (2014). *Metrics, Statistics, Tests.* PROMISE '14. [PDF via TETSUYASAKAI.info](https://www.f.waseda.jp/tetsuya/pubs/promise2014-sakai.pdf) — the canonical reference for paired significance testing on retrieval metrics; used again in mod-109.
- `ir_measures`, [*Docs*](https://ir-measur.es/en/latest/) — computes nDCG, MAP, MRR, Recall from run + qrels files; the primary alternative to MTEB's built-in metric implementations.
- Cormack, G. V., & Grossman, M. R. (2017). *Navigating Imprecision in Relevance Assessments on the Road to Total Recall.* SIGIR — on why label noise limits the discriminative power of small eval sets.
- Craswell, N., et al. (2021). *Overview of the TREC 2020 deep learning track.* — a public example of the "same corpus + fixed qrels + rotating models" design shape.
- Anthropic engineering, [*Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — a public engineering case study of a domain-fit-shape ablation (chunk + embedding + reranker), useful as a shape reference.
