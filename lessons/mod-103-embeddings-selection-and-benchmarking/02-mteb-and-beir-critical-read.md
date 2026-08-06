# Chapter 2 — Reading MTEB and BEIR Critically

## Why this chapter exists

Every embedding model card cites MTEB. Every product blog says "state of the art on BEIR." Every leaderboard is sorted on one column. The reflex reaction — pick the top row and ship — is exactly what this chapter exists to stop.

Two facts to internalize before you look at either leaderboard:

1. **MTEB is not one task.** It is a family of 50+ tasks across classification, clustering, pair classification, reranking, STS, summarization, and *retrieval* — and the aggregated leaderboard averages across them. A model that is #1 on the average can be well outside the top 10 on retrieval specifically, or vice versa.
2. **BEIR was designed to demonstrate that in-domain benchmark scores do not transfer.** The whole point of Thakur et al. 2021 was to show that a retrieval model trained on MS MARCO transfers unevenly to 18 other domains — and often loses to plain BM25 on the tail ones.

This chapter is how to read the two artifacts you will actually be pointed at, translate a leaderboard row into a claim about *your* corpus, and know when the answer is "run the benchmark yourself on my data" (which chapter 3 shows how to do and chapter 7 makes rigorous).

## MTEB — the Massive Text Embedding Benchmark

**What it is.** Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* ([arXiv:2210.07316](https://arxiv.org/abs/2210.07316)) is a benchmark suite bundled as a Python package (`mteb`) that runs a fixed set of embedding tasks and reports a per-task score plus an aggregated average. The [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) is the interactive view most people mean by "MTEB."

**The task taxonomy.** Task families in MTEB v1 (see the paper's Table 1 for the current authoritative list):

| Family | What it measures | Example datasets | Metric |
|---|---|---|---|
| Classification | Sentence → label | Amazon reviews, MTOP intent, Banking77 | Accuracy / F1 |
| Clustering | Group sentences into topical clusters | ArxivClusteringP2P, RedditClustering | V-measure |
| Pair classification | Is (sentence-A, sentence-B) a paraphrase? | SprintDuplicateQuestions, TwitterSemEval2015 | AP |
| Reranking | Given a query and a candidate list, rank them | MindSmallReranking, StackOverflowDupQuestions | MAP |
| **Retrieval** | Given a query, retrieve relevant docs from a corpus | The 15 datasets that make up **BEIR** (see below) | nDCG@10 |
| STS (Semantic Textual Similarity) | Score how similar two sentences are, correlate with human | STS-B, SICK-R | Spearman ρ |
| Summarization eval | Score similarity of a summary to references | SummEval | Spearman ρ |
| Bitext mining (v2+) | Match parallel sentences across languages | Tatoeba, BUCC | F1 |

**Why the taxonomy matters when you read the leaderboard.** If your job is retrieval — and in this track it is — then the average column on the leaderboard is *not* your metric. You want the **retrieval** column, and specifically the retrieval *subtasks* whose distribution most resembles your corpus (chapter 7 makes this concrete). A model with an outstanding STS score and a mediocre retrieval score is a good sentence-similarity model, not a good retriever.

### How to actually read the leaderboard

The [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) exposes columns for each task family. Two things to do every time you look at it:

1. **Sort by *the family you care about*, not "Average".** For this track, sort by **Retrieval Average (nDCG@10)**. The order changes.
2. **Filter by language.** MTEB has English (default), multilingual (MMTEB), Chinese (C-MTEB), French (F-MTEB), Polish (PL-MTEB), Korean (Ko-MTEB), and more. A leader on English is not automatically a leader on your language. Muennighoff et al., *MMTEB: Massive Multilingual Text Embedding Benchmark* ([arXiv:2502.13595](https://arxiv.org/abs/2502.13595)) is the current authoritative reference for the multilingual expansion.

**Bookkeeping columns that actually matter.** Beyond the score columns, look at:

- **Number of parameters / model size.** A 7B-parameter LLM-based embedder ("gte-Qwen2-7B-instruct") will beat a 33M-parameter MiniLM. That is not free — it is 10–100× the GPU memory to serve.
- **Embedding dimension.** Chapter 8 will show that dim × corpus size drives storage; 4096-dim vs. 384-dim is an order-of-magnitude cost delta before any quality signal.
- **Max tokens.** Some entries have 512-token limits; some have 8K or 32K. Your chunker (mod-102) is designed against a target token limit — pick a model whose limit matches.
- **License.** Check the model card: many "open" embedders are CC-BY-NC or Apache-2.0-with-restrictions; some (Cohere's Embed v3 open-weight distillations, some Chinese-origin models) have use-restriction clauses that block commercial deployment.

### The reproducibility caveat leaderboards do not shout

Two subtle points that will bite you if you skip them:

- **Prompt / instruction sensitivity.** Some embedders (E5, GTE-Qwen, `Snowflake Arctic Embed`) expect a *task-typed instruction* prefix — `"query: <text>"` and `"passage: <text>"`, or a per-task instruction like `"Retrieve semantically similar text: <query>"`. Reported leaderboard scores use the model card's recommended prompts. Skip the prompt and your score can drop by 5–20 nDCG points. The MTEB harness (chapter 3) uses the model card's prompt when the model is registered; a random `SentenceTransformer.encode(text)` in your code does not.
- **Score contamination.** Some embedders have been trained on data overlapping with MTEB tasks — sometimes deliberately (BGE's authors report which tasks they trained on), sometimes not. A leaderboard row is a lower bound on what the model *can* do on that specific test set, not on your corpus. Kim et al., *MTEB-French* / MMTEB v2 discussions surface the contamination issue explicitly; treat any score within noise of a peer's as "these are equivalent."

## BEIR — the out-of-domain retrieval benchmark

**What it is.** Thakur, N., et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)) — 18 retrieval datasets across 9 task types (Q&A, argument retrieval, fact-checking, biomedical, etc.), all evaluated in a **zero-shot** setting: you train (or pretrain) once and evaluate on all 18 without fine-tuning.

**The datasets you will see repeatedly.**

| Dataset | Domain | Query shape | Corpus size |
|---|---|---|---|
| MS MARCO | Web search | Natural-language queries | 8.8M passages |
| Natural Questions (NQ) | Wikipedia QA | Questions | 2.7M passages |
| HotpotQA | Multi-hop Wikipedia QA | Questions | 5.2M passages |
| TREC-COVID | Biomedical, COVID papers | Descriptive queries | 171k documents |
| BioASQ | Biomedical QA | Questions | 14.9M documents |
| NFCorpus | Nutrition, medical | Questions | 3.6k documents |
| FiQA-2018 | Financial QA | Questions | 57k answers |
| SciFact | Scientific claim verification | Claims | 5k abstracts |
| SciDocs | Scientific paper similarity | Papers | 25k papers |
| CQADupStack | Community Q&A | Duplicate questions | ~470k per subforum |
| Quora | Duplicate detection | Questions | 523k questions |
| ArguAna | Argument retrieval | Arguments | 8.6k arguments |
| Touché-2020 | Argument retrieval | Controversial questions | 383k documents |
| DBPedia | Entity retrieval | Entity-seeking queries | 4.6M entities |
| CLIMATE-FEVER | Climate-claim fact-checking | Claims | 5.4M docs |
| FEVER | Wikipedia fact-checking | Claims | 5.4M docs |
| Signal-1M | News retrieval | Tweets | 2.9M documents |
| Robust04 | News retrieval | Descriptive topics | 528k documents |

The **primary metric** across the whole benchmark is **nDCG@10**, aggregated as a simple average across the 18 datasets. (BEIR reports Recall@100 and MRR@10 secondarily.)

### The one BEIR result that changed how the field talks about retrieval

Thakur et al.'s Table 2 (the paper's central result) showed:

- **In-domain**, on MS MARCO alone, DPR (a bi-encoder) beat BM25 handily.
- **Out-of-domain**, averaged across the other 17 BEIR datasets, plain **BM25 beat DPR** — and beat several other well-cited dense retrievers.

This is the empirical basis for three of the field's now-standard practices:

- Never trust a single-corpus retrieval benchmark. Domain shift is real and large.
- Always run BM25 as your baseline; if your dense retriever cannot beat it *on your corpus*, use BM25.
- Hybrid retrieval (BM25 + dense fused) is a serious default, not a hack. mod-106 covers fusion in depth.

Contemporary embedders (BGE-M3, E5-mistral, GTE-Qwen, voyage-3, `text-embedding-3-large`) do beat BM25 on the BEIR average — but the *tail datasets* (Signal-1M, Touché-2020) still surprise, and the gap over BM25 varies wildly per dataset. Look at per-dataset columns, not just the average.

### What the BEIR score actually measures — and does not

**What it measures.** How well a retriever *pretrained on general data* transfers to 18 held-out corpora with human-labeled qrels, evaluated at nDCG@10.

**What it does not measure:**

- **Your corpus's terminology.** BEIR has biomedical (BioASQ) and financial (FiQA) subsets, but if your corpus is internal legal / customs / pharmaceutical / trading terminology, none of the 18 datasets is a proxy.
- **Your query distribution.** BEIR queries are mostly natural-language questions or descriptive topics. If your queries are code snippets, entity IDs, or JIRA-ticket-style short phrases, the BEIR score does not tell you what will happen.
- **Your document length.** MS MARCO passages are ~60 tokens; TREC-COVID docs are full abstracts (~200 tokens). Your 4000-token PDF chunks are outside both distributions.
- **Real latency budgets or storage costs.** BEIR is a quality benchmark, not a systems benchmark.

The right way to use BEIR: as a **prior**. A model that is Top-10 on BEIR average *and* Top-10 on the specific BEIR subset closest to your domain is a strong candidate to include in your own domain-fit benchmark (chapter 7). A model that is #1 on BEIR average but has never seen anything like your corpus is *also* a strong candidate — but you cannot know until you run it.

### Reading the "BEIR sub-average" trap

Public evaluations often report a "BEIR-14" or "BEIR-11" subset — some datasets (Signal-1M, TREC-News, BioASQ) were dropped for licensing or availability reasons. When you compare two model cards, first check which BEIR subset each reports on. Two numbers labeled "BEIR average" from different sources may aggregate different task lists. The [`beir` library](https://github.com/beir-cellar/beir) and the MTEB retrieval average are the two canonical reporting conventions; the numbers are close but not identical.

## The translation problem: from leaderboard score to your corpus

Here is the concrete failure mode that motivates chapter 7. Model A reports:

- MTEB Retrieval Average: **56.5 nDCG@10**
- BEIR average: **54.2 nDCG@10**

Model B reports:

- MTEB Retrieval Average: **58.1 nDCG@10**
- BEIR average: **53.9 nDCG@10**

Which one to deploy on your corpus of internal insurance policies?

The honest answer is **you cannot tell from the leaderboard**. The 1.6-point MTEB delta may be within noise, or may be driven by MS MARCO improvements that do not transfer. The 0.3-point BEIR delta is definitely within noise. What determines the answer is:

- Which model's *pretraining and fine-tuning data* is closer to insurance-policy language.
- Whether either model's training corpus overlapped with any MTEB task, inflating its number.
- Whether your queries look like MS MARCO questions or like the terse "coverage for X under Y" phrasing your actual users type.
- Storage, latency, cost per million tokens for the shape you serve at.

The only credible way to answer this question is to run *both* on a small, corpus-native, hand-labeled eval set. That is chapter 7's design work. Chapter 3 shows how to plug that eval set into the MTEB harness so you can reuse the reporting.

## Three practical rules for reading either leaderboard

**1. Sort by the axis that matches your role.** For retrieval work, sort MTEB by retrieval nDCG@10, not the top-level average. For reranker work, look at the "reranking" family (or the MS MARCO passage reranking numbers cross-encoder authors publish). For classification / clustering, use those columns.

**2. Filter models to a serving shape you can afford.** Add these implicit filters:

- Embedding dimension ≤ (your target storage per chunk) — often ≤ 1024.
- Model params ≤ (your GPU / API budget) — 7B-parameter LLM embedders are gorgeous on paper and painful in production.
- Max input tokens ≥ (your chunker's target chunk size).
- License compatible with your deployment.
- Language / multilingual support matches your corpus.

**3. Look at *variance across tasks* as a proxy for robustness.** A model that averages 55 nDCG on 15 retrieval subtasks with a per-task standard deviation of 4 is more predictable than one that averages 56 with a stdev of 12. The high-variance model has some tasks it aces and some it fails — one of the failing ones might be yours.

## Concrete: what to write down before you leave the leaderboard

For every model you shortlist for chapter 7's domain-fit benchmark, capture in a table (`docs/model-shortlist.md`):

| Model | Provenance | Params | Dim | Max tokens | MTEB retrieval nDCG@10 | BEIR average | License | Hosted-or-open |
|---|---|---|---|---|---|---|---|---|
| … | HF repo or vendor page URL | … | … | … | … | … | … | … |

That table becomes the input to chapter 7. Six to twelve rows is the right shortlist size — narrow enough to run in an evening, wide enough to catch a surprise winner.

## What is NOT in this chapter (and where it goes)

- **Actually running an MTEB task locally.** Chapter 3, complete with the code.
- **Doing your own labeling and running your own retrieval eval.** Chapter 7, and mod-109 goes deeper on judgment-set construction and paired significance testing.
- **The training methodology behind BGE / E5 / GTE / etc.** Chapter 5 surveys the model families with pointers into their papers; deep dive on training belongs to the peer `fine-tuning-engineer` track.
- **RAG-triad application-layer eval (faithfulness, answer relevance, RAGAS).** Peer track `ai-eval-engineer` (level 30) owns that.

## What you should be able to do after this chapter

- Explain that MTEB is a *taxonomy* of tasks and that the top-level average is not the number you sort on for a retrieval role.
- Explain what BEIR measures (zero-shot, 18 domains, nDCG@10) and cite the paper's central finding (BM25 beats DPR out-of-domain averaged over 17 datasets).
- Name at least six of the BEIR datasets by domain (MS MARCO, NQ, HotpotQA, TREC-COVID, FiQA, SciFact are a good starter set).
- Read a model card and pull the four numbers that go on your shortlist table: params, dim, max tokens, retrieval nDCG@10 (with a note on which subset).
- Explain why two models within ~1 point of each other on MTEB or BEIR are effectively tied and why the tie-breaker is a domain-fit run on your data.
- Name the trap of "BEIR-11" vs. "BEIR-14" reporting subsets when comparing across model cards.

## Further reading

- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316).
- Muennighoff, N., et al. (2024). *MMTEB: Massive Multilingual Text Embedding Benchmark.* [arXiv:2502.13595](https://arxiv.org/abs/2502.13595) — the multilingual expansion.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* [arXiv:2104.08663](https://arxiv.org/abs/2104.08663). NeurIPS Datasets and Benchmarks.
- Hugging Face, [*MTEB Leaderboard*](https://huggingface.co/spaces/mteb/leaderboard) — the live leaderboard; sort by Retrieval nDCG@10 for this track.
- MTEB repo, [`embeddings-benchmark/mteb`](https://github.com/embeddings-benchmark/mteb) — source of truth for the current task list, per-task metrics, and per-model prompt conventions.
- BEIR repo, [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — the reference harness, corpus / qrels download scripts, and canonical BM25 baseline scores.
- Reimers, N., [*The importance of retrieval evaluation on domain data*](https://www.sbert.net/examples/applications/information-retrieval/README.html) — sbert.net's official framing of why in-domain eval is not optional.
