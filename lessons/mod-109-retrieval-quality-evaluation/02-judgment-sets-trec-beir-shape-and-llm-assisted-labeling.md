# Chapter 2 — Judgment Sets: The TREC / BEIR Shape and How to Build One

## Why this chapter exists

Every retrieval metric in chapter 3 takes two inputs: a **run file** (queries → ranked list of doc-IDs, produced by your retriever) and a **qrels file** (queries → relevant doc-IDs, produced by humans and/or LLMs with a spot-check). The metric harness is standardized; the judgment set is what you have to build. Every hour spent tuning HNSW without a judgment set is speculative. Every hour spent on the judgment set pays interest across every future retrieval decision — model swaps, chunker changes, reranker upgrades, CI gates, drift alerts.

This chapter is the artifact: the file format that every toolchain in the field consumes, the two labeling protocols (curated and LLM-assisted), and the human spot-check that keeps LLM-assisted labels honest.

## The qrels format is TREC's, and it never changed

TREC (the Text REtrieval Conference, hosted at NIST since 1992) settled on a plain-text qrels format decades ago. Every downstream toolchain — `trec_eval`, `pytrec_eval`, `ir_measures`, BEIR, `ranx`, `pyterrier` — accepts it directly.

**Qrels file (`qrels.txt`) — one judgment per line, whitespace-separated:**

```
query_id  iteration  doc_id  relevance
q1        0          d17     1
q1        0          d92     1
q1        0          d5      2
q1        0          d33     0
q2        0          d43     1
q2        0          d128    1
```

- **`query_id`** — string identifier for the query. Must match the run file.
- **`iteration`** — an unused TREC-era field, always `0`. Present for backward compatibility.
- **`doc_id`** — string identifier for the document (or chunk) in your corpus.
- **`relevance`** — non-negative integer. `0` = irrelevant, `1` = relevant, `2` = highly relevant, etc. Binary metrics (recall, precision, hit-rate, MRR) treat any `> 0` as relevant. Graded metrics (nDCG, ERR) use the numeric value.

**Run file (`run.txt`) — one hit per line, six whitespace-separated fields (TREC format):**

```
query_id  iteration  doc_id  rank  score  run_name
q1        Q0         d17     1     0.912  bge-large-en-v1.5
q1        Q0         d5      2     0.876  bge-large-en-v1.5
q1        Q0         d92     3     0.844  bge-large-en-v1.5
...
```

- **`iteration`** — always the literal string `Q0` in run files (TREC quirk; do not use `0`).
- **`rank`** — 1-indexed position in the ranked list.
- **`score`** — retriever score (used to sort; some metrics reorder from `score` if `rank` looks wrong).
- **`run_name`** — string identifier; used when comparing multiple runs.

Both formats also come in **JSONL** shapes (BEIR's default) and a Python-dict shape (`ir_measures.Qrel`, `ir_measures.ScoredDoc`) — all convert 1:1. Pick one at the start of your project and stick with it; the friction of maintaining two shapes is not repaid.

**BEIR's flavor.** BEIR's `qrels/{split}.tsv` is the same shape with a TSV header row and tab separators. Its `corpus.jsonl` and `queries.jsonl` sit alongside — one JSON object per line, keyed by `_id`. The `ir_measures` library reads either; when you author a new set for your own corpus, follow BEIR's layout so you can drop it into any BEIR-compatible harness later.

## What the doc-ID is (the decision that bites you a year later)

Before you label a single query, decide what the `doc_id` in your qrels refers to. Three common choices, each with a downstream constraint:

1. **`doc_id` = a chunk.** The judgment records "chunk `d17` is relevant to `q1`." Metrics compute directly on the chunk list your retriever returns. Straightforward, but every re-chunking (mod-102 chunk-size sweep, new hierarchical scheme) invalidates the judgment set — the chunk-IDs change.
2. **`doc_id` = a source document (parent).** The judgment records "document `paper-42.pdf` is relevant to `q1`." Retriever hits are collapsed to their parent doc before matching against qrels (usually first hit wins per parent). Survives re-chunking; loses granularity — a retriever that returns the *right document, wrong section* scores the same as one that returns the exact passage.
3. **`doc_id` = a stable passage anchor.** A stable ID like `paper-42.pdf#page=7#para=3`, tracked as metadata across re-chunking runs. Survives re-chunking; requires the ingest pipeline (mod-102) to preserve the anchor across pipeline versions.

For this module's exercises, option 3 is the default recommendation — the passage anchor. It matches the citation anchors mod-107 assembled and mod-108 rendered, and it does not blow up on a chunker swap. If your ingest doesn't yet preserve stable anchors, option 2 (parent document) is a fine starting point; option 1 is a last resort.

Whichever you pick, write it in the qrels file's header comment (`# doc_id = passage_anchor`) so the next engineer to touch it doesn't have to reverse-engineer.

## Two protocols for producing judgments

Every relevance label in a qrels file comes from one of two places: a human wrote it (curated), or an LLM wrote it and a human spot-checked (LLM-assisted). Both are legitimate; pick based on scale, budget, and the sensitivity of the domain.

### Protocol A — Curated (SME-labeled)

**When to use.** High-stakes domains (medical, legal, compliance, financial), narrow corpora (< 10k documents), small query sets (< 300), or as the human spot-check layer on top of protocol B.

**The workflow.**

1. **Author the query set.** Sources: real user queries from logs, questions from FAQ documents, questions that domain SMEs write to test coverage, questions extracted from support tickets. Aim for 50–300 queries at first — enough for chapter 4's significance tests to have power, small enough for a human to actually label.
2. **Pool candidate documents per query.** Never label from the full corpus (too expensive, missing recall). Instead, **pool** — run three or four diverse retrievers (BM25, one dense embedder, one hybrid, optionally a reranker) top-20 each, union the resulting candidate lists. TREC's original methodology; still the field default because it bounds the labeling budget while preserving unbiased relative comparisons among the retrievers whose output was pooled. Voorhees (1998) is the reference.
3. **Present each (query, candidate) pair to the labeler.** In a simple UI (Streamlit / Label Studio / a Google Sheet is fine). Two questions: `[irrelevant / partially / relevant / highly relevant]` on a 0-3 scale, and an optional freeform note ("query is ambiguous," "wrong entity"). Graded relevance costs more time than binary but unlocks nDCG's tie-breaking.
4. **Adjudicate disagreements.** If two labelers disagree, a third breaks the tie or the pair is marked ambiguous and dropped. Track inter-annotator agreement (Cohen's κ or Fleiss's κ) so you know how noisy the ceiling is — if κ < 0.6, the labels themselves are the bottleneck, not the retriever.
5. **Write the qrels file.** One line per judged (query, doc) pair. Explicit `0` labels for pooled candidates that turned out irrelevant — this matters for precision metrics.

**Cost model.** A trained labeler handles 40–80 pairs per hour on plain-text passages; slower on structured / technical content. For a 100-query set with 4-retriever pool of top-20 each, deduplication brings the labeling load to ~40–60 unique candidates per query, or 4,000–6,000 pairs total, or 50–150 labeler-hours. Two labelers with adjudication doubles it.

### Protocol B — LLM-assisted with mandatory human spot-check

**When to use.** Broader corpora (100k–10M documents), large query sets (300–10k), rapid iteration cycles, or when SME time is the binding constraint. **Never** as the sole labeling source — the spot-check is not optional. See Faggioli et al. (2023) "Perspectives on Large Language Models for Relevance Judgment" for the field's current framing.

**The workflow.**

1. **Author (or extract) the query set.** Same as protocol A. If your corpus has zero query history, an LLM can *synthesize* candidate queries from documents (chapter 6's BEIR-subset labeling uses this too; the technique is sometimes called **query generation** or **synthetic qrels**). See Bonifacio et al., *InPars* ([arXiv:2202.05144](https://arxiv.org/abs/2202.05144)) and Dai et al., *Promptagator* ([arXiv:2209.11755](https://arxiv.org/abs/2209.11755)) for the two canonical approaches. Prompt shape: "Given this passage, write three questions a user of this system would ask that this passage answers." Human review of the *queries* is much cheaper than human review of the labels — do the query review carefully.
2. **Pool candidate documents per query.** Same as protocol A.
3. **Have the LLM label each (query, candidate) pair.** Prompt shape (a minimal starter — Faggioli et al. and MacAvaney & Soldaini (2023) contain more thoroughly tested variants):

    ```
    You are a relevance judge. Given a query and a candidate passage,
    decide the passage's relevance on this scale:

    0 = irrelevant (the passage does not address the query at all)
    1 = related but not directly answering (same topic, doesn't answer)
    2 = relevant (the passage contains part of the answer)
    3 = highly relevant (the passage directly and completely answers)

    Query: {query}

    Passage: {passage}

    Respond with a single line: LABEL: <integer 0-3>
    Then one sentence of justification: WHY: <reason>
    ```

    Use a modern instruction-tuned model (Claude Sonnet, GPT-4-class, or better). Cache aggressively — the passage half of the prompt repeats across queries and is the ideal cache-prefix candidate on Anthropic / OpenAI.
4. **Mandatory human spot-check.** Sample ~10% of the LLM's labels (stratified — 10% of each label class, not 10% of all pairs) and re-label them by hand. Compute agreement between LLM labels and human labels: overall accuracy, per-class recall, Cohen's κ. **Deploy thresholds:** if overall agreement is < 80% or κ < 0.6, the LLM judge is not reliable enough for your domain; escalate to protocol A or invest in a better prompt / better model. Publish the agreement number alongside the qrels file — every downstream user of the qrels should be able to see what the label ceiling is.
5. **Adversarial review of high-consequence labels.** For every query whose top-5 retriever result flipped `0 ↔ 3` or `1 ↔ 3` under a run comparison (chapter 3), have a human re-label the flipped pair. Sakai & Zeng (2023) and the LLMJudge track at TREC 2024 both report that LLM judges skew toward the retrieval families they were trained on — a bias that only shows up on flipped pairs.
6. **Write the qrels file** with the LLM's labels for the un-reviewed 90% and the human's labels for the spot-checked 10%. Mark spot-checked labels with a comment if your format supports it.

**Cost model.** Order-of-magnitude cheaper than protocol A on labeling, plus 5-10% human time for spot-check. The trade is *bias risk* — LLM judges have systematic preferences (recency, verbosity, specific writing styles). The spot-check is what surfaces those biases before they poison a year of retriever comparisons.

### Which to pick

| Situation | Protocol |
|---|---|
| Regulated domain (legal, medical, compliance) | A (curated) |
| < 300 queries, SME available | A (curated) |
| 300–10k queries, iterative work | B (LLM-assisted + spot-check) |
| Zero query history, need to bootstrap | B (LLM-generated queries + LLM labels + spot-check) |
| Cross-check on protocol B's honesty | A on a 100-query random subset |

Most production RAG teams end up running **both**: protocol A on a 100-query "gold" set that never moves, protocol B on a 1,000+ query "silver" set that grows over time. The gold set is the anchor for release gates (mod-111); the silver set is the wide-coverage detector for regressions the gold set does not sample.

## The query set matters as much as the labels

A judgment set is only as representative as its queries. Three failure modes to avoid:

- **Retrieve-your-own-questions.** If the queries were written by looking at the corpus ("what does document X say?"), retrievers that memorize the corpus will look great. Real users write queries *without* seeing the corpus first. Prefer queries from support tickets, chat logs, or SMEs who authored questions without paging through documents.
- **Monoculture on query length.** If all queries are 5-8 word natural-language questions, you won't measure how your retriever handles single-word terms, code snippets, entity IDs, or long descriptive queries. Track query-length distribution and match it to your real user distribution.
- **Skewed by topic.** If 80% of your queries are about product A and 5% about product B, product A's retriever regressions will dominate the metric and product B's will be invisible. Stratify sampling by topic and either weight or report per-stratum numbers.

Publish a `queries-meta.md` alongside the qrels file: sources of queries, length distribution, topic stratification, dates covered. That metadata is what makes the eval *reproducible* by the next engineer — the queries themselves don't self-explain.

## Concrete: a minimal end-to-end example

**Corpus:** 5,000 chunks of internal engineering runbooks (chunk-ID format `runbook-<slug>#chunk=<n>`).

**Queries:** 120 support-ticket questions from the last quarter, deduplicated.

**Pooling:** BM25 top-20 + `bge-base-en-v1.5` top-20 + hybrid RRF top-20, union → ~35 unique candidates per query on average.

**Labels:** Protocol B — Claude Sonnet judges every (query, candidate) pair on the 0-3 scale; a subject-matter engineer spot-checks 10% (~420 pairs).

**Files checked into git:**

```
eval/
├── qrels.tsv                    # TREC/BEIR-shape qrels
├── queries.jsonl                # BEIR-shape query set
├── corpus-manifest.md           # doc-id scheme, corpus snapshot, source
├── queries-meta.md              # query provenance, length dist, topic mix
├── labeling-protocol.md         # protocol B, prompt, model, spot-check %
├── spot-check-agreement.md      # κ, per-class recall, deployed threshold
├── pooling-runs/                # run files of the retrievers used to pool
│   ├── bm25.run
│   ├── bge-base.run
│   └── hybrid-rrf.run
└── VERSION                      # eval-set version tag (mod-111 pins to this)
```

That directory becomes the input to every subsequent chapter. Chapter 3 turns it into metrics; chapter 4 turns per-query metric arrays into significance tests; chapter 5 runs an ablation grid over it; mod-111 pins the CI gate to this version tag.

## What you should be able to do after this chapter

- Write out a qrels file in TREC / BEIR shape and explain each of the four fields.
- Explain the doc-ID choice (chunk vs. document vs. stable passage anchor) and pick one for your corpus.
- Design a pooled candidate set from ≥ 3 diverse retrievers per query.
- Run protocol A (SME-curated) for a ~100-query set with two labelers and κ tracked.
- Run protocol B (LLM-assisted) with an explicit spot-check on ≥ 10% stratified across classes, publish agreement, and refuse to deploy if κ < 0.6.
- Enumerate the three query-set failure modes (retrieve-your-own-questions, length monoculture, topic skew) and defend a query set against them.

## Further reading

- Voorhees, E. M., & Harman, D. K. (Eds.). (2005). *TREC: Experiment and Evaluation in Information Retrieval.* MIT Press. [DOI:10.7551/mitpress/6916.001.0001](https://doi.org/10.7551/mitpress/6916.001.0001) — the TREC methodology reference; the pooling protocol chapters 2 and 3 use.
- Voorhees, E. M. (1998). *Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness.* SIGIR. [DOI:10.1145/290941.291017](https://doi.org/10.1145/290941.291017) — the "pooled judgments are stable under labeler disagreement" result.
- Thakur, N., et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — BEIR qrels / queries / corpus layout, which chapter 2's example follows.
- Faggioli, G., Dietz, L., Clarke, C. L. A., Demartini, G., et al. (2023). *Perspectives on Large Language Models for Relevance Judgment.* ICTIR. [arXiv:2304.09161](https://arxiv.org/abs/2304.09161) — the community's framing of LLM-as-judge for qrels; the reference behind protocol B's spot-check discipline.
- MacAvaney, S., & Soldaini, L. (2023). *One-Shot Labeling for Automatic Relevance Estimation.* SIGIR. [arXiv:2302.11266](https://arxiv.org/abs/2302.11266) — one-shot LLM labeling and its calibration.
- Thomas, P., Spielman, S., Craswell, N., & Mitra, B. (2024). *Large Language Models can Accurately Predict Searcher Preferences.* SIGIR. [arXiv:2309.10621](https://arxiv.org/abs/2309.10621) — Microsoft's evidence that LLM labels track human labels well enough to use, with caveats matching this chapter's spot-check requirement.
- Bonifacio, L., et al. (2022). *InPars: Data Augmentation for Information Retrieval using Large Language Models.* SIGIR. [arXiv:2202.05144](https://arxiv.org/abs/2202.05144) — synthetic query generation from passages.
- Dai, Z., et al. (2022). *Promptagator: Few-shot Dense Retrieval From 8 Examples.* [arXiv:2209.11755](https://arxiv.org/abs/2209.11755) — few-shot prompting for query generation; alternative to InPars.
- Sakai, T., & Zeng, Z. (2023). *On Estimating Variances for Topic Set Size Design.* [arXiv:2304.12063](https://arxiv.org/abs/2304.12063) — how many queries you actually need for chapter 4's significance tests.
- TREC 2024 [LLMJudge track overview](https://llm4eval.github.io/LLMJudge/) — current community results on LLM-judge reliability for IR relevance.
