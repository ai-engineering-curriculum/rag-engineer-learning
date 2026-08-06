# Exercise 01 — Build a Judgment Set with Curated and LLM-Assisted Labels

**Estimated effort:** ~3 hours. Python, an LLM API (Claude Sonnet or GPT-4-class), your own corpus (from mod-102's ingestion) or a small public corpus, ~30-60 minutes of your own SME time for spot-checking. No GPU required.

## Objective

Produce a real, reusable judgment set on a corpus of your choice — TREC / BEIR shape, ≥ 60 queries, both curated (protocol A) and LLM-assisted-with-spot-check (protocol B), with the spot-check agreement number published.

By the end you have `eval/qrels.tsv`, `eval/queries.jsonl`, a corpus manifest, a labeling-protocol write-up, and the spot-check agreement metric. Every subsequent exercise in this module consumes this artifact. Mod-111's retriever CI pins to it.

## Prerequisites

- Chapter 2 read.
- A corpus of at least 500 chunks (or documents), ideally your own from mod-102's ingestion pipeline. If you don't have one, use one of these small BEIR datasets as a stand-in — download via `beir`'s `util.download_and_unzip`:
  - **NFCorpus** — 3.6k biomedical documents; small, downloads in seconds. Good starting corpus.
  - **SciFact** — 5k scientific abstracts; short queries; also small.
- At least 3 retrievers you can run over the corpus, for pooling in step 2. Any three of:
  - BM25 (via Elasticsearch, `rank_bm25`, `pyserini`, or a Lucene-family index)
  - A dense retriever with an off-the-shelf embedder (`bge-base-en-v1.5`, `e5-small-v2`, or a hosted embedding API)
  - A hybrid (BM25 + dense fused via RRF; mod-106 chapter 2)
- An LLM API key (Anthropic Claude, OpenAI GPT-4-class, or equivalent).
- Python 3.10+, `pip install ir_measures beir sentence-transformers scikit-learn`.
- 30-60 minutes of your own time in the middle of the exercise for the SME spot-check. Block it on your calendar.

## Steps

**Step 1 — Choose the corpus and doc-ID scheme (15 minutes).** In `eval/corpus-manifest.md`:

- Name the corpus source and version (git SHA, ingest timestamp, snapshot ID).
- Pick a doc-ID scheme: chunk-ID, parent doc-ID, or stable passage anchor (chapter 2's three options). Justify in one sentence.
- Report the corpus size (documents, chunks, average token count per chunk).

If you're using a BEIR dataset as a stand-in, the doc-ID scheme is fixed (`_id` in the corpus.jsonl); skip the justification but note the source.

**Step 2 — Author the query set (30 minutes).** In `eval/queries.jsonl`:

- Aim for **60-120 queries**. Under 60 will cost you significance-test power (chapter 4); over 120 lengthens the labeling budget without much marginal benefit for a first pass.
- **Preferred sources**, in order:
  1. Real user queries from logs (dedupe, PII-strip).
  2. Support-ticket questions.
  3. FAQ questions.
  4. SME-written questions where the SME did not first read the corpus.
- If you have no query history, use LLM query generation: for a stratified sample of 60 documents, prompt an LLM to write 1-2 questions the document would answer. Have a human (you) review the generated queries — drop implausible ones, edit awkward ones. Chapter 2's InPars / Promptagator references give the technique.
- One JSON object per line: `{"_id": "q1", "text": "..."}` (BEIR shape).

Publish a `eval/queries-meta.md` alongside: sources of queries, length distribution (median, min, max), topic-stratification breakdown if any.

**Step 3 — Pool candidate documents per query (30 minutes).** In `eval/pooling-runs/`:

- Run each of your 3 retrievers on all queries, top-20 each. Save as TREC run files:
  - `eval/pooling-runs/bm25.run`
  - `eval/pooling-runs/dense-{model}.run`
  - `eval/pooling-runs/hybrid-rrf.run`
- For each query, union the top-20 doc-IDs across all runs (deduplicating) → per-query candidate pool. Aim for 30-50 unique candidates per query on average.
- Persist the candidate pools as `eval/pool.jsonl`: one object per query with `{"query_id": ..., "candidates": [doc_ids...]}`.

**Step 4 — Protocol B: LLM-assisted labeling (45 minutes for setup + LLM run time).** In `label_llm.py`:

- Load queries and pooled candidates.
- For each (query, candidate) pair, call an LLM with chapter 2's minimal prompt (or a variant you tune):

    ```
    You are a relevance judge. Given a query and a candidate passage, decide the passage's relevance on this scale:
      0 = irrelevant
      1 = related but not directly answering
      2 = relevant (contains part of the answer)
      3 = highly relevant (directly and completely answers)
    Query: {query}
    Passage: {passage}
    Respond with a single line: LABEL: <integer 0-3>
    Then one sentence of justification: WHY: <reason>
    ```

- Cache aggressively (the query changes per pair; the passage repeats across queries — Anthropic/OpenAI cache-prefix on the passage side, or roll your own with a dict).
- Save labels + LLM justifications to `eval/labels-llm.tsv` — one row per (query, doc, label, why).

Cost planning: at ~60 queries × ~40 candidates = ~2,400 pairs. At Claude Sonnet 3.5 rates and ~500 tokens per pair, expect a few dollars and ~30-45 minutes wall-clock (parallelize with ~10 concurrent requests).

**Step 5 — Protocol A: SME spot-check on a stratified sample (30-45 minutes of your time).** In `label_spotcheck.py`:

- Sample **~10% of the LLM's labels, stratified by label class** — i.e., ~10% of the 0s, ~10% of the 1s, ~10% of the 2s, ~10% of the 3s. Not 10% of all labels (that would over-sample the majority class and miss disagreement on the interesting rare classes).
- Present each sampled pair to yourself (or a domain expert) in a plain UI — a Streamlit script, a Google Sheet, a printed list, whatever. Re-label without looking at the LLM's label. Save to `eval/labels-spotcheck.tsv`.
- Compute **agreement metrics** between LLM and human on the spot-check subset:
  - Overall accuracy (exact match on the 0-3 scale).
  - Cohen's κ (`sklearn.metrics.cohen_kappa_score`).
  - Per-class recall (of the class-1 pairs the human labeled, how many did the LLM also call class-1?).
- Publish to `eval/spot-check-agreement.md`.

**Step 6 — Deployment decision on protocol B (10 minutes).**

Chapter 2's thresholds:

- Overall accuracy ≥ 80% and Cohen's κ ≥ 0.6 → LLM labels are reliable enough to publish as qrels.
- Otherwise → escalate: try a better model, tune the prompt, or fall back to protocol A only. Do not publish qrels with LLM labels below this bar.

Document the decision in `eval/labeling-protocol.md`: which protocol you used, the spot-check numbers, whether you deployed the LLM labels or fell back to human-only, and *why*.

**Step 7 — Write the qrels file (15 minutes).** In `eval/qrels.tsv`:

- Header comment: `# corpus-version: <sha>`, `# doc-id: <scheme>`, `# labels: <protocol>`, `# spot-check: <accuracy, kappa>`.
- One line per judged pair: `query_id\t0\tdoc_id\trelevance`.
- Use the human's label for the ~10% spot-check subset; use the LLM's label for the un-reviewed 90%.
- Include the `0` labels (irrelevant judgments) explicitly — TREC-shape qrels need both positives and negatives from the pool, or precision metrics (chapter 3) mis-compute.

**Step 8 — Freeze and tag (10 minutes).**

- `git add eval/` and commit with a version tag: `v<YYYY-MM>-a` (matching mod-111's version-tag convention).
- Publish `eval/VERSION` with the tag. Future retriever CI runs pin to this tag; changing it invalidates historical comparisons.

## Acceptance criteria

- `eval/qrels.tsv` in TREC / BEIR shape with ≥ 60 queries and ≥ 30 judged (query, doc) pairs per query on average.
- `eval/queries.jsonl` in BEIR shape.
- `eval/corpus-manifest.md` documents corpus source + version + doc-ID scheme.
- `eval/queries-meta.md` documents query provenance and length distribution.
- `eval/labeling-protocol.md` documents which protocol (A or B) and the spot-check decision.
- `eval/spot-check-agreement.md` reports overall accuracy, Cohen's κ, per-class recall for the LLM-vs-human comparison on the stratified 10% sample.
- `eval/pooling-runs/` contains the run files from ≥ 3 retrievers used for pooling.
- `eval/VERSION` contains a version tag; the eval directory is committed to git.

## Stretch goals

- **Two-labeler protocol A.** For a 30-query subset of the pooled candidates, have *two* humans label independently. Compute inter-annotator κ. If κ < 0.6, the labels themselves are the noise floor — the retriever comparison in chapter 4 will not exceed this ceiling.
- **Adversarial subset.** Hand-author 10 queries designed to trip up specific failure modes (queries that use synonyms of corpus vocabulary; queries with typos; queries that reference an entity only mentioned once in the corpus). Label them by hand. Run your baseline retriever separately on this subset and compare to the aggregate — often surfaces regressions the general set does not.
- **Difficulty stratification.** After running your baseline retriever, mark each query with a difficulty band (top-1 relevant found: easy; top-10 relevant found: medium; not found in top-100: hard). Publish per-band metrics — chapter 5's ablation reads much more clearly with strata than without.
- **Synthetic query generation as a bootstrap.** If you have no query history, do the InPars / Promptagator approach: for 100 documents, generate 3 questions each; review; keep the survivors. Publish the LLM-side prompt so the technique is reproducible. Compare synthetic vs. any-real-query-you-do-have to see whether the synthesized distribution matches.
- **Publish as a shareable eval artifact.** If your corpus is public (or synthesizable), publish `eval/` as a GitHub-hosted BEIR-compatible dataset — a `beir`-loadable corpus.jsonl + queries.jsonl + qrels/test.tsv. Every downstream module can then pull it directly.

## Notes

- **Do not skip the spot-check.** Chapter 2's protocol B without a spot-check is not protocol B; it's untraced LLM labels. Every downstream ablation in this module amplifies whatever bias the LLM had. The 30-45 minutes of your own time is the highest-leverage investment in the module.
- **Do not label from a full corpus of thousands of docs.** Always pool. Pooling from 3 diverse retrievers surfaces the candidates most likely to be relevant, and TREC methodology (Voorhees 1998) has established that the resulting relative ranking of *pooled* retrievers is unbiased even when there are unjudged relevant docs.
- **Do not put PII in the query set.** Even if the queries come from real logs, strip identifiers, redact names, and log the redaction rules in `queries-meta.md`. The qrels file is meant to be re-read by future engineers, possibly external — hygiene matters.
- **Do not fake queries you couldn't get.** If the corpus is real but you have no user history, generate queries (LLM or SME); don't compose queries by paging through the corpus and asking "what does this doc say?" That is the "retrieve-your-own-questions" failure from chapter 2.
- **Version the qrels file.** Every change (adding queries, re-labeling a subset, expanding the pool) gets a new version tag. Older comparisons continue to pin to the old tag until they're re-run against the new one.
