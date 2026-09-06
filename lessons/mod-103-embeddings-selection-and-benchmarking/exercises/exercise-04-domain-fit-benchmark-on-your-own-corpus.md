# Exercise 04 — Domain-Fit Benchmark on Your Own Corpus

**Estimated effort:** ~3 hours (plus corpus + eval-set labor if you do not have those already).

## Objective

This is the capstone of the module. You will design and run the domain-fit benchmark chapter 7 laid out end-to-end: pin a corpus, build (or reuse) a 50+ query hand-labeled eval set, hold every other axis constant, sweep at least **five** embedding candidates spanning family shape and provenance, and produce a defensible ranked recommendation table with paired significance tests. The output is the artifact that would justify the embedding choice for this corpus in an architecture review — not a leaderboard row, not a vibe, an on-your-data measurement.

Every subsequent module in this track — vector-DB tuning (mod-104), hybrid retrieval (mod-106), reranker choice (mod-107), evaluation harness (mod-109), retrieval MLOps (mod-111) — assumes the artifact this exercise produces exists.

## Prerequisites

- Chapters 1 through 7 read once. Chapter 7 is the design template; chapter 6 is the cost lens you fold in at the end.
- Exercise-02 (MTEB harness) and exercise-03 (adapters for one hosted + one self-hosted model) complete. You will reuse both.
- A corpus. Ranked from best to worst source:
  1. The output of mod-102 exercise-03 or exercise-04 (your chunked corpus with a `chunker.yaml` you already pinned).
  2. A publicly available domain corpus (an OSS project's docs, a company handbook that is public, a set of research papers on one narrow topic).
  3. As a last resort, a BEIR dataset (FiQA, TREC-COVID, NFCorpus) treated as if it were "yours". Note it in the deliverable — the exercise is worth less if you skip the on-your-data labeling.
- Python deps: `mteb`, `sentence-transformers`, `pandas`, `scipy`, `rank_bm25` (or Pyserini), plus the SDKs for whichever hosted models you include.
- Enough time on a GPU (or patience on CPU) to embed the corpus 5–8 times. If your corpus is huge, chunk-sample it down to ≤50k chunks for the exercise; the methodology is unchanged.

## Part A — Freeze the corpus and the eval set (60 min)

Chapter 7's Phase 1. Two artifacts, checked into version control, immutable for the rest of the exercise.

### A.1 — Corpus

- Copy (or re-chunk once) into `data/corpus/chunks.jsonl` with fields `{"chunk_id": ..., "doc_id": ..., "section_path": ..., "text": ...}`. Every subsequent run reads this file — do not regenerate mid-sweep.
- Note the `chunker.yaml` (id, version, chunk_size, overlap) at the top of `docs/benchmark-config.md`. If you did not use a mod-102 chunker, describe the segmentation in enough detail that a colleague could reproduce it.

### A.2 — Queries and qrels

Aim for **≥50 queries** (30 is a hard floor; below that the ranking is noise per chapter 7). Sourcing, ranked as chapter 7 argued:

1. Real user queries from a search log.
2. Domain-expert-written questions.
3. LLM-generated + human-reviewed queries. **Not** LLM-generated + LLM-scored — that is the labeling contamination trap chapter 7 called out.

Diversity requirements (from chapter 7, Decision 1):

- Cover granularity: fact-lookup, section-summary, full-doc-scope.
- Cover vocabulary: corpus's own terms, and paraphrases a user would type.
- Cover query length: some short, some long.
- Cover query shape: at least some questions, some phrases, some entity names.
- Include ≥5 intentionally hard cases — near-synonyms, edge cases, or out-of-scope queries.

Save the queries as `data/judgments/queries.tsv` (`qid \t query_text`).

Label the qrels as `data/judgments/qrels.tsv` in TREC format:

```
qid  0  chunk_id_or_doc_id  rel
```

Pick coarse (doc-level) or fine (chunk-level) labels — chapter 7 said fine is stronger but slower; either is acceptable. Note the choice explicitly in the config doc. Binary `rel ∈ {0, 1}` is fine; if you have time for graded `{0, 1, 2}`, nDCG rewards it.

**Freeze the labels before you run any model.** If you catch a labeling error later, note it in the doc, do not silently change the qrels file mid-sweep. Bump `qrels_revision` if you must.

### A.3 — `docs/benchmark-config.md`

Chapter 7's `benchmark_config.yaml`, but as a document header:

```markdown
# Domain-fit benchmark — {corpus name}, {date}

- Chunker: {chunker_id}@{version}, chunk_size={n}, overlap={m}
- Corpus: {path}, n_chunks={n}, revision={sha or v1}
- Queries: {path}, n_queries={≥50}
- Qrels: {path}, revision=v1, labeling={coarse|fine}, graded={y|n}
- Retriever backend: pgvector (or Qdrant or in-memory) with cosine distance
- ANN: HNSW, M={m}, efConstruction={c}, efSearch={s}
- top_k: 100 retrieved per query; nDCG@10 primary
- Seed: 0
- mteb version: {version}
```

## Part B — Shortlist the candidates (30 min)

At least **five**, spanning:

- **Family.** ≥1 bi-encoder. Optionally 1 sparse-neural (SPLADE via `naver/splade-v3` on a Lucene backend if you have one) or 1 ColBERT (RAGatouille + `answerdotai/answerai-colbert-small-v1`). If you skip SPLADE/ColBERT for the exercise, note the omission.
- **Provenance.** ≥1 hosted API, ≥2 self-hosted open.
- **Size / cost tier.** Include a small model (`bge-small` at 384 dim or `text-embedding-3-small` at 256 dim MRL) and a mid-size model (`bge-large`, `voyage-3`, `text-embedding-3-large` at 1024 dim, or `snowflake-arctic-embed-l`).
- **Domain-specialized model** if one applies (`voyage-code-3` for a code corpus, `voyage-law-2` for legal, `bge-m3` for multilingual). Otherwise pick another well-cited generalist.

**Mandatory baseline row:** BM25. Chapter 2 and chapter 7 both argued this is not optional. Use `rank_bm25` for a service-free implementation (`BM25Okapi(tokenized_corpus)`), or Elasticsearch if you already run it. Note the tokenizer and stopword handling — those matter.

**Mandatory reference row (if applicable):** whatever embedding model your current stack ships. If you have no current stack, mark the row `n/a` and pick a defensible default (`BAAI/bge-large-en-v1.5`) as the reference instead.

Persist as `docs/shortlist.md` — model, provenance, family, dim, max tokens, MRL support, prompt requirement, license, cost per 1M tokens (or GPU rate for self-hosted). This is chapter 2's shortlist table upgraded with cost.

## Part C — Automate the sweep (45 min)

Write `scripts/run_sweep.py`. Two options for the harness:

- **Option 1 (preferred): MTEB.** Register the corpus + queries + qrels as an `AbsTaskRetrieval` subclass (chapter 3 pattern; exercise-02 Part D built exactly this). Then loop your shortlist through `mteb.MTEB(tasks=[YourTask()]).run(model, output_folder=...)`.
- **Option 2: `ir_measures`-based custom harness.** Embed corpus, embed queries with the right prefix, cosine top-100 with numpy, score with `ir_measures.calc(qrels, run)`. Simpler; loses MTEB's reporting shape but works.

Requirements:

- Every model gets its recommended prefix / role type / instruction. Chapter 3 covered E5, GTE-Qwen, Nomic, Cohere, Voyage, Google. Skipping the prefix is a bug, not a "controlled variable".
- Every model gets the same `top_k=100` retrieval depth.
- Every model reads the *same* frozen `data/corpus/chunks.jsonl`. Do not re-chunk between models.
- Persist per-model **per-query** nDCG@10 (not just aggregates) to `results/per_query/<model>.jsonl`. Part D depends on the per-query values.
- Log wall-clock embed time and vector-DB ingest time per model.

## Part D — Aggregate, rank, and significance-test (30 min)

Write `scripts/build_ranking.py`:

1. Aggregate per-model JSON into `docs/domain-fit-ranking.md`:

   ```markdown
   | Model | Provenance | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | offline_embed_s | native_dim |
   |---|---|---|---|---|---|---|---|
   | ... | ... | ... | ... | ... | ... | ... | ... |
   ```

   Sort by nDCG@10 descending. Include the BM25 row and the reference row.

2. Run **pairwise Wilcoxon signed-rank** on per-query nDCG deltas for the top-4 models. Chapter 7 covered this — `scipy.stats.wilcoxon`. Output a triangular p-value matrix in the same doc.

3. For each pair with `p > 0.05`, annotate "tied by paired test". Chapter 7 said: when tied, pick on cost, not on the point estimate.

## Part E — Per-query error analysis (30 min)

Aggregate scores hide where models actually differ. Do exactly what chapter 7 Phase 3 step 4 says:

- Find 5 queries where the top-1 model's nDCG@10 beats the runner-up by ≥0.5.
- Find 5 queries where the runner-up beats the top-1 by ≥0.5.
- Read the queries and the top hits from both models. Write two paragraphs in `docs/domain-fit-ranking.md` under an `## Error analysis` section — what shape of queries does each model handle better? Do the losses cluster around a specific vocabulary / query pattern / doc type?

If you find any query where **all** models score 0, the qrels for that query are likely wrong (a relevant chunk is not labeled). Chapter 7 said: fix the qrels, bump the revision, re-run only that query — do not silently ignore.

## Part F — Fold cost in and write the recommendation (30 min)

Merge the ranking table with the chapter-6 cost table (from exercise-03 methodology). Produce `docs/domain-fit-recommendation.md`:

```markdown
# Embedding recommendation — {corpus}, {date}

## Ranking (paired-test-adjusted)

| Model | nDCG@10 | Tied group | Offline embed $ | Query embed $/mo | Storage $/mo | Recommendation |
|---|---|---|---|---|---|---|
| ... | ... | A | ... | ... | ... | ✅ ship |
| ... | ... | A | ... | ... | ... | ⛔ tied but costlier |
| ... | ... | B | ... | ... | ... | ⛔ below floor |
```

Write three closing paragraphs:

1. **The recommendation.** Which model would you ship, and why. Name the paired-test winner (or the cheapest of the top tied group if there is a tie). Include the delta vs. the reference row and vs. the BM25 floor.
2. **The risks.** Chapter 7 named three failure modes (MTEB winner trap, unfair comparison, tied is tied); which one bit you hardest here, and what would you re-run to reduce the risk?
3. **The next step.** If the winner does not beat the baseline by a comfortable margin, chapter 9 argued the next move is usually hybrid retrieval or a reranker before considering fine-tuning. Name the specific mod-106 or mod-107 experiment you would run before spending training budget.

## Deliverables

- `data/corpus/chunks.jsonl`, `data/judgments/{queries.tsv,qrels.tsv}` — frozen artifacts, checked in.
- `docs/benchmark-config.md`, `docs/shortlist.md`.
- `scripts/run_sweep.py`, `scripts/build_ranking.py`.
- `results/per_query/<model>.jsonl` — per-model per-query nDCG@10 values.
- `docs/domain-fit-ranking.md` — ranking table + Wilcoxon matrix + error analysis.
- `docs/domain-fit-recommendation.md` — final ranked table with cost + recommendation.

## Acceptance criteria

- The eval set has ≥50 queries with per-query labels; the corpus is frozen (one file, one revision) across the whole sweep.
- The sweep runs ≥5 embedding models plus BM25, plus the reference (or a stated absence of one).
- Every model in the sweep uses its recommended prompt / role type / instruction. State per model which prefix was applied.
- `docs/domain-fit-ranking.md` has: the ranked table, the pairwise Wilcoxon matrix, the error-analysis paragraphs.
- `docs/domain-fit-recommendation.md` names exactly one recommended model, states the quality floor used (BM25 + Δ), and includes the cost delta vs. the reference row.
- No model in the sweep beats another *only* because it was run with a prefix that its competitor was denied — the config doc should make the per-model prompt handling auditable.

## Stretch goals

- Add a **sparse-neural** row (SPLADE via `naver/splade-v3` on Elasticsearch or Vespa) and a **ColBERT** row (`RAGatouille` + `colbertv2.0`) to the sweep. They will score differently from the dense rows; interpret them via chapter 1's role map — is the ColBERT model at reranker-quality? Would fusion of SPLADE + a top dense model beat both alone (preview of mod-106)?
- Add a **quantization ablation**: pick the winning dense model, produce (float32, int8, MRL-256, MRL-256+int8) variants using chapter 8's mechanics, run all four through the same benchmark. Confirm chapter 8's claim that int8 costs single-digit percent nDCG.
- Run the sweep on **two chunker versions** (mod-102 exercise-03 gave you at least two viable chunkers) to see how much chunker choice interacts with embedder choice. This is a preview of the ablation-grid mod-102 chapter 5 discussed.
- Compute a **95% bootstrap confidence interval** on nDCG@10 for the top-3 models — sample 50 queries with replacement 1000 times, recompute, take the 2.5th and 97.5th percentiles. Compare the intervals to the paired-test verdict — they should agree.
- Feed the winner's config directly into a `chunker.yaml`-shaped `embedder.yaml` and commit it as the source of truth for mod-104 (vector DB tuning) to consume.
