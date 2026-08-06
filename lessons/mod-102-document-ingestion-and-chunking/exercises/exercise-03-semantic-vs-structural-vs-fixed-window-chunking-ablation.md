# Exercise 03 — Semantic vs. structural vs. fixed-window chunking ablation

**Estimated effort:** ~3 hours.

## Objective

Turn "which chunker should I use?" from an opinion into a measurement. You will chunk one corpus five different ways, embed each variant, retrieve on a shared judgment set, and score with `ir_measures`. The output is a table where the winner is defensible on your corpus, and a written recommendation you would drop into a design doc.

This exercise is where chapter 4's strategy families and chapter 5's size trade-off become numbers.

## Prerequisites

- Chapters 4 (*Chunking Strategies*) and 5 (*Chunk-Size / Recall / Precision Trade-Off*) read once.
- `exercise-01` router works — you can turn a directory of files into `Chunk`s.
- pgvector from the mod-101 toolchain is up.
- Python deps: `ir_measures`, `sentence-transformers` (or `llama-index-embeddings-huggingface`), `llama-index`, `langchain-text-splitters`.

## The corpus and judgment set

Reuse `data/corpus/handbook.pdf` from `exercise-01`, or pick any corpus of at least 30 pages with clear section structure. A single well-structured document is enough; multiple related documents are better.

Build a judgment set of **at least 30 queries**, saved as `data/judgments/qrels.tsv` in TREC format:

```
qid  0  chunk_id_or_doc_id_or_span_id  rel
```

Two labeling shapes are acceptable:

- **Coarse (document-level).** `rel = 1` if the document is relevant to the query, `0` otherwise. Faster; measures document-level retrieval.
- **Fine (span-level).** For each query, name the section-path or page-range that contains the answer. `rel = 1` for chunks whose `section_path` matches. Slower; measures precision within a document.

Fine labels give you a stronger signal for chunker ablation. Use fine labels if you have the time; use coarse labels if you do not.

Save the queries themselves as `data/judgments/queries.tsv`:

```
qid  query_text
```

Diversity matters: include queries with varying granularity (some fact-lookup, some section-summary), varying vocabulary (some using the corpus's own terminology, some using synonyms), and at least a handful that should have multiple relevant chunks.

## Part A — Build the five chunker variants (60 min)

Implement each as a function `chunk_variant(passages) -> list[Chunk]`:

1. **`fixed-800-100`** — `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)` (token-aware `length_function`). The baseline.
2. **`fixed-400-40`** — same, smaller. Isolates chunk-size effect at fixed strategy.
3. **`structural-markdown`** — `MarkdownHeaderTextSplitter` (or LlamaIndex `MarkdownNodeParser`) at H1/H2/H3, recurse-split any section > 800 tokens.
4. **`semantic-95`** — LlamaIndex `SemanticSplitterNodeParser(breakpoint_percentile_threshold=95, embed_model=BAAI/bge-small-en-v1.5)`, or the LangChain `SemanticChunker` equivalent.
5. **`title-structural`** — `unstructured.chunk_by_title(elements, max_characters=1200, new_after_n_chars=800)` on the parser output. Tests the "structural chunker with a size cap" shape.

Constraints:

- The **embedding model is held constant** across all five: `BAAI/bge-small-en-v1.5` (384-dim, 512-token context).
- The **retriever is held constant**: pgvector with an HNSW index, `top_k = 20`.
- The **corpus is held constant**: same input passages from your ingest pipeline.

If you vary two axes at once (chunker *and* embedding, chunker *and* retriever) you cannot attribute the delta. Chapter 5's ablation harness is explicit about this.

## Part B — Ingest each variant into its own table (30 min)

Namespace by chunker: `chunks_fixed_800_100`, `chunks_fixed_400_40`, `chunks_structural_markdown`, `chunks_semantic_95`, `chunks_title_structural`. Each table has the schema from chapter 7, with an HNSW index on the `embedding` column.

Log per-variant ingest stats: `n_chunks`, `mean_tokens`, `p95_tokens`, `total_index_size_mb`, `wall_clock_ingest_s`. Commit as `docs/ingest-stats.md`.

## Part C — Run retrieval on the judgment set (45 min)

Write `scripts/eval_chunkers.py`:

```python
for variant in VARIANTS:
    run = {}
    for qid, query in queries.items():
        hits = retrieve(variant, query, top_k=20)  # returns [(chunk_id_or_doc_id, score), ...]
        run[qid] = {label(hit): score for hit, score in hits}
    metrics = ir_measures.calc_aggregate([
        ir_measures.Recall @ 10, ir_measures.nDCG @ 10, ir_measures.MRR @ 10,
    ], qrels, run)
    save_row(variant, metrics)
```

Notes:

- `label(hit)` maps a chunk_id to whatever your qrels use (doc_id if you labeled coarse; span_id if you labeled fine).
- Deduplicate hits by `doc_id` before computing `Recall@k` when your qrels are document-level; otherwise you can score `Recall@10` on ten chunks from the same document.

Output `docs/chunker-ablation.md`:

```markdown
# Chunker ablation — corpus: {name}, judgment set: {n} queries, {date}

| chunker              | n_chunks | mean_tokens | recall@10 | nDCG@10 | MRR@10 | ingest_s | index_mb |
|----------------------|----------|-------------|-----------|---------|--------|----------|----------|
| fixed-800-100        |          |             |           |         |        |          |          |
| fixed-400-40         |          |             |           |         |        |          |          |
| structural-markdown  |          |             |           |         |        |          |          |
| semantic-95          |          |             |           |         |        |          |          |
| title-structural     |          |             |           |         |        |          |          |
```

## Part D — Per-query error analysis (30 min)

Aggregate metrics hide where the strategies actually differ. Compute per-query `Recall@10` for the top-2 variants. Find:

- **5 queries where variant A beats variant B by ≥ 0.5**. Read them. What is common about them?
- **5 queries where variant B beats variant A by ≥ 0.5**. Same question.
- **Any query where all variants scored 0.** Something is likely wrong with the judgment (missing chunk in qrels), not the chunker. Fix and re-run.

Write two paragraphs in `docs/chunker-ablation.md` naming the query patterns each variant handles well.

## Part E — Recommendation (15 min)

Close `docs/chunker-ablation.md` with a `## Recommendation` section: one paragraph naming which chunker you would deploy for this corpus, one paragraph naming what you would try next (parent-child? different chunk size? different embedding?), and a `chunker.yaml` block in the shape chapter 5 defined:

```yaml
chunker: llama_index.MarkdownNodeParser
version: 1.0.0
chunk_size: 800
chunk_overlap: 100
notes: |
  Winner of exercise-03 ablation on {date}: nDCG@10 = 0.71 (vs. 0.65 for the
  fixed-800-100 baseline). Beats semantic-95 by 0.04 nDCG at 1/3 the ingest cost.
```

## Deliverables

- `src/chunkers/` — five chunker variants.
- `data/judgments/qrels.tsv`, `data/judgments/queries.tsv` — 30+ queries, TREC-format qrels.
- `scripts/eval_chunkers.py`.
- `docs/ingest-stats.md`, `docs/chunker-ablation.md` (with the recommendation and `chunker.yaml`).

## Acceptance criteria

- Five chunker variants all produce non-empty `Chunk` sets.
- `scripts/eval_chunkers.py` runs end-to-end and produces the ablation table with `Recall@10`, `nDCG@10`, `MRR@10` populated for each variant.
- The `Recommendation` section names one variant, not two ("go with A" — not "either A or B, depending").
- Error analysis in Part D is present and specific ("A wins on fact-lookup queries mentioning table headers; B wins on multi-sentence policy summarizations").
- Every configuration (embedding model, HNSW parameters, `top_k`) is held constant across variants and named in the doc.

## Stretch goals

- Extend the ablation to a 3 × 3 grid: three chunkers × three chunk sizes at `{400, 800, 1600}` tokens. Nine cells; report `Recall@10` for each. This is the shape that produces the "chunk-size curve" chapter 5 talks about.
- Repeat the ablation with a different embedding model (`intfloat/e5-small-v2` or `mixedbread-ai/mxbai-embed-large-v1`). Do the winners change? This is why chapter 5 warned that chunker choice interacts with embedder choice.
- Add BM25 (via Elasticsearch or `rank_bm25`) as a sixth "strategy" — same qrels, same queries, no chunker. Chapter 4 argued BM25 is the strong baseline every dense stack should beat; check whether yours does.
