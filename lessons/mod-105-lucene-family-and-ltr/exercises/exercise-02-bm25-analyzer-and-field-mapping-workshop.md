# Exercise 02 — BM25 Analyzer and Field-Mapping Workshop

**Estimated effort:** ~3 hours. Hands-on Elasticsearch or OpenSearch, plus a small corpus and an eval judgment set.

## Objective

Author a production-shaped lexical index end-to-end: an explicit mapping, a language-appropriate custom analyzer with ICU normalisation and curated synonyms, defensible BM25 parameters, a `multi_match` with field-boosting, and a `function_score` layer for business signals. Test each layer with `_analyze` and `_explain`. The goal is a lexical baseline you would ship — not a `standard`-analyzer strawman a hybrid stack silently beats.

## Prerequisites

- Chapters 2 and 3 read.
- A running Elasticsearch 8.10+ or OpenSearch 2.10+ cluster (single-node Docker is fine: [Elastic quick start](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html) or [OpenSearch quick start](https://opensearch.org/downloads.html)).
- A corpus of ~1,000–10,000 English documents you can index — pick one of:
    - Your own team's support articles, docs, or wiki.
    - [MS MARCO passage collection](https://microsoft.github.io/msmarco/) subset (public).
    - [BEIR SciFact or FiQA](https://github.com/beir-cellar/beir) — small, English, hand-labelled judgment sets.
- An eval judgment set of ~20–50 (query, relevant-doc-ids) pairs. If you're using BEIR, the qrels come with it. If it's your own corpus, hand-label ~20 queries with the top-5 truly-relevant docs each.
- Python or `curl` — the interface, not the point.

## Steps

**Step 1 — Baseline (the strawman).** Create an index with `dynamic: true` and no custom analyzer. Ingest the corpus. Run a `multi_match` query over `title` and `body` with default settings for ten queries from your judgment set. Record `recall@10` and rough nDCG@10 (an eyeball scoring on a 0–4 grade is fine — mod-109 covers rigorous scoring).

This is your "before" number. Every subsequent step you must beat it.

**Step 2 — Author an explicit mapping.** Rewrite the index with `dynamic: "strict"` and an explicit mapping. Define:

- `article_id` as `keyword`.
- `title` as `text` with a `.keyword` sub-field capped at `ignore_above: 256`.
- `body` as `text` with `term_vector: with_positions_offsets` (you'll want it for step 6 and for future LTR work).
- `tags` as `keyword`.
- Whatever numeric business-signal fields your corpus has (`view_count`, `helpful_ratio`, `updated_at`).

Reindex the corpus. Confirm ingestion succeeds and dynamic-strict rejection works by pushing a doc with an unmapped field (you should get an error).

**Step 3 — Author a custom analyzer.** Following chapter 2, build:

- A `support_index_analyzer` and `support_search_analyzer` pair.
- `html_strip` char filter (if your corpus has HTML) and a `mapping` char filter for domain terms that tokenise badly (`C#`, `C++`, `.NET`, etc. — pick ~5 relevant to your corpus).
- `icu_normalizer` (NFKC-CF) and `icu_folding` — you must install the [`analysis-icu`](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html) plugin if it isn't already.
- Language-appropriate stopwords and stemmer (`light_english` for English; use the appropriate language analyzer for another language).
- `synonym_graph` filter on the search-time analyzer only, with a curated synonym file (start with ~20 domain synonyms; extend as you find query terms without matches).

Test the analyzer with `_analyze`:

```
POST /support_articles/_analyze
{
  "analyzer": "support_search_analyzer",
  "text": "your realistic domain query with C# and Möller"
}
```

Verify the token output matches your expectation for at least ten diverse query strings. Save the ten strings + expected tokens as `analyzer_smoke_tests.md` in your work directory — you'll re-run these every time you touch the analyzer.

Reindex the corpus. Run the same ten queries from step 1. Measure `recall@10` and nDCG@10. It should beat step 1. If it doesn't, use `_analyze` and `_explain` on the queries that got worse to find why.

**Step 4 — Field boosting with `multi_match`.** Replace the naïve `match` query with a `multi_match`:

```json
{
  "query": {
    "multi_match": {
      "query": "<query>",
      "type": "best_fields",
      "fields": ["title^5", "body^1", "tags^3"],
      "tie_breaker": 0.3
    }
  }
}
```

Sweep the boost values. Try (title, body, tags) = (5,1,3), (3,1,2), (10,1,5), (2,1,1). Record `recall@10` and nDCG@10 for each. Pick the best tuple and defend the pick in one paragraph. Note whether `best_fields` outperforms `most_fields` or `cross_fields` on your corpus.

**Step 5 — BM25 parameter tuning (optional but recommended).** Author a custom `similarity` block per chapter 3 with `k1: 1.6, b: 0.5` for the `body` field. Reindex (BM25 params require reindex to take effect). Compare to defaults. Only ship the change if it measurably wins.

**Step 6 — Function scoring for business signal.** Wrap your best `multi_match` in a `function_score` that adds:

- Gaussian decay on `updated_at` with `scale: 30d, offset: 7d, decay: 0.5`.
- `log1p` on a popularity field (`view_count_30d` or whatever your corpus offers).
- If your corpus has a quality-ratio field (`helpful_ratio`), a `field_value_factor` with `weight: 2.0`.

Use `boost_mode: "multiply"` and `score_mode: "sum"` as your starting defaults. Re-run the queries and measure. Diagnose any regressions with `_explain` on the top-3 results per query.

**Step 7 — Rescore instead of scoring everything.** Move the `function_score` block into a `rescore` window on top-500 candidates. Verify the top-10 results are identical or near-identical (small differences are OK if a document just outside the top-500 was previously in the top-10 — flag them; usually they aren't the docs you care about). Measure query latency and note the improvement.

**Step 8 — Write it up (~30 min).** Author a short report (`writeup.md`) containing:

- The final mapping and analyzer definition (as JSON, checked into your work directory).
- A results table: `recall@10` and nDCG@10 for each of the seven steps (baseline through rescore).
- One paragraph per non-obvious decision (why you picked `light_english` over Porter, why you picked those five char-filter substitutions, why you chose `boost_mode: multiply` over `sum`).
- One paragraph on the top-3 lessons you'd carry into the next lexical index you build.

## Acceptance criteria

- Explicit `dynamic: "strict"` mapping deployed and verified with a schema-violation test.
- Custom analyzer with `icu_normalizer`, `icu_folding`, language stemmer, and `synonym_graph` (search-time), tested with `_analyze` on ≥ 10 realistic query strings and matching expected tokens.
- `multi_match` field-boost sweep of ≥ 4 configurations, with metrics per configuration.
- `function_score` adds ≥ 2 business signals with defended `boost_mode` and `score_mode`.
- `rescore` deployed with a measurable latency improvement and near-identical top-10.
- Final report contains a metrics table and defended decisions per step.
- The final `recall@10` and nDCG@10 beat the step-1 baseline. If they don't, the report explains why and what you would try next.

## Stretch goals

- **Multi-lingual expansion.** Add a second language to your corpus (Japanese, Chinese, or a Latin-script language other than English) and author a language-appropriate analyzer for it. Route documents to the right analyzer based on a `language` field.
- **Synonym-set API rather than file.** Migrate the synonym filter to Elasticsearch's [synonyms API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-with-synonyms.html) (8.10+) and demonstrate reloading a synonym set without reindexing.
- **Domain-specific `k1`/`b` sweep.** Grid-search `k1 ∈ {0.9, 1.2, 1.6, 2.0}` × `b ∈ {0.25, 0.5, 0.75, 1.0}` on your best-config query set. Report the winner with a delta-nDCG and defend whether the delta is significant enough to ship.
- **Debugging drill.** Pick the query in your judgment set where your final config scores *worst*. Use `_explain` on the actual-top-1 and the expected-top-1 to explain why the expected doc lost. Propose a single change that would rank the expected doc higher, and measure whether it does.
- **Port to OpenSearch (or vice versa).** If you started on Elasticsearch, translate the mapping, analyzer, and queries to OpenSearch (or Solr). Note any semantic differences that surprised you.

## Notes

- Do not skip `_analyze` and `_explain`. They are the fastest debugging loops in the Lucene family; every hour spent on them saves five hours of blind boost-tuning.
- Every analyzer change requires a reindex to apply to existing docs. Set up an alias-based dual-index cutover early (mod-104 chapter 8 pattern) so reindexing is not a blocker.
- If your metrics do not move, the problem is usually the analyzer (not the boosts). Verify tokenisation before you touch the query.
- Do not tune to a 20-query eval set past ~2 significant-figure precision — you'll overfit. mod-109 covers real eval methodology; this exercise is directional.
