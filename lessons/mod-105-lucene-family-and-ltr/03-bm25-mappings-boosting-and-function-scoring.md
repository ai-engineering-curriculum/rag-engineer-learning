# Chapter 3 — Field Mappings, BM25, Boosting, and Function Scoring: The Production Lexical Baseline

## Why this chapter exists

BM25 on a well-authored lexical index is the strong baseline every hybrid stack fuses with. It is also the retrieval component most level-30 engineers under-invest in — they reach for dense retrieval because the vendor demo said "vectors are all you need," and they end up with a fused pipeline whose lexical arm is a shrug: `standard` analyzer, no boosting, no `function_score`, default BM25 parameters. That produces a hybrid stack whose lexical arm loses to a plain BM25 baseline someone else wrote in an afternoon.

This chapter shows what a *deliberately-authored* BM25 index looks like. The analyzer from chapter 2 tokenises text; this chapter puts that text into a field mapping, tunes BM25 for the field, weights fields against each other with a `multi_match`, and adds business signal (recency, popularity, quality) on top with `function_score`. Chapter 4 adds the dense-vector field alongside. Chapter 5 replaces the hand-tuned boosts with an LTR model.

## The field mapping is where retrieval quality lives

Every document in Elasticsearch/OpenSearch is a JSON object; every field's type, analyzer, and index-time options are declared in the index's **mapping**. Solr has an equivalent schema.xml or [managed schema API](https://solr.apache.org/guide/solr/latest/indexing-guide/schema-api.html); Vespa has a `.sd` (search-definition) file per document type.

At level 30 you author the mapping, not the engine. Do not rely on **dynamic mapping** — where the engine guesses field types from first-seen values — for anything that matters. Dynamic mapping quietly turns a numeric field into `long`, a datestring into `date`, and a text field into `text` + `keyword` (both), and then bakes those choices into the index without your consent. Every production index worth its uptime has an explicit mapping authored deliberately.

### The half-dozen field types you actually use

Elasticsearch's [full type list](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html) is long; the ones that matter for a retrieval index are a short list:

- **`text`** — analyzed (tokenised) text. Goes through an analyzer (chapter 2); produces terms in an inverted index. Supports full-text queries. **Not aggregatable, not sortable, not exact-matchable.**
- **`keyword`** — exact-match string, no tokenisation. Supports term queries, aggregations, and sorting. The pattern for categorical fields (`status`, `tenant_id`, `country`, `product_category`).
- **`text` + `keyword` multi-field.** The idiomatic pattern is a `text` field with a `.keyword` sub-field: `title` runs through the English analyzer for full-text search; `title.keyword` is the exact string for aggregations and exact-match. Elasticsearch's dynamic mapping does this by default; you should do it explicitly.
- **`long`, `integer`, `short`, `byte`, `double`, `float`, `half_float`, `scaled_float`** — numeric types. Pick the narrowest that fits; each byte matters at scale.
- **`date`** — parsed against an explicit format list. Do not rely on default format parsing; always declare `format: "strict_date_optional_time||epoch_millis"` or the format list you actually receive.
- **`boolean`** — one bit per doc; cheap.
- **`nested`** — for arrays of objects that need per-object query semantics (chapter 3 of mod-102 discusses this from the chunking side). Costlier than flat objects; use only when you need it.
- **`dense_vector`** — chapter 4.

### The three per-field switches worth naming

Every field has three switches that you set deliberately:

- **`index: true | false`** — is this field searchable? Default `true`. Set `false` for fields you store for retrieval but never query on (e.g., raw source URL for display only). Saves index size and refresh cost.
- **`doc_values: true | false`** — is this field usable for sort/aggregation/script? Default `true` for numeric/keyword/date; default `false` for `text`. Doc values are a column-store representation on disk; disabling saves disk at the cost of making the field unsortable/unaggregatable.
- **`store: true | false`** — is the raw field value stored separately from `_source`? Default `false` — you retrieve fields from `_source`. Only set `true` when you need to disable `_source` and still fetch specific fields (a specialised optimization).

Two more switches that come up:

- **`norms: true | false`** — should the field participate in length-normalised scoring (BM25's `b` parameter)? Default `true` for `text`; you rarely turn it off. Set `false` for `text` fields that will only be term-existence-filtered (never scored).
- **`term_vector`** — for fields that need term-position or term-frequency data at query time (e.g., for LTR features or highlighting). `with_positions_offsets` is a common choice for LTR + highlighting.

### A concrete mapping for a support-article index

```json
PUT /support_articles
{
  "settings": { "analysis": { /* the support_index_analyzer / support_search_analyzer from chapter 2 */ } },
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "article_id":     { "type": "keyword" },
      "tenant_id":      { "type": "keyword" },
      "title":          {
        "type": "text",
        "analyzer": "support_index_analyzer",
        "search_analyzer": "support_search_analyzer",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "body":           {
        "type": "text",
        "analyzer": "support_index_analyzer",
        "search_analyzer": "support_search_analyzer",
        "term_vector": "with_positions_offsets"
      },
      "product":        { "type": "keyword" },
      "category":       { "type": "keyword" },
      "tags":           { "type": "keyword" },
      "updated_at":     { "type": "date" },
      "view_count_30d": { "type": "integer" },
      "quality_score":  { "type": "float" },
      "helpful_ratio":  { "type": "float" }
    }
  }
}
```

Points worth naming:

- **`dynamic: "strict"`** — refuse to accept documents with unmapped fields. This turns "the schema silently mutated in production" into an indexing error, which is exactly what you want.
- **`title.keyword` with `ignore_above: 256`** — titles longer than 256 chars won't be stored as keyword (aggregation would be meaningless anyway).
- **`body` with `term_vector: with_positions_offsets`** — needed for LTR feature extraction and highlighting (chapter 5).
- **Business-signal fields** (`view_count_30d`, `quality_score`, `helpful_ratio`) — numeric fields that we'll consume from `function_score` below and eventually from LTR features.
- **`tenant_id` as `keyword`** — cheap-to-filter categorical field; used for per-tenant isolation (mod-112).

## BM25: the scoring function you inherit

Elasticsearch and OpenSearch use **BM25** as the default similarity function since 5.0 (2016), inherited from Lucene 6. The Solr default is also BM25 as of 6.0. Vespa uses [`bm25` as a first-class rank feature](https://docs.vespa.ai/en/reference/bm25.html) available in ranking expressions.

The BM25 score of a document `d` for a query `q` with terms `t ∈ q` is:

```
BM25(d, q) = Σ_t IDF(t) · f(t, d) · (k1 + 1) / (f(t, d) + k1 · (1 - b + b · |d|/avgdl))
```

Where:

- **`f(t, d)`** — term frequency: how many times term `t` appears in document `d`.
- **`|d|`** — length of document `d` in tokens.
- **`avgdl`** — average document length across the field's corpus.
- **`IDF(t)`** — inverse document frequency, `log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)`, where `N` is total docs and `n(t)` is docs containing `t`.
- **`k1`** — term-frequency saturation. Lucene default: **1.2**.
- **`b`** — length normalisation strength. Lucene default: **0.75**.

The [original BM25 paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (Robertson and Zaragoza, 2009) is a readable one-afternoon investment; do not skip it if you plan to tune the parameters.

### Tuning `k1` and `b`

Both parameters are tunable per field via the `similarity` block:

```json
PUT /support_articles/_settings
{
  "index": {
    "similarity": {
      "long_body_bm25": {
        "type": "BM25",
        "k1": 1.6,
        "b": 0.5
      }
    }
  }
}
```

Then apply the custom similarity to the field mapping:

```json
"body": {
  "type": "text",
  "analyzer": "...",
  "similarity": "long_body_bm25"
}
```

The intuition:

- **`k1` (default 1.2)** — controls how fast term-frequency saturates. Low `k1` (near 0.5): a term appearing 5 times counts almost as much as appearing 20 times. High `k1` (near 2.0): more repetitions matter more. For long-form content where an author who mentions the term many times is genuinely emphasising it, higher `k1` (~1.5–1.8) often helps.
- **`b` (default 0.75)** — controls how much long documents get penalised. `b=1.0`: full length normalisation (fully divide term-frequency by document length). `b=0.0`: no length normalisation. For fields where document length is a real quality signal (titles are always short; a 200-token title vs. a 20-token title is a data-quality flag, not "less relevant"), lower `b` (~0.3–0.5) often helps.

The rule of thumb: **defaults are fine for most workloads**. Tune only after you have an eval set (mod-109) and a measurement loop that gives you nDCG@10 for a parameter sweep. Do not tune `k1` and `b` by eyeballing top-10 results on ad-hoc queries; you will overfit to your own intuition.

### Sanity-checking scores with `_explain`

Elasticsearch's [`_explain` API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-explain.html) returns the full BM25 breakdown for a single document against a single query. When a document ranks lower (or higher) than you expected, `_explain` tells you which term contributed what, which field dominated, and what the norms were. It is the fastest debugging loop for BM25 anomalies. Solr has [`debugQuery=true`](https://solr.apache.org/guide/solr/latest/query-guide/common-query-parameters.html#debug-parameter) for the same thing.

## Boosting: `multi_match` across fields

Real documents have multiple text fields (title, body, tags, product). BM25 alone will not know that a term match in the title is worth 5× a term match in the body. **`multi_match`** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html)) is the query that combines matches across fields with a per-field boost:

```json
{
  "query": {
    "multi_match": {
      "query": "reset admin password",
      "type": "best_fields",
      "fields": ["title^5", "body^1", "tags^3"],
      "tie_breaker": 0.3
    }
  }
}
```

The `^N` syntax is a **per-field boost** — multiply the field's BM25 score by `N` before combining. Boosts are relative; only ratios matter (`title^5, body^1` and `title^50, body^10` produce identical rankings).

### The four `multi_match` types

- **`best_fields`** (default) — score each field, take the max, add `tie_breaker × sum(other_field_scores)`. Right when the query is likely to be a single-field match ("reset admin password" probably matches title *or* body, not both meaningfully).
- **`most_fields`** — sum all matching field scores. Right when the query benefits from matching in many fields (e.g., different analyzers on the same content indexed multiple times).
- **`cross_fields`** — treat all fields as one big field for scoring purposes. Handles "the first name is in `first_name` and the last name is in `last_name`" correctly. Requires all fields share an analyzer.
- **`phrase`** and **`phrase_prefix`** — phrase-oriented variants.

For most retrieval workloads, `best_fields` with a moderate `tie_breaker` (0.1–0.3) is the right default. Switch to `cross_fields` when your content is genuinely split across fields (address parts, name parts).

### How to pick the boost weights

Do not eyeball. The credible options in order of increasing rigor:

1. **Guess and check on a small eval set.** Fastest; overfits.
2. **Grid search over an eval set** — mod-109 gives you the eval set; iterate title/body/tags boosts on a 3×3×3 grid and pick the tuple with the best nDCG@10.
3. **Learning-to-rank replaces the boosts with model-learned feature weights** — chapter 5. This is the graduation move; once you have LTR, you delete the hand-tuned `^N` values.

The most common mistake: **using boosts to compensate for a bad analyzer**. If your title field is not matching a query, first check the analyzer with `_analyze` (chapter 2); do not paper over an analyzer bug with a bigger boost.

## Function scoring: adding business signal

Pure text scoring returns the most textually-similar documents. Real production ranking almost always wants to blend text score with **business signal**: recency (more recent > older), popularity (more clicked > never clicked), quality (higher-rated > lower-rated), authorship (staff articles > user articles).

**`function_score`** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html)) is the query that blends a base query's text score with per-document functions. Solr's equivalent is [function queries](https://solr.apache.org/guide/solr/latest/query-guide/function-queries.html) combined with `edismax`'s `bf` (boost function) parameter; Vespa expresses the same idea directly in a [ranking expression](https://docs.vespa.ai/en/ranking.html).

A representative `function_score` for the support-article corpus:

```json
{
  "query": {
    "function_score": {
      "query": {
        "multi_match": {
          "query": "reset admin password",
          "type": "best_fields",
          "fields": ["title^5", "body^1", "tags^3"],
          "tie_breaker": 0.3
        }
      },
      "functions": [
        {
          "gauss": {
            "updated_at": {
              "origin": "now",
              "scale":  "30d",
              "offset": "7d",
              "decay":  0.5
            }
          },
          "weight": 1.0
        },
        {
          "field_value_factor": {
            "field":     "view_count_30d",
            "modifier":  "log1p",
            "missing":   1
          },
          "weight": 1.0
        },
        {
          "field_value_factor": {
            "field":     "helpful_ratio",
            "modifier":  "none",
            "missing":   0.5
          },
          "weight": 2.0
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply",
      "min_score": 0.1
    }
  }
}
```

Points worth naming:

- **Gauss decay on `updated_at`** — recent documents score higher, with a smooth Gaussian falloff. `origin: "now"`, `scale: "30d"`, `offset: "7d"` means: full score for anything within 7 days of now, then the score decays with a Gaussian whose scale is 30 days.
- **`log1p` on `view_count_30d`** — log-scale popularity boost. Prevents "the one article with 1M views" from dominating everything; a 10× view difference translates to ~2.4× score difference under `log1p`.
- **`helpful_ratio` with `weight: 2.0`** — the "did users say this article was helpful?" signal, weighted 2× because it's a stronger quality signal than views (a heavily-viewed article can still be unhelpful).
- **`score_mode: "sum"`** — how the three function outputs combine into one function score. Options: `multiply`, `sum`, `avg`, `first`, `max`, `min`.
- **`boost_mode: "multiply"`** — how the combined function score combines with the base query's text score. `multiply` is the classical choice; `sum` and `replace` are also available.

### Script scoring: when you need custom math

For blending that `function_score` can't express (custom quality formulas, per-tenant weightings, temporal windows), **[`script_score`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html)** runs a Painless script per candidate document:

```json
{
  "query": {
    "script_score": {
      "query": { "multi_match": { "query": "reset password", "fields": ["title^5", "body"] } },
      "script": {
        "source": "_score * (1.0 + Math.log(1.0 + doc['view_count_30d'].value)) * doc['helpful_ratio'].value"
      }
    }
  }
}
```

Powerful; slow per-doc. Two ways to keep it fast: (1) put the `script_score` inside a `function_score` or rescore window so it only runs on top-K candidates, not the whole result set; (2) use `rescore` (below) rather than scoring every candidate.

### Rescoring: hand-tuning at top-K

**[`rescore`](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#rescore)** re-scores only the top-N candidates from the base query with a more expensive scoring function. Standard pattern: base query is a cheap `multi_match`; rescore is a more expensive `function_score` or `script_score` that runs on top-1000. The user sees the top-10 after rescoring; the engine only paid the expensive scoring cost 1000 times, not on every doc.

```json
{
  "query": { "multi_match": { "query": "reset password", "fields": ["title", "body"] } },
  "rescore": {
    "window_size": 1000,
    "query": {
      "score_mode": "multiply",
      "rescore_query": {
        "function_score": {
          "query": { "match_all": {} },
          "functions": [ { "gauss": { "updated_at": { "origin": "now", "scale": "30d" } } } ]
        }
      }
    }
  }
}
```

Rescore is the pattern LTR sits inside — chapter 5 puts the LTR model in a rescore step over the top-1000 BM25 candidates.

## The three anti-patterns to avoid

1. **Using `standard` analyzer with default BM25 and calling it a "baseline."** It is a shrug. A real baseline is chapter-2-shaped: language-appropriate analyzer, ICU normalisation, curated synonyms, and — if the domain warrants — tuned `k1`/`b`. Anything less is a strawman that a hybrid stack will silently outperform without you learning anything.
2. **Stacking a dozen `function_score` functions with hand-tuned weights.** After the third function, hand-tuning stops converging on the eval set. That is the signal to graduate to LTR (chapter 5); the model learns feature weights better than you do.
3. **Running `script_score` on every candidate.** If the script runs on 10 million docs per query, the cluster falls over. Move it into a `rescore` window on the top-1000 candidates.

## What is NOT in this chapter (and where it goes)

- **Dense-vector fields alongside these text fields** — chapter 4.
- **LTR features derived from the fields defined here** — chapter 5.
- **Sharding, replicas, index size** — chapter 6.
- **Hybrid retrieval fusion (RRF, weighted-sum, learned fusion)** — mod-106.
- **Retriever eval — nDCG, MRR, statistical significance** — mod-109.

## What you should be able to do after this chapter

- Author an explicit index mapping (`text` + `keyword`, numeric types, dates, `nested` where needed) with `dynamic: "strict"` and the analyzers from chapter 2 wired in.
- Explain the BM25 formula and the roles of `k1` and `b` well enough to defend a per-field similarity override.
- Write a `multi_match` across three or more fields with defended per-field boosts, and know when to reach for `best_fields` vs. `cross_fields` vs. `most_fields`.
- Add a `function_score` block that combines recency (Gaussian decay), popularity (`log1p`), and quality signal, and defend the choice of `score_mode` and `boost_mode`.
- Use `rescore` to keep expensive scoring functions off the whole result set.
- Debug an unexpected rank with `_explain` and diagnose whether the cause is analyzer, mapping, or scoring.

## Further reading

- Robertson, S. and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* [Foundations and Trends in IR](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — the reference paper.
- Elastic: [*BM25 similarity module*](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html) — the `k1`/`b` knobs and the alternatives (DFR, LMDirichlet, LMJelinekMercer, boolean).
- Elastic: [*Multi-match query*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html) — the field-boosting query.
- Elastic: [*Function score query*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html) — the business-signal blending query.
- Elastic: [*Script score query*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html) — custom scoring formulas.
- Solr: [*Function queries*](https://solr.apache.org/guide/solr/latest/query-guide/function-queries.html) — the Solr-side equivalents.
- Turnbull, D. and Berryman, J. (2016). *Relevant Search: With Applications for Solr and Elasticsearch* (Manning). The standard book-length treatment of hand-tuned Lucene-family relevance engineering.
