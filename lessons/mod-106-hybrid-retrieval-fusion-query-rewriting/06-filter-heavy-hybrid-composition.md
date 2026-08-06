# Chapter 6 — Filter-Heavy Hybrid: Structured Predicates Without Collapsing to One Arm

## Why this chapter exists

Almost every enterprise retrieval query has a structured-filter dimension. "Show me support articles for **product = Acme 4402, updated in the last 90 days, in English**, ranked by relevance to the user's query." "Retrieve product listings **where price < $50, in stock**, matching the query." "Retrieve documents **the user has read access to**, matching the query."

Hybrid retrieval interacts with those filters in three specific ways that go wrong more often than the two-arm-plus-fusion case chapters 2 and 3 covered. This chapter names each failure mode and the compose-with-metadata pattern that keeps the hybrid arm from silently collapsing to a single-arm result. Chapter 7 marks the boundary to the peer track and closes the module.

## The four ways filters interact with hybrid retrieval

For any query, there are four moments the filter can enter the pipeline:

1. **Pre-filter (before retrieval).** The engine narrows the candidate set to filter-matching docs *first*, then runs BM25 / dense over that reduced set. Semantics: only in-scope docs can be returned; scoring is over the filtered candidates.
2. **Post-filter (after retrieval).** Retrieve top-N ignoring the filter, then drop out-of-scope docs. Semantics: filter applied at return time; the top-K after filtering may be much smaller than K, or empty.
3. **Filter-as-fusion-arm.** The structured predicate is treated as another retriever whose "score" is 1 for matching docs, 0 otherwise. RRF or weighted-sum combines it with the BM25 and dense arms. Semantics: matching the filter is a positive signal, not a hard requirement.
4. **Boost-not-filter (soft filter).** The filter is a scoring modifier (via `function_score` in the Lucene family): matching docs get a bonus; non-matching docs are still eligible. Semantics: filter is a preference, not a rule.

Which of the four is "correct" depends on the query semantics — but the wrong choice can silently break hybrid retrieval in ways that fusion cannot repair. Sections below walk each.

## Failure mode 1: Post-filtering an ANN retriever

The most common failure. HNSW / IVF-Flat retrievers are trained to return top-*k* by similarity; if you retrieve top-100 and then filter, you may end up with only 4 in-scope docs — because 96 of the 100 nearest neighbors were out of scope. Recall on the filtered candidate set collapses.

The example that comes up: a query "printer troubleshooting" over a corpus of a million support articles, filtered to `product = printer, updated_at > 2024-01-01`. If the corpus has only a few thousand articles in scope and you retrieve top-100 unfiltered nearest neighbors, most of them may be about other products; after post-filtering, you have 5 in-scope candidates and a broken retrieval.

**The fix: use pre-filtered ANN.** All modern vector engines support filtered ANN in the search call: Elasticsearch's `knn` query with `filter` ([docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-search-filter-example)), OpenSearch's `knn` filter ([docs](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/)), Qdrant's `filter` ([docs](https://qdrant.tech/documentation/concepts/filtering/)), Weaviate's `where` ([docs](https://weaviate.io/developers/weaviate/search/filters)), pgvector's `WHERE ... ORDER BY embedding <=> query LIMIT ...` — all express the same shape. The engine either walks the HNSW graph while skipping non-matching nodes, or intersects the ANN result with a bit-set of filter-matching doc IDs.

Two things to watch, per engine:

- **HNSW's graph is not filter-aware by construction.** Filtered ANN with tight predicates (< 1% of docs match) can degrade recall dramatically — the search may exhaust the visitable graph before finding enough in-filter candidates. Engines have different heuristics (pgvector's [iterative HNSW search](https://github.com/pgvector/pgvector#iterative-index-scans), Weaviate's [flat / IVF fallback](https://weaviate.io/developers/weaviate/config-refs/schema/vector-index#hnsw-index-parameters), Qdrant's [payload index and filter estimator](https://qdrant.tech/documentation/concepts/indexing/)). Read your engine's docs on this; the failure mode is silent low recall.
- **The engine's default may still be post-filter.** Some frameworks still default to post-filter for backward compatibility. Verify by construction — do not assume "there's a filter parameter" means it is applied pre-ANN.

Mod-104 (vector databases) covers this in detail from the vector-DB side; this chapter names it because filter-heavy queries into hybrid stacks are where most retrieval engineers first meet the problem.

## Failure mode 2: The dense arm loses to the lexical arm on tight filters

Even with proper pre-filtered ANN, tight filters can silently unbalance the fusion. The pattern:

- BM25 arm: retrieves top-100 among filter-matching docs. Solid.
- Dense arm: retrieves top-100 among filter-matching docs, but with a tight filter the ANN degrades (as above), returning noisy candidates.
- RRF: fuses the two arms. The dense arm's noise doesn't help much, but its presence in the fusion still contributes rank votes. Net effect: the fused result is BM25-dominant, and the dense arm is silently useless.

You can miss this for months if you only track aggregate metrics. Per-query-class metrics (queries with tight filters vs. queries with loose filters) reveal the problem.

**The fix: measure per filter-tightness.** Split your eval set by filter tightness — how many docs match the filter — and report recall@100 per bucket. If the dense arm's contribution is nil on tight-filter queries, either:

- Increase the ANN `num_candidates` for tight-filter queries (Elasticsearch's `num_candidates` is the internal candidate pool before the top-*k* selection; raising it improves filtered-ANN recall at query cost).
- Fall back to BM25-only on very-tight-filter queries (< 1% of the corpus matches). RRF over a BM25 arm and a broken dense arm is worse than BM25-only.
- Route (chapter 5) tight-filter queries to a BM25-only retriever.

## Failure mode 3: The collapse to a single arm after filtering

A subtler failure. Suppose your fusion default is weighted-sum with normalization. When a tight filter reduces the candidate set, the per-query normalization (min-max, z-score) becomes noisy — you may only have 10 candidates in one arm, and the min-max range is dominated by a single high or low score. The normalized-score distribution warps, and the arm that produced fewer candidates gets over- or underweighted arbitrarily.

**The fix — the "compose-with-metadata" pattern:**

1. **Author the filter into the retrieval call, not the fusion.** Both arms retrieve with the same filter; the fusion combines apples-to-apples ranked lists over the same candidate set.
2. **Use RRF over weighted-sum on filter-heavy queries.** RRF's rank-based scoring is not affected by the per-arm-size warp; weighted-sum with per-query normalization is. This is one of the concrete cases where RRF's robustness wins.
3. **Keep the filter in metadata, not in embedding.** Do not encode filter values (product ID, date, permission) into the embedding text. That is fragile (filters change without re-embedding); use metadata fields on the doc and apply the filter at query time.
4. **Score the metadata match separately, if you need it as a signal.** If matching the filter is a *stronger* signal than raw retrieval score (a rare case: e.g., matching a security clearance level), use it as a boost, not just a hard filter. See "Boost-not-filter" below.

## Failure mode 4: ACL and permission filters silently returning wrong docs

Permission filters are a special case: they are non-negotiable. A returned doc that the user cannot see is a security incident, not a relevance miss. Two anti-patterns come up:

- **Post-filtering permissions after retrieval.** The retrieval log shows top-10 docs the user shouldn't see; the response drops them; the ranking after filtering is broken. The user sees only 3 hits; the response feels sparse. This is the recall-collapse failure again, but the impact is worse because the filter is legally required.
- **Encoding permissions in the embedding.** "the acl for this doc is: engineering, security-cleared, us-only" as a prefix in the doc text. Fragile: permission changes require re-embedding; permission checks depend on the embedder correctly separating permission tokens from content tokens.

The right pattern for ACL-aware retrieval:

1. **Author permissions as a structured field on every doc.** `acl: ["engineering", "security-cleared", "us-only"]` on the doc; `user_groups: ["engineering", "security-cleared"]` at query time.
2. **Pass the user's groups into the retrieval call as a pre-filter.** Both arms retrieve with the ACL predicate as pre-filter.
3. **Include the ACL check in the eval set.** Adversarial queries that would surface not-allowed docs if the filter were missing; verify they return nothing.

Mod-112 (Enterprise Retrieval: ACL, structured, graph, multimodal) covers ACL-aware retrieval in depth; this chapter names it because the compose-with-metadata pattern is the same across ordinary metadata filters and permission filters — the stakes just differ.

## The four filter shapes: which one to use when

| Shape | Semantics | When to use |
|---|---|---|
| **Pre-filter (hard)** | Only in-scope docs eligible; scoring over the filtered set. | Permissions, tenant isolation, product / language routing, "the user asked for X-year model" — anything binary and non-negotiable. |
| **Post-filter** | Retrieve, then drop out-of-scope. | Almost never for hybrid stacks. Only if the filter is a rare fallback (< 1% queries) and you cannot get pre-filter working. |
| **Filter-as-fusion-arm** | Filter match is a positive signal; non-matching docs still eligible. | Preference-not-requirement: "prefer 2024 docs but return older if they match strongly." Emulate with weighted-sum + a rank contribution from the filter. |
| **Boost-not-filter (soft filter via `function_score`)** | Matching docs get a scoring bonus; non-matching still returned. | Preference-not-requirement expressed via ranking. E.g., recency preference via Gaussian decay on `updated_at` (chapter 3 of mod-105). |

The mistake most level-30 engineers make is treating everything as a hard filter and then wondering why recall craters. Half of real "filters" are actually preferences and belong in the scoring, not the filtering.

## The compose-with-metadata pattern (worked example)

Concrete: a support-articles hybrid retrieval with the query "how do I reset the admin password," filtered to `product = "Acme Dashboard"`, `language = "en"`, boosted toward `updated_at` recency, and permission-filtered by user group.

**Elasticsearch RRF retriever composing all four:**

```json
POST /support_articles/_search
{
  "size": 20,
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "bool": {
                "must": {
                  "multi_match": {
                    "query": "reset admin password",
                    "type": "best_fields",
                    "fields": ["title^5", "body^1", "tags^3"]
                  }
                },
                "filter": [
                  { "term":  { "product":  "Acme Dashboard" } },
                  { "term":  { "language": "en" } },
                  { "terms": { "acl": ["support", "user_readable"] } }
                ]
              }
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector_builder": {
              "text_embedding": {
                "model_id": "sentence-transformers__all-minilm-l12-v2",
                "model_text": "reset admin password"
              }
            },
            "k": 100,
            "num_candidates": 500,
            "filter": [
              { "term":  { "product":  "Acme Dashboard" } },
              { "term":  { "language": "en" } },
              { "terms": { "acl": ["support", "user_readable"] } }
            ]
          }
        }
      ],
      "rank_constant": 60,
      "rank_window_size": 100
    }
  }
}
```

Points worth naming:

- **Both arms carry the same filter clauses.** BM25 `bool.filter` and the `knn.filter` both apply pre-retrieval. Filters are identical, so both arms score over the same candidate set.
- **`num_candidates: 500` for the dense arm.** Higher than `k: 100` so filtered ANN has room to search. If the filter is tight, bump this to 1000 or 2000; measure recall@100.
- **Recency preference is *not* here** — it belongs in a `function_score` wrapper or in a rescore step. Do not conflate hard filters with soft preferences.

To add recency as a soft preference, wrap the BM25 arm's inner query in `function_score`:

```json
{
  "function_score": {
    "query": { "multi_match": { "query": "reset admin password", "fields": ["title^5", "body"] } },
    "functions": [
      {
        "gauss": {
          "updated_at": { "origin": "now", "scale": "90d", "offset": "7d", "decay": 0.5 }
        }
      }
    ],
    "boost_mode": "multiply"
  }
}
```

The dense arm typically does not use `function_score` (the score-scale-mismatch problem chapter 1 named applies inside a single arm too — mixing kNN scores with function-score modifiers is fragile). If you want recency to affect the dense arm's contribution, apply it in a rescore step over the fused top-100, or express recency as a metadata bonus in a fused ranking step outside the two arms.

## Native hybrid query in OpenSearch with filters

OpenSearch's [hybrid query](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) composes similarly:

```json
POST /support_articles/_search?search_pipeline=rrf-hybrid-pipeline
{
  "size": 20,
  "query": {
    "hybrid": {
      "queries": [
        {
          "bool": {
            "must": {
              "multi_match": {
                "query": "reset admin password",
                "fields": ["title^5", "body", "tags^3"]
              }
            },
            "filter": [
              { "term":  { "product":  "Acme Dashboard" } },
              { "term":  { "language": "en" } },
              { "terms": { "acl": ["support", "user_readable"] } }
            ]
          }
        },
        {
          "knn": {
            "embedding": {
              "vector": [/* query embedding */],
              "k": 100,
              "filter": {
                "bool": {
                  "must": [
                    { "term":  { "product":  "Acme Dashboard" } },
                    { "term":  { "language": "en" } },
                    { "terms": { "acl": ["support", "user_readable"] } }
                  ]
                }
              }
            }
          }
        }
      ]
    }
  }
}
```

Same shape: filters duplicated across both arms, RRF (via the search pipeline) fuses.

## The two anti-patterns worth naming

1. **Filter on one arm, not the other.** Silently produces asymmetric candidate sets — one arm retrieves within-scope docs, the other retrieves globally. RRF over asymmetric sets is dominated by the arm that retrieved the in-scope docs; the other arm is decorative. Always mirror the filter across arms.
2. **Encoding filters into embeddings.** Doc text `"tenant_id=42 language=en updated=2024-11-01 body: how to reset..."`. Fragile (filter changes require re-embedding), leaky (the LLM may echo filter values back at generation time), and the embedder has no reason to respect the filter structure. Keep filters in metadata; keep semantic content in the embedding.

## What is NOT in this chapter (and where it goes)

- **The retriever interface contract that the peer track consumes** — chapter 7.
- **The filtered-ANN recall problem in detail (pre- vs. post-filter tradeoffs per vector engine)** — mod-104 (chapters on HNSW / IVF and filter semantics).
- **ACL-aware retrieval at enterprise scale (row-level security, multi-tenant, dynamic ACLs)** — mod-112.
- **Structured retrieval (text-to-SQL), graph retrieval, multimodal retrieval** — mod-112.
- **The eval methodology (per-query-class metrics, recall@k with filter tightness)** — mod-109.

## What you should be able to do after this chapter

- Name the four moments a filter can enter a retrieval pipeline (pre-filter, post-filter, filter-as-fusion-arm, boost-not-filter) and pick the right one per query.
- Diagnose the recall-collapse failure that post-filtering an ANN retriever produces and the fix (pre-filtered ANN with adequate `num_candidates`).
- Recognise the dense-arm-becomes-decorative failure on tight-filter queries and the workarounds (num_candidates bump, tight-filter routing, BM25-only fallback).
- Author a compose-with-metadata hybrid query in Elasticsearch or OpenSearch with mirrored filters across both arms.
- Distinguish permission (hard filter) from preference (soft boost via `function_score`) and defend the choice per query type.
- Refuse to encode filter values into embeddings and explain why.

## Further reading

- Elasticsearch: [*Filter for kNN search*](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-search-filter-example) — the pre-filtered ANN shape.
- Elasticsearch: [*RRF retriever*](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — with filter clauses on each sub-retriever.
- OpenSearch: [*Filtered k-NN search*](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/) — the pre-filtered ANN shape for OpenSearch.
- OpenSearch: [*Hybrid query*](https://opensearch.org/docs/latest/query-dsl/compound/hybrid/) — the native two-arm hybrid shape.
- Weaviate: [*Filters*](https://weaviate.io/developers/weaviate/search/filters) — Weaviate's `where` semantics for filtered vector search.
- Qdrant: [*Filtering*](https://qdrant.tech/documentation/concepts/filtering/) — Qdrant's payload-index and filter-estimator model.
- pgvector: [*Iterative index scans*](https://github.com/pgvector/pgvector#iterative-index-scans) — the postgres-side answer to the tight-filter-with-HNSW recall problem.
- Wang, X., Zhou, Y., and Deng, F. (2023). *Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters.* Microsoft Research. [arXiv:2401.01715](https://arxiv.org/abs/2401.01715) — for the reader who wants the algorithms literature behind filter-aware ANN.
