# Chapter 7 — Metadata Filtering: Pre-Filter vs. Post-Filter, and the Correctness Trap of Small `k`

## Why this chapter exists

A retrieval engineer who has been on-call for a real RAG system has debugged this bug at least once: "our search worked fine on the demo, but when tenant Acme with a small permission scope queries it, the answer is missing the obvious document." Nine times out of ten, that bug is **post-filter with a selective predicate at a small `k`** — a correctness trap that is not visible in ANN benchmarks (which never filter) and not visible in end-to-end recall tests that use unrestricted queries. It only surfaces in production, on real predicates, at production-typical `k`.

This chapter names the trap, shows the mechanics, walks the vendor-by-vendor pattern for pushing predicates into the index, and gives you the drill exercise 04 turns into a hands-on lab. The stakes are high because filter bugs are silent — the system still returns *k* results; they are just not the *right* k.

## The two shapes: pre-filter vs. post-filter

The vector index returns the top-`k` vectors most similar to `q`. A filter `p(doc)` is a boolean predicate on the payload (tenant, permission, freshness, category, ...). There are exactly two places to apply it.

**Post-filter** (a.k.a. "filter after search"):

```
    q ── ANN top-k ──▶ [d₁, d₂, ..., d_k] ──▶ apply p(·) ──▶ [d_i : p(d_i)]
```

- The ANN scan is unaware of `p`. It returns `k` candidates.
- `p` is applied to the `k` candidates and any that fail are dropped.
- **If none of the top-`k` satisfy `p`, the result is empty** — even if there are many corpus vectors that satisfy `p` and are close to `q`.

**Pre-filter** (a.k.a. "filtered ANN," "constrained ANN"):

```
    q, p ── ANN over {v : p(v)} ──▶ top-k' subject to p ──▶ [d_i : p(d_i)]
```

- The ANN scan is *filter-aware*: it only considers vectors satisfying `p`, or it re-enters the scan whenever the candidate set is exhausted.
- Returns `k` results that satisfy `p` (up to the corpus's supply of such vectors).
- **Cost model depends on the filter's selectivity** — very selective filters can be more expensive because the scan has to explore more of the graph/partition to find `k` matching vectors.

The correctness trap: **post-filter under a selective predicate at small `k` silently drops recall.** The number of results returned is not `k` — it is `k' ≤ k`, and `k'` may be zero even when the "true" recall (on the filtered corpus) should be perfect.

## The mechanics of the trap — a concrete walkthrough

Suppose the corpus has `N = 10⁶` vectors and a predicate `p(doc) = doc.tenant_id == 42` matches 0.1% of the corpus (`10³` vectors). You retrieve `k = 10` for a query `q`.

**Post-filter, `k = 10`:**

1. ANN returns the 10 vectors closest to `q` across the *whole* corpus.
2. In expectation, 10 · 0.001 = 0.01 of them belong to tenant 42.
3. Almost every query returns an empty result for tenant 42.

**Post-filter, `k = 100`:**

1. ANN returns the top-100 across the whole corpus.
2. Expected matches: 100 · 0.001 = 0.1. Still usually empty.

**Post-filter, `k = 10000`:**

1. Expected matches: ~10.
2. But you have to score 10⁴ vectors from a 10⁶ index and then throw most away. Latency inflates, and you still miss any tenant-42 documents that were merely close-*enough* to `q` (i.e. not in the global top-10000).

**Pre-filter, `k = 10`:**

1. Scanner walks only tenant-42 vectors.
2. Returns the top-10 among the 10³ that match. Every one satisfies `p`.

The post-filter shape only works when your predicate is *loose* (matches, say, > 20% of the corpus). Every RAG system with even mild multi-tenancy or permission scoping has predicates that fail that criterion — sometimes drastically. That is why chapter 2's rubric asked "is filtering pre- or post-scan" as one of the seven questions.

## Why the post-filter shape ever existed

Post-filter is the default in engines whose vector index was built without knowledge of the metadata. FAISS's `IndexIVFPQ` returns IDs; you attach metadata elsewhere; the only place to apply `p` is after the search. That is fine for research benchmarks (which do not filter) but wrong for multi-tenant production.

pgvector's older versions (< 0.5) had the same shape for HNSW: the planner used the ANN index to find candidates, then evaluated the `WHERE` clause. If the `WHERE` was very selective, recall crashed. pgvector 0.5+ added **iterative index scan** for HNSW — the planner iterates the ANN graph, evaluating the predicate inline, and reissues the scan if too few candidates satisfy `p`. Correctness is restored at the cost of some latency on selective filters.

The general fix that every specialist database now implements is some form of **filter-aware traversal**: teach the ANN algorithm about the predicate so its exploration budget is spent on matching vectors.

## Filter-aware traversal — how the specialists do it

Two flavors, both widely deployed:

### 1. Bitmap-intersected HNSW (Weaviate, Milvus, some pgvector configs)

Maintain a **Roaring bitmap** (or similar) per filterable attribute value indicating which vector IDs match. At query time:

1. AND the bitmaps for the requested predicate → the allowed-IDs bitmap.
2. Run HNSW traversal; at each node, check the bitmap; only expand to allowed neighbors.
3. Continue until top-`k` matching vectors are found or the graph is exhausted.

Roaring bitmaps are memory-efficient and fast to intersect. The correctness holds because the traversal actually visits matching vectors; the cost holds because bitmap check is O(1) per node.

Weaviate documents this exact shape in [*Filters*](https://weaviate.io/developers/weaviate/concepts/filtering); Milvus's [*Filtered Search*](https://milvus.io/docs/filtered-search.md) covers the same idea.

### 2. Payload-indexed segments with filter-aware graph (Qdrant)

Qdrant keeps **payload indexes** on individual fields (keyword, integer, float, geo, boolean, datetime). When a query with a `filter` arrives, Qdrant uses the payload indexes to identify the matching subset per segment; the HNSW traversal is filter-aware and stays inside the matching subset. For very selective filters, Qdrant may fall back to an exact scan on the filtered subset (which is cheaper than exploring an unfiltered graph and finding zero matches). The [*Filtrable HNSW*](https://qdrant.tech/articles/filtrable-hnsw/) article is the authoritative writeup.

### 3. Iterative index scan (pgvector 0.5+)

Postgres's planner drives an iterative loop:

1. Ask the HNSW index for the next candidate.
2. Evaluate the `WHERE` predicate.
3. If matched, emit; otherwise, ask for the next candidate.
4. Continue until `k` matches are emitted or the graph is exhausted.

Set with `SET hnsw.iterative_scan = strict_order;` in pgvector 0.7+ (or `relaxed_order`, which allows scoring out-of-order for speed). Details in the pgvector README.

## The per-vendor "how do I pre-filter?" pattern

| System | How to specify a filter | Filter-aware backend |
|---|---|---|
| pgvector | `WHERE ... AND ...` in SQL; combine with ANN `ORDER BY ... LIMIT k` | HNSW iterative scan (0.5+); IVFFlat still needs care with selective filters |
| Pinecone | `filter` object in the `query` request | Serverless index type is filter-aware; older pod index types have restrictions |
| Weaviate | `where` clause on GraphQL / REST | Roaring-bitmap-intersected HNSW; per-tenant HNSW for multi-tenancy |
| Qdrant | `filter` in `search` request; payload indexes on individual fields | Filter-aware HNSW; fallback to exact on very selective filters |
| Milvus | Boolean expression string in `expr` param | Filter-aware traversal (`iterative` mode in 2.4+) |
| FAISS | `IDSelectorBatch`/`IDSelectorBitmap` passed to `search` | Restricted to the ID subset the selector defines |
| Chroma | `where` (metadata), `where_document` (substring on doc body) | Pre-filter via metadata pushdown |

The point of the table is not to memorize each syntax — it is to know that each vendor has *some way* to push a predicate into the ANN scan, and you need to use it. The most common mistake is to `SELECT ... FROM t ORDER BY embedding <=> $q LIMIT 10` followed by application-side filtering. That is post-filter with all its failure modes.

## Selectivity — the number every plan hinges on

**Selectivity** = `|{docs : p(doc)}| / N` — the fraction of the corpus that satisfies the predicate.

| Selectivity regime | Post-filter safe? | Pre-filter cost | Typical example |
|---|---|---|---|
| > 20% | Usually yes | Cheap | "category IN (a, b, c)" on a large-vocabulary corpus |
| 1% – 20% | No — recall degrades | Moderate | "tenant_id = X" in a mid-sized multi-tenant system |
| < 1% | No, catastrophically | Sometimes falls back to exact | "tenant_id = X" on a large system; "doc_id IN (…)" for permissions |

The operational rule: **know your selectivity distribution**. Sample a hundred real queries with their filters, compute selectivity, plot the histogram. If a meaningful fraction sits below 20%, post-filter is not an option.

## When pre-filter itself is expensive — the two shapes to know

Filter-aware HNSW is not free. On very selective filters, the traversal may explore much of the graph looking for matching vectors — sometimes it is cheaper to abandon the graph and do an **exact scan on the filtered subset**. Both Qdrant and Milvus implement this fallback; you rarely need to configure it, but you need to know it exists so the latency shape ("selective filter → longer p95 for that query class") is not a surprise.

The second expensive shape is **very-many-filter-attribute combinations**. If your predicate is a disjunction over five keyword indexes and two datetime ranges, the bitmap intersection can dominate cost. The fix is application-side: precompute common filter sets, cache the bitmap, or partition data physically so a common filter reduces to "scan segment X" rather than "scan-and-filter everywhere."

## The multi-tenant special case

Multi-tenancy is the single most common source of the post-filter trap. Two patterns handle it cleanly:

**1. Namespace / partition per tenant** — the physical layout is already partitioned by tenant, so a "tenant_id = X" query naturally scans only tenant X's vectors. Pinecone namespaces, Milvus partitions, Weaviate's `multiTenancyConfig`, Qdrant sharding by tenant.

- **Pro.** Zero recall risk; each tenant's HNSW / IVF-PQ is tuned for its data.
- **Con.** Overhead scales linearly with tenant count. Not great for 100k+ small tenants.

**2. Tenant-ID payload field with pre-filter** — one big index, `tenant_id` as an indexed payload field, every query pre-filters on it.

- **Pro.** One index amortizes across many tenants.
- **Con.** Requires filter-aware traversal. Correctness depends on the vendor doing this right (chapter 2 rubric cell).
- **Isolation caveat.** Payload-based tenancy is a *logical* boundary. A bug in the filter machinery can leak across tenants. mod-112 covers the ACL implications.

Choose based on tenant count and size distribution. A handful of large tenants: namespace-per-tenant. Many small tenants: shared index + payload filter. Mixed: hybrid (large tenants each in their own namespace, small tenants sharing).

## The correctness drill — the pattern exercise 04 codifies

The drill for a new deployment or a new predicate:

1. **Enumerate the predicates.** Every `WHERE` / `filter` clause the query API supports. Do not skip the "internal" ones (permission scope, freshness).
2. **Estimate selectivity per predicate.** Sample real queries or profile production logs.
3. **Classify each predicate as "loose" (> 20%) or "selective" (< 20%).**
4. **For every selective predicate, verify the pre-filter path.** Point at the vendor doc that says "this filter is pushed into the index." If you cannot find that doc, run the correctness test in the next section.
5. **Add a CI test that runs a canary query with a selective predicate and asserts recall.** Details in mod-111.

## A concrete correctness test — the shape

The test walks the trap the same way it happens in production.

```python
import numpy as np

def build_and_query(engine, predicate_selectivity, k=10, N=10_000):
    # Corpus of N random vectors, each with a tenant_id.
    # `predicate_selectivity` of the corpus is tenant_id = 42.
    dim = 128
    xb = np.random.randn(N, dim).astype("float32")
    tenant_ids = np.where(
        np.random.rand(N) < predicate_selectivity, 42, 0
    )
    engine.load(xb, tenant_ids)

    # Ground truth: brute-force top-k on the tenant-42 subset.
    q = np.random.randn(dim).astype("float32")
    tenant42_mask = tenant_ids == 42
    xb_tenant = xb[tenant42_mask]
    if len(xb_tenant) == 0:
        return None
    exact_scores = xb_tenant @ q
    exact_top_k = np.argsort(-exact_scores)[:k]

    # ANN query with filter=tenant_id==42
    ann_ids = engine.query(q, k=k, filter={"tenant_id": 42})
    ann_scores = xb[ann_ids] @ q
    # Recall = |ANN ∩ exact| / k on the (tenant-42) index space
    exact_ids = np.where(tenant42_mask)[0][exact_top_k]
    return len(set(ann_ids) & set(exact_ids)) / k

# Sweep selectivity
for s in [0.5, 0.2, 0.05, 0.01, 0.001]:
    r = build_and_query(engine, predicate_selectivity=s)
    print(f"selectivity={s:.3%}  recall@10={r:.3f}")
```

Run this against both an engine you claim is filter-aware and a naïve post-filter engine (e.g. FAISS with post-filter in Python). The naïve one crashes to near-zero recall at `s = 0.001`; a correctly filter-aware one stays near 1.0. Exercise 04 turns this shape into a runnable lab across pgvector, Qdrant, and FAISS.

## What is NOT in this chapter (and where it goes)

- **ACLs and permission-scope filters as security boundaries** — mod-112.
- **Filter-based routing across tenants at platform scale** — level 40 territory; chapter 9's boundary.
- **Hybrid retrieval (BM25 ⊕ dense) and how filters compose across both legs** — mod-106.

## What you should be able to do after this chapter

- Draw the two filter shapes (pre and post) and predict what each does at low, medium, and high selectivity.
- State the correctness trap in one sentence and name at least three real-world predicates in which it triggers.
- Point at each of the seven systems' pre-filter machinery and give the one-line "how you specify a filter" for each.
- Design a correctness test that would have caught the trap before production.

The next chapter takes the tuned, sized, correctly-filtered index and puts it on the operational shelf: versioning, dual-index cutover, and the connection to retriever CI.

## Further reading

- Qdrant, [*Filtrable HNSW*](https://qdrant.tech/articles/filtrable-hnsw/) — the authoritative writeup of filter-aware graph traversal.
- Weaviate, [*Filters*](https://weaviate.io/developers/weaviate/concepts/filtering) — Roaring-bitmap-intersected HNSW.
- Milvus, [*Filtered Search*](https://milvus.io/docs/filtered-search.md) — iterative filter mode and expression syntax.
- pgvector README, sections on [HNSW iterative scan](https://github.com/pgvector/pgvector#iterative-index-scans) — the fix that made pgvector safe for selective predicates.
- Pinecone, [*Understanding metadata*](https://docs.pinecone.io/guides/data/understanding-metadata) — filter push-down behavior across index types.
- Wang, C., et al. (2024). *AnalyticDB-V: A Hybrid Analytical Engine Towards Query Fusion for Structured and Unstructured Data.* VLDB 2024 — one of the recent papers treating filtered-ANN correctness as a first-class research problem.
- Chen, P., et al. (2023). *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.* SIGMOD 2024 — filter-aware ANN with a general predicate handler.
