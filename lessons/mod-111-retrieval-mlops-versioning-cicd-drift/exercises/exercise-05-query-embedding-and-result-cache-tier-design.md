# Exercise 05 — Query-Embedding and Result Cache Tier Design

**Estimated effort:** ~2.5 hours. Python, Redis (or an in-process LRU that spills to disk), the retriever + cost dashboard from exercise 04, and the cutover scripts from exercise 01.

## Objective

Implement the three cache tiers from chapter 6 in a working retriever pipeline — query-embedding cache, top-k result cache, and generation cache — with complete keys, correct TTLs, invalidation hooks driven by exercise-01's cutover script, single-flight stampede protection, and a security-review-worthy tenant-scope story. Confirm each tier by driving a mixed traffic profile through the pipeline and reading the hit rates back off the cost dashboard.

By the end you have:

- `caches/keys.py` — one `cache_key()` function per tier; the *only* place cache keys are constructed.
- `caches/tiers.py` — the three cache implementations (Redis-backed or in-process; interface identical).
- `retriever/pipeline.py` — the cascaded pipeline from chapter 6's pseudocode.
- Wired-in invalidation: exercise-01's `flip.py` triggers the top-k and generation cache purges.
- Single-flight coalescing on at least the top-k tier.
- A hit-rate report per tier for a mixed synthetic traffic profile.
- A short security note showing why the tenant-scope predicate is in the key.

## Prerequisites

- Chapter 6 read; chapter 2 read (the invalidation hook).
- Exercise 01 done (cutover automation) and exercise 04 done (the cost dashboard + `INDEX_ID`-emitting retriever).
- Redis running locally (`docker run -p 6379:6379 redis:7`) or an in-process LRU (`cachetools`).
- Python 3.10+, `pip install redis cachetools` (or your preferred equivalents).

## Steps

**Step 1 — Author the key functions (25 minutes).**

`caches/keys.py` — one function per tier. Every field that affects the output must be in the key. Follow chapter 6's rules:

```python
def query_embed_key(query, model_id, model_revision):
    return sha256(json.dumps({
        "q": normalize(query),
        "m": model_id,
        "r": model_revision,
    }, sort_keys=True).encode()).hexdigest()

def top_k_key(query, index_id, retrieval_params, tenant_id):
    return sha256(json.dumps({
        "q": normalize(query),
        "i": index_id,
        "p": retrieval_params.canonical(),   # top_k, efSearch, filters, RRF k
        "t": tenant_id,
    }, sort_keys=True).encode()).hexdigest()

def generation_key(query, retrieved_context_hash, prompt_template_version, model_id, model_revision, tenant_id):
    return sha256(json.dumps({
        "q": normalize(query),
        "c": retrieved_context_hash,
        "pt": prompt_template_version,
        "m": model_id,
        "r": model_revision,
        "t": tenant_id,
    }, sort_keys=True).encode()).hexdigest()
```

Implement `normalize()`: lowercase, strip trailing punctuation, collapse whitespace, NFKC. Publish the normalization contract in `caches/README.md`; a change to it invalidates the cache.

Write a unit test that fails if any tier's key function is missing a field that the pipeline uses. This test is the mechanical defense against the "cache key incompleteness" bug from chapter 6's Failure A.

**Step 2 — Implement the three cache tiers (25 minutes).**

`caches/tiers.py`:

- **`QueryEmbedCache`** — get/set/purge on a Redis `HSET` or an in-memory LRU keyed by the query-embed key; value is the embedding vector serialized as bytes.
- **`TopKCache`** — get/set/purge; value is a list of `(doc_id, score)` tuples serialized as JSON.
- **`GenerationCache`** — get/set/purge; value is the generated answer as a string.

Each tier exposes:

- `get(key) -> Optional[value]`
- `set(key, value, ttl_seconds)`
- `purge()` — invalidates the whole tier (used by the cutover hook).
- `hits` and `misses` counters.

Publish per-tier TTLs in `caches/config.yaml`:

- `query_embed`: `None` (effectively forever; the model_id in the key handles model-bump invalidation).
- `top_k`: `1800` (30 min) or your project value.
- `generation`: `3600` (1 hour) or your project value.

Add TTL jitter (±10%) to the top-k and generation tiers.

**Step 3 — Wire the cascaded pipeline (25 minutes).**

`retriever/pipeline.py` follows chapter 6's pseudocode:

```python
def retrieve_and_generate(query, tenant_id, retrieval_params):
    # Tier 3: generation cache
    ctx_hash = "unknown"   # populate after retrieval; look up initially with a placeholder
    gk = generation_key(query, ctx_hash, PROMPT_TEMPLATE_VERSION, MODEL_ID, MODEL_REVISION, tenant_id)
    if answer := gen_cache.get(gk):
        emit_metric("cache_hit_generation")
        return answer

    # Tier 2: top-k cache
    tk = top_k_key(query, INDEX_ID, retrieval_params, tenant_id)
    if top_k := top_k_cache.get(tk):
        emit_metric("cache_hit_top_k")
    else:
        # Tier 1: query-embed cache
        qe = query_embed_key(query, EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
        if v := query_embed_cache.get(qe):
            emit_metric("cache_hit_query_embed")
        else:
            v = embed_api.embed(query)
            query_embed_cache.set(qe, v, ttl_seconds=None)

        top_k = ann.search(v, top_k=retrieval_params.top_k, filters=tenant_filter(tenant_id))
        top_k_cache.set(tk, top_k, ttl_seconds=TOP_K_TTL)

    # Re-compute context hash for the generation cache key (see Notes)
    ctx_hash = hash_context(top_k)
    gk = generation_key(query, ctx_hash, PROMPT_TEMPLATE_VERSION, MODEL_ID, MODEL_REVISION, tenant_id)
    if answer := gen_cache.get(gk):
        emit_metric("cache_hit_generation")
        return answer

    answer = generator.generate(query, top_k, PROMPT_TEMPLATE)
    gen_cache.set(gk, answer, ttl_seconds=GEN_TTL)
    return answer
```

Confirm the pipeline runs end-to-end on a single query with an actual embedding + retrieval + generation call (stub the generator if you don't have LLM access — a deterministic function of `top_k` is fine).

**Step 4 — Wire the cutover invalidation hook (15 minutes).**

Extend exercise-01's `flip.py`:

```python
def flip_alias(alias, target):
    execute_atomic_flip(alias, target)
    top_k_cache.purge()      # doc-IDs may be different in the new index
    gen_cache.purge()        # generation depends on retrieved context
    # DO NOT purge query_embed_cache unless the embedding model changed
    if new_manifest.embedding_model_id != old_manifest.embedding_model_id:
        query_embed_cache.purge()
    audit_log("caches purged after cutover", flipped_to=target)
```

Run the flip from exercise 01. Confirm the two caches purged; confirm the query-embedding cache did NOT purge (unless the model actually changed). Cite chapter 6's Failure B in a comment — the "post-cutover, users saw stale results for hours" incident this hook defends against.

**Step 5 — Add single-flight to the top-k tier (20 minutes).**

Wrap `top_k_cache.get_or_compute(key, fetch_fn)` with a coalescer: when N concurrent requests miss on the same key, only one calls `fetch_fn`; the rest wait on the future.

Redis-based approaches use `SETNX` + a short expiry; in-process approaches use `threading.Lock` per key. Chapter 6's `singleflight` reference is the pattern. `cachetools`'s `TTLCache` + a per-key `threading.Condition` is one straightforward implementation.

Simulate: fire 50 concurrent requests for the same query when the top-k cache is cold. Confirm only ONE embedding API call and ONE ANN search fire; the other 49 wait and return.

**Step 6 — Tenant-scope security drill (15 minutes).**

This step is the defense against chapter 6's Failure A ("the top-k cache leaked results between tenants").

- Author two synthetic tenants A and B with disjoint corpora (or shared corpus but with per-tenant filter predicates).
- Fire the same query from both tenants.
- Confirm each tenant sees its own top-k, not the other's.
- Delete `tenant_id` from `top_k_key` momentarily; fire again; confirm the leak is now visible.
- Restore `tenant_id`; run the unit test from step 1; confirm the test would have caught the deletion.

Write a one-paragraph note in `caches/README.md` — "tenant-scope predicate is part of retrieval_params and is in the key; unit-tested; deleting it is a security incident."

**Step 7 — Traffic profile and hit-rate report (15 minutes).**

Fire a mixed traffic profile through the pipeline:

- 100 unique queries.
- 20 queries with high repetition (~10x each) — the tail that caches capture.
- 30 queries mixed across the two tenants.
- 50 queries that vary slightly ("password reset" vs. "reset password" vs. "How do I reset my password?") — the normalization test.

Report from the metrics you emit:

- Query-embedding cache hit rate.
- Top-k cache hit rate.
- Generation cache hit rate.
- Wall-clock speedup on cache hits vs. cold requests.
- Cost drop per request when the top-k cache hits (should be the embed call + the ANN call combined — chapter 5's cost model).

Publish in `caches/hit-rate-report.md`.

**Step 8 — Write up (10 minutes).**

In `caches/README.md`, publish:

- The three tiers with keys, values, TTLs, and invalidation triggers as a table.
- The cutover invalidation hook and its rationale (chapter 6 Failure B).
- The tenant-scope defense (chapter 6 Failure A) with the unit test citation.
- The stampede-protection choice (single-flight, TTL jitter, and/or refresh-ahead) per tier.
- The hit-rate report from step 7.
- A note on the *when-not-to-cache* rules (temperature > 0 generations, time-sensitive queries, session-specific personalization) and how your pipeline surfaces the `no_cache=True` opt-out.

## Acceptance criteria

- All three cache tiers exist, run, and are backed by a real store (Redis or a persistent LRU).
- Every cache key includes every field that affects the output, enforced by a unit test.
- The cascaded pipeline demonstrates the correct order (generation → top-k → query-embed).
- The cutover flip triggers the correct purges; the query-embed cache is preserved when the embedding model didn't change.
- Single-flight is proven with the concurrent-request test in step 5.
- The tenant-scope security drill in step 6 demonstrates the defense (with and without the predicate in the key).
- `caches/hit-rate-report.md` shows the hit rates and cost / latency deltas from the traffic profile in step 7.
- `caches/README.md` documents the design, the failures the design defends against, and the opt-out policy.

## Stretch goals

- **Refresh-ahead for hot keys.** For the top-N most-requested keys, refresh the cache asynchronously before TTL expiry. Downstream never sees the miss. Measure the tail-latency win.
- **Vendor-side prompt caching.** Enable Anthropic's or OpenAI's prompt-caching feature alongside your generation cache. Chapter 6 warns not to confuse the two; confirm they compose and record the combined savings.
- **Change-data-capture invalidation.** Hook chapter 4's incremental ingest to invalidate the top-k cache for keys whose top result *might* have changed (chunk-ID-level surgical invalidation). Measure hit-rate impact vs. the simple purge-everything approach.
- **`no_cache=True` on session-scoped queries.** Wire a rule where a query with a session-context token skips the generation cache; confirm the pipeline honors it and the metric shows the skipped-cache count.
- **Multi-region cache.** If you have two Redis nodes, shard by `tenant_id` (consistent hashing per Karger 1997) and observe per-shard hit rates.
- **Cache-size vs. hit-rate curve.** Set the top-k cache to progressively larger sizes; plot hit rate vs. size. Locate the knee of the curve; put the number in the write-up.

## Notes

- **`cache_key()` is the single source of truth for keys.** Do not construct a key inline in the pipeline. The unit test in step 1 is what catches "someone added a new retrieval parameter and forgot to add it to the key."
- **Two `generation_key` lookups is not a bug.** The generation-cache key depends on `retrieved_context_hash`, which isn't known until after retrieval. The pseudocode looks up the generation cache twice — once optimistically with a placeholder (which will miss until retrieval fills the hash), once after retrieval. If your workload benefits, restructure into a top-k-first shape; the two-lookup form is the safest starting point.
- **Do not batch cache invalidation.** The flip script invalidates in-line, synchronously. "Invalidate on the next batch" is chapter 6's Failure B in disguise.
- **Tenant-scope in the key is a security control, not an optimization.** Treat it as a hard requirement; document it as such; make deletion of the field a CI-failing change.
- **Redis with client-side caching (RESP3) is a good default.** Redis's [client-side caching](https://redis.io/docs/manual/client-side-caching/) invalidates the client's local copy on server-side write. Combine with your cache tiers for a lower-tail-latency read path.
- **The `no_cache=True` opt-out is easy to forget.** Make it a required field on the retrieval request in a strict mode; the missing-flag failure is easy to catch in a code review and hard to notice in production.
