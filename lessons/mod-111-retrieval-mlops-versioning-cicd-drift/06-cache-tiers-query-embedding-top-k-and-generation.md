# Chapter 6 — Cache Tiers: Query-Embedding, Top-k Result, and Generation

## Why this chapter exists

Chapter 5 named the six cost / latency line items and identified caches as the "cheap, hits both cost and latency, sometimes hits quality" lever. This chapter is the design of those caches: where they sit, what they key on, what they invalidate on, and how the three tiers compose.

The three tiers are:

1. **Query-embedding cache** — caches the vector produced by embedding a query text. Keyed on the normalized query text.
2. **Top-k result cache** — caches the retriever's ranked list of doc-IDs for a query. Keyed on `(query, INDEX_ID, retrieval_params)`.
3. **Generation cache** — caches the LLM's final answer for a query. Keyed on `(query, retrieved_docs, prompt_template, model)`.

Each tier sits at a different point in the request path; each has different TTL, invalidation, and cost characteristics; each has a different hit-rate expectation. The right shape uses all three, wired so a cache miss at one tier still benefits from a hit at another.

Boundary: chapter 5 covered *what* to cache (why caching hits the cost dashboard). This chapter covers *how* to cache (design, invalidation, safety).

## The request path with caches

```
    query text                                                                             answer
        │                                                                                    ▲
        ▼                                                                                    │
    ┌──────────────────┐          hit ────────►                                     ┌────────┴────────┐
    │ generation cache │───────────────────────────────────────────────────────────►│  return cached  │
    │ (query,ctx,mdl)  │                                                              │  answer         │
    └────────┬─────────┘                                                              └─────────────────┘
             │ miss                                                                              ▲
             ▼                                                                                    │
    ┌──────────────────┐          hit ─────────────────────────►                       ┌─────────┴───────┐
    │ top-k cache      │──────────────────────────►                                    │ generator (LLM) │
    │ (q, INDEX_ID)    │                                                                │                  │
    └────────┬─────────┘                                                                └──────────────────┘
             │ miss                                                                              ▲
             ▼                                                                                    │
    ┌──────────────────┐          hit ─────────►                    ┌────────────────┐ ┌────────┴───────┐
    │ query-embed cache│────►                    ┌──────────────┐   │  ANN search    │ │ reranker (mod- │
    │ (normalized q)   │                          │ embedding API│──►│  (chunks →     │─►│ 107)           │
    └────────┬─────────┘  miss                    └──────────────┘   │   top-N docs)  │ │                 │
             ▼                                                        └────────────────┘ └────────────────┘
       call embed API
```

The three tiers cascade: generation-cache hit ends the request; top-k-cache hit skips the retriever + reranker; query-embedding-cache hit skips the embedding call. Every hit reduces cost and latency on its part of the pipeline.

## Tier 1: query-embedding cache

**Position in pipeline.** Before the embedding API call.

**Key.** The **normalized** query text. Normalization is critical — see the pitfalls below.

**Value.** The embedding vector (a fixed-length float32 array; e.g. 1024 floats).

**TTL.** Long. Effectively forever *for a fixed embedding model + revision*. A query's vector doesn't change until the model changes; when the model changes, invalidate all entries.

**Invalidation.** On embedding-model-or-revision change (a tuple bump on the embedding field, chapter 4). Simplest is to key the cache internally as `(model_id, query_text)` so a bump silently misses the old entries; they age out.

**Store.** Redis or an in-memory LRU with disk-backed spill for large embedding tables. Payload size = number-of-dims × 4 bytes; 1024-dim floats are 4 KB per entry. A million cached queries at 1024-dim is ~4 GB — trivial for Redis.

**Hit rate.** Highly variable. User-facing surfaces with strong query repetition (support chatbots) see 40–70% hit rates; exploratory-analytics surfaces see much less. Instrument the hit rate (chapter 5's dashboard) and let it inform whether the tier is worth keeping.

**Normalization pitfalls.** Without normalization, `"how do I reset my password"` and `"How do I reset my password?"` cache-miss on each other. Standard normalization pipeline:

1. Lowercase.
2. Strip trailing punctuation.
3. Collapse repeated whitespace.
4. Optional: Unicode-normalize (NFKC).
5. Optional: strip stopwords or apply a light stemmer *if* your embedding model is stopword-invariant (most modern semantic embedders are; do NOT do this for exact-match systems).

Publish the normalization function alongside the cache; treat a change to it as a cache-invalidation event.

**When NOT to cache.** For queries that include a **timestamp or session-specific context** (e.g. "what did I just say about X?"), caching by query text alone is wrong — the embedding is the same but the answer isn't. Cache on the retrieval side (top-k) at your own risk; the generation cache at the top is *almost always* wrong for those queries. The safe rule: strip time-varying context before hashing for the cache key, and require the retriever library's caller to pass a `no_cache=True` flag for the queries where you can't reliably strip it.

## Tier 2: top-k result cache

**Position in pipeline.** Before the ANN search.

**Key.** `(normalized_query_text, INDEX_ID, retrieval_params)`. `retrieval_params` includes `top_k`, `efSearch` / `nprobe`, filter predicates, RRF `k`, and any other query-time knob that changes what the retriever returns.

**Value.** The ranked list of doc-IDs (and scores, if the reranker uses them as input). Not the doc contents — those live in the payload store; the top-k cache just stores IDs.

**TTL.** Shorter than the embedding cache. The top-k result *changes* when:
- The corpus changes (a new relevant doc appears; a stale one is deleted).
- The index changes (`INDEX_ID` bumps).
- Retrieval params change.

**Invalidation.**
- On tuple change → `INDEX_ID` changes → the cache key changes → old entries silently miss. Include `INDEX_ID` in the key exactly for this reason. Chapter 2's cutover flip triggers a *cache purge* to reclaim the space immediately rather than waiting for TTL expiry.
- On incremental ingest (chapter 4) → the corpus changes but `INDEX_ID` may not, if you keep the tuple's `corpus-snapshot` field date-fuzzy. Two options: bump `INDEX_ID` on every incremental (higher cache churn), or accept staleness up to the next tuple bump (lower fidelity for fresh queries). Project-specific choice; document it.
- On `retrieval_params` change → different key → silent miss.

**Store.** Redis is the standard. Payload = a list of ~50–200 doc-IDs; small. TTL 5–60 minutes is a common range for user-facing surfaces.

**Hit rate.** Comparable to the query-embedding cache — same queries repeat. Because the top-k cache is *downstream* of the query-embedding cache, a hit here also saves the embedding call *and* the ANN search.

**Freshness trade.** The main downside of top-k caching is stale results after a small corpus change. A user creating a new relevant doc, then immediately querying for it, might see the cached (pre-doc) result. Options:
- Short TTL (5 minutes) accepts a small staleness window.
- Cache invalidation via a change-data-capture event that hooks into the incremental ingest.
- Exclude specific queries from the cache (e.g. authenticated user's own docs).

**When NOT to cache.** For tenant-scoped or personalized retrieval (mod-104 chapter 7, mod-112 ACL) the key must include the tenant-scope predicate; if it doesn't, one tenant's cached results leak to another. This is a **security** issue, not a fidelity issue — get it right.

## Tier 3: generation cache

**Position in pipeline.** At the very top of the pipeline; before any retrieval or embedding.

**Key.** `(normalized_query, retrieved_context_hash, prompt_template_version, model_id, model_version)`. If the retrieved context can change (it can — corpus updates, incremental ingest), the context-hash must reflect what was retrieved.

**Value.** The LLM's generated answer.

**TTL.** Shortest of the three tiers. Model behavior can shift (hosted vendors quietly update models), user intent shifts, and factuality decays. Common: hours to a day.

**Invalidation.**
- On model version change → key change → silent miss.
- On prompt-template change → key change → silent miss.
- On retrieval-side change → key change → silent miss.
- On explicit TTL expiry.

**Store.** Redis or the LLM provider's built-in prompt cache. Anthropic and OpenAI both offer prompt-caching primitives that reduce the *cost per call* of a repeated prompt without a client-side cache — check current vendor docs for the shape.

**Hit rate.** Highly workload-dependent. FAQ-style surfaces see high hit rates; open-ended surfaces see low ones. Instrument.

**Confusion with prompt caching.** "Prompt caching" (Anthropic / OpenAI) reduces the *token cost* of repeated prompt prefixes on the vendor side; the vendor still runs the model, just cheaper. "Generation caching" (this tier) returns a cached answer without calling the model at all. Both are useful; they compose. Don't confuse them.

**When NOT to cache.**
- Any generation with **randomness / temperature > 0** — the caller may expect variety, and the cached "one right answer" defeats that.
- Any generation with **user-specific personalization** — mixing users at the cache key is a leak.
- **Time-sensitive queries** ("what's the current status of X") — the cache TTL must be shorter than the freshness expectation.

## How the three tiers interact — invariants and correctness

Two invariants keep the composition correct.

**Invariant 1: an upstream miss must never *silently* be masked by a downstream hit.** If the top-k cache hits but is stale (the corpus changed and TTL didn't catch it), returning a stale generation over stale retrieval is a compound stale. Chapter 2's cutover flip is the once-per-rebuild event that reliably purges the top-k cache; the top-k TTL is the between-rebuild bound.

**Invariant 2: cache keys must be complete.** Every input that affects the output must be in the key. The most common bug is forgetting `INDEX_ID` in the top-k cache key. The second-most-common is forgetting `prompt_template_version` in the generation cache key.

The mechanical way to enforce completeness: the retriever library defines a `cache_key()` method that constructs the full key. Never construct the key ad-hoc at the call site.

```python
# retriever.py
def cache_key(self, query, params):
    return hashlib.sha256(
        json.dumps({
            "query": normalize(query),
            "index_id": self.config.get("INDEX_ID"),
            "params": params.canonical_form(),
        }, sort_keys=True).encode()
    ).hexdigest()
```

Every cache put/get goes through `cache_key()`. Any new cache-affecting field is added to `cache_key()` — the compiler / test suite tells you what breaks.

## Cache invalidation on cutover — the chapter-2 hook

Chapter 2's flip script must trigger cache invalidation. The pattern:

```python
# flip_alias.py (from chapter 2), extended
def flip_alias(alias, target):
    # ... perform the atomic flip ...
    invalidate_top_k_cache(alias=alias)                    # purge top-k cache
    # generation cache: purge if the answers depend on retrieval (they do for RAG)
    invalidate_generation_cache(alias=alias)
    # query-embedding cache: DO NOT purge unless embedding model changed
    audit_log("caches purged after cutover", alias=alias, target=target)
```

Query-embedding cache typically survives cutover (the embedding function didn't change unless the tuple's `embedding-model` field did — check the manifest); the top-k cache is invalidated because the doc-IDs may differ; the generation cache is invalidated because the retrieved context feeds it.

## Stampede protection

When the cache invalidates and every request suddenly misses, downstream services take a load spike (the "thundering herd"). Mitigations:

- **Request coalescing / single-flight.** Only one request per key hits the downstream at once; concurrent requests for the same key wait. Go's [`singleflight`](https://pkg.go.dev/golang.org/x/sync/singleflight), Redis-based [request-coalescing patterns](https://redis.io/docs/manual/patterns/distributed-locks/), and the [`cachetools`](https://cachetools.readthedocs.io/) Python library are common implementations.
- **Staggered TTLs.** Add jitter (±10-20%) to TTL so entries don't all expire at the same second.
- **Refresh-ahead.** For hot keys, refresh the cache asynchronously before the TTL expires. Downstream doesn't see a miss.

Every retrieval cache should have at least single-flight in place.

## Cache sizing — the storage vs. hit-rate trade

Caches obey a diminishing-returns curve: doubling the cache size increases hit rate less than double. The right size is where marginal storage cost equals marginal savings from the additional hits.

Practical shape:

1. Instrument hit rate at the current size.
2. Estimate hit rate at 2× size from the tail of your query distribution (a request-frequency histogram makes this a straight calculation).
3. Compute the storage-cost delta and the API-cost delta at 2×.
4. Ship the larger cache if the delta is positive.

Redis is cheap per GB; the calculation usually says "make it larger" until you hit the memory ceiling of the box.

## Concrete example — the three tiers as pseudocode

```python
# retriever_pipeline.py (illustrative)

def retrieve_and_generate(query, tenant_id):
    # Tier 3: generation cache (top of pipeline)
    gen_key = generation_cache_key(query, tenant_id, PROMPT_TEMPLATE_VERSION, MODEL_ID)
    if answer := gen_cache.get(gen_key):
        emit_metric("cache_hit_generation", tenant_id=tenant_id)
        return answer

    # Tier 2: top-k result cache
    tk_key = top_k_cache_key(query, tenant_id, INDEX_ID, retrieval_params)
    if top_k := top_k_cache.get(tk_key):
        emit_metric("cache_hit_top_k", tenant_id=tenant_id)
    else:
        # Tier 1: query-embedding cache
        qe_key = query_embed_cache_key(query, EMBEDDING_MODEL_ID)
        if q_vec := query_embed_cache.get(qe_key):
            emit_metric("cache_hit_query_embed", tenant_id=tenant_id)
        else:
            q_vec = embedding_api.embed(query)
            query_embed_cache.set(qe_key, q_vec, ttl=None)

        top_k = ann.search(q_vec, top_k=RETRIEVE_N, filters=tenant_id)
        top_k_cache.set(tk_key, top_k, ttl=TOP_K_TTL)

    # (Optional rerank step — chapter 5)
    reranked = reranker.rerank(query, top_k, keep_k=RERANK_K)

    answer = generator.generate(query, reranked, PROMPT_TEMPLATE)
    gen_cache.set(gen_key, answer, ttl=GEN_TTL)
    return answer
```

Two things to observe: every tier is *checked before it is populated below*, and every cache key includes everything that affects the output. The `emit_metric` calls populate the cost/latency dashboard from chapter 5.

## Two failures the cache design defends against

### Failure A: cache-key incompleteness leaked results between tenants

The top-k cache key was `(query_text)` and forgot `tenant_id`. Two tenants who happened to ask the same question saw each other's results. Discovered by a security review, six weeks after ship, with no way to know how many bad results were returned in the interim.

**Defense.** The `cache_key()` method as the single source of truth for what goes into the key. Tenant-scope predicates are part of `retrieval_params`; they *must* be in the key. A unit test enforces that the key includes every field the caller relies on.

### Failure B: post-cutover, users saw stale results for hours

A cutover flipped the alias; the retriever library started serving from `v2`; the top-k cache was not invalidated. For hours, users saw cached results from `v1` because the cache still returned them. Only new queries hit `v2` fresh.

**Defense.** Chapter 2's flip script must invalidate the top-k cache (and the generation cache) as part of the flip. Not "eventually"; not "on TTL expiry"; immediately, in the same script.

## What you should be able to do after this chapter

- Draw the request path with the three cache tiers in position (generation-cache at top, top-k in the middle, query-embedding at the bottom).
- Author a complete `cache_key()` function for each tier and name every field that must be in it.
- State the TTL and invalidation policy for each tier and explain why the tier's characteristics dictate them.
- Wire chapter 2's cutover script to invalidate the top-k and generation caches while leaving the query-embedding cache intact.
- Name at least three stampede-protection mechanisms (single-flight, TTL jitter, refresh-ahead) and pick one for a given cache.
- Distinguish "generation caching" (returns cached answer, no LLM call) from "vendor-side prompt caching" (still calls the LLM, cheaper).
- Recognize when a query must be excluded from caching (personalized, time-sensitive, high-variance / creative generation).

## Further reading

- Redis: [*Client-side caching*](https://redis.io/docs/manual/client-side-caching/) — the reference on the Redis-native caching primitives.
- Redis: [*Distributed locks*](https://redis.io/docs/manual/patterns/distributed-locks/) — used for single-flight patterns.
- Anthropic docs: [*Prompt caching*](https://docs.anthropic.com/claude/docs/prompt-caching) — the vendor-side prompt-cache primitive that composes with this chapter's generation cache. Confirm current URL and semantics in the docs.
- OpenAI docs: [*Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching) — the OpenAI vendor-side equivalent.
- Nishtala, R., Fugal, H., Grimm, S., et al. (2013). *Scaling Memcache at Facebook.* NSDI. [PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf) — canonical reference on production caching including stampede protection.
- Go standard library: [`singleflight`](https://pkg.go.dev/golang.org/x/sync/singleflight) — the reference implementation of single-flight coalescing; the pattern is portable.
- Chip Huyen (2022). *Designing Machine Learning Systems*, Ch. 7 (Model Deployment and Prediction Service) — general treatment of prediction-time caching that this chapter specializes.
- Karger, D. et al. (1997). *Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web.* STOC. [DOI:10.1145/258533.258660](https://doi.org/10.1145/258533.258660) — foundational reference; useful if the top-k cache is sharded across nodes.
- `cachetools` Python library. [docs](https://cachetools.readthedocs.io/) — commonly used for in-process LRU with single-flight.
