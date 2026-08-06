# Exercise 01 — Permission-Aware Retrieval: Query-Time vs. Index-Time Drill

**Estimated effort:** ~3 hours. Python, a vector DB you can run locally (Qdrant / Weaviate / pgvector), a small synthetic corpus of ~200–500 chunks with ACL metadata, a handful of synthetic user profiles at multiple clearance levels.

## Objective

Implement permission-aware retrieval in three enforcement configurations — query-time filter, index-time filter (partitioned), and per-tenant isolation — and compare their correctness, freshness, and cost properties on the same synthetic corpus. Add a belt-and-braces Python-side ACL post-check and prove it catches at least one bug you deliberately plant. Wire an adversarial fuzz test that measures leak rate, and treat leak-rate = 0 as the CI gate.

By the end you have:

- `corpus/` — a synthetic corpus with per-chunk ACL metadata (`tenant_id`, `sensitivity`, `allowed_groups`, `allowed_users`).
- Three retriever variants — `retriever/query_time.py`, `retriever/index_time.py`, `retriever/per_tenant.py`.
- `retriever/post_check.py` — the Python ACL post-check.
- `evals/fuzz.py` — the adversarial ACL fuzz test that computes leak rate per user × per retriever variant.
- A short write-up (`RESULTS.md`) comparing the three on leak rate, latency, storage overhead, and freshness on a simulated permission-revocation event.

## Prerequisites

- Chapters 2 and 3 read.
- mod-104 chapter 7 (metadata filtering) as background — the "pre-filter vs. post-filter" distinction matters here.
- Docker (or a local install) for Qdrant / Weaviate; or Postgres 15+ with the `vector` extension for pgvector.
- Python 3.10+; `pip install qdrant-client` (or the client for your engine) and `sentence-transformers`.

## Steps

### Step 1 — Build the synthetic corpus (30 minutes)

Author `corpus/build.py` that generates ~200–500 chunks with structured ACL metadata. Suggested schema:

```json
{
  "chunk_id": "chunk-000123",
  "doc_id": "doc-42",
  "text": "Confidential update on the Q3 project alpha budget: ...",
  "tenant_id": "tenant_a",
  "sensitivity": "confidential",
  "allowed_groups": ["grp_finance", "grp_project_alpha"],
  "allowed_users": ["u_ceo", "u_cfo"]
}
```

- Two tenants: `tenant_a`, `tenant_b`. ~50–60% of chunks in each.
- Three sensitivity levels: `public`, `internal`, `confidential`.
- 4–6 groups per tenant; users are members of 1–3 groups each.
- Generate 5–10 chunks per doc; the text can be short lorem-ipsum plus a topic label so vector search actually clusters.

Also generate ~8–12 user profiles across both tenants at all three clearance levels.

Save the corpus + users to JSONL so every subsequent step reads from a stable input.

### Step 2 — Implement query-time filtering (35 minutes)

`retriever/query_time.py`: single collection holding all chunks (both tenants, all sensitivities). At query time, translate the caller's policy context into a backend filter expression.

- Embed chunks with a small text embedder (`all-MiniLM-L6-v2` is fine).
- Upload to your vector DB in one collection.
- On `retrieve(query, policy_ctx)`:
  - Build a filter that intersects `tenant_id == policy_ctx.tenant_id` AND `(allowed_groups ∩ policy_ctx.principals) is non-empty OR allowed_users ∩ policy_ctx.principals is non-empty`.
  - Confirm the filter is applied *pre-filter* on your engine (Qdrant is; check the docs for the others).
  - Return top-10.

Verify by hand-issuing three queries as three different users; confirm each user gets results whose ACL includes them.

### Step 3 — Implement index-time filtering with partitioning (30 minutes)

`retriever/index_time.py`: one collection per `(tenant, sensitivity)` combination. Six collections total (2 tenants × 3 sensitivity levels).

- On ingest, assign each chunk to the collection matching its `(tenant_id, sensitivity)`.
- On `retrieve(query, policy_ctx)`:
  - Compute which collections the caller can read (all `tenant == policy_ctx.tenant_id` AND `sensitivity <= policy_ctx.clearance`).
  - Fan out queries to each allowed collection, take top-10 per collection, RRF-merge to top-10 overall.

Note: within a partition you still need a *finer* filter for `allowed_groups`/`allowed_users` — either do that as a query-time filter within each partition, or accept that this partitioning scheme is coarser than what chapter 2's shape-B denormalization would give you. Document the choice in `RESULTS.md`.

### Step 4 — Implement per-tenant isolation (25 minutes)

`retriever/per_tenant.py`: one collection per tenant, with query-time filtering *within* the tenant for per-doc ACLs (matches chapter 2's recommended production shape).

- On `retrieve(query, policy_ctx)`:
  - Resolve `collection_name = f"chunks_tenant_{policy_ctx.tenant_id}"`.
  - Issue the query against exactly that collection with the group/user filter from step 2.
  - Assert (in code) that every returned chunk has `tenant_id == policy_ctx.tenant_id`; raise if any doesn't.

### Step 5 — The Python-side ACL post-check (15 minutes)

`retriever/post_check.py`:

```python
def user_can_read(chunk_acl, policy_ctx):
    principals = set(policy_ctx["principals"])
    if principals & set(chunk_acl.get("denied_users", [])):
        return False
    if principals & set(chunk_acl.get("denied_groups", [])):
        return False
    allowed = set(chunk_acl.get("allowed_users", [])) | set(chunk_acl.get("allowed_groups", []))
    return bool(principals & allowed)
```

Wrap each of the three retrievers so the post-check runs after retrieval; drop chunks that fail; log the drop with the chunk_id, the caller, and which retriever it came from.

### Step 6 — Plant a bug and prove the post-check catches it (15 minutes)

Introduce a deliberate bug in *one* of the retriever backends' filter expression — e.g., in the query-time retriever, drop the `allowed_users` half of the OR so the filter only checks `allowed_groups`. This will cause user-scoped shares to leak.

Add a fuzz-test case that reveals it: a chunk whose only allowed principal is a specific user, queried by another user. The primary filter returns it (bug); the post-check drops it (belt-and-braces works).

Fix the bug afterward; keep the fuzz-test case as a regression guard.

### Step 7 — Build the adversarial fuzz test (25 minutes)

`evals/fuzz.py`:

- For each user in your profile set, run 20–50 queries (pick from a set of realistic queries that would surface both allowed and disallowed content).
- For each `(user, retriever_variant)` pair, compute `leak_rate = disallowed_chunks_returned / total_chunks_returned`.
- Also compute `over_restriction_rate = allowed_chunks_dropped_by_post_check` (as a sanity check — should be exactly 0 if the primary filter is correct).
- Publish a table: rows = retriever variant, columns = leak rate, over-restriction rate, avg p50 latency, storage overhead.

### Step 8 — Simulate a permission revocation and measure freshness (15 minutes)

- Pick a user with access to a group.
- Note the current returned set for a query that surfaces group-scoped chunks.
- Revoke the user's group membership in the ACL store.
- **Query-time retriever:** should reflect the revocation on the very next request. Measure the elapsed time (near-zero).
- **Index-time retriever (partitioning):** the change might not require reindex if the partition is on sensitivity; if on group, revocation requires re-embedding all affected chunks. Simulate whichever your partition scheme requires and measure the "rebuild lag."
- **Per-tenant:** within-tenant behaves like query-time (immediate). Cross-tenant is not applicable (the user was already restricted to one tenant).

Document the freshness lag per variant in `RESULTS.md`.

### Step 9 — Write up `RESULTS.md` (15 minutes)

Publish the comparison:

- Leak rate per retriever variant (target: 0 after the deliberate bug is fixed).
- Latency (p50, p95) per variant.
- Storage overhead per variant (chunks × any duplication factor).
- Freshness on the revocation event.
- One paragraph per variant on when you'd pick it in production.

Reference chapter 2's decision matrix and note where your numbers agree or diverge.

## Acceptance criteria

- All three retriever variants implemented and runnable against the same corpus.
- Python post-check wraps each variant.
- Leak rate is exactly 0 for all three variants on the fuzz test *after* the deliberate bug is fixed.
- The deliberate bug in step 6 is caught by the post-check (documented in `RESULTS.md`).
- Freshness measurement shows the expected pattern — query-time near-instant, index-time bounded by rebuild.
- `RESULTS.md` publishes the comparison table and a recommendation paragraph per variant.

## Stretch goals

- **Add the per-tenant + query-time hybrid** as a fourth variant (chapter 2's recommended production shape). Show it inherits the best of both.
- **Implement group-membership caching** with a short TTL and measure the latency delta.
- **Fuzz an adversarial query.** Author queries deliberately worded to surface disallowed content (semantic similarity attack); verify no leak.
- **Replay ACL changes from a delta stream.** Simulate SharePoint / Drive delta-query events; measure end-to-end delay from event → retriever-side effect.
- **Add a second engine.** Repeat the drill on pgvector and Qdrant; note the differences in filter behavior (pre-filter vs. adaptive vs. post-filter).

## Notes

- **Do not skip the post-check.** The whole point of belt-and-braces is that the primary filter is not the only defense. Wiring it in from step 5 makes step 6's bug-plant meaningful.
- **The fuzz test is the CI gate.** In production, this is what refuses to let a PR that regresses leak rate merge. Wire it as a pytest that fails on leak-rate > 0 so you can run it locally with `pytest evals/`.
- **Prefer real ACL shapes over toy ones.** If you have access to your org's ACL data (anonymized), use it instead of fully-synthetic — the failure modes on real data are the ones you actually need to catch.
- **Log the effective ACL on every response.** Chapter 2's "log the effective ACL" recommendation is worth making a habit; add it now and it'll be there when you actually need it.
