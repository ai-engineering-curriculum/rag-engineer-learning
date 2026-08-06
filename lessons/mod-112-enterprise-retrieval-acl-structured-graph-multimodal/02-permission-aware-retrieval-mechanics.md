# Chapter 2 — Permission-Aware Retrieval: Query-Time, Index-Time, and Per-Tenant Isolation

## Why this chapter exists

Every retriever from modules 101–111 answered the question "given a query, return the top-k most relevant chunks." An enterprise retriever answers a different one: "given a query **and a caller with a specific set of permissions**, return the top-k most relevant chunks *that the caller is allowed to see*." Getting that second clause wrong is the single failure mode most likely to end a RAG project — a cross-tenant leak or a "user X saw the CEO's compensation memo" incident.

There are three places to enforce the permission clause, each with a distinct correctness / cost / freshness trade-off. This chapter names them, walks the failure modes of each, and gives a decision matrix for picking one (or, more often, layering two).

The three enforcement points:

1. **Query-time filtering** — the caller's ACL is a filter passed alongside the query. The full index holds everything; the retriever restricts results per request.
2. **Index-time filtering (personalized / partitioned indexes)** — the ACL is baked into the index at build time. Either indexes are partitioned by the ACL dimension (per-group, per-project), or every chunk carries denormalized ACL metadata that the retriever exact-matches.
3. **Per-tenant index isolation** — each tenant gets a physically separate index. No cross-tenant query is possible because there is no shared surface to leak through.

These are not mutually exclusive — production stacks routinely use per-tenant isolation as the coarse boundary and query-time filtering inside each tenant for finer-grained document-level ACLs. But they *are* distinct mechanisms with distinct properties, and confusing them is how leaks happen.

## Enforcement point 1 — query-time filtering

The caller's ACL travels with the query. The retriever appends it as a filter clause to whatever backend it is talking to.

Concretely, in vector-DB terms:

```python
# Qdrant example
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

def search_with_acl(client, query_vec, user_groups, tenant_id, top_k=10):
    acl_filter = Filter(
        must=[
            FieldCondition(key="tenant_id", match=MatchAny(any=[tenant_id])),
            FieldCondition(key="allowed_groups", match=MatchAny(any=user_groups)),
        ]
    )
    return client.search(
        collection_name="chunks",
        query_vector=query_vec,
        query_filter=acl_filter,
        limit=top_k,
    )
```

Every major vector DB has the equivalent primitive: [Pinecone metadata filter](https://docs.pinecone.io/guides/data/filter-with-metadata), [Weaviate `where` filter](https://weaviate.io/developers/weaviate/api/graphql/filters), [Qdrant `Filter`](https://qdrant.tech/documentation/concepts/filtering/), [Milvus expression filter](https://milvus.io/docs/boolean.md), [pgvector `WHERE`](https://github.com/pgvector/pgvector#querying) on adjacent columns, [Elasticsearch / OpenSearch bool filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) around a `knn` query.

**Correctness trap 1 — pre-filter vs. post-filter.** mod-104 chapter 7 covered this in ANN-tuning terms; here it is a *security* issue. If the backend applies the filter *after* returning top-k (post-filter), and the caller has few permissions but the query is highly-similar to a lot of chunks they *cannot* see, the returned set can be small — or empty — because most of the top-k were filtered out. A user experience bug ("no results for X, but I know it's in the corpus") is the symptom; a *silent* logic bug in the filter applies zero constraints, and the failure mode is a leak, not an empty page. Always confirm your engine is applying the filter *before* the ANN traversal (pre-filter) or that it iteratively expands `ef` until top-k post-filter results exist (some engines call this "adaptive" or "filter-aware search").

**Correctness trap 2 — the filter is *inclusive* of the caller's ACL, not the doc's ACL.** The common mistake: chunks are tagged with a single `owner_group` and the filter is `owner_group in user.groups`. This is wrong for any real ACL — a doc is often shared with multiple groups *and* specific users *and* inherits from a parent that has its own ACL. The chunk needs the full *effective ACL* denormalized onto it (chapter 3 walks the SharePoint / Drive / Salesforce shape of this), and the filter checks that the intersection is non-empty. Do not conflate "the group that owns the doc" with "the groups permitted to read the doc."

**Correctness trap 3 — the caller's ACL is stale.** The retriever knows `user.groups = ["engineering", "project-A"]` because that is what the caller passed. But the caller was removed from `project-A` five minutes ago and the JWT / session cookie still says they are in it. The retriever must either fetch the caller's current groups fresh (adds a round-trip, but is the honest answer) or trust a token with a short TTL (minutes, not hours) and accept a bounded window of staleness. Long-lived tokens with hardcoded group claims are a leak by construction.

**Freshness properties.** Query-time filtering is *maximally fresh with respect to permissions*: the filter is evaluated at query time, so a permission grant that lands in your metadata store 500ms before the query takes effect immediately. This is the killer feature — enterprise permissioning changes constantly, and rebuilding the index every time a doc is re-shared is not viable.

**Cost properties.** Query-time filtering pays at *query* time. The filter can shrink the searchable space to a tiny fraction of the index, which the ANN traversal must navigate around. On HNSW indexes with a very restrictive filter, this can drop recall dramatically or force the engine into a slow fallback (post-filter with `ef` inflation, or a brute-force fallback below some density threshold). Test your specific engine on your specific filter distribution before you assume "the filter is free."

**When query-time is the right choice.** Fine-grained per-doc ACLs. Frequently-changing permissions. Filters that are dense enough that pre-filter is not degenerate (a user has thousands of permitted docs, not five). Multi-tenant SaaS where tenant boundaries are enforced elsewhere and query-time is for within-tenant per-doc ACLs.

## Enforcement point 2 — index-time filtering (partitioned or ACL-denormalized indexes)

The ACL is baked into the index at build time. Two shapes:

**Shape A — partitioning.** The index is physically partitioned along the ACL dimension. One partition per group, one per project, one per tenant. A query for a caller in groups `[A, B, C]` fans out to partitions `A`, `B`, `C`, merges the results (RRF or weighted-sum from mod-106), and returns top-k. There is no cross-partition surface — a group `D` partition simply is not queried.

**Shape B — ACL-denormalized documents.** The index is a single collection, but each chunk is *duplicated* for each group / user with access, or an equivalent per-permission-set copy exists. A caller queries only the partition matching their exact effective ACL set (or a small set of overlapping sets). This is the extreme end — it works when the number of distinct ACL sets is small (a handful of "sensitivity classes" — public / internal / confidential / restricted).

**Correctness trap 1 — the denormalized ACL goes stale.** The doc's ACL changed; the index still reflects the old ACL. Until the next rebuild, the retriever serves stale-permission results — sometimes over-restrictive (annoying), sometimes over-permissive (a leak). Freshness is bounded by the rebuild cadence.

**Correctness trap 2 — the merge across partitions is wrong.** If a user has access to 12 groups and you fan-out to 12 partitions asking for top-10 each, then merge to top-10 with RRF, the merge is correct but the result set is 120 candidates to rerank — that changes your latency budget. Reranking (mod-107) is where this cost surfaces.

**Correctness trap 3 — a chunk's ACL is many-to-many with the partition scheme.** SharePoint's model puts a doc in a *library* that has an ACL, but the doc can also be individually re-shared or inherit-broken to have its own ACL. A partition scheme keyed on library will over-return (docs the library ACL allows but the doc's own ACL forbids) or under-return (docs re-shared to the caller that the library ACL doesn't cover). Denormalization must reflect the *effective* ACL, not the parent's.

**Freshness properties.** Index-time filtering is *as fresh as the last rebuild that touched the affected chunks*. If you are running nightly rebuilds, permissions drift for up to 24 hours. Incremental rebuilds (mod-111 chapter 4) that re-index a chunk on ACL change get you to minutes; near-real-time re-indexing gets you to seconds — but at query-time-filter cost with none of query-time-filter's flexibility.

**Cost properties.** Index-time filtering pays at *build* time. Duplication (shape B) can inflate index size by 10× or more if ACL sets are diverse. Partitioning (shape A) has near-zero storage overhead but pays at query time with the fan-out-and-merge cost.

**When index-time is the right choice.** Coarse-grained ACL boundaries that map cleanly to partitions (a small number of long-lived groups; a fixed set of sensitivity classes). Permissions that change rarely (a legal-hold class, an executive-only class). Query-time filter cost is too high on your engine or your filter distribution.

## Enforcement point 3 — per-tenant index isolation

Each tenant gets a physically separate index. Tenant A's queries never touch tenant B's storage. This is the *strongest* boundary and the *only* one that survives a bug in the query-time filter or the ACL denormalization — because there is no shared collection to leak from.

Concretely:

- One Qdrant collection per tenant. One Elasticsearch index (or index alias) per tenant. One Postgres schema per tenant, with the pgvector table in that schema.
- Retriever service reads the caller's `tenant_id`, resolves to `collection_name = f"chunks_tenant_{tenant_id}"`, and every request goes to exactly that collection.
- Within a tenant, apply query-time filtering or index-time filtering for the fine-grained per-doc ACL.

**Correctness trap 1 — the routing layer resolves the wrong tenant.** A bug in the routing layer sends tenant A's query to tenant B's collection. Now every result is a leak. The mitigation: every chunk carries a `tenant_id` field, and the retriever asserts every returned chunk's `tenant_id == request.tenant_id` before responding. This is a *defense in depth* pattern — the collection separation is the primary defense; the assertion is a check the primary defense didn't misfire.

**Correctness trap 2 — a cross-tenant global index for search coalescing.** Someone (well-meaning) builds a "search across all tenants for internal admin support tickets" collection. The moment a user request hits that collection, the tenant boundary is gone. If you must have such a thing, run it in a *different service* with *different auth* — do not colocate.

**Correctness trap 3 — schema evolutions that require rebuilding many tenants at once.** A new field, a new embedding model, a new chunker — you cannot rebuild 5,000 tenants simultaneously. Cross-tenant orchestration under partial-completion has to be planned. mod-111's dual-index cutover extends per-tenant.

**Freshness properties.** Same as within-tenant enforcement — depends on whether you layer query-time or index-time filtering inside each tenant.

**Cost properties.** Storage cost is *n_tenants × overhead* — each collection has a fixed per-collection cost (metadata, index structures) on top of vector storage. For engines with a hard cap on collection count (some managed services), per-tenant does not scale to 100 K+ tenants without workarounds (multi-tenant-per-collection with a query-time filter — which is what you were trying to avoid). This is where the boundary to `ai-infra-senior-engineer` (level 40) kicks in — platform-scale per-tenant isolation is a distinct discipline.

**When per-tenant isolation is the right choice.** Multi-tenant SaaS where tenants are *legally* separate (GDPR data-residency, HIPAA, SOC 2 with contractual isolation). The blast radius of a routing bug is smaller than the blast radius of a query-time-filter bug. Small tenant counts (dozens to low thousands) where the per-collection overhead is affordable.

## The decision matrix

| Concern | Query-time filter | Index-time filter | Per-tenant isolation |
|---|---|---|---|
| Freshness on permission change | Immediate | Bounded by rebuild cadence | Immediate for tenancy; within-tenant depends on within-tenant strategy |
| Storage overhead | None | Duplication factor (shape B) or none (shape A) | n_tenants × per-collection overhead |
| Query latency | Filter cost (can be high on restrictive filters) | Partition fan-out + merge (shape A); low (shape B) | Very low (small collection) |
| Blast radius of implementation bug | Cross-partition leak in one collection | Cross-partition leak in one collection | Bug in routing layer only |
| Fine-grained per-doc ACLs | Native | Only if denormalized effectively | Combine with query-time within tenant |
| Coarse-grained boundaries (tenant, org unit, sensitivity class) | Works but wastes the strong-boundary property | Native fit | Native fit |
| Suits highly-dynamic permissioning | Yes | No (needs frequent rebuild) | Yes for tenant; within depends |
| Scales to 100 K+ boundaries | Yes | Not shape B | Not without an ai-infra-senior-engineer's help |

**Common production shape:** per-tenant isolation as the outer boundary (a bug in the middle layers cannot leak across tenants) + query-time filtering *within* a tenant for per-doc ACLs (matches the freshness need of dynamic permissioning). This gives you the leak-blast-radius of per-tenant with the freshness of query-time.

## Denormalizing effective ACLs onto chunks

Whichever enforcement point you pick (except the coarsest per-tenant-only), you need each chunk to carry an *effective* ACL — the set of principals that can read it *right now*, after all group memberships and inheritance chains are resolved.

The naive shape:

```json
{
  "chunk_id": "abc123",
  "doc_id": "doc-42",
  "text": "...",
  "vector": [...],
  "acl": {
    "allowed_groups": ["engineering", "project-alpha"],
    "allowed_users": ["u_josh"],
    "denied_groups": [],
    "denied_users": []
  },
  "tenant_id": "acme"
}
```

This works when ACLs are simple (no negative permissions, no role hierarchy, no time-bound access, no dynamic rules). Real enterprise ACLs are usually more complex — chapter 3's SharePoint / Drive / Salesforce sections show why. When you cannot faithfully denormalize (e.g., Salesforce's row-visibility depends on the caller's *current* role hierarchy, which changes independently of the doc), the honest answer is: *do not denormalize the ACL onto the chunk; look it up at query time against the source system.* That is expensive but correct. Optimize later (chapter 3 has caching strategies).

**The check evaluation.** Given a request from user `U`, and a chunk with effective ACL `A`, the retriever must answer: is user `U` in the effective read-set of `A`? A common shape:

```python
def user_can_read(user, chunk_acl):
    user_principals = {user.id} | set(user.groups) | set(user.roles)
    if user_principals & set(chunk_acl.get("denied_users", [])):
        return False
    if user_principals & set(chunk_acl.get("denied_groups", [])):
        return False
    allowed = set(chunk_acl.get("allowed_users", [])) | set(chunk_acl.get("allowed_groups", []))
    return bool(user_principals & allowed)
```

For query-time filtering, you translate this into a backend-specific filter expression (Qdrant `Filter`, Elasticsearch `bool` query, pgvector `WHERE`). For post-retrieval verification (belt-and-braces), you re-run this check in Python before returning results.

## Belt-and-braces — always run the check twice

The strongest permission-aware retrieval stacks apply the ACL twice:

1. **Inside the retriever** — the backend filter (pre-filter at ANN time). This keeps the recall high and the latency low.
2. **After retrieval, before returning** — a Python-side check that re-verifies every returned chunk against the caller's ACL, using the *same* effective ACL fields the filter used.

The second pass is O(top_k) and costs microseconds. It catches:

- Bugs in the backend filter expression (a typo in a field name).
- Denormalization staleness — the chunk was indexed with an old ACL that includes the caller, but the source-system check (fetched fresh) says the caller was recently removed.
- Post-filter bugs where the backend silently returned pre-filter results.
- Cross-tenant routing bugs where the `tenant_id` doesn't match the caller.

The Python check is the last line of defense before the LLM sees the chunk. Fail closed — if the check throws or returns anything ambiguous, drop the chunk. Log the drop; investigate the cause.

## An eval loop for permission-aware retrieval

The recall / nDCG numbers from mod-109 do not catch a permission leak. You need a permission-specific eval that is orthogonal to relevance:

- **Adversarial ACL fuzz test.** Build a synthetic corpus with clear ACL groupings (docs tagged with `sensitivity in {public, internal, confidential}`). Author user profiles at each sensitivity level. For every user × doc pair, retrieve with the user's ACL and assert that every returned chunk is in the user's allowed set. The metric is **leak rate** — number of returned chunks with `acl.sensitivity > user.clearance`, divided by total returned chunks. Target: exactly zero.
- **Consistency across enforcement points.** Retrieve the same query with query-time filter, with index-time filter, and (if you have it) with per-tenant isolation. Assert the results are identical (up to relevance-ordering ties). Divergence means one of the enforcement points is buggy.
- **Freshness lag test.** Revoke a user's access to a doc at time `t`; poll the retriever every second; measure the wall-clock delay until the doc stops appearing in that user's results. Compare against your SLO.
- **Staleness detection.** Run the ACL fuzz test against a *stale* denormalized index (permissions changed but index not rebuilt) and confirm your Python-side post-check catches the newly-forbidden docs.

Wire these into the mod-111 retriever CI as a *gate* — a PR that regresses leak rate from zero to nonzero cannot merge. This is the retrieval-side equivalent of a "does not break the ACL contract" invariant test.

## What you should be able to do after this chapter

- Enumerate the three enforcement points (query-time filter, index-time filter, per-tenant isolation) and name the freshness / cost / blast-radius trade-off of each.
- Explain the pre-filter vs. post-filter distinction as a *security* issue, not just a performance one, and confirm which mode your vector DB uses under a restrictive filter.
- Name the effective-ACL denormalization shape and why "owner group" is wrong for real ACLs.
- Argue why the common production shape is per-tenant + query-time inside, and name the two properties that combination provides.
- Author a post-retrieval Python ACL check that is a defense-in-depth backstop against the primary filter.
- Design the four-part permission-aware eval loop (fuzz test, cross-enforcement consistency, freshness lag, staleness detection) and describe how it plugs into the mod-111 retriever CI.

## Further reading

- Qdrant docs: [*Filtering*](https://qdrant.tech/documentation/concepts/filtering/) and [*Payload and filtering*](https://qdrant.tech/documentation/concepts/payload/) — the reference for pre-filter behavior on HNSW.
- Elasticsearch docs: [*Filter search results*](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html) and [*kNN search*](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html) — Elastic's document filter semantics interact with `knn` in ways that are worth reading carefully before you assume "pre-filter."
- Pinecone: [*Filter with metadata*](https://docs.pinecone.io/guides/data/filter-with-metadata) — Pinecone's filter semantics and the caveats around highly-selective filters.
- Weaviate: [*Multi-tenancy*](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy) — Weaviate's first-class multi-tenancy feature is a good reference for per-tenant isolation done natively.
- OpenSearch: [*k-NN plugin — filtering*](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/) — OpenSearch's variant, notable for its "exact vs. approximate" fallback under restrictive filters.
- OWASP: [*Broken Access Control*](https://owasp.org/Top10/A01_2021-Broken_Access_Control/) — the general-web version of the failure this chapter's checks defend against; the recommended controls (deny by default, enforce ACLs on the server) apply to retrievers verbatim.
- Sanderson, M., & Croft, W. B. (2012). *The History of Information Retrieval Research.* Proceedings of the IEEE — the IR community has thought about "authorized results" since the enterprise-search era; the concept is not new even if the vector-DB implementation is.
