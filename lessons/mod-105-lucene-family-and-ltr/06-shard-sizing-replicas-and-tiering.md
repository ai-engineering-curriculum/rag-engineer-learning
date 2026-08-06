# Chapter 6 — Shard Sizing, Replica Placement, and Hot-Warm-Cold Tiering

## Why this chapter exists

A retrieval engineer who authors a beautiful analyzer, a defensible mapping, a strong BM25 baseline, a well-tuned dense-vector field, and a solid LTR model — and then deploys it on a badly-sharded cluster — has done all the work for a system that will page someone at 3 a.m. Sharding is the operational shape of a Lucene-family cluster; it is where indexing throughput, query latency, memory pressure, refresh cost, and rebalancing pain all live. Get it wrong and no amount of retrieval-quality work rescues you.

Elasticsearch, OpenSearch, and Solr all use the same sharding model (a logical index is split into `N` primary shards, each replicated `R` times across the cluster). Vespa uses a related but distinct model (content clusters with groups of nodes). This chapter walks the shared model, gives the heuristics that Elastic's own docs publish, and shows how hot-warm-cold tiering shapes a real cluster's storage and query cost.

This is a level-30 chapter. Everything below fits inside a single-region deployment on a well-known engine. Cross-DC replication, GPU-accelerated ANN capacity planning, and platform-scale HA are chapter 7's boundary; those belong to `ai-infra-senior-engineer` (level 40) and `staff-ml-engineer` (level 50).

## The shard, one paragraph

A **primary shard** is a self-contained Lucene index — its own segments, its own inverted index, its own doc-values, its own HNSW graph if you have a `dense_vector` field. A logical Elasticsearch index with `N` primary shards means the corpus is partitioned across `N` Lucene indexes, and any query fans out to all `N` shards, gets partial top-K from each, and merges. **Replicas** are copies of primary shards on other nodes: they give you redundancy (a node loss doesn't kill a shard) and read scaling (search requests can hit any copy). Shards are the unit of parallelism, the unit of replication, the unit of recovery, and — after the index is created — mostly immutable in count.

That last clause is the one that catches teams: **the primary shard count is fixed at index creation**. You can add nodes, add replicas, split (limited), and shrink (limited) — but you cannot reshard a live index cheaply. Get the count wrong on day one and you carry the wrong count for the life of the index.

## The 20–40 GB shard size heuristic

Elastic's [shard-size guidance](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html) — repeated in nearly every capacity-planning blog post they've published — is:

> Aim for shards **between 10 GB and 50 GB** in size. For search-heavy workloads, aim for the **20–25 GB** end; for log-heavy time-series workloads (where old shards are rarely queried), 40–50 GB is fine.

The upper bound comes from **shard-recovery time**. When a node goes down, its shards' replicas on other nodes are promoted to primary, and new replicas are built by copying from the surviving primaries. A 200-GB shard takes 5–10× longer to copy than a 20-GB one; during that window your cluster is one node loss away from data loss. Above ~50 GB, recovery windows start dominating your MTTR.

The lower bound comes from **per-shard overhead**. Each shard is a separate Lucene index with its own heap allocations, its own file handles, its own segment merges, its own refresh-worker threads. Ten thousand 1-GB shards eat far more cluster resources than a hundred 100-GB shards. Elastic's own [sizing docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html) call out ~20 shards per GB of JVM heap as a rough ceiling for per-node shard count.

The rule of thumb: **plan primary shard count so that each shard sits in 20–40 GB when the index reaches its steady-state size**. Not its day-one size — its steady state after the corpus has grown for a year.

### The sizing math

Given: expected steady-state corpus size `S` (in GB, including analyzer expansion, doc_values, HNSW graph, and _source).

Pick primary shard count: `N_primary ≈ S / 30` (targeting ~30 GB per shard, mid-range).

Pick replica count: `N_replica = 1` for most single-region deployments (one replica = one node loss survivable); `N_replica = 2` for critical workloads or read-heavy scaling; `N_replica = 0` for cost-optimised non-critical (do not do this for anything that matters).

Total shard count across the cluster: `N_total = N_primary × (1 + N_replica)`.

Node count: `N_nodes ≥ N_total / max_shards_per_node`, where `max_shards_per_node` is bounded by your heap size (~20 shards/GB heap, from above). And you want the primary shards to be spread evenly across nodes, so `N_nodes ≥ N_primary` if you want at most one primary per node for the biggest indexes.

Two concrete cases:

- **Support-article corpus, 15 GB after one year.** Primary shards: 1 (the whole thing fits in one 15-GB shard, well below the 20-GB lower bound). Replicas: 1. Deployment: 2 nodes minimum, 3 for HA (allowing one loss without a rebuild).
- **Log-style time-series corpus, 500 GB per day, retained 30 days = 15 TB steady state.** Primary shards per daily index: 500 / 40 = 13 (rounded up). Replicas: 1. Daily total: 26 shards. Retention window: 30 days × 26 = 780 shards. Node count driven by disk (15 TB × 2 replicas = 30 TB) and heap (780 shards × 2 = 1560 shards total; at 20 shards/GB heap you need ~78 GB of aggregate heap, likely 4–8 nodes with 16–32 GB heap each).

### When you break the heuristic

- **Very small indexes.** If your entire index is < 5 GB, one shard with one replica is the right answer even though it violates the 20-GB lower bound. Don't force splits.
- **Very large individual documents.** If each doc is ~500 KB (long articles with dense-vector fields), you hit the 40-GB shard cap faster than doc-count intuition suggests. Recount.
- **Update-heavy indexes.** High update rates cause segment merging cost that scales with shard size. Smaller shards (~15 GB) reduce merge cost per update.
- **Read-heavy time-series.** If old shards are still queried, they behave like search shards, not log shards; keep them in the 20-GB range.

## Primary shard count is (almost) immutable

Once created, you cannot arbitrarily change an index's primary shard count. The three escape hatches:

- **[Shrink API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-shrink-index.html)** — reduce shard count to a divisor of the current count (10 → 5, 6 → 3, 6 → 2, 6 → 1). Requires the source index to be read-only, all shards on one node, health green. Cheap-ish; still a maintenance window.
- **[Split API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-split-index.html)** — increase shard count to a multiple of the current count (must have been created with `index.number_of_routing_shards` at creation, which is the default for indexes created ≥ 7.0). Requires source read-only. Also maintenance-window work.
- **Reindex to a new index with the desired shard count.** Full cutover; the pattern from [mod-104 chapter 8](../mod-104-vector-databases-and-ann-tuning/08-index-versioning-and-dual-index-cutover.md) with dual-index cutover. Always available; more expensive but no per-shard-count constraints.

The rule of thumb: **the reindex-and-cutover pattern is what you use for real shard-count changes**. Shrink and split are for narrow cases (post-hot-phase shrinking of daily log indexes is a common shrink use).

## Replicas: what one, two, or zero actually mean

- **`replicas: 0`.** No redundancy. A single node loss loses shards permanently until you re-ingest. Only acceptable for indexes you can rebuild in minutes from a source-of-truth. Never acceptable for a production retrieval index.
- **`replicas: 1`.** The default. One replica of each primary. One node loss is survivable; two simultaneous losses risk data loss. Read throughput is 2× single-replica for search-heavy workloads.
- **`replicas: 2` or more.** More read throughput, more redundancy, more disk. The right pick when you have a large read fleet and disk is cheap.

Replica placement is handled by the cluster's [shard allocation awareness](https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-allocation-awareness.html): you tag nodes with attributes (`zone: a`, `zone: b`, `zone: c`) and configure allocation to spread primary + replica pairs across zones. **Never put a primary and its replica on the same rack, zone, or host**. Getting this wrong is a common cause of "we had three replicas and still lost data" incidents.

## Hot-warm-cold: the tiering shape

Log and time-series workloads generate massive amounts of data whose query rate collapses with age. A one-day-old error log is queried a hundred times an hour; the same log at day-30 is queried maybe once a week. Serving both at the same latency and cost is silly. **Hot-warm-cold tiering** ([Elastic's data-tier docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html)) matches storage class to query rate.

### The three tiers

- **Hot tier.** Ingest happens here. SSD-backed, high CPU, generous RAM. Recent, frequently-queried data. `p50` latency in tens of milliseconds; `p99` in low hundreds. Typical retention: hours to days.
- **Warm tier.** Read-only after ingest. HDD or slower SSD, less RAM per GB of data. Recent-but-not-hot data. Latency an order of magnitude higher than hot. Typical retention: days to weeks.
- **Cold tier.** Read-only, high-density storage (large HDDs, sometimes searchable snapshots on object storage). Queried rarely; when queried, latency in seconds is fine. Typical retention: weeks to months.
- **Frozen tier** (Elastic-specific extension). Data lives entirely on object storage (S3, GCS); queried by mounting on demand via [searchable snapshots](https://www.elastic.co/guide/en/elasticsearch/reference/current/searchable-snapshots.html). Cheapest per GB by far; slowest per query. Typical retention: months to years for regulatory hold.

### How tiering shapes the cluster

Each node gets a `data_hot`, `data_warm`, `data_cold`, or `data_frozen` role. **Index lifecycle management (ILM)** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html)) is the automation that moves an index (or a shard of one) from tier to tier as it ages:

```json
PUT _ilm/policy/support_articles_policy
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": { "max_size": "50GB", "max_age": "7d" }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": { "number_of_shards": 1 },
          "forcemerge": { "max_num_segments": 1 },
          "allocate": { "require": { "data": "warm" } }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "allocate": { "require": { "data": "cold" } }
        }
      },
      "delete": {
        "min_age": "365d",
        "actions": { "delete": {} }
      }
    }
  }
}
```

Points worth naming:

- **Rollover on hot** — a new backing index is created when the current one hits 50 GB or 7 days, whichever first. This is how time-series indexes stay bounded.
- **Shrink + force-merge on warm** — once the index is read-only, collapse to one shard and force-merge to one segment. Recovers a lot of per-shard and per-segment overhead. Post-shrink, the shard is denser but no longer under active update.
- **Allocate on warm/cold** — moves the shard to nodes with the corresponding `data_*` tier attribute. This is where storage class actually changes.
- **Delete after 365 days** — retention enforcement. Match your compliance window.

OpenSearch has an [equivalent Index State Management (ISM) plugin](https://opensearch.org/docs/latest/im-plugin/ism/index/) with a similar shape but different DSL. Solr's [autoscaling / tiering story](https://solr.apache.org/guide/solr/latest/deployment-guide/solrcloud-autoscaling-overview.html) is thinner and less integrated. Vespa's [content cluster grouping](https://docs.vespa.ai/en/reference/services-content.html) does tiering differently — by group / hierarchy — and is out of scope here.

### The tiering trade-off matrix

| Tier | Latency (typical) | $/GB/month (typical) | Query rate | Update rate |
|---|---|---|---|---|
| Hot | 10–100 ms | high (SSD + RAM) | high | writable |
| Warm | 100 ms – 1 s | medium | medium | read-only |
| Cold | 1–5 s | low | low | read-only |
| Frozen | 5–30 s | very low (object storage) | very-low | read-only |

The costs and latencies are order-of-magnitude sketches — your cloud, instance types, and workload will move them. The order relationship is what stays fixed.

## Query latency implications of shard count

More shards = more parallelism = higher single-query throughput *only if you have enough CPU cores to service the fanout*. Two dynamics push in opposite directions:

- **More shards ⇒ smaller shards ⇒ more parallel scan.** Each shard's scan runs on its own thread; a query hitting 16 shards runs 16 partial scans concurrently. If your cluster has 16+ available cores, this is a latency win.
- **More shards ⇒ more merging overhead per query.** The coordinating node has to fetch top-K from each shard, merge, and re-rank. At 100+ shards per query, this merge cost dominates the scan cost.

The rule of thumb: **for interactive retrieval workloads, target 8–20 primary shards per query** — enough parallelism to amortise the scan across cores, not so much that the merge dominates. For time-series where a query may search 30 daily indexes, use [cross-cluster search](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cross-cluster-search.html) or `filter_path` optimisation to prune shards before scan (see [`_search` with `pre_filter_shard_size`](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-preference)).

## Refresh interval: the throughput/freshness knob

Every write to Elasticsearch/OpenSearch goes into an in-memory buffer and is invisible to search until a **refresh** flushes it to a searchable segment. The default refresh interval is 1 second — near-real-time, but expensive.

For high-throughput ingestion (log-style workloads), extend the refresh interval:

```json
PUT /support_articles/_settings
{
  "index": {
    "refresh_interval": "30s"
  }
}
```

Higher `refresh_interval`:

- Fewer segment creations per unit time → less merge overhead.
- More indexing throughput.
- Higher indexing-to-search latency (documents take up to `refresh_interval` to appear).

For search-heavy indexes where near-real-time is required (support articles, user content), keep `1s`. For log-style ingestion where a 30-second delay is acceptable, use `30s`. For pure batch loads, set `-1` (no auto-refresh) during load and refresh manually after.

## The three anti-patterns to avoid

1. **Over-sharding for imagined future scale.** "Let's use 50 shards for our 2-GB corpus in case we grow." No — 50 shards for 2 GB means 40 MB per shard. All the per-shard overhead, none of the parallelism benefit. Size for your one-year steady state, plus 2× headroom.
2. **Ignoring shard allocation awareness.** Every replica lives on a different node from its primary, but nothing prevents both from landing in the same rack, zone, or host by default. Tag nodes with `zone` attributes and configure `cluster.routing.allocation.awareness.attributes: zone`. Test node loss in staging.
3. **Refresh interval `1s` on log-style ingestion.** You are paying near-real-time latency cost for data that will be looked at once every 3 hours. Extend the interval, save the merge cost.

## What is NOT in this chapter (and where it goes)

- **Cross-DC replication (cross-cluster replication in ES, cross-cluster search).** Chapter 7.
- **GPU-accelerated ANN capacity planning.** Chapter 7; belongs to `ai-infra-senior-engineer`.
- **Platform-scale HA (active-active, quorum for cluster state).** Chapter 7.
- **Multi-tenant capacity planning at 10,000+ tenants.** Chapter 7 and mod-112.
- **The retriever CI that guards against sizing regressions.** mod-111.

## What you should be able to do after this chapter

- Recite the 20–40 GB primary-shard-size heuristic and defend a shard count for a given steady-state corpus size.
- Explain why primary shard count is immutable-in-practice and know the shrink / split / reindex escape hatches.
- Author an ILM policy that moves a log-style index through hot → warm → cold → delete on defensible age boundaries.
- Configure shard-allocation awareness so primary/replica pairs never share a failure domain.
- Recognise the trade-off between shard count and query-time merge cost.
- Choose an `refresh_interval` appropriate to the workload (search-heavy vs. log-heavy vs. batch).
- Identify the point at which sizing / tiering conversations cross into `ai-infra-senior-engineer` territory (chapter 7).

## Further reading

- Elastic: [*Size your shards*](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html) — the primary reference for the 20–40 GB heuristic.
- Elastic: [*Data tiers*](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html) — the hot/warm/cold/frozen model.
- Elastic: [*Index lifecycle management*](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html) — the ILM policy DSL.
- Elastic: [*Shard allocation awareness*](https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-allocation-awareness.html).
- Elastic: [*Searchable snapshots*](https://www.elastic.co/guide/en/elasticsearch/reference/current/searchable-snapshots.html) — the frozen-tier storage model.
- OpenSearch: [*Index State Management*](https://opensearch.org/docs/latest/im-plugin/ism/index/) — OpenSearch's ILM equivalent.
- OpenSearch: [*Cluster sizing*](https://opensearch.org/docs/latest/tuning-your-cluster/) — sizing guidance for OpenSearch clusters.
- Solr: [*Distributed search / sharding*](https://solr.apache.org/guide/solr/latest/deployment-guide/cluster-types.html) — the SolrCloud sharding docs.
- Vespa: [*Content clusters*](https://docs.vespa.ai/en/reference/services-content.html) — Vespa's alternative to shard/replica.
- Andrew Cholakian, *Exploring Elasticsearch* (2019, self-published) — the classic operator-focused introduction; specifically the chapter on shard sizing.
