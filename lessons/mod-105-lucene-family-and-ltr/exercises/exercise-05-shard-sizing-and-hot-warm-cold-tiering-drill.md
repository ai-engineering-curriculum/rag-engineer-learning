# Exercise 05 — Shard Sizing and Hot-Warm-Cold Tiering Drill

**Estimated effort:** ~2 hours. Mostly a design-and-writing exercise; a small hands-on ILM policy dry-run on a single-node cluster.

## Objective

Design defensible cluster shapes for three workloads that stress different corners of the shard-sizing and tiering guidance from chapter 6, then dry-run an ILM (or OpenSearch ISM) policy that would automate the hot → warm → cold → delete lifecycle for the time-series workload. The point is to produce artefacts (a sizing spreadsheet, an ILM policy JSON, a rollout checklist) that would survive a platform-team review — not to serve real traffic.

## Prerequisites

- Chapter 6 read.
- Elastic's [*Size your shards*](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html) and [*Data tiers*](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html) pages open in a browser.
- A running Elasticsearch 8.10+ or OpenSearch 2.10+ single-node cluster for the ILM/ISM dry-run in step 4. Docker single-node is fine.
- Spreadsheet, Markdown table, or Jupyter notebook — whichever you author capacity plans in.

## The three workloads

Design a cluster shape for each. Do not skip one; they are chosen to hit different corners of the heuristic.

### Workload A — Small support-articles corpus, near-real-time

- Corpus: ~50,000 support articles, growing ~1,000 articles/month.
- Per-doc size: ~10 KB text, plus a 768-dim `int8_hnsw` vector field.
- Query rate: ~50 QPS steady, ~500 QPS peak (post-incident traffic spike).
- Update rate: ~200 updates/day (editorial edits); near-real-time visibility desired.
- Retention: forever (the article set is the product's memory).
- Uptime target: 99.9% (~8 hours downtime/year allowed).
- Cloud posture: single AWS region, three AZs available.

### Workload B — Product catalogue for a mid-size marketplace

- Corpus: ~5 million products.
- Per-doc size: ~4 KB text + structured attributes + a 384-dim `int8_hnsw` vector field + a `keyword`-heavy tenant field (1,000 tenants).
- Query rate: ~2,000 QPS steady, ~10,000 QPS peak (Black-Friday-style).
- Update rate: ~50,000 updates/day (inventory + price sync); freshness within 30 seconds is acceptable.
- Retention: forever (until a product is delisted; ~10% churn per year).
- Uptime target: 99.95%.
- Cloud posture: single region, three AZs.

### Workload C — Application error-log ingest, time-series

- Corpus: ~500 GB/day ingested; retained 30 days hot-queryable, 90 days queryable-with-latency, 365 days regulatory hold on cheap storage.
- Per-doc size: ~2 KB JSON, no vectors.
- Query rate: ~5 QPS on last-24-hour data (dashboards + oncall greps); ~0.1 QPS on data older than 7 days.
- Update rate: append-only after ingest.
- Retention: 30 days hot, 60 days warm, 275 days cold, deleted at 365 days.
- Uptime target: 99.9% for ingest; queries can tolerate seconds-scale latency on cold data.
- Cloud posture: single region, three AZs.

## Steps

**Step 1 — Sizing math per workload (~30 min).** For each of the three workloads, compute and present:

- **Steady-state size in one year** (be explicit about the growth model — write it as "50K docs × 10 KB × ingest amplification factor ≈ X GB", not just "≈ X GB"). Include the vector field's contribution if the workload has one; use the sizing math from chapter 4's "Sizing dense-vector fields" section for int8_hnsw fields.
- **Primary shard count**, derived from the ~20–40 GB shard-size heuristic. Show the arithmetic. If you round to something other than `S / 30`, defend the round.
- **Replica count**, defended against the uptime target and the read-throughput requirement.
- **Node count minimum**, derived from `N_primary`, `1 + N_replica`, and a per-node shard cap of `20 × heap_GB` (assume 32 GB heap per node unless you defend a different number).
- **Refresh interval**, defended against the update rate and the freshness requirement.
- **`index.number_of_routing_shards`** if you want to preserve a future `split` option.

Present all three as a single table (workload × parameter) or as three worked sections. Show the math; do not just report the numbers.

**Step 2 — Failure-domain layout (~15 min).** For each workload, sketch the shard-allocation-awareness config:

- Node role tags (`data_hot`, `data_warm`, `data_cold`, or single-tier).
- Zone tags (`zone: a`, `zone: b`, `zone: c`).
- The `cluster.routing.allocation.awareness.attributes` setting.
- The primary-plus-replicas layout across zones, described in a sentence ("each primary lands in `zone: a`, replicas land in `zone: b` and `zone: c` — one AZ loss leaves the index green").

You do not have to spin up three-node clusters. A diagram, table, or bullet list is fine as long as it is unambiguous.

**Step 3 — Immutability trap and cutover plan (~15 min).** Write a paragraph per workload answering:

- *If your one-year steady-state estimate is wrong by 3× (in either direction), what do you do?*
- *Which of the three escape hatches (shrink, split, reindex-and-cutover) applies to this workload, and at what threshold do you invoke it?*

For workload A, note that a 3× over-estimate probably means "reindex is overkill, live with the smaller shards." For workload B, a 3× under-estimate means "start a dual-index cutover using the mod-104 chapter 8 pattern before the shards cross ~60 GB." For workload C, rollover-driven daily indexes make the immutability trap less painful — but the *rollover threshold* (`max_size: 50GB, max_age: 7d`) is itself a sizing decision.

**Step 4 — ILM policy for workload C (~30 min).** Author an ILM (or OpenSearch ISM) policy that automates the workload C lifecycle:

- **Hot phase.** Rollover at 50 GB or 7 days, whichever first.
- **Warm phase.** After 30 days, force-merge to 1 segment, shrink to `1` primary shard, allocate to `data: warm` nodes.
- **Cold phase.** After 90 days, allocate to `data: cold` nodes. If you can, layer a searchable-snapshot cold phase (which requires an object-storage repository configured — a `type: fs` snapshot repo on the local disk is a fine dry-run stand-in).
- **Delete phase.** After 365 days.

Deploy the policy to your single-node cluster with `PUT _ilm/policy/errors_policy` (or the OpenSearch `PUT _plugins/_ism/policies/errors_policy` equivalent). Attach it to an index template so new backing indexes pick it up (`PUT _index_template/errors_template`). Force a rollover manually (`POST <alias>/_rollover`) and confirm the new backing index appears. Simulate an aged index (`PUT _ilm/policy/errors_policy?rollup=true` and the [`_ilm/move`](https://www.elastic.co/guide/en/elasticsearch/reference/current/ilm-move-to-step.html) API) to demonstrate that the policy's warm-phase actions run.

You will not have a real hot/warm/cold node topology in a single-node dry-run. That is fine — the goal is to prove the policy is syntactically correct, its phases fire in the right order, and its actions do what you expect. Capture the `_ilm/explain` output before-and-after your simulated move.

**Step 5 — Refresh and rollover interval defence (~10 min).** For each workload, defend a `refresh_interval` value against the trade-off matrix from chapter 6:

- Workload A: `1s` (default), because near-real-time editorial visibility is a product requirement.
- Workload B: something between `5s` and `30s` — defend which one against the 30-second freshness spec and the 50K updates/day throughput.
- Workload C: `30s` (or higher). Defend against the ingest volume.

Write one sentence per workload with the number and the justification. Explicitly state the trade-off you are accepting.

**Step 6 — Peak-load capacity check (~10 min).** For workload B (10,000 QPS peak) and workload A (500 QPS peak), sketch:

- The number of primary + replica shards that would be servicing each query at peak.
- Whether your node count from step 1 has enough CPU cores to fan out queries without oversubscribing (assume 8 cores per node available for search work).
- Whether more replicas would help; whether more primary shards would help; which is cheaper.

You do not need a load test. Napkin math is the point. Cite the chapter 6 heuristic "target 8–20 primary shards per query" and defend where your workload sits.

**Step 7 — Rollout checklist (~10 min).** Author a one-page `rollout_checklist.md` that a platform team would run through before promoting one of your three cluster shapes to production. Cover at minimum:

- Cluster-state check (nodes in the right roles, zone tags in place, `awareness.attributes` set).
- Template check (index template applied, ILM policy attached, mapping matches your spec).
- Backup / snapshot repository configured and tested with a restore drill.
- Node loss drill (kill one data node in staging, watch the cluster recover to green in bounded time).
- Query latency SLO baseline captured (`p50, p95, p99` on a representative query set).
- Kill-switch / rollback plan if the new cluster misbehaves.

Match the checklist items to the workload where they are strictest (e.g., snapshot restore-time for workload A is the "product's memory," so a longer RPO is unacceptable there).

**Step 8 — Write it up (~5 min).** Bundle the artefacts into a single `writeup.md`:

- The sizing table from step 1 (all three workloads).
- The failure-domain sketches from step 2.
- The immutability-trap paragraphs from step 3.
- The ILM policy JSON from step 4, plus the `_ilm/explain` before/after outputs.
- The refresh-interval defences from step 5.
- The peak-load napkin math from step 6.
- The rollout checklist from step 7.

## Acceptance criteria

- Sizing math shown (not just answers reported) for all three workloads. Vector-field contribution called out where relevant.
- Primary shard count for each workload defended against the 20–40 GB heuristic with the arithmetic present.
- Replica count and node count for each workload defended against the uptime target and heap-per-node cap.
- Failure-domain sketch for each workload names the node roles, zone tags, and awareness attribute.
- One paragraph per workload on the immutability trap and the cutover threshold.
- ILM (or ISM) policy JSON for workload C deployed to a running cluster; `_ilm/explain` output captured before and after a phase transition simulation.
- `refresh_interval` defended for each workload with the trade-off named.
- Peak-load fanout math present for workloads A and B with an explicit tie to the "8–20 shards per query" heuristic.
- Rollout checklist with ≥ 6 items, adapted to at least one of the three workloads.
- No claim in the writeup uses a $/GB or ms-latency number without either (a) a citation to a docs page, (b) a note that the number is an order-of-magnitude sketch from chapter 6, or (c) a measurement you took yourself.

## Stretch goals

- **Vector-workload sizing add-on.** Redo workload B's sizing if you switched from `int8_hnsw` to `bbq_hnsw` (chapter 4's binary quantization). What does the shard count become? Where do you have to compensate (more `num_candidates`, a rescore stage) to keep recall parity?
- **Cross-cluster search topology.** For workload C, sketch a cross-cluster-search topology where hot lives in one cluster and cold + frozen live in another. What does that buy you? What does it cost operationally? Cite [Elastic's cross-cluster search docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cross-cluster-search.html).
- **Searchable snapshots cold phase.** Configure a real (local `fs` type) snapshot repository, extend workload C's ILM policy with a frozen phase that mounts a searchable snapshot, and run a query against a mounted snapshot index. Note the latency difference vs. a warm-tier index.
- **OpenSearch ISM port.** Translate the workload C ILM policy to an OpenSearch ISM policy. Note the structural differences (state machine explicit in ISM vs. phase-based in ILM). Which representation reads more clearly? Defend a preference.
- **Failure-mode simulation on a multi-node cluster.** Spin up a real three-node cluster (Docker Compose with three ES/OS containers, each tagged with a different `zone` attribute). Deploy one of your three workload configurations. Kill one node. Time the recovery. Report `time_to_green` and any shards that did not re-replicate as expected.

## Notes

- Every number in this exercise is an *estimate*. You will not get shard sizing exactly right on paper; the point is to be roughly right, know why, and have a plan for when reality disagrees.
- Do not skip the `_ilm/explain` output capture in step 4 even if the phase transition worked visibly. That output is the primary debug artefact when ILM misbehaves in production — you want the muscle memory of reading it now.
- The three workloads intentionally do not include a "10⁹-vector, cross-DC, GPU-ANN, active-active" case. Those live in chapter 7's boundary discussion and belong to `ai-infra-senior-engineer` (level 40) and `staff-ml-engineer` (level 50). If your real workload looks like that, this exercise is not the right sizing input — escalate.
- If your ILM policy deploys but the phase transitions do not fire in your simulation, check three things first: the policy is attached to the index template *and* the alias is pointed at the current write index; the phase `min_age` is measured from rollover, not from index creation; and `_ilm/explain` will name the specific step that is stalled.
- The rollout checklist in step 7 is not a template — write it for the workload you would be most nervous shipping. That is where the value is.
