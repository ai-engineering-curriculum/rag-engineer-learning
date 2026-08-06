# Chapter 9 — Boundary to `ai-infra-senior-engineer` (Level 40): What This Track Does Not Own

## Why this chapter exists

Every level-30 role on the ladder has a peer or higher-level role whose scope is easy to accidentally claim. For the retrieval engineer that adjacent role is **`ai-infra-senior-engineer` (level 40)** — the person who owns the *platform* that vector databases run on top of. Knowing where their scope starts is important for two reasons: (1) you avoid re-solving problems they own more expertly than you do; (2) you know what to hand off cleanly when the corpus, QPS, or availability target crosses into their territory.

This chapter is the ownership contract between the two roles. Everything above the line lives in this track; everything below the line is a conversation with the platform team. Nothing here is a criticism of one level or the other; both roles are essential, and neither can succeed alone.

## What this track owns

You own the retrieval system as-a-product: a tuned, sized, correctly-filtered, correctly-versioned index that meets a `recall@k` and `p95 latency` target on a defined corpus, deployed on infrastructure you can operate.

Concretely:

- **Algorithm choice and tuning** — HNSW vs. IVF-PQ vs. hybrid, and the parameter sweep that defends it (chapters 3–5).
- **Vendor / engine selection** — pgvector vs. Qdrant vs. Milvus vs. Pinecone; the seven-question rubric (chapter 2).
- **Sizing plan** — RAM per million vectors, shard count, replica count, refresh cadence (chapter 6).
- **Metadata filter correctness** — pre-filter vs. post-filter, per-vendor pushdown patterns (chapter 7).
- **Index versioning and cutover** — the tuple convention and dual-index deployment (chapter 8).
- **Recall / MRR / nDCG measurement** — the eval judgment set + statistical significance (mod-109).
- **Retriever CI, drift monitoring, MLOps** — the automation of the above (mod-111).
- **Hybrid retrieval, reranking, prompt-slice** — the pipeline around the retriever (mod-106, mod-107, mod-108).

Anything you can defend within a single-region deployment on a well-known vendor is on your side of the line.

## What `ai-infra-senior-engineer` owns

They own the *platform* the retrieval system runs on — the infrastructure and reliability engineering that keeps the shelf running under adversarial conditions and enterprise-scale demands. Concretely:

### GPU-accelerated ANN

Building or operating GPU-backed vector indexes (FAISS-GPU, cuVS/RAFT, Milvus `GPU_*` indexes, NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) library) at the scale where the GPU pool itself needs capacity planning is platform work. You should *know* GPU ANN exists and roughly what it accelerates (index build, and per-query scan for very high QPS). You should not be the person authoring the GPU pool's scaling policy — that is capacity planning across many workloads, not just yours.

**Where you hand off.** When your corpus or query volume makes a GPU-backed rebuild the difference between "hours" and "minutes," or when the query-time cost of scanning a 10⁹-vector IVF-PQ index would drop meaningfully on a GPU, you write the requirements ("build must complete in ≤ 30 min on X GB working set") and the platform team designs the GPU allocation.

### Sharded cross-DC replication

Cross-data-center replication of a vector index — with consistency guarantees under regional failover, cross-region write cost management, and the reconciliation semantics after a partition — is platform work. Every specialist engine (Milvus, Weaviate, Qdrant) has a cross-DC story, but *operating* that story at production reliability is a full-time role.

**Where you hand off.** When the SLA requires "one region failing should not impact retrieval," you specify the RPO/RTO you need and the platform team designs replication. You keep owning the *shape* of the index (chapters 4–8); they own the *replication*.

### High-availability plans

The platform-scale HA plan — active-active vs. active-passive, quorum for the metadata store, coordinator-node failover in Milvus, Raft repair procedures in Weaviate/Qdrant — is platform work.

**Where you hand off.** You provide the failure-behavior contract you need from the retrieval system ("during a node loss, `p95` must degrade by no more than X, and recall must not fall below Y"). The platform team implements the HA plan that satisfies it.

### Multi-tenant capacity planning at platform scale

Tenant fanout when tenants number in the tens of thousands, with heavily skewed size distribution, is platform work. You know how to shape a good per-tenant vs. shared-namespace decision for a small tenant count (chapter 7). Deciding that "at 50,000 tenants across seven regions we run a two-tier system with a small-tenant shared pool and a large-tenant per-cluster deployment, sharded like this" is platform architecture.

### Cost-per-query at the platform level

You track the recall/latency/RAM trade-off of a specific configuration. The platform team tracks the full cost-per-query across the stack — compute, storage, cross-AZ traffic, embedding inference, reranker inference, LLM inference — and sets the cost-attribution rules. You *use* those numbers; you do not usually set them.

### Multi-workload capacity planning and noisy-neighbor management

When your retrieval system shares infrastructure with other AI workloads (batch training, other inference services, non-AI databases), the noisy-neighbor and capacity-arbitration story is platform work.

## The grey zone — where you collaborate

Some concerns straddle both roles. The productive shape is *joint ownership with a hand-off point*:

- **Quantization strategy** — you know PQ, BQ, SQ (chapters 5–6) and can prototype them. Deploying binary quantization at a scale where it changes the platform's memory-planning model is a joint decision.
- **DiskANN and disk tiering** — you understand the shape (chapter 6). Deciding to move to DiskANN because the platform team measured that RAM cost was unsustainable is joint work.
- **Recall SLO on top of the availability SLO** — the availability SLO is theirs; the recall SLO is yours; the combined SLO ("99.9% availability *and* recall@10 ≥ 0.9") requires both.
- **Choosing a new vendor** — you drive the choice on retrieval merits (rubric, benchmarks). The platform team drives the choice on operational merits (does it fit our K8s/observability/SRE stack?). Both vetoes matter.
- **Cross-region client routing** — you know that a replicated index needs a client that routes to the nearest healthy region. Building the routing layer is platform work.

## The three anti-patterns to avoid

1. **Building your own replication.** If your engine has replication (Weaviate Raft, Qdrant Raft, Milvus's distributed architecture, Pinecone's managed), do not write a homegrown syncer. It is far more work than it looks, and the platform team will inherit the operational cost.
2. **Sizing for the moon.** "Let's just deploy a 100-node Milvus cluster for our 5 M-vector corpus" is a red flag. Match the sizing plan to the actual scale, and let the platform team push you up when the scale changes.
3. **Ignoring the platform team's constraints.** If they run a specific observability stack, a specific Kubernetes distribution, or a specific set of blessed regions, your vendor choice has to fit. A perfect-on-paper vendor that doesn't ship a Datadog/Prometheus/OpenTelemetry integration is an ongoing operational burden the platform team will resent.

## The two escalation triggers

Two concrete signals that a project has grown into `ai-infra-senior-engineer` territory:

1. **The corpus + QPS combined product hits a scale that requires cross-node coordination for a single query.** Once a single query fans out to multiple shards, you are in distributed systems territory; latency variance, straggler handling, and result merging get harder than a chapter-4 sweep can help with. That is when you loop the platform team in.
2. **The reliability requirement exceeds what a single-region deployment offers.** "One region can go down" is when replication is not enough; you need cross-region failover, cross-region write reconciliation, and a runbook that a human on-call can execute at 3 a.m. That is platform work.

Trigger either of these and the conversation shifts from "which HNSW `M` do we pick?" to "how does the platform serve retrieval at this scale?" — a completely different conversation with a completely different set of experts.

## The one-page contract you can put on a wiki

The shape of the artifact:

> **Retrieval engineering (level 30) owns:** algorithm choice and tuning; vendor selection; sizing plan for a single-region deployment; metadata filter correctness; index versioning and cutover; recall/MRR/nDCG measurement; retriever CI and drift; the retriever library that other services call.
>
> **AI infra senior engineering (level 40) owns:** GPU pool sizing for ANN; cross-DC replication and failover; platform-scale HA plans; capacity planning across many AI workloads; cost-attribution and cost-per-query at the platform level; the observability, deploy, and on-call stack the retrieval system inherits.
>
> **Joint decisions:** vendor selection (both retrieval-fit and platform-fit vetos); quantization at platform-affecting scale; the combined availability + recall SLO; the choice to move to disk-tiered or GPU-backed indexes.

Every level-30 project should ship with that one-pager filled in before the first index is built.

## What this track covers vs. what mod-111 and mod-112 cover

Two of this track's own downstream modules also drill into ownership questions this chapter touches on:

- **mod-111 (Retrieval MLOps: versioning, CI/CD, drift)** picks up where chapter 8 leaves off — retriever CI machinery, drift monitoring, orchestration integration.
- **mod-112 (Enterprise retrieval: ACL, structured, graph, multimodal)** covers the enterprise-scale multi-tenancy correctness and ACL enforcement that chapter 7's filter correctness only scratches.

Both stay firmly within retrieval engineering. Neither crosses into platform-owned scope; they cover the *retrieval-side* of concerns that the platform layer supports.

## What you should be able to do after this chapter

- Draw the line between "single-region deployment on a well-known vendor" (your scope) and "GPU pool, cross-DC replication, platform HA" (their scope).
- Recognize the two escalation triggers and know what to hand off when either fires.
- Author the one-page ownership contract for a real project.
- Not accidentally claim work that belongs to the platform team, and not accidentally punt work that belongs to you.

That closes mod-104. mod-105 shifts to the Lucene family (Elasticsearch, OpenSearch, Solr, Vespa) and learning-to-rank — the other half of the retrieval world that hybrid stacks depend on.

## Further reading

- The track's own [`CURRICULUM.md`](../../CURRICULUM.md) — the module map and the level-ladder positioning.
- The track's own [`PREREQUISITES.md`](../../PREREQUISITES.md) — the peer/adjacent-track links.
- NVIDIA cuVS / RAFT documentation: [`github.com/rapidsai/cuvs`](https://github.com/rapidsai/cuvs) — the reference for GPU-accelerated ANN as a platform concern.
- Milvus docs: [*GPU Index*](https://milvus.io/docs/gpu_index.md), [*Cluster architecture*](https://milvus.io/docs/architecture_overview.md) — for the cross-node coordination shape.
- Google SRE workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — for the joint availability + recall SLO discussion.
