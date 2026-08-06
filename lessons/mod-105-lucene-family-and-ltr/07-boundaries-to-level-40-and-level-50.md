# Chapter 7 — Boundaries to `ai-infra-senior-engineer` (Level 40) and `staff-ml-engineer` (Level 50)

## Why this chapter exists

Every level-30 role has upstream roles whose scope is easy to accidentally claim. For the retrieval engineer touching Lucene-family engines and LTR, there are two: **`ai-infra-senior-engineer` (level 40)** owns the platform the cluster runs on, and **`staff-ml-engineer` (level 50)** owns the web-scale ML training infrastructure and the modelling frontier past classical LTR. This chapter marks the two boundary lines explicitly so you know what to hand off and what to keep.

The mod-104 boundary chapter drew the line to `ai-infra-senior-engineer` for vector-database platform work. This chapter re-draws the same line for Lucene-family clusters (with different specifics — the topology is different) and adds the line to `staff-ml-engineer` for LTR and ranking-ML work.

Both lines are ownership contracts, not walls. Retrieval engineers collaborate with both roles constantly; the point of the contract is that everyone knows who is on the hook for what when something breaks.

## What this track owns

You own the retrieval product: a tuned, sized, correctly-analyzed, correctly-sharded, correctly-tiered lexical (and optionally dense) index with a defensible ranking model, meeting a `recall@k` / nDCG / `p95 latency` target on a defined corpus, deployed on infrastructure you can operate.

Concretely, chapters 1–6:

- **Engine selection** — ES vs. OS vs. Solr vs. Vespa on the four-question framework (chapter 1).
- **Analyzer authoring** — tokenisation, stemming, ICU normalisation, synonym strategy (chapter 2).
- **Field mapping, BM25 tuning, boosting, function scoring** — the production lexical index (chapter 3).
- **Dense-vector field integration** — quantization pick, hybrid retrieval as a native query (chapter 4).
- **LTR feature engineering, model training, deployment via the LTR plugin** (chapter 5).
- **Shard sizing, replica placement, hot-warm-cold ILM policies** — for a single-region deployment (chapter 6).
- **Retriever eval, MLOps, retriever CI, drift monitoring** — mod-109, mod-111.
- **Hybrid retrieval fusion, prompt-context assembly, reranking** — mod-106, mod-107, mod-108.
- **Enterprise retrieval concerns — ACL, structured, graph, multimodal** — mod-112.

Anything you can defend within a single-region deployment of a well-known engine, running at a scale your JVM heap can hold, is on your side of both lines.

## Line 1: what `ai-infra-senior-engineer` (level 40) owns

They own the *platform* the search cluster runs on — the infrastructure and reliability engineering that keeps the shelf running under adversarial conditions and enterprise-scale demand. The topics below sit above the retrieval engineer's ceiling.

### Web-scale sharding

Sharding a single logical index across thousands of shards, or across hundreds of nodes, is platform work. The chapter-6 heuristics stop scaling around a few hundred shards per index and a low-tens-of-nodes cluster; past that the correct answers involve multi-cluster architectures, [cross-cluster search](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cross-cluster-search.html), searchable-snapshot-backed frozen tiers, and shard-splitting strategies that require deep platform knowledge.

**Where you hand off.** When your corpus size or query fanout means a single-cluster ES/OS deployment is at its limits (thousands of shards per cluster, tens of nodes struggling to hold cluster state), you write the requirements ("this index must survive N-region loss, must serve X QPS at Y latency, must retain Z years of data") and the platform team designs the multi-cluster shape.

### Cross-DC replication and remote-cluster orchestration

Cross-data-center replication of Elasticsearch or OpenSearch — via [cross-cluster replication (CCR)](https://www.elastic.co/guide/en/elasticsearch/reference/current/xpack-ccr.html) on Elastic's commercial license, or via [OpenSearch cross-cluster replication](https://opensearch.org/docs/latest/tuning-your-cluster/replication-plugin/index/), or via SolrCloud with ZooKeeper across DCs, or via Vespa's global content clusters — is platform work.

**Where you hand off.** When your SLA requires "one region failing should not impact retrieval," or "writes accepted in region A must be visible in region B within N seconds," you specify RPO/RTO and the platform team designs replication topology, network peering, and failover. You keep owning the *shape* of the index (chapters 2–5); they own the *replication*.

### Cluster orchestration at platform scale

Running one three-node cluster is your job. Running fifty three-hundred-node clusters — with a paved-road operator (ECK, [opensearch-k8s-operator](https://github.com/opensearch-project/opensearch-k8s-operator), custom controllers), fleet-wide upgrade rollouts, cross-cluster observability, capacity forecasting, and multi-tenant chargeback — is platform work.

**Where you hand off.** When your project is "the Nth workload on the shared search platform," you conform to the platform's onboarding, observability, and upgrade cadence; you do not fork the operator or run your own K8s cluster.

### Multi-tenant capacity planning at platform scale

The chapter-6 sizing math handles single-workload sizing. Managing a shared search platform used by hundreds of workloads — with per-workload quotas, noisy-neighbour isolation, chargeback, and per-tenant SLOs — is platform work.

**Where you hand off.** You provide a capacity requirement ("this workload is X GB steady-state, Y write QPS, Z read QPS"); the platform team decides how to fit it into the shared pool.

### Observability and on-call at platform scale

Your monitoring of your specific index — search latency, indexing throughput, refresh cost, LTR model p99 — is yours. The platform-wide observability of cluster health, hot spots, shard rebalancing storms, JVM garbage-collection anomalies, and the on-call rotation that responds to platform incidents is platform work.

**Where you hand off.** You define the SLIs and alerts that fire on *your* workload; the platform team owns the underlying cluster-health SLIs and on-call.

### Storage-tier economics at platform scale

Chapter 6's hot/warm/cold picks are per-workload. Choosing what storage classes the platform offers, what searchable-snapshot object-storage backend to use, what disk types to provision, and what the chargeback rates are — is platform work.

**Where you hand off.** You pick from the tiers the platform offers; you do not spin up your own S3 bucket for searchable snapshots.

## Line 2: what `staff-ml-engineer` (level 50) owns

Level-50 ML engineers own the modelling frontier and the ML platform underneath it. The topics below sit above the retrieval engineer's ceiling on the ML side.

### Ranking-model research and novel architectures

Classical LTR (LambdaMART / LightGBM / XGBoost) is your job at level 30. Designing a novel ranking architecture — a two-tower model with hard-negative mining, a listwise transformer, a multi-task learning setup that jointly optimises click-through and satisfaction — is level-50 work.

**Where you hand off.** When your project needs a model architecture past the LightGBM/XGBoost ceiling, you write the retrieval-quality target and the acceptable feature set; a staff ML engineer designs and trains the architecture. You wire it into the serving pipeline (chapter 5 shape, adapted for the new model format).

### Personalization at fleet scale

Per-user personalization signals in the LTR feature set are yours (chapter 5). Building the *personalization platform* — a real-time user-embedding service, a session-context store with millisecond lookup, a feature store that maintains fresh user features across web-scale traffic — is staff-ML-platform work.

**Where you hand off.** You define what personalization signal you'd like as an LTR feature; the ML platform team builds and serves the signal.

### Web-scale training infrastructure

Training a LightGBM/XGBoost LambdaMART model on 10⁷ (query, doc) pairs on your laptop is your job. Training a ranking model on 10⁹ pairs with distributed GPUs, spot-instance orchestration, hyperparameter search at fleet scale, and daily retraining automation is staff-ML work.

**Where you hand off.** You produce a training loop that fits on a workstation; if the eventual scale needs distributed training, the ML platform team builds the training infra.

### Click-model research and unbiased learning to rank

Applying position-bias correction to your click logs (chapter 5) is your job. Designing a novel click model (dynamic Bayesian click model, cascade model, or a click-through-rate propensity estimator specific to your product) is level-50 work.

**Where you hand off.** You use the debiased labels the ML platform team produces; you do not design the debiasing model.

### LLM-as-judge, cross-encoder-at-scale, and RAG-quality frontier work

Mod-107 covers neural rerankers (cross-encoders, ColBERT, LLM-as-judge) at a level-30 depth: how to use them, how to eval them, how to fit them into the pipeline. The *research* on new reranker architectures, the *cost economics* of running a cross-encoder at fleet scale, the *distillation* of a large reranker into a small serving model — that lands in level-50 scope, often in collaboration with an `applied-ai-engineer` or a dedicated retrieval-research group.

**Where you hand off.** You pick from the reranker menu the ML/research team has vetted; you do not design new rerankers.

### ML observability at platform scale

Your monitoring of your specific LTR model's online metrics (position-1 click-through, session satisfaction, model latency, feature drift) is yours. The platform-wide ML observability stack — a feature store with monitoring, a model registry, model-fairness dashboards, cross-workload drift detection — is platform-ML work.

**Where you hand off.** You emit your model's metrics into the platform's ML observability stack; you do not build the stack.

## The three-role Venn: where all three collaborate

Some concerns touch all three roles:

- **The vendor / engine choice for a search platform.** Retrieval engineer drives the retrieval-fit veto (chapter 1). AI infra drives the operational-fit veto (chapter 6 shape, plus platform integration). Staff ML weighs in when the choice constrains future ML work (e.g., "can we bolt on custom rerankers?" — favours Vespa, ES with ML Commons, or a specialist).
- **The retriever-quality vs. serving-cost frontier.** Retrieval engineer picks index/model config; ai-infra prices the compute; staff ML weighs in on whether a more expensive but higher-quality model justifies the cost.
- **A cross-region search product.** Retrieval engineer specifies the recall/latency contract across regions; ai-infra owns the cross-DC topology and consistency; staff ML weighs in on personalization / model-freshness across regions.

The productive pattern: **shared design docs with named owners per section, and a single go/no-go review that all three roles sign off on for a launch**.

## The three anti-patterns to avoid

1. **Building your own cross-cluster replication.** ES CCR, OS CCR, Vespa's global content clusters — the vendors ship replication. Do not roll your own; the operational cost will fall on the platform team.
2. **Deploying a novel ranking architecture without an ML-platform sign-off.** A random-forest that works on your workstation may not work on the platform's serving stack, may not have observability, may not have a retraining pipeline. Loop staff ML in early.
3. **Ignoring the platform team's blessed regions, K8s distribution, or observability stack.** Chapter 1's "which engine you already run" question applies here too — a technically-perfect engine choice that doesn't fit your platform's paved road is an ongoing operational burden.

## The two escalation triggers

Two concrete signals that a project has crossed one of the boundaries above:

1. **The scale (corpus size × query fanout × availability requirement) is beyond a single-region single-cluster deployment.** You are in `ai-infra-senior-engineer` territory. Loop them in early — the topology question changes the retrieval design.
2. **The ranking-model complexity or personalization ambition is beyond LambdaMART on engineered features.** You are in `staff-ml-engineer` territory. Loop them in early — the model architecture changes the feature engineering and the serving path.

Either trigger: stop, escalate, and re-scope. Do not fight scale or complexity alone at level 30; the failure modes get harder faster than the tools scale.

## The one-page contract you can put on a wiki

The shape of the artifact:

> **Retrieval engineering (level 30) owns:** engine selection for a single-region deployment; analyzer authoring; field mapping, BM25 tuning, function scoring; dense-vector field integration; LTR feature engineering, training, and plugin deployment; single-cluster shard sizing, replica placement, and hot-warm-cold ILM; retriever eval, CI, drift; the retriever product that other services call.
>
> **AI infra senior engineering (level 40) owns:** search platform operator and paved road; multi-cluster architectures; cross-DC replication; cross-cluster search topology; multi-tenant capacity planning; on-call and observability at platform scale; storage-tier economics and blessed regions.
>
> **Staff ML engineering (level 50) owns:** ranking-model research past classical LTR; personalization platform (user embeddings, session-context features, feature store); web-scale training infrastructure; click-model research and unbiased LTR; the reranker frontier (cross-encoders at scale, LLM-as-judge, distillation).
>
> **Joint decisions:** engine selection (retrieval-fit × operational-fit × ML-fit vetoes); the combined recall + availability + cost target; novel ML deployments; cross-DC ranking-model consistency; the choice to move a workload to a shared platform.

Every level-30 project past demo scale should ship with that one-pager filled in and signed by named owners before the first index is built.

## What this module hands off to the rest of the track

Two of this track's own downstream modules pick up threads this chapter deferred:

- **mod-106 (Hybrid retrieval, fusion, query rewriting)** — takes the ES RRF / OS hybrid query / Vespa rank expression from chapter 4 and covers the fusion methodology (RRF, weighted-sum, learned fusion) end-to-end.
- **mod-107 (Reranking and context assembly)** — covers neural rerankers as the level-30 counterpart to LTR from chapter 5. Where LTR combines engineered features, the rerankers there compute fresh (query, document) representations.
- **mod-109 (Retrieval quality evaluation)** — covers the eval methodology behind the labels your LTR training needs, and the offline eval that gates model promotion.
- **mod-111 (Retrieval MLOps, versioning, CI, drift)** — covers the automation of retriever CI, model versioning, and drift monitoring at level-30 depth. Chapter 6's ILM is one piece; mod-111 covers the ML-side lifecycle.
- **mod-112 (Enterprise retrieval, ACL, structured, graph, multimodal)** — covers ACL enforcement (which is a filter-correctness concern this chapter did not touch) and enterprise data topology.

All five stay firmly within retrieval engineering. Neither crosses the level-40 or level-50 lines drawn above.

## What you should be able to do after this chapter

- Draw the two boundary lines (level-30 → level-40 platform; level-30 → level-50 ML) for a Lucene-family retrieval project.
- Recognise the two escalation triggers and know what to hand off when either fires.
- Author the one-page ownership contract for a real project with named owners in each role.
- Not accidentally claim work that belongs to platform or ML, and not accidentally punt work that belongs to you.

## Further reading

- The track's own [`CURRICULUM.md`](../../CURRICULUM.md) — the module map and level-ladder positioning.
- The track's own [`PREREQUISITES.md`](../../PREREQUISITES.md) — the peer/adjacent-track links.
- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — the joint availability + recall SLO discussion.
- Elastic: [*Cross-cluster replication*](https://www.elastic.co/guide/en/elasticsearch/reference/current/xpack-ccr.html) — an example of what platform work looks like above chapter 6's line.
- OpenSearch: [*Cross-cluster replication*](https://opensearch.org/docs/latest/tuning-your-cluster/replication-plugin/index/) — the OpenSearch equivalent.
- Karpukhin, V., Oğuz, B., Min, S., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — canonical modelling paper of the kind level-50 ML engineers extend; useful for you to know exists as the frontier context above chapter-5's ceiling.
