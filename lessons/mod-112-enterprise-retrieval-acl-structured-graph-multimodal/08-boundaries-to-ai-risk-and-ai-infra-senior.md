# Chapter 8 — Boundaries: `ai-risk-engineer` and `ai-infra-senior-engineer`

## Why this chapter exists

Enterprise retrieval brushes up against two adjacent disciplines whose scope this module deliberately does not annex. Both are load-bearing for a production RAG at real scale; both have their own tracks that go deeper than a retrieval-engineer's time budget can afford. This chapter draws the boundary explicitly so you don't misappropriate one of those tracks' work and so you know exactly what to hand over.

The two peers:

- **`ai-risk-engineer` (peer, level 25, Governance family)** owns *content-safety and responsible-AI evaluation methodology*. This module covers ACL *mechanics*; it does not cover PII detection, prompt-injection defense evaluation, guardrail effectiveness scoring, or fairness/bias evaluation of retrieval outputs.
- **`ai-infra-senior-engineer` (level 40, Infra family)** owns *retrieval-serving infrastructure at platform scale*. This module covers per-tenant index isolation *concepts*; it does not cover running 5,000 per-tenant Qdrant collections in a Kubernetes cluster with per-tenant SLOs, or GPU-accelerated ANN sharding across data centers.

Getting these boundaries right keeps this module scoped to what a level-30 retrieval-engineering IC can be responsible for. Getting them wrong bloats the module into a governance-family or infrastructure-family curriculum it can't actually deliver.

## Boundary 1 — content safety and PII redaction (`ai-risk-engineer`)

### What `ai-risk-engineer` owns

- **PII detection methodology.** Given text, identify names, emails, phone numbers, SSNs, credit-card numbers, IBANs, medical record numbers, and hundreds of other entity types. Evaluate detectors on precision, recall, and per-entity-class calibration. Reference tools: [Microsoft Presidio](https://github.com/microsoft/presidio), [AWS Comprehend PII](https://docs.aws.amazon.com/comprehend/latest/dg/pii.html), [Google Cloud DLP](https://cloud.google.com/security/products/dlp).
- **Redaction methodology.** Replace / mask / hash / synth-substitute detected entities with per-context-appropriate strategies. Evaluate whether the redaction is reversible-when-needed, non-reversible-when-required, and doesn't destroy meaning for downstream consumers.
- **Prompt-injection defense evaluation.** Whether input filters, output filters, and structural defenses (like tool-use gating) reduce the success rate of adversarial prompts. Reference: [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/).
- **Guardrail effectiveness scoring.** Does a content-safety guardrail (toxicity, self-harm, illegal content) catch what it should while not over-blocking? False-positive / false-negative rates on labeled sets.
- **Fairness / bias evaluation.** Systematic differences in retrieval or generation outcomes across demographic slices.
- **Content-safety evaluation platforms.** Guardrails-AI, NeMo Guardrails, LlamaGuard, ShieldGemma — as content-safety artifacts, not as retrieval infra.

### What this module owns

- **The retrieval mechanism** (chapters 2–7). ACL filtering, tenancy isolation, text-to-SQL safety guards, graph-retriever policy honoring, router post-checks.
- **The hooks where redaction plugs in.** Two places matter:
  - **Ingestion-time redaction.** Text goes through a PII detector before it becomes a chunk; detected entities are redacted per policy; the redacted text is what gets embedded and stored. `ai-risk-engineer` owns the detector; this module owns the pipeline stage where it plugs in.
  - **Return-time redaction.** Retrieved chunks pass through a PII filter before they hit the generator or the user. Useful when the ingestion-time redaction can't be complete (a doc whose classification was raised after ingestion; a tenant whose policy is stricter than the corpus-wide default).

### What the handoff looks like

The retriever-engineer's contract with `ai-risk-engineer`:

- **Ingestion hook.** "Before chunk `C` is embedded and stored, call `pii_scrub(C.text, policy=T.policy)` and store the result. The scrubber returns `(scrubbed_text, entities_removed_metadata)`; store both — the metadata is what enables an audit."
- **Retrieval hook.** "Before chunk `C` is returned by the router, call `pii_check(C.text, requester=U)`. If the check flags disallowed content for user `U`, drop the chunk and log the drop."
- **Evaluation hook.** "The retrieval-CI (mod-111) gates on retrieval quality *and* on a leak-rate metric provided by `ai-risk-engineer`. A regression on either blocks the merge."

The retriever-engineer *does not* build the PII detector or evaluate its per-entity precision/recall. If you find yourself designing the SSN-detection regex, delegate.

### A specific anti-pattern

**"We ship redaction ourselves because it's easier than integrating with the risk team's service."** Common. Also wrong. A homegrown detector will:

- Miss entity types (a colleague's new IBAN format, a foreign SSN equivalent).
- Over-block (a doc about "Social Security policy" gets its every mention scrubbed).
- Not be independently audited.
- Drift as regulations change (new GDPR guidance, new US state-level PII definitions).

The right shape is: **use the org's PII detector (behind a stable interface) or use one of the reference tools (Presidio is a common default) and stay in touch with the risk team about scope drift.** Don't build.

## Boundary 2 — platform-scale cross-tenant isolation (`ai-infra-senior-engineer`)

### What `ai-infra-senior-engineer` owns

- **Per-tenant physical isolation at scale.** Hundreds to thousands of tenants, each with their own storage, encryption keys, network isolation. GPU-accelerated ANN with per-tenant capacity allocation. Cross-region replication of per-tenant indexes.
- **Multi-region deployment topology** for retrieval — active-active vs. active-passive, data-residency compliance (data cannot leave EU, cannot leave a specific customer VPC), disaster-recovery playbooks with per-tenant RPO/RTO SLAs.
- **Retrieval serving on shared infrastructure at high concurrency** — connection pooling, tenant-aware autoscaling, noisy-neighbor mitigation, per-tenant rate limiting.
- **Encryption-at-rest and in-transit keying** — customer-managed keys (BYOK), key rotation without index rebuild, envelope encryption of vectors.
- **Compliance certifications** as they apply to the retrieval infra — SOC 2 controls for the retrieval service, HIPAA BAA scope, FedRAMP boundary.

### What this module owns

- **The three enforcement points** (chapter 2). Query-time, index-time, per-tenant isolation as *concepts* — the correctness trade-offs, the layering, the belt-and-braces post-check.
- **Per-tenant isolation up to the "collection per tenant" implementation** — one Qdrant collection per tenant, one Elasticsearch index alias per tenant, one Postgres schema per tenant.
- **The retriever-side contract** — every retriever accepts a policy context, honors it, logs drops.

### What the handoff looks like

The retriever-engineer's contract with `ai-infra-senior-engineer`:

- **"We give you a retriever that accepts `tenant_id` and never returns cross-tenant data. You give us the operational envelope — the physical shards, the encryption keys, the SLA, the rate limits."**
- The retriever's *shape* (one collection per tenant, sharded by tenant if collection count exceeds engine limits) is owned by this module.
- The retriever's *scale* (which shards live where, which encryption key covers which tenant, how many replicas per shard, how to fail over when a shard dies) is `ai-infra-senior-engineer`'s.

### A specific anti-pattern

**"We shipped one giant collection with `tenant_id` metadata because we couldn't afford per-tenant collections."** Fine on the small side (dozens of tenants), risky in the middle (hundreds), and reckless at scale (thousands of tenants, tens of millions of docs each). The single-collection approach:

- Puts every tenant at risk of a single filter bug.
- Makes per-tenant SLA impossible (a heavy tenant slows every tenant).
- Complicates per-tenant deletion (a GDPR "delete this tenant" request becomes a mass-delete-with-filter, which is expensive and hard to verify).

The right shape at scale is: per-tenant collection with a control-plane that routes queries to the right shard. Building that control plane at scale is `ai-infra-senior-engineer`'s territory. This module tells you what shape it should serve; the peer builds the shape.

## Two things this module explicitly does not deliver (and cites the peer for)

### 1. A PII-redaction pipeline

You know that ingestion-time and return-time redaction *plug in* to the retriever pipeline (chapter 2's `pii_check` hook; chapter 3's Salesforce Field-Level Security handling). You don't know how to build a state-of-the-art PII detector, evaluate it, or defend its false-negative rate to auditors. When your project needs redaction:

- Reach for the `ai-risk-engineer`-authored PII service (or Presidio / Comprehend PII / Cloud DLP if you don't have one).
- Wire the pipeline stage per their contract.
- Pass their leak-rate metric as a CI gate alongside your retrieval-quality gates.

### 2. Platform-scale isolation

You know that per-tenant isolation is a coarse boundary with excellent blast-radius properties (chapter 2). You know Weaviate has first-class multi-tenancy and other engines do it via collection-per-tenant. You do not know how to run thousands of per-tenant collections in production with SLA guarantees. When your project scales past a few hundred tenants:

- Bring `ai-infra-senior-engineer` in for the platform layer.
- Your retriever surface stays the same — accept policy context, refuse cross-tenant queries — but the underlying storage, sharding, encryption, and replication are theirs.
- Your CI/CD scaffolding (mod-111) hands off to their multi-region deployment pipeline.

## The tracks you might confuse this boundary with

- **`applied-ai-engineer` (peer, level 30)** owns the *product-surface* consumption of a retriever — RAG-triad instrumentation (faithfulness / answer relevance / context precision-recall from `ai-eval-engineer`'s tools), user-facing prompt shape, product decisions about when to use retrieval. If you find yourself designing the user's UI around retrieval, that's `applied-ai-engineer`.
- **`ai-eval-engineer` (peer, level 30)** owns application-layer RAG eval (RAGAS, TruLens, DeepEval) and general LLM tracing (LangSmith, Langfuse, Arize, Phoenix). This module owns retrieval-only metrics (recall@k, MRR, nDCG); the peer owns end-to-end eval.
- **`agentic-ai-engineer` (level 40)** owns agent orchestration that *consumes* the retriever as a tool. The router this module builds is a retriever surface; the agent that decides *when* to call it is the peer's.

Chapter 1 also cited these; the point of restating them here is that a boundary you can articulate is a boundary you can defend when a project inevitably tries to expand your scope into them.

## The handoff artifacts

When you finish a project in this track and hand off to the two peers, ship them:

**To `ai-risk-engineer`:**

- The list of retrieval hooks where redaction plugs in (ingestion-time, return-time), with call signatures.
- The set of chunk-metadata fields available to the redaction service (`sensitivity_label`, `source_system`, `tenant_id`, `is_public`).
- The current leak-rate baseline from your fuzz test — the number their metric must not regress against.

**To `ai-infra-senior-engineer`:**

- The per-tenant isolation shape (collection-per-tenant, control-plane routing).
- The policy contract (chapter 7) — the fields every retriever accepts.
- The retrieval-quality gates (mod-111) — the metrics that must not regress under whatever storage / sharding / replication topology they implement.
- The freshness SLA (how quickly a permission change must propagate) — the SLO their system serves against.

Handoffs go smoothly when both sides have written down what they own and what they need. That is what these chapters are.

## What you should be able to do after this chapter

- Draw a two-column boundary diagram: on the left, what this module owns; on the right, what each peer owns. Populate for both `ai-risk-engineer` and `ai-infra-senior-engineer`.
- Refuse to build a PII detector when a project asks; cite `ai-risk-engineer` and describe the hook.
- Refuse to design a per-tenant sharding topology at platform scale; cite `ai-infra-senior-engineer` and describe the interface.
- Author a handoff artifact for each peer that lists the hooks, contracts, gates, and SLAs your retriever exposes to them.
- Recognize the two anti-patterns (homegrown redaction, single-collection multi-tenant at scale) and name the peer that would prevent each.

## Further reading

- Microsoft Presidio: [*presidio*](https://github.com/microsoft/presidio) — the OSS PII detection / redaction toolkit; a common baseline your `ai-risk-engineer` peers work from.
- OWASP: [*LLM Top 10*](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — the canonical taxonomy of LLM-specific risks (prompt injection, insecure output handling, sensitive info disclosure, model DoS). The retrieval-side items (Sensitive Information Disclosure, Insecure Plugin Design) map to what this module's ACL and safety chapters own.
- NIST: [*AI Risk Management Framework (AI RMF 1.0)*](https://www.nist.gov/itl/ai-risk-management-framework) — the general risk-taxonomy framework `ai-risk-engineer` operationalizes; useful context for the vocabulary you'll trade at handoff time.
- Google Cloud: [*Data loss prevention (DLP)*](https://cloud.google.com/security/products/dlp) and AWS: [*Amazon Comprehend PII detection*](https://docs.aws.amazon.com/comprehend/latest/dg/pii.html) — the hyperscaler PII services your risk-engineering peers may plug in.
- Weaviate: [*Multi-tenancy*](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy) — engine-native multi-tenancy as the closest thing to "cross-tenant isolation as a first-class primitive."
- Google Cloud: [*Multi-tenancy in enterprise SaaS*](https://cloud.google.com/architecture/best-practices-for-enterprise-organizations) — general architectural guidance for multi-tenant systems; the retrieval slice is what your `ai-infra-senior-engineer` peers implement.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly. Chapter 11 on stream processing and chapter 12 on replication cover the general shape of the platform-scale problems `ai-infra-senior-engineer` owns.
