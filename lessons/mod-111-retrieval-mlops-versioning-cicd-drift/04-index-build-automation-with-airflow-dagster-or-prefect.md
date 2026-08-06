# Chapter 4 — Index-Build Automation with Airflow / Dagster / Prefect

## Why this chapter exists

Chapter 2 automated the cutover (build → verify → flip → cool-down). Chapter 3 automated the CI (a PR → an eval → a gate). This chapter automates the **build itself** — the batch job that turns corpus changes into new chunks, embeds them, ships them into the index, and hands the result to the cutover state machine.

The domain is orchestration — Airflow, Dagster, Prefect, and the cloud-native equivalents (AWS Step Functions, GCP Workflows / Vertex AI Pipelines, Azure Data Factory / Databricks Workflows). Choosing between them and authoring the DAG shape is `ai-infra-mlops` (level 25) territory; this chapter assumes you know that. What this chapter owns is the **retrieval-shaped** questions on top:

- Which corpus deltas trigger an incremental build vs. a full rebuild?
- Where in the DAG does the embedding pass sit, and how do you retry it when the hosted API rate-limits?
- What is the *ingestion-lag SLO* — the promised time between "a document was updated" and "it is retrievable" — and how does the DAG enforce it?
- What does a backfill look like when you need to re-embed the entire corpus with a new model?
- What is the shape of the asset graph so a Dagster / Airflow operator can see it and reason about it?

Answering those is what this chapter does.

## The asset graph — the retrieval index's dependencies

Whether you use Dagster's asset abstraction, Airflow's task graph, Prefect's flow, or a cloud-native pipeline definition, the shape is the same. The retrieval index depends on a specific chain of upstream assets:

```
                                                                      ┌─────────────────┐
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │                 │
    │ raw_corpus   │───►│ cleaned_docs │───►│ chunked_docs │──┬──────►│ embedded_chunks │──┬──►
    │              │    │              │    │              │  │       │                 │  │
    │ (S3, Confl., │    │ (normalized, │    │ (chunker.v4) │  │       │ (embed model    │  │
    │  Notion,     │    │  dedup'd,    │    │              │  │       │  bge-large-1.5) │  │
    │  file share) │    │  parsed)     │    │              │  │       │                 │  │
    └──────────────┘    └──────────────┘    └──────────────┘  │       └─────────────────┘  │
                                                              │                            │
                                                              │       ┌─────────────────┐  │
                                                              └──────►│ payload_records │◄─┘
                                                                      │                 │
                                                                      │ (schema.v2:     │
                                                                      │  doc_id, page,  │
                                                                      │  section, ACL)  │
                                                                      └────────┬────────┘
                                                                               │
                                                                               ▼
                                                                      ┌─────────────────┐
                                                                      │ built_index     │
                                                                      │ (ANN config,    │
                                                                      │  HNSW M32-      │
                                                                      │  ef200)         │
                                                                      └────────┬────────┘
                                                                               │
                                                                               │  hand-off to
                                                                               │  chapter 2's
                                                                               ▼  cutover automation
                                                                          verify + flip
```

Every arrow is a computed dependency: change the upstream, the downstream is stale. Dagster's [asset abstraction](https://docs.dagster.io/concepts/assets/software-defined-assets) models this natively (an asset knows what it depends on and whether it is stale); Airflow's [dataset](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html) abstraction is the closest equivalent; Prefect's [flow / task](https://docs.prefect.io/latest/concepts/flows/) shape hosts the same graph.

The upstream boundary — where does `raw_corpus` come from — is the ingestion pipeline from mod-102. This chapter picks up from `cleaned_docs`.

## Incremental vs. full — the trigger matrix

The single most-important operational question is when to run *incremental* (embed and index only the deltas) vs. *full* (re-embed and re-index everything).

| Event | Incremental | Full |
|---|---|---|
| New documents added to corpus | ✅ | rare |
| Existing document *updated* | ✅ (delete old chunks, embed new) | never |
| Existing document *deleted* | ✅ (delete chunks; no embed) | never |
| Chunker config changed | ❌ | ✅ (every chunk is stale) |
| Chunker code changed (behavior differs) | ❌ | ✅ |
| Embedding model / revision changed | ❌ | ✅ (every vector is stale) |
| ANN config changed | ❌ | ✅ (index structure changes) |
| Schema changed (breaking) | ❌ | ✅ |
| Schema changed (additive-only, unused at query time) | ✅ (backfill new field) | scheduled full |
| Scheduled cadence (drift-refresh, safety rebuild) | — | ✅ (weekly / monthly, project-specific) |

The rule underneath the table: **any change that touches the index tuple's non-corpus-snapshot fields (`embedding-model`, `chunker-version`, `schema-version`, `ANN-config`) is a full rebuild.** Only the corpus snapshot can be updated incrementally.

## Incremental-build shape

An incremental run reads a corpus delta and mutates the *live* index. The safe shape:

1. **Read the delta.** The source system's change-feed (Confluence has a `since` parameter on its content API, Notion has a webhook, git has `log --since`, S3 has [object versioning + a manifest](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)). If the source has no change-feed, checksum-diff a manifest of paths + mtimes + hashes — expensive but universal.
2. **Compute the chunk-level delta.** New documents produce new chunks; updated documents produce (deleted-old-chunks, new-chunks); deleted documents produce deleted-chunks.
3. **Embed only new chunks.** The embedding-API cost per doc (chapter 5) makes this the biggest cost lever the pipeline has.
4. **Upsert the new chunks and delete the removed chunks in the live index.** Every engine supports both operations (`upsert` and `delete` on pgvector / Weaviate / Qdrant / Milvus / Pinecone / Chroma / OpenSearch). The order matters: upsert first, then delete, so no request during the window sees fewer chunks than it should.
5. **Emit a delta manifest** to the audit log. The delta manifest is what the drift monitor (chapter 7) reads to compute corpus drift.

Incremental runs do *not* trigger a full cutover — the live index is being mutated in place. The `INDEX_ID` does not change (the corpus-snapshot field bumps its date, but the vendor collection name stays put — you can also treat that as a tuple change and cutover; project-dependent).

**When incremental is NOT safe**: if the source system does not give reliable change-feed, incremental drifts (documents that were deleted upstream but the deletion event was missed) will accumulate. A **periodic full rebuild** (weekly or monthly) is the safety net that cleans up drift; every mature retrieval pipeline runs one on a cadence.

## Full-rebuild shape

Full-rebuild is the shape chapter 2's cutover automation was designed for. The DAG shape:

1. **Snapshot the corpus.** Freeze the state at a specific timestamp. Every downstream step reads from that snapshot, not from live-updating source. This is what makes the rebuild reproducible.
2. **Chunk.** Run the entire corpus through the chunker. Produces `chunked_docs` — an asset the embedding step consumes.
3. **Embed.** Batch the chunks through the embedding pass. This is the longest step and the most-expensive. See "the embedding-pass shape" below.
4. **Build payload records.** Add metadata (doc-id, ACL scope, section path, updated-at, tenant-id) per the schema. Produces `payload_records`.
5. **Build the ANN index.** Ingest `payload_records` into a *new* collection whose name is the new tuple. Some engines are fast to build (pgvector), others slow but memory-efficient (HNSW at high `efConstruction`), others need explicit training passes (IVF-PQ — mod-104 chapter 5).
6. **Write the manifest** (mod-104 chapter 8).
7. **Hand to chapter 2's cutover automation** — `verify` runs the health checks, `flip` moves the alias, `cool-down` waits, `drop v1` at the end.

Every step is an asset in Dagster / a task in Airflow / a step in Prefect. Every step's output is persisted, so a failure in step 5 does not require re-running steps 1–4.

## The embedding pass — retry, batch, cost

The embedding pass deserves its own subsection because it is the dominant cost and the most-common source of failure.

- **Batch size.** Hosted embedding APIs accept batches (OpenAI's `/v1/embeddings` accepts up to a configurable limit per request; Voyage, Cohere, and hosted models on Vertex / Bedrock each have their own limits). Pick a batch size close to the vendor's max — fewer requests means less overhead — but not so large that a single failed batch retries an expensive amount of work.
- **Concurrency.** Multiple in-flight batches speed up wall-clock but risk rate-limits. Start conservative (e.g. 4–8 concurrent), watch for HTTP 429 responses, back off.
- **Retry with exponential backoff.** Standard library — the [`tenacity`](https://tenacity.readthedocs.io/) package is a common choice — or the orchestrator's built-in retry primitive. Idempotency comes from the fact that embedding the same text twice produces the same vector (for a fixed model).
- **Checkpointing.** For a rebuild that takes hours, save partial results every N minutes. If the pipeline dies at hour 3 of a 5-hour rebuild, resume from the last checkpoint, not from scratch.
- **Cost accounting.** Emit the number of tokens embedded per batch to metrics. Chapter 5's cost model reads this.

Every mature pipeline has *all five* of these — skipping any is a runbook away from a bad night.

## Backfill runs

A backfill is a rebuild triggered by *changing the tuple*, not by corpus deltas. Examples:

- Upgrading the embedding model from `bge-large-en-v1.5` to `bge-large-en-v1.6`.
- Rechunking with a new chunker version.
- Migrating to a new schema.

The shape is the same as a full rebuild, plus:

- **A ship gate.** Never flip a backfilled index without running the retriever CI's judgment-set eval (chapter 3) *against the backfill result at full corpus*, not just against the eval slice. A backfill that passed CI on 10k chunks may regress on 10M — always confirm at scale.
- **A cost estimate up front.** Chapter 5's cost model tells you what a backfill will cost. A `bge-large` re-embed of 10M chunks at ~$0.00013 per 1K tokens and ~250 tokens per chunk is ~$325 in embedding cost alone (illustrative — check current vendor prices). Get the sign-off before you start.
- **A staged shadow.** Run the backfilled index in shadow mode (chapter 2) for a warm-up period before the flip, especially when it changes the embedding model.

Backfills are the largest single-event cost in a retrieval pipeline and are often the least well-planned. Treat them as project-scale events, not weekly ops.

## The ingestion-lag SLO — the freshness contract

The ingestion-lag SLO is the promise from the retrieval pipeline to the rest of the org: *when a source document is updated, it will be retrievable within `L` time*.

Common shapes:

- **`L = 24 hours`** — nightly incremental. Common for corpora that update slowly (an internal wiki, a knowledge base with periodic docs updates).
- **`L = 1 hour`** — hourly incremental. Common for support-ticket corpora, log corpora, anything where fresh content matters within a business hour.
- **`L = 5 minutes`** — near-real-time. Common for chat / conversation corpora where a newly-sent message must be immediately findable. Requires event-driven ingestion (webhooks, change-data-capture) rather than polling.

The SLO shapes several downstream decisions:

- **Trigger cadence.** Nightly / hourly / event-driven. The orchestrator's [schedule](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/timetable.html) or [sensor](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html) picks up the delta.
- **Wall-clock budget for the incremental pipeline.** Chapter 2's verify + flip take some time; the ingestion pipeline itself takes some time; the incremental run must complete within `L − (verify_time + flip_time + safety_margin)`.
- **Alerting.** An `ingestion_lag` gauge — the time between the newest document in the corpus and now — is a first-class metric emitted by the orchestrator. Alert when it exceeds `L`.

Every SLO also has a corresponding error budget — how much of the month is the pipeline allowed to violate the SLO. Read the Google SRE book chapter on SLOs for the general shape; specialize `L` and the budget for your product.

## Orchestrator choice — brief

The orchestrator is not the interesting part of this chapter (`ai-infra-mlops` teaches you how to author DAGs in each). A short summary of the tradeoffs specific to retrieval:

- **[Airflow](https://airflow.apache.org/)** — mature, huge community, mature operator ecosystem. The [`KubernetesPodOperator`](https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html) is the common way to run embedding batches. Weakness: asset-oriented reasoning is bolted on ([datasets](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html) landed in 2.4) — Dagster does it more naturally.
- **[Dagster](https://dagster.io/)** — asset-oriented from the start. The asset graph above maps directly to Dagster's model. The retrieval graph looks like a Dagster demo. Weakness: smaller ecosystem than Airflow.
- **[Prefect](https://www.prefect.io/)** — Pythonic, low-ceremony, good local dev experience. Weakness: less asset-oriented than Dagster; scale story less battle-tested than Airflow.
- **AWS Step Functions / EventBridge / Batch** — cloud-native, deeply integrated with AWS. Good if the rest of the stack is AWS.
- **GCP Vertex AI Pipelines / Workflows** — cloud-native on GCP. Vertex AI Pipelines runs [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) under the hood.
- **Databricks Workflows / Azure Data Factory** — reasonable defaults if you already live in Databricks / Azure.

Pick what the rest of the org uses. Retrieval is not weird enough to justify a bespoke orchestrator; every mature system has retrieval pipelines built on the mainstream ones.

## Concrete example — a Dagster asset graph shape

Illustrative (Dagster syntax; the shape is portable):

```python
# assets.py (illustrative)
from dagster import asset, AutoMaterializePolicy, FreshnessPolicy

@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=60))
def cleaned_docs(context, raw_corpus): ...

@asset
def chunked_docs(cleaned_docs, chunker_config): ...

@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=60))
def embedded_chunks(chunked_docs, embedding_config):
    # Batched embedding pass with retry / checkpointing.
    ...

@asset
def payload_records(embedded_chunks, schema_config): ...

@asset(auto_materialize_policy=AutoMaterializePolicy.eager())
def built_index(payload_records, ann_config, tuple_manifest):
    # Emits an index_id + writes the manifest.
    ...

@asset(deps=[built_index])
def cutover_verified(built_index):
    # Chapter 2's health checks.
    ...

@asset(deps=[cutover_verified])
def cutover_flipped(cutover_verified):
    # Chapter 2's flip script.
    ...
```

Dagster's [freshness policies](https://docs.dagster.io/concepts/assets/asset-materialization#freshness-policies) map directly to the ingestion-lag SLO. Dagster's [auto-materialize policy](https://docs.dagster.io/concepts/assets/auto-materialize) drives the "run incremental when upstream changes" behavior. Airflow's equivalent shape is a DAG with `Dataset`-driven scheduling; Prefect's is a flow with `depends_on`.

The point of the example is not that Dagster is the answer — it's that once you have the asset graph, every orchestrator has a way to run it.

## Reprocessing runs — the "we changed the parser" case

Sometimes the *chunker* didn't change but the *cleaner* did (an HTML parser fix, a PDF layout heuristic change). The clean-upstream now produces different `cleaned_docs`, which produce different `chunked_docs`, which change every downstream vector.

Reprocessing is a subclass of backfill:

- Mark the affected upstream asset as stale.
- Let the asset graph re-materialize downstream from that point.
- Backfill's ship gate still applies — evaluate at full scale before promoting.

Dagster and Airflow both surface "these downstream assets are stale because this upstream changed" as a first-class query on the graph, which is the operational reason to prefer asset-oriented orchestrators for retrieval work.

## Two failures the automation defends against

### Failure A: the silent partial rebuild

The nightly rebuild crashed on step 4 (payload records) with a parser exception on 3% of documents. Someone silenced the alert, the rebuild "finished" (with 97% of documents), the flip happened, and 3% of the corpus is now unretrievable. Users complain about specific missing documents; no one connects it to the rebuild.

**Defense.** The DAG's step 4 output is compared against the input count. A drop of >0.5% fails the step. `structural sanity` in chapter 2's verify catches whatever slips through — `count(vectors)` in the new index vs. `manifest.corpus_snapshot.n_chunks` from step 1.

### Failure B: the embedding-pass rate-limit that fell over on Friday night

The hosted embedding API rate-limited at 3 AM Saturday. The pipeline had no backoff; the retry storm looked like an attack; the vendor throttled harder; wall-clock ballooned; the rebuild missed the ingestion-lag SLO by 8 hours.

**Defense.** Exponential backoff, jittered retries, and a *slow* concurrency ramp-up (start at 2 concurrent batches, grow only if latency is stable). Combined with a *cost accounting* that would have surfaced the retry storm as an unusual token spend on the vendor dashboard — chapter 5's cost dashboard is the second half of this defense.

## What you should be able to do after this chapter

- Sketch the retrieval index's asset graph (raw_corpus → cleaned → chunked → embedded → payload → built) and name what each arrow depends on.
- Read the incremental-vs-full trigger matrix and answer for a given change ("we bumped `efConstruction` from 200 to 400") which shape it triggers.
- Describe the embedding-pass's five must-have properties (batch, concurrency, retry, checkpoint, cost account) and why skipping any is a runbook away from a bad night.
- Author an ingestion-lag SLO for a specific corpus (with a value and a rationale) and translate it into an orchestrator schedule / sensor / freshness policy.
- Distinguish incremental, full rebuild, backfill, and reprocessing runs; know when each shape applies.
- Point at Dagster's asset abstraction, Airflow's datasets, and Prefect's flows as the three mainstream orchestrator hosts for this graph, without opinionating on which.

## Further reading

- Apache Airflow: [*Datasets and data-aware scheduling*](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html) — Airflow's asset-oriented shape.
- Dagster: [*Software-defined assets*](https://docs.dagster.io/concepts/assets/software-defined-assets) and [*Freshness policies*](https://docs.dagster.io/concepts/assets/asset-materialization#freshness-policies) — the asset graph and the SLO.
- Prefect: [*Flows*](https://docs.prefect.io/latest/concepts/flows/) and [*Deployments*](https://docs.prefect.io/latest/concepts/deployments/) — the Prefect shape.
- Google SRE Book, Ch. 4 (*Service Level Objectives*). [Full text](https://sre.google/sre-book/service-level-objectives/) — the general SLO framework specialized here as the ingestion-lag SLO.
- Kubeflow Pipelines documentation. [`kubeflow.org/docs/components/pipelines/`](https://www.kubeflow.org/docs/components/pipelines/) — the pipeline abstraction Vertex AI Pipelines uses.
- Chip Huyen (2022). *Designing Machine Learning Systems.* Chapter 10 (Infrastructure for ML) — general treatment of the pipeline / orchestrator layer this chapter specializes.
- OpenAI [Rate-limits](https://platform.openai.com/docs/guides/rate-limits) documentation — a specific case of the vendor-side constraint the embedding pass must respect.
- `tenacity` docs. [`tenacity.readthedocs.io`](https://tenacity.readthedocs.io/) — the go-to retry-with-backoff library for Python embedding batches.
- The [`ai-infra-mlops`](https://github.com/ai-engineering-curriculum) curriculum — the foundational track that teaches the orchestrators this chapter assumes.
