# Chapter 8 — Index Versioning Conventions and Dual-Index Cutover

## Why this chapter exists

An index that returns great numbers on your judgment set today is a liability tomorrow if you cannot tell which build produced those numbers, cannot rebuild it deterministically, and cannot swap it into or out of production without an outage. Retrieval engineers who have owned an index for six months have all learned the same lesson: **the index is a versioned artifact keyed on more than "the code that was current at build time."**

This chapter codifies the convention every downstream module assumes: an index is identified by the tuple `(embedding-model, chunker-version, schema-version, corpus-snapshot, ann-config)`, deployed via a dual-index cutover pattern, and gated by a CI check that runs a recall regression on every PR that touches any part of that tuple. mod-111 deep-dives the CI machinery; this chapter is where the *convention* lives, because it is a pre-requisite for the operational patterns in mod-111 to make sense.

## What is actually in an index — the tuple that identifies it

At level 20, "the index" is a Chroma collection. At level 30 an index is:

```
    index_id = (embedding_model, chunker_version, schema_version, corpus_snapshot, ann_config)
```

Each field is load-bearing:

- **`embedding_model`** — the exact model + version + revision. `bge-large-en-v1.5` at Hugging Face commit `d4aa6907d3e50f7c...` is a different index from the same model retrained tomorrow. Hosted APIs (`text-embedding-3-large`) get a semver + provider name; the vendor sometimes silently re-versions, which is why you also record the *date* of the embedding pass.
- **`chunker_version`** — the chunker code that produced the passages that were embedded. Change the boundary policy, and every downstream chunk is different; the old index cannot answer new queries meaningfully.
- **`schema_version`** — the payload schema. Add a field to the payload, and old queries that filter on it will silently miss. Rename a field, and every filter clause needs updating.
- **`corpus_snapshot`** — the exact set of source documents. A snapshot ID (commit hash of the source repo, a corpus-ingest date, a Data Version Control tag) tells you what documents the index knew about.
- **`ann_config`** — `(M, efConstruction)` for HNSW; `(nlist, pq_m, pq_nbits)` for IVF-PQ; whatever the chapter-4/5 tuning settled on. Runtime knobs (`efSearch`, `nprobe`) are *not* in the tuple; they are per-request.

The name of a good index in a real system looks like:

```
    hr_wiki__bge-large-v1.5__chunker-v3__schema-v2__snap-2026-08-01__hnsw-m32-ef200
```

Long, but unambiguous. Every automation you build hangs off this string.

## Why the tuple has to be complete

Skipping any field bites you in exactly one way:

- **Skip `embedding_model`** and a silent model bump on the hosted side degrades retrieval overnight with no code change.
- **Skip `chunker_version`** and an "improve chunk boundaries" PR passes CI on a small dataset and destroys retrieval on production when the old chunks in the index no longer match the newly-chunked queries at retrieval time.
- **Skip `schema_version`** and a filter renamed from `tenant_id` to `tenant` starts returning empty results with no error.
- **Skip `corpus_snapshot`** and a "rebuild the index from source" job produces a numerically different index because upstream sources moved.
- **Skip `ann_config`** and you cannot answer "why is this index using so much more RAM than the last one?" without diffing configuration files by hand.

Every one of these has been a real production incident somewhere; the convention exists because the alternative is expensive.

## The one-index-per-tuple rule

For each unique tuple, build a *separate* index. Do not update in place. Do not "just change `efConstruction` on the live index." Two live indexes on the same corpus are fine; that is the dual-index pattern below.

The reason this is a rule and not a suggestion: **evaluation results are only comparable when the tuple is fixed**. If you change the embedding model *and* the ANN config in the same rebuild, you cannot tell which change moved recall. mod-111's CI walks changes one dimension at a time exactly so cause and effect stay attributable.

## The dual-index cutover pattern

Every production rebuild follows this shape.

**State A — Steady state.** Live traffic hits `index_v1`. The retriever library reads a config value `INDEX_ID = v1_id` at query time.

**State B — Build.** A batch job (Airflow / Dagster / Prefect) builds `index_v2` with the new tuple. This runs off the read path — it takes as long as it takes; production is unaffected. Storage during build is `size(v1) + size(v2)` (chapter 6 noted the doubling).

**State C — Verify.** Retriever CI (mod-111) runs the eval judgment set against `index_v2`:

- `recall@k` must be within a defined delta of `index_v1` (or better).
- `p95 latency` must be within a defined delta.
- All canary queries (including the selective-predicate canaries from chapter 7) must return non-empty.
- If any check fails, `index_v2` is discarded; alarms fire; a human decides.

**State D — Flip.** If the CI passes, the retriever's `INDEX_ID` flips from `v1_id` to `v2_id`. The flip is atomic (a single config change, or an alias update in the engine).

**State E — Cool-down.** `index_v1` stays warm for a rollback window (24 h – 7 d, project-dependent). If prod telemetry sees a quality regression the flip is reversible by flipping `INDEX_ID` back. Once the window closes, `index_v1` is dropped.

**Why the pattern.** Every state is reversible. Every state is observable. Nothing that fails CI reaches production. The cost is one index of extra storage during the build + cool-down window, and one CI job per rebuild.

## The alias / pointer flip — vendor-by-vendor

The atomic flip in State D depends on the engine:

- **pgvector.** A view over `chunks_v1`; `CREATE OR REPLACE VIEW chunks AS SELECT * FROM chunks_v2;` is the flip. Or a table-name lookup in the application config, updated via a deploy.
- **Weaviate.** [Collection aliases](https://weaviate.io/developers/weaviate/manage-data/collections#alias) — `alias("hr_wiki").set_class("hr_wiki_v2")` flips the pointer.
- **Qdrant.** [Collection aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases) — a rename plus alias creation gives you the same effect.
- **Elasticsearch / OpenSearch.** Index aliases — the [alias API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-aliases.html) is the reference; `POST _aliases` with an add/remove pair is atomic per alias.
- **Milvus.** [Aliases](https://milvus.io/docs/create_alias.md) — `create_alias("hr_wiki", "hr_wiki_v2")`, or `alter_alias`.
- **Pinecone.** Index names are the pointer; you either flip your client's config or use two indexes and switch client-side (Pinecone does not expose engine-side aliases for indexes as of this writing).
- **Chroma.** OSS Chroma does not have aliases; collection-name-in-config flip is the practical pattern.
- **FAISS.** Library — the pointer is a variable in your process; hot-swap it.

The point of the table is that **every vendor has some machinery for a client-transparent flip**. If yours does not, put a config-file indirection between the retriever and the engine.

## Green–blue vs. shadow deploys — two variants worth knowing

The dual-index pattern above is **green–blue**: two indexes exist, traffic flips over cleanly. Two close cousins are worth naming:

- **Shadow deploy** — `index_v2` receives a mirrored copy of live queries in parallel with `index_v1` for a warm-up window. Only `v1`'s results are returned to users; `v2`'s results are logged for offline comparison against `v1`. Useful when you want production-representative telemetry before the flip.
- **Canary flip** — a small fraction of live traffic is routed to `index_v2` for a warm-up window before the full flip. Useful when you want production-representative telemetry *plus* a real user-facing sanity check. Feature-flag machinery (LaunchDarkly, Statsig, Unleash, or a home-grown percentile router) drives the fraction.

Green–blue is the default; shadow and canary are options when the rebuild changes something risky (a new embedding model, a chunker rewrite).

## The retriever-library contract

The retriever library — the code that owns "given a query, return top-k chunks" — should:

1. **Read `INDEX_ID` from config at each request start.** Not at process start. That way an alias flip propagates without a redeploy.
2. **Emit `INDEX_ID` on every log line and metric.** So a p95 regression can be attributed to a specific build.
3. **Refuse to serve** if the config's `INDEX_ID` does not exist in the engine. Fail loud, not silent.
4. **Refuse to serve** if the engine's schema does not match the config's `schema_version`. Same reason.

These four are cheap to add on day one and impossible to retrofit after the first incident.

## The naming convention — writing it down

Adopt one convention across the codebase and never deviate. A working shape:

```
    {corpus}__{embedding_family}-{version}__chunker-{version}__schema-{version}__snap-{YYYY-MM-DD}__{algo}-{key-params}
```

Example:

```
    hr_wiki__bge-large-en-1.5__chunker-3__schema-2__snap-2026-08-01__hnsw-M32-ef200
```

The database's naming rules constrain the format (some allow hyphens, some do not; some cap the length). Standardize on double-underscore as the field separator so a shell one-liner can `cut` it apart. Store the full manifest — every field's value plus the exact commit hash of every piece of code — as a JSON blob alongside the index (chapter 6's "back up the built index" step).

## The manifest — the JSON that lives with every index

Every built index ships with a manifest:

```json
{
  "index_id": "hr_wiki__bge-large-en-1.5__chunker-3__schema-2__snap-2026-08-01__hnsw-M32-ef200",
  "corpus": "hr_wiki",
  "corpus_snapshot": {
    "source": "confluence://space/HR",
    "snapshot_at": "2026-08-01T04:12:00Z",
    "n_documents": 12482,
    "n_chunks": 89321
  },
  "embedding": {
    "model": "BAAI/bge-large-en-v1.5",
    "revision": "d4aa6907d3e50f7c...",
    "dim": 1024,
    "distance": "cosine"
  },
  "chunker": {
    "version": "3.1.0",
    "commit": "8a4c9f1..."
  },
  "schema": {
    "version": 2,
    "fields": ["doc_id", "section_path", "page", "updated_at", "permission_scope", "tenant_id"]
  },
  "ann_config": {
    "algorithm": "hnsw",
    "M": 32,
    "efConstruction": 200,
    "dtype": "float32"
  },
  "build": {
    "started_at": "2026-08-01T04:20:00Z",
    "finished_at": "2026-08-01T04:52:00Z",
    "hardware": "c6i.4xlarge",
    "peak_ram_gb": 8.2
  },
  "eval_at_build": {
    "recall@10": 0.947,
    "mrr@10": 0.812,
    "p95_ms": 32.7,
    "judgment_set": "hr_wiki_eval_v4"
  }
}
```

The manifest is the source of truth for "what is this index." It is what an incident review reads to answer "what changed between the last-good and first-bad build."

## What flags a rebuild

A rebuild is required when *any* field of the tuple changes:

- **Embedding model bump** (either the model or the revision). Absolutely a rebuild — every vector is stale.
- **Chunker change** (boundary policy, size, overlap, metadata attached to a chunk). Full rebuild — every chunk that was indexed is stale.
- **Schema change** (new field, renamed field, removed field). Full rebuild if the change affects what queries expect at query time; incremental if it is purely additive metadata that queries do not use yet.
- **Corpus refresh** (nightly ingest picked up new documents). Usually incremental; periodic clean rebuild per chapter 6's cadence policy.
- **ANN config change** (`M`, `efConstruction`, `nlist`, `pq_m`). Full rebuild.
- **`efSearch` / `nprobe` change.** No rebuild — these are per-request.

That six-item checklist is the pre-CI gate every retriever PR passes.

## The connection to retriever CI (mod-111 preview)

This chapter defines the convention; mod-111 defines the CI that enforces it. In short, a PR that touches the chunker, embedding config, or ANN config triggers:

1. A rebuild of the index on a small representative corpus slice.
2. An eval against the pinned judgment set.
3. A regression check: recall@k, MRR@k, p95 latency vs. the current-live tuple.
4. A pass/fail on statistical significance (paired randomization tests — Smucker, Allan & Carterette 2007 [DOI:10.1145/1321440.1321528](https://dl.acm.org/doi/10.1145/1321440.1321528)).
5. On pass, the promoted build is registered in the index registry and eligible for cutover.

You cannot build that CI without the tuple convention in this chapter — the CI's very first question is "which tuple am I building?" — so the convention is the foundation.

## The four failure modes this chapter prevents

1. **Silent embedding drift.** Hosted vendor bumps model minor version; retrieval quality changes; nobody notices because the index name did not change. The tuple's `embedding_model.revision` field catches it.
2. **Chunker drift.** A PR "improves" chunk boundaries; the live index still holds the old chunks; queries against new chunks retrieve stale-looking passages. The tuple's `chunker.version` field forces a rebuild.
3. **Schema drift.** A field is renamed in code; queries that filter on it silently return empty. The retriever library's schema-version check catches this at startup.
4. **Recall regression on cutover.** A rebuild degrades recall in a way the eval judgment set catches, but nobody ran the eval before the flip. The dual-index cutover + CI check makes the eval unskippable.

None of the four is a novel insight — every mature retrieval team encodes some version of this convention because the alternative is fighting these bugs in production.

## What is NOT in this chapter (and where it goes)

- **The CI machinery itself** (which orchestrator runs it, how the judgment set is versioned, how paired significance testing is implemented). mod-111 owns this.
- **Corpus / query / quality drift monitoring in production.** mod-111 also.
- **Semantic-versioning of the schema** (breaking-change vs. additive semantics). Application-level; live outside this module.
- **Cross-DC replication of the index during cutover.** Level-40 platform concern; chapter 9.

## What you should be able to do after this chapter

- Write out the five fields of the index tuple and give an example value for each on your own project.
- Draw the dual-index cutover state machine (steady → build → verify → flip → cool-down) and name the failure recovery at each transition.
- Author a manifest JSON for a hypothetical index build.
- Point to the mod-111 chapters that will pick up where this chapter leaves off.

The final chapter marks the boundary to `ai-infra-senior-engineer` (level 40) — the platform-scale concerns you should *know about* but not own.

## Further reading

- Weaviate docs: [*Collection aliases*](https://weaviate.io/developers/weaviate/manage-data/collections#alias).
- Qdrant docs: [*Collection aliases*](https://qdrant.tech/documentation/concepts/collections/#collection-aliases).
- Milvus docs: [*Aliases*](https://milvus.io/docs/create_alias.md).
- Elasticsearch docs: [*Aliases*](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html).
- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM 2007. [DOI:10.1145/1321440.1321528](https://dl.acm.org/doi/10.1145/1321440.1321528) — the paired-test methodology mod-111 builds on.
- Zaharia, M., et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin — for the general "artifact + manifest" pattern that this chapter's manifest is an instance of.
- MLflow docs, [*Model Registry*](https://mlflow.org/docs/latest/model-registry.html) — a mainstream implementation of the artifact-plus-manifest pattern (models rather than indexes, same shape).
