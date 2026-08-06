# Chapter 2 — Scripted Dual-Index Cutover with Health Checks and Rollback

## Why this chapter exists

mod-104 chapter 8 defined the **convention** — an index is identified by the tuple `(embedding-model, chunker-version, schema-version, corpus-snapshot, ANN-config)`, one index per tuple, dual-index cutover as the deployment shape, manifest JSON as the source of truth. That chapter assumed you would run each state (build → verify → flip → cool-down) by hand.

This chapter turns the convention into **automation**. The goal is that a promoted index moves through the state machine without a human at each transition, that every state's health check has a machine-checkable pass/fail, that rollback is a scripted single command (not a lookup-and-hope), and that the `INDEX_ID` config contract is enforced by the retriever library, not by a runbook.

If you have not read mod-104 chapter 8, read it first — this chapter assumes the tuple, the manifest, and the vendor-specific alias mechanisms. Everything here is operational scaffolding on top.

## The five states as a scripted state machine

The mod-104 state machine, made concrete:

```
    ┌────────┐  build asset      ┌────────┐  eval passes?      ┌────────┐  alias flip      ┌────────┐
    │  A     │ ────────────────► │  B     │ ─────────────────► │  D     │ ────────────►    │  E     │
    │ steady │                    │ verify │                    │  flip  │                    │ cool-  │
    │        │  ◄──────────────── │        │  ◄──────────────── │        │  ◄─────────────── │ down   │
    └────────┘  rollback         └────────┘  discard v2        └────────┘  auto-rollback     └────────┘
                (v1 stays live)   (v1 stays live)               (window t_rb elapses; drop v1)
```

Each transition is a script — not a human. The scripts and their exit codes are the contract:

| From | To | Trigger | Exit-code contract |
|---|---|---|---|
| A → B (steady → build) | build script | corpus delta ≥ threshold OR PR that changes tuple lands OR schedule (chapter 4) | `0` = manifest written, `n` = build error |
| B → verify | verify script | build finished | `0` = all health checks pass, `n` = at least one failed |
| verify → discard | discard script | verify returned non-zero | fires alert; `0` on cleanup |
| verify → flip (D) | flip script | verify passed | `0` = alias moved, `n` = alias didn't move (retry / alert) |
| D → cool-down (E) | timer | flip completed | rollback available for window `t_rb` |
| E → drop v1 | cleanup script | `t_rb` elapsed with no auto-rollback | `0` = v1 dropped |
| E → rollback | rollback script | quality alert during window `t_rb` | `0` = alias moved back to v1 |

Every one of those scripts belongs in the repo, not in a runbook. The runbook cites them.

## Health checks in state B (verify) — the machine-checkable list

Verify is the state that separates dual-index-cutover from ship-and-pray. The health checks it runs are:

### 1. Structural sanity

- The engine reports `count(vectors)` within tolerance of `manifest.corpus_snapshot.n_chunks` (e.g. within 0.5% — allow for known-drop parser exceptions, log the rest).
- The embedding dim reported by the engine equals `manifest.embedding.dim`.
- The distance metric configured on the collection equals `manifest.embedding.distance`.
- The schema fields present on a sampled point match `manifest.schema.fields`.

These are cheap checks — one API call each — and they catch entire classes of build-time errors (silent doc drops, dim mismatch after a model bump, schema drift).

### 2. Retrieval eval on the pinned judgment set

Run the mod-109 judgment set against `index_v2` with the retriever configured to hit `v2`. Compute the metric triple `(recall@N, nDCG@10, MRR@10)`. Gate on:

- `recall@N(v2) ≥ recall@N(v1) − ε_recall`
- `nDCG@10(v2) ≥ nDCG@10(v1) − ε_ndcg`
- Paired significance: if the delta is negative, is it statistically distinguishable from noise? Use the paired Wilcoxon on per-query score deltas (mod-109 chapter 4). If a negative delta is significant, fail; if it is not significant, warn but pass.

`ε_recall` and `ε_ndcg` are the *regression thresholds*. Chapter 3 goes into how to pick them — for now, treat them as tuple-specific project constants (e.g. `ε_recall = 0.01`, `ε_ndcg = 0.005`).

### 3. Latency canary

Run a fixed canary query set (small — 20–50 queries) against `v2` and record p50 / p95. Gate on:

- `p95(v2) ≤ p95(v1) × (1 + δ_latency)` where `δ_latency` is the project's latency-regression budget (commonly 20% for a rebuild that changes ANN config, 10% otherwise).

The canary set is deliberately smaller than the judgment set — it exists to catch operational regressions (a new ANN param that blew up query time), not statistical ones.

### 4. Structural spot-checks

A short list of hand-chosen queries whose top result is *known* to be a specific doc-ID. These exist for two reasons: they catch schema-level drift that eval numbers can miss ("the query filters on `tenant_id` but the field is now `tenant`, so the top-1 is silently wrong"); and they catch tuple-typo bugs that are otherwise hard to notice ("we rebuilt with the wrong chunker version by accident").

Publish the spot-check set alongside the judgment set in the eval repo, versioned the same way.

### 5. Selective-predicate canaries

If the schema supports filtering (mod-104 chapter 7), run a small set of filtered queries — one per major predicate — and assert non-empty results. Filter-side regressions ("the new schema doesn't index `updated_at` for pre-filter") show up here and nowhere else.

## The `INDEX_ID` config contract — enforced at request time

mod-104 chapter 8 named the four requirements for the retriever library:

1. Read `INDEX_ID` from config at each request start (not process start).
2. Emit `INDEX_ID` on every log line and metric.
3. Refuse to serve if the config's `INDEX_ID` does not exist in the engine.
4. Refuse to serve if the engine's `schema_version` does not match the config's.

Making those enforceable at request time — not at deploy time — is what lets the flip be atomic. Concretely:

```python
# retriever.py (illustrative)

class Retriever:
    def __init__(self, config_source, engine):
        self.config = config_source   # provides INDEX_ID on read
        self.engine = engine

    def search(self, query, top_k=10):
        index_id = self.config.get("INDEX_ID")   # re-read every request
        collection = self.engine.get_or_none(index_id)
        if collection is None:
            raise IndexNotAvailable(index_id)     # 5xx; alerts fire
        if collection.schema_version != self.config.get("SCHEMA_VERSION"):
            raise SchemaMismatch(index_id)        # 5xx; alerts fire
        results = collection.search(query, top_k=top_k)
        emit_metric("retriever_hit", index_id=index_id, ...)
        return results
```

The `re-read every request` piece matters more than it looks. If `INDEX_ID` comes from a mutable config source (a Consul key, a Redis key, a database row, a feature-flag service), then the flip in state D updates the config value and every subsequent request switches to `v2`. No pod redeploy, no rolling restart, no cold cache on the retriever side.

The `re-read every request` piece also matters for rollback — the same mechanism used to flip forward is used to flip back.

## The alias-flip script — one per engine

mod-104 chapter 8 enumerated the vendor-specific alias APIs. The *script* is what makes them atomic in your automation:

```python
# flip_alias.py (illustrative — pgvector variant)

def flip_pgvector_view(view_name, target_table):
    """
    Atomic pointer flip for pgvector-backed indexes.
    Single DDL; no window of half-updated state.
    """
    execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {target_table};")
    audit_log(action="flip", view=view_name, target=target_table)
```

Equivalent shapes for each engine:

- **Weaviate**: [`alias.update`](https://weaviate.io/developers/weaviate/manage-data/collections#alias) with the new class name.
- **Qdrant**: [`update_aliases`](https://qdrant.tech/documentation/concepts/collections/#collection-aliases) with an atomic `CreateAlias` + `DeleteAlias` in the same call.
- **Elasticsearch / OpenSearch**: [`POST _aliases`](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html) with an `add` and a `remove` in the same body — atomic per the docs.
- **Milvus**: [`alter_alias`](https://milvus.io/docs/create_alias.md).
- **Pinecone**: no engine-side alias; the flip is client-side config, so make sure the "flip" is a single write to the config source.
- **Chroma (OSS)**: no engine-side alias; same as Pinecone — flip the config source atomically.
- **FAISS**: library — the pointer is a variable in your process; hot-swap via an atomic reference update, guarded by a lock.

Every script:

- Reads which target it should flip *to* from the CI's manifest of the promoted build.
- Executes the vendor-specific atomic API.
- Verifies post-flip that the alias now points where it should.
- Emits an audit-log entry (chapter 4's automation reads this).
- Exits non-zero if any step failed; the state machine handles the failure.

## The rollback script

The rollback script is *the same script* as the flip, with the target set to the last-good `INDEX_ID`. Two properties matter:

1. **Cheap to invoke.** A single command with no extra flags — a human under alert-fatigue-conditions must not have to reason about the invocation.
2. **Idempotent.** Running it twice is a no-op; if the first invocation partially completed, the second finishes it.

Rollback is *not* the same as discard-in-verify. Rollback happens in state E (cool-down) — `v2` is already live; `v1` is warm; production traffic has been on `v2` for minutes to hours. Discard happens in state B/verify — nothing shipped; the failed build is thrown away.

The distinction matters because rollback triggers a downstream cache-invalidation cascade (chapter 6): the query-embedding cache is unaffected, but the top-k result cache must be invalidated because the results it holds are keyed on `v2`, which is no longer serving.

## Cool-down window — how long?

The default is *long enough for the drift monitor (chapter 7) to catch a slow regression*.

- **24 hours** — for retrieval-only changes on a corpus that is queried in a single business-hours cycle. Two full cycles (48 h) is common.
- **7 days** — for changes that touch the reranker or the retriever's fusion, because rerankers are more likely to expose regressions on a weekly-repeating query pattern.
- **A full monitoring period** — for changes that also change the schema; schema regressions surface as user complaints, and complaint arrival is Poisson-slow.

During cool-down, `v1` continues to occupy storage. That is the price of instant reversibility. When cool-down elapses with no rollback signal, `v1` is dropped by the cleanup script; storage goes back to steady state.

## When the pattern breaks — three cases to plan for

### Case 1: the build changes the embedding dim

Old: 768-dim. New: 1024-dim. The engine's collection dim is set at create time and cannot be altered on most engines. This is a *destructive rebuild* — the new tuple gets a whole new collection with its own name, and the alias flip is the pointer that hides that from callers. Cool-down works normally; the old collection stays until dropped.

### Case 2: the build changes the schema in a way that breaks queries

If `tenant_id` becomes `tenant`, the retriever library's request-time `SCHEMA_VERSION` check fires and the retriever refuses to serve. The flip has effectively taken production down. Two mitigations:

- Ship the query-time change (rename in the query builder) *first*, in a compatibility mode that reads both names. Then flip the index. Then drop the old name.
- Or ship a **shadow deploy** (mod-104 chapter 8) — `v2` receives queries but does not return them — until the query-time code is proven ready.

### Case 3: the flip succeeded but a downstream cache is stale

The top-k result cache (chapter 6) is keyed on `(query, INDEX_ID)`. If a rogue caller has *not* passed `INDEX_ID` in the cache key, the flip returns old-index results after the flip. The retriever library's cache-key contract must include `INDEX_ID`. Chapter 6 goes into the invalidation pattern; make sure the flip script triggers whatever cache-purge hook the caches expose.

## The manifest, revisited — for automation

mod-104 chapter 8 defined the manifest JSON. The fields that this chapter's automation *reads*:

- `index_id` — the string that becomes the alias target.
- `eval_at_build.recall@10`, `eval_at_build.mrr@10` — the baseline for the health-check regression thresholds.
- `build.finished_at` — the timestamp the ingestion-lag SLO (chapter 4) compares against.
- `corpus_snapshot.snapshot_at` — the timestamp the "how stale is the current live index" query answers.
- `chunker.commit`, `embedding.revision`, `ann_config` — the fields that determine whether a PR touching any of them requires a rebuild (chapter 3's CI trigger reads these).

Make the manifest machine-parseable (JSON, one file per index) and version it alongside the index in whatever registry your team uses (MLflow's model registry, W&B, a plain S3 bucket with a naming convention).

## A concrete example — a state machine as pseudocode

Making it all fit together (illustrative, not runnable — chapter 4 shows the orchestrator-specific shape):

```python
# cutover.py — the automation the CI kicks off

def build_and_promote(tuple_spec):
    # State A → B
    index_id = build_index(tuple_spec)
    manifest = write_manifest(index_id, tuple_spec)

    # State B: verify
    checks = run_health_checks(index_id, baseline=current_live())
    if not checks.passed():
        discard(index_id)
        alert("verify failed", checks.report())
        return "discarded"

    # State D: flip
    flip_alias(alias="chunks", target=index_id)
    audit_log("flipped", index_id=index_id, prior=current_live())
    invalidate_top_k_cache()

    # State E: cool-down (async)
    schedule_cleanup(prior=current_live(), after=cool_down_window)
    return "promoted"

def rollback(alert_reason):
    prior = last_good_index_id()
    flip_alias(alias="chunks", target=prior)
    audit_log("rolled_back", to=prior, reason=alert_reason)
    invalidate_top_k_cache()
    return "rolled_back"
```

The two functions above are the entire operator interface for the cutover. Chapter 4 wires them into the orchestrator; chapter 3 wires the CI's judgment-set eval into `run_health_checks`.

## What you should be able to do after this chapter

- Enumerate the five states of the cutover state machine and name the script that owns each transition.
- List the five health-check families that run in state B (structural, retrieval-eval, latency canary, structural spot-check, selective-predicate canary) and explain what each catches.
- Explain the `INDEX_ID` config contract and why re-reading it every request (not every process start) is what makes the flip atomic.
- Distinguish rollback (state E, live traffic on `v2`) from discard (state B, `v2` never shipped) and name the downstream cache implication of each.
- Author a manifest-driven rollback command that a human under alert-fatigue can invoke correctly.

## Further reading

- mod-104 chapter 8 — the convention this chapter automates.
- Sato, D., Wider, A., & Windheuser, C. (2019). *Continuous Delivery for Machine Learning.* martinfowler.com. [Article](https://martinfowler.com/articles/cd4ml.html) — the general CD-for-ML deployment shape this chapter specializes to retrieval.
- Fowler, M. (2010). *BlueGreenDeployment.* [Article](https://martinfowler.com/bliki/BlueGreenDeployment.html) — the pattern the dual-index cutover instantiates.
- Google SRE Book, Ch. 8 (*Release Engineering*). [Full text](https://sre.google/sre-book/release-engineering/) — hermetic builds, gated rollouts, rollback as a first-class capability.
- Weaviate: [*Aliases*](https://weaviate.io/developers/weaviate/manage-data/collections#alias). Qdrant: [*Collection aliases*](https://qdrant.tech/documentation/concepts/collections/#collection-aliases). Milvus: [*Aliases*](https://milvus.io/docs/create_alias.md). Elasticsearch: [*Aliases*](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html) — the vendor references for the atomic-flip primitive.
- MLflow: [*Model Registry*](https://mlflow.org/docs/latest/model-registry.html) — the general artifact + manifest + stage pattern; this chapter's manifest is an instance of the same shape, and MLflow works fine as the registry backend for indexes if your team already uses it for models.
