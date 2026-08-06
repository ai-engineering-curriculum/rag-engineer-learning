# Exercise 01 — Index Versioning and Dual-Index Cutover Drill

**Estimated effort:** ~2 hours. Python, a vector DB you can run locally (pgvector / Qdrant / Weaviate / Chroma), a small corpus (500–5,000 chunks), the mod-109 judgment set (or a stand-in of ~30 labeled queries), a config store that can atomically update (Redis key / a JSON file the process re-reads).

## Objective

Build the scripted dual-index cutover state machine from chapter 2 and run it through all five states plus one intentional discard and one intentional rollback. Produce the audit log, the manifest for each built index, and the retriever-side config-contract check.

By the end you have:

- `cutover/build.py`, `cutover/verify.py`, `cutover/flip.py`, `cutover/rollback.py`, `cutover/cleanup.py` — the five scripts.
- `cutover/manifest.py` — the manifest writer/reader.
- `retriever/service.py` — the retriever library that reads `INDEX_ID` at each request and enforces the schema-version contract.
- `audit.log` — the append-only log every script writes to.
- A recorded run through all five states, plus a discard, plus a rollback.

## Prerequisites

- Chapters 2 and 4 read; chapter 3 helpful (verify uses the eval).
- mod-104 chapter 8 read — the tuple, manifest, and alias-flip patterns are prerequisites.
- mod-109 exercise 01 done, or an equivalent judgment set of ~30–120 queries. If you don't have one, hand-author 20 queries against a corpus you can label yourself in an hour.
- A local vector DB. Recommended: **pgvector** or **Qdrant** — both let you script the alias/pointer flip cleanly (pgvector via a view, Qdrant via `update_aliases`). Weaviate and Milvus work too; Chroma requires a client-side config indirection because it has no engine-side alias.
- Python 3.10+, `pip install ir_measures`, plus the client for your vector DB.

## Steps

**Step 1 — Set up two tuples on the same corpus (20 minutes).**

Author `configs/tuples.yaml`:

```yaml
tuples:
  v1:
    embedding_model: <your model + revision>
    chunker_version: "1.0.0"
    schema_version: 1
    corpus_snapshot: "<date>"
    ann_config: {algorithm: hnsw, M: 16, efConstruction: 100}
  v2:
    embedding_model: <same model, same revision>
    chunker_version: "1.0.0"
    schema_version: 1
    corpus_snapshot: "<date>"
    ann_config: {algorithm: hnsw, M: 32, efConstruction: 200}  # <-- ANN config change
```

`v1` and `v2` differ only in ANN config. This is the simplest tuple change and the one that most clearly exercises the state machine.

Compute the tuple string for each per mod-104 chapter 8's naming convention. Write them into `configs/tuples.yaml` as `index_id_v1`, `index_id_v2`.

**Step 2 — Author `cutover/build.py` (20 minutes).**

- Reads a tuple from `configs/tuples.yaml`.
- Creates a new collection in the vector DB with the tuple's `index_id` as the collection name.
- Chunks the corpus (or reuses cached chunks), embeds each chunk, uploads to the new collection.
- Writes `manifests/{index_id}.json` following mod-104 chapter 8's manifest schema.
- Exits `0` on success, non-zero on any failure.
- Appends to `audit.log`: `{timestamp, action: "build", index_id, tuple, status}`.

Build `index_id_v1` and confirm it exists.

**Step 3 — Author the retriever-side config contract (20 minutes).**

`retriever/service.py`:

```python
class Retriever:
    def __init__(self, config, engine):
        self.config = config    # reads INDEX_ID and SCHEMA_VERSION from a mutable source
        self.engine = engine

    def search(self, query, top_k=10):
        index_id = self.config.get("INDEX_ID")            # re-read every request
        collection = self.engine.get_or_none(index_id)
        if collection is None:
            raise IndexNotAvailable(index_id)
        if collection.schema_version != self.config.get("SCHEMA_VERSION"):
            raise SchemaMismatch(index_id)
        return collection.search(query, top_k=top_k)
```

The config source can be a file that the process `stat()`s + re-reads on modification, a Redis key, or a database row. What matters is that a flip in step 4 changes the config without a service restart.

Set `INDEX_ID = index_id_v1`; confirm the retriever serves from `v1`.

**Step 4 — Author `cutover/verify.py` (25 minutes).**

Runs against a candidate `index_id`:

1. **Structural sanity.** Confirm the collection's vector count matches the manifest's `n_chunks` within 0.5%; confirm the embedding dim matches; confirm the distance metric matches.
2. **Retrieval eval.** Load your judgment set. Run the retriever pointed at the candidate `index_id`. Compute `recall@k`, `nDCG@10`, `MRR@10` (mod-109 chapter 3's `ir_measures` snippet is the template).
3. **Regression gate.** Compare to a baseline (`index_id_v1`'s eval). If any metric is worse by more than a project threshold (`ε_hard = 0.02` recall, `0.01` nDCG for this exercise), exit non-zero.
4. **Latency canary.** Run 20 canary queries; record p50 / p95; gate on p95 within 20% of baseline.
5. **Structural spot-checks.** 3–5 hand-picked queries whose top-1 result is a known doc-ID; assert.

On any failure, exit non-zero, log to audit.log, and don't proceed to flip.

**Step 5 — Author `cutover/flip.py` and drive the happy path (15 minutes).**

- Reads a target `index_id`.
- Executes the atomic flip (engine-specific — pgvector view, Qdrant alias, Weaviate alias, Chroma config indirection).
- Post-flip, verify the alias now points where it should.
- Appends to `audit.log`: `{timestamp, action: "flip", from, to, status}`.
- Publish the new `INDEX_ID` to the config source (Redis key / file).

Build `index_id_v2`. Run verify — it should pass. Run flip. Confirm the retriever now serves from `v2` (issue one query; check the retriever's emitted `INDEX_ID` metric).

**Step 6 — Force a discard (15 minutes).**

- Author a synthetic bad `v3`: same tuple as `v2` but populate the collection with corrupted vectors (embed random noise instead of real content, or drop 20% of chunks by exiting the build loop early).
- Run `cutover/build.py` → `v3`. Run `cutover/verify.py` → it should exit non-zero.
- Confirm `v3` was discarded (or marked as failed in the registry); confirm the retriever still serves from `v2`.
- Confirm audit.log shows the failed verify with the specific reason.

**Step 7 — Force a rollback (15 minutes).**

- With `v2` currently serving, run `cutover/rollback.py --to v1`.
- Confirm the retriever now serves from `v1` after the rollback (test with a query, check the emitted `INDEX_ID`).
- Confirm the audit.log shows the rollback with the alert reason field populated.
- Note in your write-up: what should have triggered the rollback in production (a drift alert, a user complaint, a quality dashboard signal — chapter 7 sources).

**Step 8 — Author the cleanup script and run through cool-down (10 minutes).**

- `cutover/cleanup.py --older-than <window>` drops indexes older than the cool-down window that are not currently serving.
- For this exercise, use a short window (10 seconds) so you can watch the cleanup fire.
- Run cleanup; confirm the discarded `v3` is gone and `v1` is gone (since we rolled to `v1` and then... wait — if we rolled to `v1`, `v1` is now serving; `v2` should be the drop candidate. Trace this carefully in your audit.log to make sure the cleanup script consults "who is serving?" before dropping anything.)

**Step 9 — Write up (10 minutes).**

In `cutover/README.md`, publish:

- The state machine diagram (chapter 2's ASCII is fine).
- The five scripts and what each does.
- A trace through your recorded run (happy path → discard → rollback → cleanup) with timestamps from `audit.log`.
- One paragraph on what would trigger each transition in production (chapter 7's drift alerts are the answer for rollback; chapter 3's CI is the answer for build).

## Acceptance criteria

- All five scripts exist and are runnable independently.
- A single command runs the happy path (build → verify → flip → cool-down → cleanup) end to end.
- A single command runs the discard case (build → verify → discarded; retriever unchanged).
- A single command runs the rollback (rollback → verify from rolled-to index → cleanup).
- The retriever service re-reads `INDEX_ID` on each request (verified by editing the config source live and issuing two requests without a restart).
- `audit.log` shows every state transition with timestamps and status.
- Manifests are written per index and are parseable JSON.
- `cutover/README.md` documents the state machine, the scripts, and the recorded runs.

## Stretch goals

- **Add a canary flip.** Instead of flipping 100% to `v2`, route a percentage of queries to `v2` (using a feature-flag library like [Unleash](https://www.getunleash.io/) or a hand-rolled deterministic hash on `query_id`). Verify quality on the canary before the full flip.
- **Add a shadow deploy.** Mirror live queries to `v2` in parallel with `v1`; log both results; compare offline. Requires the retriever to fan-out non-blocking.
- **Add the cache-invalidation hook.** Wire chapter 6's top-k cache; make the flip script trigger a purge; observe cache-hit-rate before/after in a dashboard panel.
- **Handle dim change.** Cook up a `v3` with a different embedding dim; verify the retriever library's schema check catches it *before* serving; confirm the flip script also refuses (defense in depth).
- **Compare vendors.** Repeat the drill on two different vector DBs (pgvector + Qdrant, say); compare how much of the code is common vs. vendor-specific.

## Notes

- **The whole point is that the flip is atomic and reversible.** If your rollback takes multiple steps that a human might get wrong at 3 AM, refactor until it's a single command.
- **Do not simulate the discard by just skipping verify.** Wire the actual failure path — an incomplete build, a corrupted collection, a genuinely regressed eval — so the state-machine hardening is real.
- **The audit.log is the artifact.** In production, this is what an incident review reads to answer "what changed between last-good and first-bad." Write it as if you'll be reading it at 3 AM.
- **Do not skip the config-contract test.** The "the flip took but the retriever didn't pick it up because it reads INDEX_ID at process start" bug is real and painful; verify the live-reread with a live retriever process before you call this done.
