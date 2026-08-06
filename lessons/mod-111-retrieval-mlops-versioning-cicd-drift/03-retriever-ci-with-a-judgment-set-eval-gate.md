# Chapter 3 — Retriever CI with a Judgment-Set Eval Gate

## Why this chapter exists

mod-109 built the judgment set and the metric harness. mod-104 chapter 8 named the tuple that identifies an index. Chapter 2 of this module scripted the cutover. This chapter is the piece that makes those artifacts *unskippable*: a CI job that runs every PR that touches the chunker, the embedding config, the retriever, or the reranker, evaluates the resulting retriever against the pinned judgment set, and blocks merge if the retrieval-quality metrics regress beyond a defined threshold.

Without this CI, the judgment set is a nice-to-have — someone runs it before the ship, quality regresses the sprint after, no one is blamed because no one owned "run the eval on every change." With the CI in place, the eval is the *default* — any retrieval-affecting PR either shows the numbers moved in the right direction or explains why the merge is warranted anyway. The judgment set becomes a load-bearing part of the merge process.

Boundary reminder: this chapter builds the *retrieval-specific* CI gate. `ai-infra-mlops` teaches you GitHub Actions / GitLab CI / Buildkite in general; this chapter says which PRs trigger the gate, which metrics gate it, and how the wall-clock budget stays sane.

## What triggers the CI — the path-based filter

The CI runs on PRs whose diff touches at least one of:

- **Chunker code** (`ingest/chunker/**`, `src/chunkers/**`, whatever your path is). Changes the chunks that go into every future embedding.
- **Chunker config** (`configs/chunker.yaml` — anything that changes chunk size, overlap, hierarchy policy).
- **Embedding config** (`configs/embedding.yaml` — model, revision, prefix, normalization).
- **Retriever code / config** (`src/retriever/**`, `configs/retriever.yaml` — score function, hybrid fusion weight, RRF `k`, top-N).
- **Reranker code / config** (`src/reranker/**`, `configs/reranker.yaml`).
- **The judgment set itself** (`eval/qrels.tsv`, `eval/queries.jsonl`) — because a change to the eval set changes the meaning of "regression."

GitHub Actions [`paths` / `paths-ignore` filters](https://docs.github.com/en/actions/using-workflows/triggering-a-workflow#using-filters), GitLab CI's [`rules:changes`](https://docs.gitlab.com/ee/ci/yaml/#ruleschanges), and Buildkite's [monorepo path-based triggers](https://buildkite.com/docs/pipelines/monorepo) all support this. Keep the filter narrow — every PR that runs the eval CI competes for the same wall-clock and cost budget, so runs that don't need it should skip.

Explicitly *not* triggering the CI: PRs that touch only unrelated services, docs-only PRs, and PRs to non-retrieval configs. The path filter is the first cost lever.

## What the CI runs — a five-step pipeline

Every triggered CI run does the same five steps.

### Step 1 — Detect what changed and pick the tuple to build

The CI reads the diff and constructs a *proposed* index tuple `(embedding-model, chunker-version, schema-version, corpus-snapshot, ANN-config)`:

- `embedding-model` — from `configs/embedding.yaml`.
- `chunker-version` — from `configs/chunker.yaml`, plus the commit hash of the chunker code.
- `schema-version` — from `configs/schema.yaml`.
- `corpus-snapshot` — the eval corpus snapshot pinned in `eval/corpus-manifest.md`. This is *not* the production corpus; it is a stable, small (chapter 4 goes into how small) representative slice.
- `ANN-config` — from `configs/retriever.yaml`.

The tuple string is the build-cache key (step 2).

### Step 2 — Build (or reuse) the index on the eval corpus slice

Building the full production index in CI is a wall-clock killer. Instead:

- **Use a fixed, small eval corpus slice.** ~1–10k chunks is the range most teams pick. Small enough to build in minutes; large enough that recall@50 numbers are stable.
- **Cache the built index by tuple.** GitHub Actions `actions/cache`, Buildkite artifact cache, or a plain S3 read-through cache keyed on the tuple string. A PR that only changes reranker config reuses the base index — CI wall-clock drops from tens of minutes to seconds.
- **Persist the built index only as long as the PR is open.** Old caches age out; a merged PR's cache is promoted to a longer-lived one; abandoned PR caches expire fast.

The corpus-snapshot slice must be the same one the *baseline* run used — otherwise the metric delta compares apples to oranges. Chapter 4 covers how to pin the snapshot.

### Step 3 — Run the judgment-set eval

Load `eval/qrels.tsv`. Run every query through the retriever configured to hit the freshly-built (or cached) index. Compute the metric triple (mod-109 chapter 3):

```python
# ci_eval.py — the core of the CI eval step (illustrative)
import ir_measures
from ir_measures import nDCG, RR, R

qrels = ir_measures.read_trec_qrels("eval/qrels.tsv")
run   = ir_measures.read_trec_run("out/pr.run")

per_query = list(ir_measures.iter_calc(
    [nDCG@10, RR@10, R@50],
    qrels, run,
))
# Save per-query values to a JSON so Step 4 can significance-test.
```

Save the run file *and* the per-query metric values. Both are inputs to the next step; both are attached to the PR as artifacts.

### Step 4 — Compare against the baseline and gate

Load the *baseline*: the metrics from the current `main`-branch tuple, computed against the same qrels version. The baseline is either:

- The result of a nightly job that re-runs the eval on `main` and stores the numbers in an artifact store, or
- A cached run from the last merge into `main`, keyed by tuple string.

Compute the delta:

```
Δ recall@50 = recall@50(PR) − recall@50(baseline)
Δ nDCG@10   = nDCG@10(PR)   − nDCG@10(baseline)
Δ MRR@10    = MRR@10(PR)    − MRR@10(baseline)
```

Then apply the gating rules.

**Rule 1 — Hard fail on a large regression.** If any metric regresses by more than `ε_hard` (project-specific; e.g. `ε_hard_recall = 0.02`, `ε_hard_ndcg = 0.01`), the CI fails, period. No override.

**Rule 2 — Soft fail on a statistically significant regression.** If a metric regresses by less than `ε_hard` but the regression is statistically significant (paired Wilcoxon `p < 0.05` on per-query deltas — mod-109 chapter 4), the CI fails, with an *override label* the author can add to the PR (e.g. `retrieval-regression-approved`) to bypass. The override requires a comment justifying it (a stakeholder decision, a known-improved-elsewhere trade-off, etc.).

**Rule 3 — Pass on statistically-indistinguishable or positive deltas.** If the mean delta is positive, or the negative delta is not statistically significant, pass. Post the numbers as a PR comment either way.

Two dial-up options many teams use:

- **Dashboarded historical trend.** Post a link to a dashboard (Grafana, an internal service) that shows the metric over the last N merges. A per-PR pass with a slowly-drifting downward trend is a *bigger* signal than any single-PR regression.
- **Per-slice significance.** If the judgment set is stratified (query difficulty, topic, tenant), report deltas per slice. A wash on aggregate that hides a 20-point recall regression on one topic is a bug the aggregate hides.

### Step 5 — Post the report and either block or pass

Post a Markdown comment on the PR with the shape:

```markdown
### Retriever CI — mod-111 gate

**Judgment set:** `eval/qrels.tsv` @ `v2026-08-a` (120 queries)
**Baseline tuple:** `hr_wiki__bge-large-en-1.5__chunker-3__schema-2__snap-2026-07-15__hnsw-M32-ef200` (main)
**Proposed tuple:** `hr_wiki__bge-large-en-1.5__chunker-4__schema-2__snap-2026-07-15__hnsw-M32-ef200` (PR)

| Metric | Baseline | PR | Δ | 95% CI (BCa) | Wilcoxon p | Gate |
|---|---|---|---|---|---|---|
| recall@50 | 0.803 | 0.811 | +0.008 | [-0.003, +0.019] | 0.128 | pass |
| nDCG@10 | 0.487 | 0.481 | -0.006 | [-0.017, +0.005] | 0.253 | pass (not significant) |
| MRR@10 | 0.612 | 0.618 | +0.006 | [-0.008, +0.020] | 0.407 | pass |

**Verdict:** pass (no significant regression). Full per-query CSV: `pr-artifacts/per-query.csv`.
```

If a hard fail: the CI job exits non-zero and the merge is blocked by the branch-protection rule.

If a soft fail: the CI job exits non-zero *unless* the override label is present, and the PR comment says clearly *what to do* to override.

## Picking the regression thresholds

`ε_hard_recall`, `ε_hard_ndcg`, and `ε_hard_mrr` are project-specific. Pick them by:

1. **Measure the run-to-run noise of your eval.** Run the *same* baseline retriever against the *same* judgment set 10 times with different random seeds (or, if deterministic, on 10 rolling windows of the corpus). Compute the std-dev of each metric across runs.
2. **Set `ε_hard` to ~3× that std-dev.** Any smaller and you'll fail on noise; any larger and you'll ship real regressions. 3σ is a common heuristic; adjust up if false failures cost more than false passes and down if the opposite.

Publish the thresholds in the CI config *with a comment* explaining how they were picked and when they were last reviewed. Re-review them quarterly (or whenever the judgment set changes materially).

## What the CI must NOT do

- **Do not compute application-layer metrics.** RAGAS faithfulness, TruLens RAG triad, answer relevance — those belong to `ai-eval-engineer` (peer, level 30) and should run in a separate pipeline that consumes the same retriever's output. Two reasons: (a) they require a generator call, which multiplies wall-clock and cost; (b) they mix concerns — a retriever-quality regression and a generator-instability signal need different owners.
- **Do not gate on latency in this pipeline.** Latency is unstable on a shared CI runner (noisy neighbors, cold caches). The latency canary belongs to the cutover state machine (chapter 2's state B), running on production-shaped infrastructure.
- **Do not run against the full production corpus.** The CI's slice-only eval is a proxy for retrieval quality; the full-corpus verify happens in chapter 2's state B on real infrastructure. Two-layer defense.
- **Do not gate on aggregate mean without significance.** A 0.001 delta is not a regression, and gating on it will train the team to reflexively override. Only significant deltas fail.

## Cost and wall-clock — keeping the CI under budget

An eval CI that takes 45 minutes wall-clock per PR breaks the feedback loop. A few knobs, ranked by impact:

- **The corpus slice size.** Halving the slice halves the index-build time and roughly halves the embedding-API cost. Pick the smallest slice that still gives stable metric numbers on the noise-measurement run above.
- **The tuple-keyed cache.** A PR that changes only reranker config should reuse the entire built index. Cache hit rates of 60–80% are achievable if the cache key is precise.
- **Retriever-only for retriever-only changes.** A PR that touches only the reranker doesn't need to run the retriever eval — it can use the *cached* retrieval run from the baseline and re-rank only. This can drop the run to a fraction of its full-pipeline time.
- **Reduced query count for the fast lane.** A "fast" lane runs the CI on a 30-query subset for a first-pass sanity check; the full 120-query eval runs only if the fast lane passes. Post both results; block on the full eval.
- **Concurrency.** Split the eval across per-query shards if the runner supports it.

Whatever the shape, publish the CI's *typical wall-clock* in the repo README. Anyone whose PR hits an unusually slow run should know what "normal" looks like.

## Concrete example — GitHub Actions workflow shape

Illustrative (not a copy-paste template — engine-specific bits vary):

```yaml
# .github/workflows/retriever-ci.yml
name: retriever-ci
on:
  pull_request:
    paths:
      - "src/chunkers/**"
      - "src/retriever/**"
      - "src/reranker/**"
      - "configs/chunker.yaml"
      - "configs/embedding.yaml"
      - "configs/retriever.yaml"
      - "configs/reranker.yaml"
      - "eval/qrels.tsv"
      - "eval/queries.jsonl"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Compute tuple string
        id: tuple
        run: python .ci/tuple.py > $GITHUB_OUTPUT

      - name: Restore index cache
        uses: actions/cache@v4
        with:
          path: .cache/index/${{ steps.tuple.outputs.tuple_id }}
          key:  ${{ steps.tuple.outputs.tuple_id }}

      - name: Build index if not cached
        run: python .ci/build_slice_index.py --tuple ${{ steps.tuple.outputs.tuple_id }}

      - name: Run eval
        run: python .ci/run_eval.py --tuple ${{ steps.tuple.outputs.tuple_id }} \
                                    --qrels eval/qrels.tsv --queries eval/queries.jsonl \
                                    --out out/pr-eval.json

      - name: Compare to baseline and gate
        run: python .ci/gate.py --pr out/pr-eval.json \
                                --baseline main-eval-artifact-url \
                                --thresholds configs/ci-thresholds.yaml \
                                --pr-number ${{ github.event.pull_request.number }}
```

The three key files are `.ci/tuple.py` (constructs the tuple string), `.ci/run_eval.py` (runs the mod-109 eval), and `.ci/gate.py` (compares to baseline, posts the PR comment, exits with the right code). Every team's version of these is short — a few hundred lines total.

## Two failures the CI defends against

### Failure A: the "seemingly harmless" chunker tweak that halved recall

A PR reduces the chunker's max chunk size from 512 tokens to 300 tokens because someone measured that shorter chunks improve one query's answer. Merged without eval. Weeks later, a broad complaint pattern shows up — questions whose answer spans two paragraphs are now split across chunks, and neither chunk retrieves.

The CI catches this on the PR itself. Retrieval on a 120-query judgment set will show `recall@50` dropping by several points (queries whose answers spanned the old-size boundary now split). The gate fails; the PR is discussed rather than merged.

### Failure B: the embedding model bump that "was fine on our test queries"

A PR bumps the embedding config to a newer model revision. The author tries five queries manually; they look fine. Merged. Two weeks later, one specific query slice — technical jargon-heavy questions — has a silent regression because the new model was trained on a corpus that under-represented that jargon.

The CI catches this if the judgment set is stratified (mod-109 chapter 2's advice) — the per-slice report shows the regression on the jargon slice even when the aggregate looks flat. Fail on that slice; discuss on the PR.

## What you should be able to do after this chapter

- List the file-path filters that trigger the retriever CI on your project.
- Describe the five-step CI pipeline (tuple → build → eval → compare → post) and name what each step outputs.
- Distinguish hard-fail thresholds (`ε_hard`) from significance-based soft-fail thresholds and know how to pick each.
- Enumerate at least three CI cost/wall-clock levers (slice size, tuple-keyed cache, per-slice fast lane, retriever-only vs. re-rank-only paths).
- State the boundary: what the retriever CI runs (retrieval-only metrics from mod-109) and what it delegates (application-layer metrics to `ai-eval-engineer`, latency canaries to chapter 2's state B).
- Author a PR-comment template that shows the metric triple, deltas, CIs, p-values, and the gate verdict in the shape a reviewer can scan in seconds.

## Further reading

- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM 2007. [DOI:10.1145/1321440.1321528](https://dl.acm.org/doi/10.1145/1321440.1321528) — the paired-test methodology this chapter's gate relies on.
- Sakai, T. (2014). *Metrics, Statistics, Tests.* [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — mod-109's anchor for significance testing; also the reference for regression-threshold choice.
- GitHub Actions: [*Trigger workflows on specific paths*](https://docs.github.com/en/actions/using-workflows/triggering-a-workflow#using-filters) — how to author the path-based filter.
- GitHub Actions: [*Caching dependencies to speed up workflows*](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows) — how to cache the built index by tuple.
- GitLab CI: [*`rules:changes`*](https://docs.gitlab.com/ee/ci/yaml/#ruleschanges) — GitLab equivalent of the path-based filter.
- Karpathy, A. (2019). *A Recipe for Training Neural Networks.* [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/) — the general "eval-on-every-change" discipline this chapter operationalizes.
- Continuous Delivery Foundation: [*CD for Machine Learning*](https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-ml-best-practices-sig/) — the general framing this chapter's retriever-specific gate specializes.
- `ai-eval-engineer` track — the peer's CI/gate for the *application-layer* RAG metrics that this chapter deliberately does not run.
