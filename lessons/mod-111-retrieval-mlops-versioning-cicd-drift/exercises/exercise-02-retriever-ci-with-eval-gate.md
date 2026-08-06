# Exercise 02 — Retriever CI with an Eval Gate

**Estimated effort:** ~3 hours. Python, a GitHub or GitLab repo, GitHub Actions (or GitLab CI / Buildkite), the mod-109 judgment set, and the artifacts from exercise 01 (build/verify/flip scripts + a corpus slice).

## Objective

Build the retriever CI pipeline from chapter 3: a workflow that triggers on the right file changes, rebuilds an index on an eval-corpus slice, runs the mod-109 judgment-set eval, compares to a baseline, and gates the merge on a recall@k / nDCG@10 regression threshold with paired significance. Post the metric report as a PR comment on every run.

By the end you have a workflow that:

- Fires only on PRs that touch retrieval-relevant paths.
- Caches the built index by tuple string so repeat runs are fast.
- Emits a PR comment with the metric triple, deltas, CIs, and gate verdict.
- Blocks merge on a significant regression; allows merge on non-significant deltas.
- Runs end-to-end in under the wall-clock budget you set.

## Prerequisites

- Chapters 2 and 3 read.
- mod-109 chapters 3 and 4 read; exercise 01 done (a real judgment set).
- Exercise 01 of this module done (build/verify scripts available).
- A GitHub repo you can push to, with Actions enabled. GitLab / Buildkite work with parallel wording.
- Python 3.10+, `pip install ir_measures scipy`.

## Steps

**Step 1 — Baseline snapshot on `main` (20 minutes).**

- On the `main` branch of your repo, commit an eval-corpus slice (~1–5k chunks) at `eval/corpus/`.
- Commit your judgment set at `eval/qrels.tsv` and `eval/queries.jsonl`.
- Run the build + eval once, save the resulting per-query values as `eval/baseline/main.per-query.csv` and the aggregates as `eval/baseline/main.json`. Commit.

The baseline is what the CI will compare each PR to.

**Step 2 — Author `.ci/tuple.py` (15 minutes).**

Reads the repo's current configs (`configs/embedding.yaml`, `configs/chunker.yaml`, `configs/schema.yaml`, `configs/retriever.yaml`, and the current commit hash of chunker/retriever source code) and constructs the tuple string per mod-104 chapter 8's naming convention. Outputs the tuple as a GitHub Actions `$GITHUB_OUTPUT` line:

```
tuple_id=hr_wiki__bge-base-en-1.5__chunker-3__schema-1__snap-2026-08-01__hnsw-M16-ef100
```

**Step 3 — Author `.ci/build_slice_index.py` (25 minutes).**

- Takes `--tuple <tuple_id>` and `--corpus eval/corpus/`.
- Builds a small index for that tuple over the eval corpus.
- Persists the index to `.cache/index/<tuple_id>/`.
- Skips if the cache is populated already (the cache action in the workflow does the restore; this script just no-ops if the target dir exists).

Reuse exercise-01's build logic.

**Step 4 — Author `.ci/run_eval.py` (20 minutes).**

- Takes `--tuple <tuple_id> --qrels eval/qrels.tsv --queries eval/queries.jsonl --out out/pr-eval.json`.
- Loads the index for that tuple, runs each query, produces a TREC run file at `out/pr.run`.
- Computes per-query metrics (`nDCG@10, RR@10, R@50`) via `ir_measures`.
- Writes both `out/pr.run` and `out/pr-eval.json` (aggregates + per-query arrays).

**Step 5 — Author `.ci/gate.py` (35 minutes).**

The gate is the whole point of the exercise. Takes `--pr out/pr-eval.json --baseline eval/baseline/main.json --thresholds configs/ci-thresholds.yaml --pr-number N`.

Logic:

1. Load PR aggregates + per-query arrays.
2. Load baseline aggregates + per-query arrays (from `eval/baseline/main.per-query.csv`).
3. For each metric (`nDCG@10, RR@10, R@50`):
   - Compute delta = PR − baseline.
   - Compute paired Wilcoxon `p` on per-query deltas (mod-109 chapter 4; `scipy.stats.wilcoxon`).
   - Compute 95% BCa bootstrap CI on the mean delta (`scipy.stats.bootstrap`; 10,000 resamples is fine).
4. Apply the gating rules from chapter 3:
   - **Hard fail** if |delta| > `ε_hard` (from `configs/ci-thresholds.yaml`).
   - **Soft fail** if delta < 0 AND `p < 0.05` AND the override label is not present on the PR.
   - **Pass** otherwise.
5. Post a Markdown comment on the PR with the chapter-3 report shape (metric table + verdict + a link to `out/pr-eval.json`).
6. Exit code: 0 on pass, non-zero on fail.

Use the GitHub REST API (via [`gh`](https://cli.github.com/) or `requests`) to post the comment. Auth via the `GITHUB_TOKEN` the Actions runtime provides.

**Step 6 — Author `configs/ci-thresholds.yaml` (5 minutes).**

Pick your `ε_hard` values by the chapter 3 procedure:

- Run baseline eval 5–10 times with different random seeds if any part of the pipeline has stochasticity; compute std-dev per metric.
- Set `ε_hard = 3 × std-dev`.
- If your pipeline is deterministic, pick sensible defaults (e.g., `ε_hard_recall = 0.02`, `ε_hard_ndcg = 0.01`) and note in a comment that you'd re-derive with the noise measurement if it existed.

Commit the file with a comment explaining how the thresholds were picked and when they were last reviewed.

**Step 7 — Wire the GitHub Actions workflow (25 minutes).**

`.github/workflows/retriever-ci.yml`:

```yaml
name: retriever-ci
on:
  pull_request:
    paths:
      - "src/chunkers/**"
      - "src/retriever/**"
      - "configs/chunker.yaml"
      - "configs/embedding.yaml"
      - "configs/retriever.yaml"
      - "eval/qrels.tsv"
      - "eval/queries.jsonl"

jobs:
  eval:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write   # to post the PR comment
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.10"}
      - name: Install deps
        run: pip install -r requirements.txt

      - name: Compute tuple
        id: tuple
        run: python .ci/tuple.py >> $GITHUB_OUTPUT

      - name: Restore index cache
        uses: actions/cache@v4
        with:
          path: .cache/index/${{ steps.tuple.outputs.tuple_id }}
          key:  index-${{ steps.tuple.outputs.tuple_id }}

      - name: Build index if not cached
        run: python .ci/build_slice_index.py --tuple ${{ steps.tuple.outputs.tuple_id }} --corpus eval/corpus/

      - name: Run eval
        run: python .ci/run_eval.py --tuple ${{ steps.tuple.outputs.tuple_id }} \
                                    --qrels eval/qrels.tsv --queries eval/queries.jsonl \
                                    --out out/pr-eval.json

      - name: Gate
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python .ci/gate.py --pr out/pr-eval.json \
                                --baseline eval/baseline/main.json \
                                --thresholds configs/ci-thresholds.yaml \
                                --pr-number ${{ github.event.pull_request.number }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: eval-pr-${{ github.event.pull_request.number }}
          path: |
            out/pr.run
            out/pr-eval.json
```

**Step 8 — Drive three PRs (25 minutes).**

Open three PRs, watch the workflow on each:

1. **A pass-clean PR:** touch `configs/retriever.yaml` in a harmless way (e.g., a comment). The workflow should run, gate should pass, comment should show deltas near zero.
2. **A soft-fail PR:** change a retriever config so the metrics dip slightly but not by more than `ε_hard`. Confirm the gate flags a soft-fail if the paired Wilcoxon reports `p < 0.05`; confirm the override label unblocks.
3. **A hard-fail PR:** change something egregious (drop `top_n` from 50 to 5, or downshift the embedding model). Confirm the gate hard-fails; confirm the override label does NOT unblock.

Record the three PR URLs (or diffs and workflow-run URLs) in your write-up.

**Step 9 — Add cost/wall-clock instrumentation (10 minutes).**

- Log `time.time()` at start and end of each of the four scripts.
- Emit a summary line at the end of the gate script showing wall-clock breakdown.
- Add a small note in the PR comment: "CI wall-clock: X seconds (build cached: yes/no)."

Confirm the tuple-cache hit path is dramatically faster than the miss path.

## Acceptance criteria

- `.github/workflows/retriever-ci.yml` triggers on the right paths and no others.
- The workflow runs end-to-end and completes on a small eval slice.
- A pass-clean PR gets a "pass" comment.
- A soft-fail PR gets a soft-fail comment with the override-label instructions; the override label lets the PR through.
- A hard-fail PR gets a hard-fail comment and blocks merge (branch-protection rule enabled).
- The index cache is used on repeat runs (verified by two consecutive runs on the same tuple).
- The PR comment includes the metric triple, deltas, CIs, p-values, and the verdict, in a scannable table.
- The tuple-cache path completes in dramatically less wall-clock than the cold path (record actual numbers).
- `configs/ci-thresholds.yaml` documents how `ε_hard` was picked.

## Stretch goals

- **Per-slice reporting.** If your judgment set is stratified (mod-109 stretch goal), report per-slice deltas and gate per-slice. Catch regressions that the aggregate hides.
- **Two-lane CI.** A fast lane (30-query subset) runs first; if it fails, the full lane doesn't run. Post both.
- **Retriever-only re-rank lane.** For PRs that touch *only* the reranker, use the cached retrieval output and only re-rank; drop wall-clock further.
- **Slack notification on failure.** Hook the gate script to post to a Slack channel on hard-fail, with the PR link and the failing metric.
- **Nightly baseline refresh.** Add a scheduled workflow that re-runs the eval on `main` and updates the baseline artifact. The PR workflow reads the freshest baseline.
- **Multi-metric override policy.** Allow the override label to unblock a soft-fail on *one* metric but not two. Discuss the trade-off in your write-up.

## Notes

- **Do NOT install RAGAS / TruLens / DeepEval and gate on faithfulness in this CI.** That crosses the boundary to `ai-eval-engineer`. Chapter 8 explains the failure mode. This CI is retrieval-only.
- **Do NOT gate on latency in this CI.** CI runners are noisy; latency gating belongs to chapter 2's cutover verify on production-shaped infra. See chapter 3.
- **The tuple string is the cache key.** If it isn't correct, the wrong index gets reused. Verify by hand on the three PRs.
- **The PR comment is the artifact reviewers will actually read.** Optimize it for scan-in-5-seconds: the verdict is the first thing, the numbers are the second, the CI-internals link is last.
- **Do not skip the noise-measurement step for `ε_hard`.** Thresholds picked without measured noise either false-fail (training the team to override reflexively) or silently ship regressions.
