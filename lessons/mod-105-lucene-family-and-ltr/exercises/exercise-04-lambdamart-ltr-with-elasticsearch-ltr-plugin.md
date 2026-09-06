# Exercise 04 — LambdaMART LTR with the Elasticsearch LTR Plugin

**Estimated effort:** ~4 hours. Hands-on Elasticsearch (or OpenSearch) plus Python for training. Uses a labelled BEIR dataset for judgments.

## Objective

Take the lexical + hybrid index from Exercises 02 and 03 and layer a **LambdaMART** rescore model on top of it. Do it the way it is done in production: define a feature-set inside Elasticsearch (so training-time and serving-time features come from the same code path), log per-(query, document) feature vectors on a labelled training set, train a LightGBM LambdaMART model offline, upload it, and wire it into a `rescore` window over BM25 top-K. Measure the win with `nDCG@10` on a held-out test set.

The point of the exercise is not to squeeze the last two percentage points out of the model — it is to walk the full LTR pipeline end-to-end, produce evidence that the model beats the hand-tuned baseline on your data, and internalise the training/serving-skew discipline that keeps LTR models honest.

## Prerequisites

- Chapter 5 read.
- Exercises 02 and 03 complete.
- A running Elasticsearch 8.12+ single-node cluster with the built-in LTR (`_ltr/`) endpoints available. OpenSearch 2.10+ with the [`opensearch-learning-to-rank-base`](https://github.com/opensearch-project/opensearch-learning-to-rank-base) plugin is an equivalent path — pick one and stick with it.
- Python with `lightgbm >= 3.3`, `pandas`, `numpy`, and the ES/OS Python client of your chosen cluster.
- A labelled dataset. Two recommended picks:
    - **[BEIR SciFact](https://github.com/beir-cellar/beir)** — small (~5K docs, ~1K judged claims), English, hand-labelled qrels. Fast to iterate on.
    - **[MS MARCO passage](https://microsoft.github.io/msmarco/) small subset** — larger; use a 100K-doc slice. More realistic feature signal.

    Do not use your own corpus for this exercise unless you already have a labelled qrels file with ≥ 500 (query, doc) judgments. Small, hand-labelled sets do not train useful LTR models.
- The corpus embedded with the same model as Exercise 03 (so you have a cheap semantic-similarity feature to include).

## Steps

**Step 1 — Index the labelled corpus (~15 min).** Ingest the BEIR (or MS MARCO subset) corpus into a fresh index using the Exercise 02 mapping and analyzer, plus the Exercise 03 dense-vector field. Confirm a basic `multi_match` and a basic `knn` query both return sensible hits for a hand-picked query from the qrels.

**Step 2 — Split judgments train / valid / test (~10 min).** From the qrels, produce three disjoint **query** splits (not document splits — never split within a query). A 70/15/15 split of the query IDs is fine. Save them as `qids_train.txt`, `qids_valid.txt`, `qids_test.txt`. The test split does not get touched again until step 8.

**Step 3 — Author the feature-set (~30 min).** Define ~15–25 features inside Elasticsearch's LTR feature-set API. Chapter 5's four categories are the spine; suggested starting features:

- **Query-only:** query length in tokens; a flag for whether the query is a natural-language question (heuristic: contains a `?` or starts with `who/what/when/where/why/how`).
- **Document-only:** document length in tokens; document age in days (if the corpus has a date field); a `has_abstract` flag.
- **Query-doc interaction:** BM25 on `title`; BM25 on `body`; BM25 on `title.exact` (a keyword sub-field for exact-match); count of unique query terms matched in title; count of unique query terms matched in body; longest matching phrase in title; cosine similarity of the query embedding and the `body_embedding` field (use `script_score` with a `cosineSimilarity` scoring script, wrapped in the feature template).

Register the feature-set:

```
PUT _ltr/_featureset/beir_features
{ "featureset": { "features": [ ... your feature templates ... ] } }
```

Test each feature individually by running a `sltr` query with that one feature and inspecting the score with `_explain` on a hand-picked (query, doc) pair. If a feature returns `NaN` or `0.0` where it should not, fix it now; you will not diagnose it later once it is buried inside a feature-vector file.

**Step 4 — Log features for the training queries (~30 min).** For each `qid` in `qids_train.txt` ∪ `qids_valid.txt`, and for each judged document under that qid (both relevant and non-relevant judgments), ask Elasticsearch to compute the feature-set. Write the results to a LibSVM-shaped file:

```
<grade> qid:<qid> 1:<feature_1> 2:<feature_2> ... N:<feature_N>
```

Two structural requirements: rows for a given qid must be **contiguous**, and features must be in the same **ordinal order** across every row. Get either wrong and LightGBM trains a nonsense model.

Log two files: `train.svm` and `valid.svm`.

**Step 5 — Sanity-check the feature file (~15 min).** Before you train anything, spend 15 minutes looking at your feature file:

- Are grades distributed the way you expect (mostly `0`s with a small tail of `2`/`3`/`4`s, per BEIR/MS MARCO)?
- Are there any features that are constant across all rows? Drop them — they add zero information.
- Are there any features that are `NaN` or `-inf`? Fix the template or add a `missing` default.
- Are the feature values roughly in comparable ranges? LambdaMART is scale-invariant per-feature (trees split on any threshold) but wildly out-of-range values often indicate a bug in the template.

This 15 minutes catches ~80% of the bugs that would otherwise take an hour to unwind at training time.

**Step 6 — Train the LambdaMART model (~30 min).** Load `train.svm` and `valid.svm` into LightGBM and train:

```python
import lightgbm as lgb

train = lgb.Dataset("train.svm")
valid = lgb.Dataset("valid.svm", reference=train)

model = lgb.train(
    params={
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "lambdarank_truncation_level": 10,
    },
    train_set=train,
    num_boost_round=500,
    valid_sets=[valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=25)],
)
```

Two things to record from the training log:

- The `nDCG@10` on the validation set at the best iteration.
- The `lgb.plot_importance(model, importance_type="gain")` output — the ranked feature importances.

If a single feature dominates the importance (> 60% of gain), that is a **feature-leak warning sign**. Investigate whether the feature is implicitly correlated with the label (e.g., you accidentally included a "clicked=1" flag from the qrels themselves). Fix and retrain if so; the model is memorising the leak.

**Step 7 — Upload the model to Elasticsearch (~15 min).** Dump the model to the JSON format the LTR plugin expects (`model.dump_model()` in LightGBM ≥ 4.0), wrap it in the plugin's model envelope, and upload:

```
POST _ltr/_featureset/beir_features/_createmodel
{
  "model": {
    "name": "beir_ltr_v1",
    "model": {
      "type": "model/lightgbm+json",
      "definition": <the LightGBM JSON dump>
    }
  }
}
```

Verify with `GET _ltr/_model/beir_ltr_v1` that it round-trips.

**Step 8 — Evaluate on the held-out test split (~30 min).** For each `qid` in `qids_test.txt`, run three queries:

- **Baseline A — BM25 top-10.** Your best `multi_match` from Exercise 02.
- **Baseline B — BM25 top-10 with function_score.** Your best `function_score` from Exercise 02.
- **LTR — BM25 top-500, `sltr` rescore, top-10 after rescore.** Use the pattern from chapter 5:

  ```json
  { "query": <your multi_match>,
    "rescore": {
      "window_size": 500,
      "query": {
        "rescore_query": {
          "sltr": { "params": { "query": "<the query text>" },
                    "model": "beir_ltr_v1" } },
        "score_mode": "total",
        "query_weight": 0,
        "rescore_query_weight": 1
      }
    } }
  ```

  Setting `query_weight: 0` makes the LTR model the sole ranker within the rescore window; a real production deployment often uses `query_weight: 1` and lets the LTR score add. Try both and note the delta.

Compute `nDCG@10` (and optionally `MRR@10`, `Recall@100`) for each of the three shapes over the test-split qids. Present in one table.

**Step 9 — Diagnose one loss (~20 min).** Pick the query in the test split where your LTR model scored **worst relative to the BM25 baseline** (biggest `nDCG@10` regression). Log the feature vectors for that query's top-10 candidates (using the feature-set logging query from step 4). By hand:

- Compare the LTR-top-1 and the qrels-top-1 feature vectors. Which feature values drove the difference?
- Is a feature you defined behaving unexpectedly (e.g., `title_bm25` is 0 because the title tokenization dropped a term)?
- Is the model over-weighting a feature that is a poor signal for this query type?

Write two paragraphs: what the LTR model got wrong on this query, and what you would change (a new feature? a re-labelling? a training-set expansion?) to fix it. This is the diagnosis skill that makes LTR ownership real; without it, LTR is a black box that periodically regresses in production.

**Step 10 — Training/serving skew audit (~15 min).** Write down the answer to each of these questions for **your** pipeline; do not paraphrase chapter 5:

- If someone changes the `body` analyzer next week, will `body_bm25` at serving time still equal `body_bm25` at training time? If not, what would you do to catch it?
- The `recency` feature you likely used: is its origin `now` at both training time and serving time? If you re-log features tomorrow, will the same (query, doc) pair get the same `recency` value?
- Your cosine-similarity feature: are you re-embedding queries at serving time with the same model version that produced the document embeddings? What breaks if the embedding model changes?
- Your `missing:` defaults: are they identical between the feature templates and any client-side pre-computation?

Save this as `training_serving_skew_audit.md`. It is the checklist you re-run every time you touch the model.

**Step 11 — Write it up (~20 min).** Author `writeup.md` with:

- Your final feature-set (as JSON) and a one-line description per feature.
- The training-log summary: best iteration, best `nDCG@10` on validation, feature-importance ranking.
- The results table from step 8: BM25 vs. function_score vs. LTR on the held-out test set.
- The loss-diagnosis paragraphs from step 9.
- The skew audit from step 10.
- One paragraph: given what you observed, would you ship this model? What is the next feature or label investment that would most move the metric?

## Acceptance criteria

- ≥ 15 features registered in the LTR feature-set API, at least one from each of chapter 5's four categories.
- A `train.svm` and `valid.svm` file exist with contiguous per-qid rows, no `NaN` values, and consistent feature ordering.
- A LightGBM LambdaMART model trained with early stopping, `objective: "lambdarank"`, `metric: "ndcg"`. Feature-importance plot included in the writeup.
- Model uploaded to Elasticsearch's LTR (or OpenSearch equivalent) and callable via `sltr`.
- Held-out `nDCG@10` measured for BM25, function_score, and LTR. LTR row must be present even if it does not win — the diagnosis is required either way.
- One losing-query diagnosis with the feature-vector comparison and a concrete "what I would change" proposal.
- Training/serving skew audit completed with a written answer to each of the four questions.
- No queries from the test split appear in the training or validation splits — the split is documented and reproducible.

## Stretch goals

- **XGBoost cross-check.** Retrain the same feature set with XGBoost's `objective: "rank:ndcg"`, upload as `model/xgboost+json`, and compare `nDCG@10`. Report which library won and by how much. Usually the delta is small; the point is to prove the pipeline is library-agnostic.
- **Fuse LTR over hybrid features.** Add two new features derived from Exercise 03: the pure-kNN score (top of the `knn` clause) and a normalised RRF-style rank position. Retrain. If nDCG improves, you have found the reason chapter 5 says "LTR is downstream of a good candidate set" is not the same as "LTR ignores the candidate-set details."
- **Position-bias sim.** BEIR/MS MARCO judgments are editorial, so there is no position bias to correct — but you can *simulate* it. Sample 20% of the (query, doc) pairs and set their grades to `0` if the doc was outside position 3 in a hypothetical BM25 result. Retrain and evaluate. Watch how the model's `nDCG@10` on the (unbiased) test set drops. This is the practical shape of the click-bias problem chapter 5 warns about.
- **Post-hoc explainability.** Use [SHAP](https://shap.readthedocs.io/) to compute per-prediction feature attributions for five test queries. Present a small table: for each query's top-1 result, which two features contributed most to the LTR score? Cross-reference with the feature-importance ranking from step 6 — do the per-prediction attributions match the global importance, or are they meaningfully query-dependent?
- **OpenSearch port.** Repeat the pipeline on OpenSearch with the `opensearch-learning-to-rank-base` plugin. Note where the API differs from Elasticsearch 8.12+ built-in LTR (the endpoints, the model-upload envelope, the `sltr` shape). Confirm the trained LightGBM model file works on both engines without retraining.

## Notes

- LTR on a tiny corpus (SciFact, ~5K docs, ~1K queries) will produce a model with weak feature learning and noisy metrics. That is OK — the exercise is the pipeline, not the ceiling. Report the numbers you get; do not paper over a small delta.
- Never split within a query. Even one document from a query in both training and test contaminates the eval. Split at the qid level.
- If the LTR model **loses** to BM25 on the test set, do not treat that as failure. The most common cause on small datasets is that BM25 is already a strong baseline and the label noise dominates. The exercise is graded on the pipeline and the diagnosis, not on a positive delta.
- Do not blow past 50 features on your first pipeline. Chapter 5 is emphatic about this — LambdaMART tolerates redundant features but noise degrades the training signal. Grow the set once you have a working baseline you trust.
- Keep the feature-set JSON in version control from day one. When it drifts between training and serving, that JSON diff is your fastest debugger.
