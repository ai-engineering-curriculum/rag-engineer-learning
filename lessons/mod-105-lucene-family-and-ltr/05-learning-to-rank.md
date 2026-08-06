# Chapter 5 — Classical Learning-to-Rank: LambdaMART, LightGBM/XGBoost, and the Elasticsearch/OpenSearch LTR Plugins

## Why this chapter exists

By the end of chapter 3 you have a `multi_match` query with hand-tuned field boosts and a `function_score` with recency, popularity, and quality signals. That is the ceiling of what hand-tuning gets you. Past the third or fourth signal, hand-tuned weights stop converging on any eval-set metric; the interactions between features are more subtle than a human weight-picker can defend. **Learning-to-rank (LTR)** replaces the hand-tuned weights with a model that learns feature weights (and interactions) from labelled data.

Classical LTR — LambdaMART / LambdaRank / RankNet with gradient-boosted trees — remains the workhorse in production search systems in 2026, alongside newer neural rerankers (mod-107). Its virtues: interpretable feature importance, fast inference (~microseconds per document), robust to noisy signals, and cheap to retrain. Its limits: it does not learn representations, and it requires labelled data (judgments or click-model-derived labels). This chapter walks the algorithm family, the model formats, the feature engineering, and the Elasticsearch / OpenSearch plugin machinery for deploying trained models on top of the lexical index from chapter 3.

This is the classical-LTR story. Neural reranking (cross-encoders, ColBERT re-scoring, LLM-as-judge rerankers) lives in mod-107. The distinction: LTR reranks top-K candidates by combining engineered features into a scalar; a neural reranker computes a fresh (query, document) representation. Both are second-stage ranking; classical LTR is cheaper and more predictable.

## The three algorithms in the family

LTR-with-gradient-boosted-trees descends from a three-paper lineage from Microsoft Research (Chris Burges' group). Understanding the lineage — even at one-paragraph depth — helps you read model configs and choose sensibly:

### RankNet (2005)

[Burges et al., *Learning to Rank using Gradient Descent*, ICML 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/). The first paper to frame ranking as a pairwise classification: for a document pair (A, B) with known preference (A > B), train a neural network `f(x_i) → score` such that `f(x_A) > f(x_B)`. The loss is the cross-entropy of the sigmoid of the score difference. Elegant; the gradient of the loss over the whole training set is what you compute.

### LambdaRank (2007)

[Burges et al., *Learning to Rank with Non-Smooth Cost Functions*, NIPS 2006](https://www.microsoft.com/en-us/research/publication/learning-to-rank-with-nonsmooth-cost-functions/). RankNet optimises pairwise classification; the metric you actually care about (nDCG, MAP) is a rank-based metric that RankNet's loss does not directly optimise. LambdaRank's insight: **multiply the RankNet gradient (`λ`) for a pair by the change in nDCG that swapping the pair would produce**. You never write the metric's loss (it isn't differentiable); you weight the pairwise gradients by the metric's sensitivity. Empirically this gets you close to directly-optimising nDCG.

### LambdaMART (2010)

[Wu et al., *Adapting boosting for information retrieval measures*, Information Retrieval 2010](https://www.microsoft.com/en-us/research/publication/adapting-boosting-for-information-retrieval-measures/). Replaces the neural-network base model of LambdaRank with a **Multiple Additive Regression Trees (MART) / gradient-boosted trees** base. Each tree fits the LambdaRank gradients of the current ensemble's residuals. Result: the tree ensemble learns the LambdaRank-shaped ranking function, and you get all the gradient-boosted-tree virtues (feature importance, fast inference, robustness to feature scales).

**LambdaMART is what production LTR usually means in 2026**. You will hear "we use LTR" — 95% of the time that means LambdaMART, trained with LightGBM or XGBoost, deployed via the Elasticsearch or OpenSearch LTR plugin.

The three-line lineage:

- **RankNet** — pairwise loss with a neural network.
- **LambdaRank** — pairwise loss × ΔnDCG weight, with a neural network.
- **LambdaMART** — same loss/weight, with gradient-boosted trees.

## The training libraries: LightGBM and XGBoost

Two mature open-source libraries implement LambdaMART. Either works; the LTR plugins support both.

### LightGBM ranking

[Microsoft's LightGBM](https://lightgbm.readthedocs.io/) supports LambdaMART directly via the `objective: "lambdarank"` parameter. Reference implementation and generally the faster of the two for ranking-specific workloads.

```python
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train,               # feature matrix (n_samples, n_features)
    label=y_train,         # relevance grades (0..4 typical)
    group=group_train,     # per-query group sizes, sums to n_samples
)

model = lgb.train(
    params={
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "lambdarank_truncation_level": 20,
    },
    train_set=train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)],
)
```

Points worth naming:

- **`group` parameter is the per-query group sizes** — this is how LightGBM knows which documents belong to which query. For query with 30 candidates, group value is 30; the rows must be contiguous per query. Get this wrong and your model trains but the metric is meaningless.
- **`lambdarank_truncation_level: 20`** — LambdaRank's ΔnDCG weight is truncated at position 20 (i.e., only pairs within the top-20 contribute their full weight). This matches typical top-K use.
- **`ndcg_eval_at`** — the positions at which nDCG is reported. Use the same K as your serving system's top-K.

### XGBoost ranking

[DMLC's XGBoost](https://xgboost.readthedocs.io/) supports both LambdaMART (`objective: "rank:ndcg"` or `"rank:pairwise"`) and a MAP-optimising variant (`objective: "rank:map"`). Its ranking API takes a `qid` column that identifies query IDs — different shape from LightGBM's `group` array but the same idea.

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)  # or set_uint_info("group_ptr", ...)

model = xgb.train(
    params={
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 20,
    },
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=50,
)
```

Choose based on team familiarity; the trained models are portable across both plugins because both are serialised to a supported model-dump format.

## Features: what the model actually sees

An LTR model does not see raw text or embeddings. It sees a **feature vector per (query, document) pair** — a fixed-size vector of numbers that the model's trees split on. Feature engineering is where you spend most of your LTR time.

The features that matter, roughly categorised:

### Query-only features

- Query length in tokens.
- Query type flags (does it look like a natural-language question? a symbol? an ID?).
- Query language detection.
- Query intent classification (informational / navigational / transactional).
- Query frequency in the log (head / torso / tail).

### Document-only features

- Document age (days since publish, days since last-updated).
- Document length in tokens.
- Structured quality signals: view count, helpful_ratio, star rating, editor pick flag.
- Document type (article / doc / FAQ / community post).
- Authorship (staff / community).
- PageRank-style link signals if applicable.
- Language of the document.

### Query-document interaction features (the bulk of the signal)

- BM25 score on the title field.
- BM25 score on the body field.
- BM25 score on tags / other secondary fields.
- Number of query terms matched in the title.
- Number of query terms matched in the body.
- Longest matching phrase length in the title.
- Cosine similarity of the query embedding and the document body embedding.
- Match features per field (exact, prefix, phrase, span).
- Position of the leftmost query-term match.
- Query-doc language match flag.

### Personalization / context features (if applicable)

- User's tenant ID (as one-hot or as a feature-cross with document tenant).
- User's language.
- User's session context.
- Time-of-day / day-of-week.

You want ~20–100 features for a first LTR model. Not thousands; the LambdaMART lineage does not benefit from feature explosion the way deep learning does. Focus feature engineering effort on the query-document interaction features — those carry most of the signal.

## Labels: relevance grades and where to get them

An LTR model needs **relevance labels** — a grade (typically 0–4) per (query, document) pair indicating how relevant the document is to the query. Two paths:

### Editorial judgments

Human raters look at (query, document) pairs and assign grades. This is the [TREC / MS MARCO](https://microsoft.github.io/msmarco/) way. Advantages: high quality, no click-model bias. Disadvantages: expensive, small coverage of the query space, static.

Typical grading scale (from TREC and MS MARCO):

- **4 — Perfectly relevant.** Answers the query directly.
- **3 — Highly relevant.** Contains the answer plus context.
- **2 — Moderately relevant.** Related but not the answer.
- **1 — Somewhat relevant.** Tangentially related.
- **0 — Not relevant.**

### Click-model-derived judgments

Infer relevance from user click behaviour. The naive "clicks = relevant" model is badly biased: position bias (users click top results even if bad), presentation bias, and click sparsity (most queries have zero clicks). Position-bias correction ([Chapelle & Zhang, 2009](https://www.researchgate.net/publication/221153934_A_dynamic_bayesian_network_click_model_for_web_search_ranking) — DBN click model) and click-through-rate normalisation are minimum bars.

The reference paper for using click logs to train LTR is [Joachims et al., *Unbiased Learning-to-Rank with Biased Feedback*, WSDM 2017](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf) — inverse propensity scoring for LTR.

Most production systems use a blend: click-derived labels for coverage and freshness, editorial labels for the head queries where quality matters most and for eval-set anchoring. mod-109 covers the eval-set methodology in depth.

## The Elasticsearch LTR flow

Elasticsearch LTR is built-in since 8.12 (previously the o19s [`elasticsearch-learning-to-rank`](https://github.com/o19s/elasticsearch-learning-to-rank) plugin). The [Elasticsearch LTR docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/learning-to-rank.html) are the canonical reference.

The pipeline:

### 1. Define features as templated queries

Each feature is a **query template** — a stored query that produces a score for a (query, document) pair. The feature-store API lets you register named templates.

```
PUT _ltr/_featureset/support_features
{
  "featureset": {
    "features": [
      {
        "name": "title_bm25",
        "params": ["query"],
        "template_language": "mustache",
        "template": {
          "match": { "title": "{{query}}" }
        }
      },
      {
        "name": "body_bm25",
        "params": ["query"],
        "template": {
          "match": { "body": "{{query}}" }
        }
      },
      {
        "name": "recency",
        "template": {
          "function_score": {
            "query": { "match_all": {} },
            "gauss": {
              "updated_at": { "origin": "now", "scale": "30d" }
            }
          }
        }
      },
      {
        "name": "helpful_ratio",
        "template": {
          "function_score": {
            "query": { "match_all": {} },
            "field_value_factor": { "field": "helpful_ratio", "missing": 0.5 }
          }
        }
      }
    ]
  }
}
```

### 2. Log features for training queries

For each (query, document) in your training set, ask Elasticsearch to compute the feature-set scores. This is the `sltr` query wrapped in a special logging query, or in 8.12+ the built-in feature extraction API.

The output: a per-(query, document) feature vector, exported to a file in **LibSVM-shaped** format:

```
1 qid:1 1:5.32 2:2.11 3:0.85 4:0.92
0 qid:1 1:1.20 2:0.44 3:0.32 4:0.10
4 qid:1 1:8.75 2:6.12 3:0.99 4:0.98
```

Columns: label (relevance grade), `qid:<query id>`, then `<feature_ordinal>:<value>` for each feature.

### 3. Train the model offline

Feed the LibSVM-shaped file into LightGBM or XGBoost as shown above. Serialise the trained model to the format the LTR plugin accepts.

### 4. Upload the trained model

```
PUT _ltr/_featureset/support_features/_createmodel
{
  "model": {
    "name": "support_ltr_v1",
    "model": {
      "type": "model/lightgbm+json",
      "definition": "<the LightGBM JSON dump>"
    }
  }
}
```

Or in Elasticsearch 8.12's built-in LTR:

```
PUT _ml/trained_models/support_ltr_v1
{
  "input": { "field_names": ["title_bm25", "body_bm25", "recency", "helpful_ratio"] },
  "definition": { "trained_model": { "tree": { ... } } }
}
```

### 5. Deploy the model in a `rescore` query

The trained model runs on the top-K candidates from a cheap first-stage query:

```
POST /support_articles/_search
{
  "query": {
    "multi_match": { "query": "reset password", "fields": ["title^5", "body"] }
  },
  "rescore": {
    "window_size": 500,
    "query": {
      "rescore_query": {
        "sltr": {
          "params": { "query": "reset password" },
          "model": "support_ltr_v1"
        }
      },
      "score_mode": "total"
    }
  }
}
```

The first-stage BM25 returns top-500 candidates; the LTR model re-scores those 500 using the feature-set and its trained tree ensemble; the user sees the top-10 after re-scoring.

## The OpenSearch LTR flow

OpenSearch's [`opensearch-learning-to-rank-base`](https://github.com/opensearch-project/opensearch-learning-to-rank-base) plugin is the direct descendant of the o19s Elasticsearch plugin (it was forked at the same time as OpenSearch itself). The API surface is nearly identical to the pre-8.12 Elasticsearch LTR plugin:

- Same `sltr` query.
- Same feature-set / model API (`_ltr/_featureset/...`, `_ltr/_model/...`).
- Same LightGBM/XGBoost model formats (`model/xgboost+json`, `model/lightgbm+json`, `model/ranklib`).
- Same LibSVM-shaped training data export.

The pattern you learn on Elasticsearch's pre-8.12 LTR ports over to OpenSearch verbatim. Elasticsearch 8.12+'s built-in LTR uses the ML Commons API instead; the concepts are the same but the endpoints differ.

## Solr and Vespa LTR: brief pointers

- **Solr** ships an [LTR contrib module in-tree](https://solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html) since 6.4 (2017). Supports LambdaMART via RankLib model format, plus MART, linear, and neural network models. Feature store is a JSON file per feature-set; models are uploaded via `/schema/model-store` API.
- **Vespa** does not have an LTR plugin because ranking is first-class. [Vespa's LightGBM guide](https://docs.vespa.ai/en/lightgbm.html) shows how to train a LambdaMART model in LightGBM and deploy it as a Vespa ranking expression that Vespa evaluates natively. No separate plugin, no separate feature-store API — the same `.sd` file that defines your fields defines your ranking expression.

The Vespa approach is elegant if you're already on Vespa. Do not switch engines *just* to get it.

## Feature importance and interpretability

LambdaMART's virtue over deep rerankers: you can look at the trained model and see which features matter. LightGBM and XGBoost both expose per-feature importance:

```python
import lightgbm as lgb
import pandas as pd

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=feature_names
).sort_values(ascending=False)
print(importance.head(20))
```

Use the importance ranking to prune features (features with near-zero gain add noise, not signal), to defend the model to stakeholders ("the top three signals driving ranking are…"), and to catch feature bugs (a feature that dominates all others is often a leak — e.g., a click-count feature that's implicitly correlated with the label). SHAP values ([shap.readthedocs.io](https://shap.readthedocs.io/)) give per-prediction feature attribution; useful for debugging why a specific (query, document) pair scored the way it did.

## The training / serving skew trap

The single most common LTR failure mode: **the feature values seen at training time do not match the feature values seen at serving time**.

Concrete instances:

- **Query normalisation differs.** Training queries were pre-processed differently than serving queries (e.g., diacritic-folded at training, not at serving). Model sees different features than it was trained on.
- **Feature templates changed silently.** Someone tweaked the analyzer on the title field between training and serving. `title_bm25` returns different numbers.
- **Missing feature defaults differ.** Training pipeline filled missing values with the median; serving pipeline uses `0.0`. Model interprets a real-zero and a missing-value identically.
- **Time-varying features aren't time-consistent.** The `recency` feature at training time was computed relative to a fixed snapshot date; at serving time it's `now`. Model has learned to expect one distribution and sees another.

The rule of thumb: **the feature-extraction code at training time and at serving time must be the same code, called through the same path**. The LTR plugins help here — training-time feature logging goes through the same feature-set the serving-time `sltr` query uses, avoiding drift by construction. When you do feature engineering *outside* the engine (e.g., pre-computing features in your ingest pipeline), your discipline must replace the plugin's structure.

## When LTR is the wrong tool

Two situations where LTR is not the right next step:

1. **You have no labels and no click log.** LTR needs labels. If your product is pre-launch or has no scale, mod-107's neural rerankers (which can work zero-shot) are the better first move; add LTR later when the click log exists.
2. **Your quality gap is representational, not ranking-order.** If BM25 finds the wrong 500 candidates because it can't handle a semantic gap ("reset my password" doesn't lexically match a doc titled "Recovering access to your account"), no LTR model can rescue you — the right document isn't in the candidate set. Dense retrieval or hybrid retrieval (mod-106) is the fix; LTR is downstream of a good candidate set.

The general rule: **LTR is a rank-quality tool; it does not compensate for a bad candidate set**. Get the retriever right first, then LTR the top-K.

## The three anti-patterns to avoid

1. **Training LTR on click logs without position-bias correction.** The model learns "the top result gets clicked" as a feature signal, not because it's more relevant. Use IPS or a click model; do not use raw clicks as labels.
2. **Deploying a new LTR model without an offline eval and an online A/B.** LTR is a machine-learning system; you need the same rigour as any other ML deployment. Offline eval on a held-out set (mod-109) gates promotion to A/B; A/B gates promotion to 100%.
3. **Blowing up the feature set to 500+ features because "more is better."** LambdaMART tolerates redundant features but does not gain from them. Prune to the 20–100 features with real gain; retrain less often; iterate on features that measurably move the metric.

## What is NOT in this chapter (and where it goes)

- **Neural rerankers (cross-encoders, ColBERT, LLM-as-judge)** — mod-107.
- **Hybrid retrieval fusion (RRF, learned fusion, LTR-over-hybrid-features)** — mod-106.
- **The eval methodology behind the labels (nDCG, MRR, statistical significance)** — mod-109.
- **Deploying LTR models via retriever CI, model versioning, offline eval regression tests** — mod-111.
- **Web-scale personalization / cross-DC ML serving** — chapter 7 (this module's boundary chapter) and the `staff-ml-engineer` (level 50) track.

## What you should be able to do after this chapter

- Explain the RankNet → LambdaRank → LambdaMART lineage in three sentences.
- Choose between LightGBM ranking and XGBoost ranking on team-familiarity grounds and defend the choice.
- Author an initial feature set of ~30 features covering query-only, document-only, query-document interaction, and (optionally) personalization signals.
- Deploy a trained LightGBM/XGBoost LambdaMART model via the Elasticsearch or OpenSearch LTR plugin, wire it into a `rescore` window on top of a first-stage BM25 or hybrid query.
- Diagnose training/serving skew and defend feature-store discipline against it.
- Recognise the two situations (no labels; bad candidate set) where LTR is not the right next step.

## Further reading

- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research technical report MSR-TR-2010-82. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) — the reference paper for the algorithm family.
- Wu, Q., Burges, C. J., Svore, K. M., & Gao, J. (2010). *Adapting boosting for information retrieval measures*. Information Retrieval, 13(3), 254–270. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/adapting-boosting-for-information-retrieval-measures/) — the LambdaMART paper.
- Joachims, T., Swaminathan, A., & Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased Feedback*. WSDM 2017. [ACM DL](https://dl.acm.org/doi/10.1145/3018661.3018699) — the reference for click-model-derived labels.
- Elasticsearch: [*Learning to Rank overview*](https://www.elastic.co/guide/en/elasticsearch/reference/current/learning-to-rank.html) — the 8.12+ built-in LTR docs.
- OpenSearch LTR: [github.com/opensearch-project/opensearch-learning-to-rank-base](https://github.com/opensearch-project/opensearch-learning-to-rank-base).
- Solr: [*Learning to Rank*](https://solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html).
- Vespa: [*LightGBM for ranking*](https://docs.vespa.ai/en/lightgbm.html) and [*Ranking framework*](https://docs.vespa.ai/en/ranking.html).
- LightGBM: [*LambdaRank ranking objective*](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) — the training-time reference.
- XGBoost: [*Learning to Rank*](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
- OpenSearch Labs: [*LTR on OpenSearch*](https://opensearch.org/blog/introducing-learning-to-rank/) — an accessible getting-started write-up.
