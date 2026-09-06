# Exercise 06 — Hand-Off: The Fine-Tuning Brief to the Fine-Tuning Engineer

**Estimated effort:** ~2 hours.

## Objective

Chapter 9 argued that fine-tuning is rarely the first move — hybrid retrieval, domain-specialized off-the-shelf models, and better rerankers close most gaps at 10–100× less cost. But when a domain-fit benchmark leaves a persistent, corpus-specific gap after those cheaper options are exhausted, fine-tuning is the answer, and the retrieval engineer owns the handoff. This exercise produces the artifact that handoff needs: a **fine-tuning brief** the peer `fine-tuning-engineer` track can pick up cold.

The brief is *not* a training recipe (that is the peer track's job). It is a scoped, evidence-backed ask: here is the failure the domain-fit benchmark exposed, here is the training data, here is the frozen eval set, here is the target metric, here is the deployment constraint. A team that produces this brief gets a trained model back. A team that hands over "our search is bad, please make it better" does not.

## Prerequisites

- Chapter 9 read once.
- Exercise-04 (domain-fit benchmark) complete. You need its ranking table and per-query error analysis — they are the evidence that fine-tuning is even the right move.
- Access to the `fine-tuning-engineer-learning` track repo (or its README) as the audience reference. The brief you write should be legible to someone on that track who has never seen your corpus.
- Python deps to mine hard negatives: `sentence-transformers`, `numpy`, `pandas`, `datasets`.

## The scenario

Assume exercise-04's benchmark returned a ranking where every candidate — including the best off-the-shelf model, hybrid retrieval (if you tried the mod-106 preview stretch), and reranker rerank (if you tried mod-107) — is stuck at nDCG@10 = 0.55 on your corpus. Business needs 0.70 or better. The gap is persistent, corpus-specific, and reproduces across the diverse-vocabulary queries you deliberately included. Chapter 9's *failure mode 3* applies. Time to hand off.

If your real exercise-04 numbers already look great and no fine-tune is warranted, that is fine — write the brief anyway as if the gap existed, using the actual corpus you benchmarked. Also add a short "why we would not actually pursue this today" paragraph at the top. The exercise is about the *handoff artifact*; you can practise it on a hypothetical.

## Part A — Justify the trigger (30 min)

Write `docs/fine-tuning-brief/00-why.md`:

1. **Which failure mode applies.** Chapter 9 named three; only failure mode 3 justifies fine-tuning. Explicitly name which of the three (with a paragraph explaining why it is not modes 1 or 2 — the latter would send you to hybrid retrieval or a domain-specialized model first).
2. **The evidence from exercise-04.** Cite the specific rows of the ranking table. Include:
   - The nDCG@10 gap between your best off-the-shelf candidate and the target.
   - The per-query error patterns from exercise-04's Part E analysis — which query shapes fail? What vocabulary trips the models?
   - The results of the "before you fine-tune, try this" list from chapter 9: hybrid BM25 fusion (mod-106), domain-specialized model (if one exists), stronger reranker (mod-107). Even if you have not run these yet, list them with a two-sentence "here is why we do not expect this to close the gap" or "here is what we still need to try" — either is honest; guessing is not.
3. **A ceiling / stopping rule.** Chapter 9 said this must be explicit. Something like: "If after 3 fine-tuning runs we have not moved nDCG@10 by ≥5 points over the current baseline of 0.55, we stop and revisit the retriever architecture."

## Part B — Produce the training data (45 min)

Chapter 9 Deliverable 1. The peer track needs `(query, positive_doc)` pairs, ideally with hard negatives.

Sourcing options (rank in this order):

1. **Real user queries + clicked results** from your product's logs — the strongest signal.
2. **Domain-expert-written questions + labeled relevant chunks** — likely what exercise-04's eval set was built from. **You must not reuse the eval set queries here.** Split them: half go to train, half stay in eval. Chapter 9 was explicit about this.
3. **LLM-generated queries over corpus chunks** — cheap; systematic bias risk. Use only as filler once options 1 and 2 are exhausted, and mark those examples explicitly in the JSONL.

Aim for **≥10k (query, positive) pairs** if pursuing the supervised recipe (chapter 9 shape A). Below that, the brief should ask for the unsupervised warm-up recipe (chapter 9 shape B) as a first pass instead — record that in the brief.

**Mine hard negatives.** Chapter 9 stressed that in-batch random negatives are a weak signal compared to hard negatives. Write `scripts/mine_hard_negatives.py`:

- Embed the training queries with your best off-the-shelf embedder (e.g., `BAAI/bge-large-en-v1.5`).
- For each query, retrieve top-50 from your corpus.
- Filter out the labeled positives.
- Keep 3–5 remaining top hits as hard negatives per query.
- **Sanity-check a random 30 of them.** Some hard negatives will be false negatives (relevant but unlabeled). Chapter 9 called out GISTEmbedLoss as the peer track's answer to this; your job here is just to disclose the rate — "spot-check found N/30 apparent false negatives."

Persist the training data as `data/fine_tune/train.jsonl` in the shape chapter 9 sketched:

```json
{"query": "coverage for laser eye surgery under gold plan",
 "positive": {"chunk_id": "handbook.gold-plan.vision-benefits.3",
              "text": "Refractive procedures including LASIK are covered..."},
 "hard_negatives": [
   {"chunk_id": "handbook.silver-plan.vision-benefits.2",
    "text": "Silver plan covers cataract surgery but excludes refractive..."},
   {"chunk_id": "handbook.gold-plan.dental-benefits.1",
    "text": "Gold plan dental benefits include..."}
 ],
 "source": "expert_labeled | click_log | llm_generated"}
```

Include a `docs/fine-tuning-brief/01-training-data.md` that describes:

- N pairs, per-source breakdown.
- Hard-negative mining recipe (which embedder, top-K, filters).
- The false-negative rate from the spot-check.
- The train / eval split — which queries are in each, how you split (e.g., by document, by time — do **not** split by query alone if two queries can share a positive doc; that leaks).

## Part C — Package the eval set and target (20 min)

Chapter 9 Deliverables 2 and 3.

Copy your exercise-04 frozen benchmark into `data/fine_tune/eval/`:

- `corpus.jsonl` (same file — do not re-chunk).
- `queries.tsv` — the eval-half queries only (not the training half).
- `qrels.tsv` — the labels for the eval-half queries.

Write `docs/fine-tuning-brief/02-eval-and-target.md`:

- Primary metric and threshold: `nDCG@10 ≥ 0.70 on the frozen eval set` (or whatever your target is).
- Secondary constraints: `Recall@100 ≥ current-model-value` (no regression); `inference latency ≤ current-model-value + 20%`; `served dim ≤ 1024`.
- Deployment constraints: `must load into TEI` (or your serving stack); `weights ≤ X GB`; `license compatible with commercial use`.
- Ceiling / stopping rule (from Part A).
- The exact `mteb` / `ir_measures` command the peer track should run against `eval/` to reproduce your baseline number. Not "some retrieval eval script" — the concrete command.

## Part D — Name the recipe (15 min)

Chapter 9 named three fine-tuning shapes:

- Shape A: supervised with `sentence-transformers` (MultipleNegativesRankingLoss / GISTEmbedLoss).
- Shape B: unsupervised warm-up (TSDAE / SimCSE / GPL).
- Shape C: parameter-efficient (LoRA / adapters).

Write `docs/fine-tuning-brief/03-recommended-recipe.md` naming which shape you *think* fits your situation, and why. Rules:

- ≥10k labeled pairs → probably A, or A after B.
- <10k labeled pairs and a lot of unlabeled domain text → probably B first, then A.
- Want per-tenant customization or fast iteration on many small experiments → probably C.

You are recommending, not prescribing — the peer track owns the training craft. Frame it as "we suggest starting with shape X because …" and leave the door open for the peer track to counter-propose.

## Part E — The post-handoff plan (10 min)

Chapter 9's final section listed what the retrieval engineer owns *after* the trained model comes back. Write `docs/fine-tuning-brief/04-post-handoff.md`:

- Re-run exercise-04's domain-fit benchmark with the new model as an added row.
- Compute the exercise-03 cost delta — does the fine-tuned model change the query-time latency or storage story?
- New `index_version_key` (mod-111): something like `internal-bge-{corpus}-v1@{sha}:dim=1024:quant=int8`.
- Dual-index cutover plan: keep the old index running, mirror traffic, compare in-product metrics, only decommission when the new index is validated. Reference mod-111 here even though it lives further down the track.
- Retraining trigger: what corpus / query drift signal (from mod-111) would tell you it is time to retrain? A one-paragraph plan is enough.

## Part F — Assemble the brief (10 min)

Combine the five files into `docs/fine-tuning-brief/README.md`:

```markdown
# Fine-tuning brief: {corpus name}, {date}

**Author:** {you}, retrieval-engineer track
**Audience:** fine-tuning-engineer track
**Status:** proposal / active / complete

## Why now
See [00-why.md](00-why.md).

## Training data
See [01-training-data.md](01-training-data.md). Data path: `data/fine_tune/train.jsonl`.

## Eval and target
See [02-eval-and-target.md](02-eval-and-target.md). Eval path: `data/fine_tune/eval/`.

## Recommended recipe
See [03-recommended-recipe.md](03-recommended-recipe.md).

## Post-handoff plan
See [04-post-handoff.md](04-post-handoff.md).
```

That README is the entry point the peer track reads first.

## Deliverables

- `docs/fine-tuning-brief/README.md` and the five sub-docs `00-why.md` through `04-post-handoff.md`.
- `data/fine_tune/train.jsonl` — ≥10k (query, positive, hard_negatives) records (or a smaller set with an explicit "we should do shape B first" note in the brief).
- `data/fine_tune/eval/{corpus.jsonl,queries.tsv,qrels.tsv}` — the frozen eval slice.
- `scripts/mine_hard_negatives.py`.

## Acceptance criteria

- The brief names which of chapter 9's three failure modes applies and gives a one-paragraph justification (not vibes).
- The training data has ≥10k pairs OR the brief explicitly asks for the unsupervised warm-up shape as the first pass; not "some fine-tuning."
- Train and eval queries are cleanly split — the brief documents the split rule.
- Hard-negative mining recipe is named (embedder used, top-K, filters). A false-negative spot-check rate is stated.
- The target section states a **numerical** primary threshold, a secondary no-regression constraint, and a deployment constraint. Not "improve retrieval."
- A ceiling / stopping rule is present so the fine-tuning engineer knows when to stop trying.
- The post-handoff plan names the exact `index_version_key` field name and the cutover pattern (dual-index, mirror traffic, decommission on validation).

## Stretch goals

- Build the training data twice: once with the domain-fit eval set's positives repurposed (the wrong way — a leak) and once with a clean split (the right way). Prove the leak inflates the reported gain by running a small fine-tune yourself and eval'ing both — this is the peer track's turf but is a strong lesson to feel firsthand.
- Add **query-side hard positives** (paraphrases of the same query that should still retrieve the same doc). This is the GISTEmbedLoss angle chapter 9 named — the guide model tells you which of your negatives are actually positives, which turns them into extra positive examples instead of contaminating negatives.
- Add a **contamination check**: for every training query, search the eval qrels for that query text or a near-duplicate. Any match is a leak; report the count.
- Write the counter-brief a fine-tuning engineer would send back: what data is missing, what target is unrealistic, what deployment constraint conflicts with the recipe. This trains you to anticipate the peer track's pushback before you send the brief.
- If your organization has a `fine-tuning-engineer` in real life, hand them this brief and iterate on it. This is the exercise most worth doing on a real project rather than the toy corpus.
