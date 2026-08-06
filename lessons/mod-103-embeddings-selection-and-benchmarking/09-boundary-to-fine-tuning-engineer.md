# Chapter 9 — Boundary to `fine-tuning-engineer`: When to Hand Off Embedding Training

## Why this chapter exists

Every chapter so far has been about *selecting* and *evaluating* an already-trained embedding model. This chapter is the point where that stops — where the answer is not "pick a different model from the shortlist" but "we need to fine-tune the retriever on our data." That work is not owned by this module. It is owned by the peer track [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) (level 30, ML Engineering family), which teaches the training craft — contrastive-loss design, hard-negative mining, LoRA / PEFT on encoders, SFT via `sentence-transformers`, TSDAE for unsupervised warm-up, MultipleNegativesRankingLoss and GISTEmbedLoss for supervised heads, distillation for smaller inference footprints.

The retrieval engineer's job at this boundary is not to *do* the training. It is to:

1. **Recognize when off-the-shelf embedders are the ceiling.** Not every retrieval problem is fixed by fine-tuning; many are fixed by better chunking (mod-102), a hybrid retriever (mod-106), or a reranker (mod-107). Fine-tuning is expensive to build and expensive to maintain — the trigger has to be genuine.
2. **Produce the training-data and eval-set artifacts the fine-tuning engineer needs.** A retrieval-engineering team that hands over "we want a better model" gets nothing back. A team that hands over a labeled dataset, a fixed eval harness, and a target metric gets a model.
3. **Own the deployment path back.** The fine-tuning engineer trains and ships weights; the retrieval engineer re-runs the domain-fit benchmark (chapter 7), decides whether to swap, and manages the index-version cutover (mod-111).

This chapter is the shape of that handoff.

## When fine-tuning is actually the right call

The question to answer first — before any handoff — is *why* the off-the-shelf model is not sufficient. There are three failure modes that a domain-fit benchmark (chapter 7) exposes; only one of them is solved by fine-tuning.

### Failure mode 1: no candidate beats the baseline by enough

Symptom: your shortlist of 5–8 embedders returns a table where the best model is only 0.02 nDCG above the second-best, and 0.05 above BM25.

Cause: your queries and your corpus are already well-served by well-trained general embedders — the gap between them is inside noise. A better embedder is not going to unlock a step change.

Not fixed by fine-tuning. Fixed by hybrid retrieval + reranking (mod-106, mod-107), by better chunking (mod-102), or by accepting that you have solved the retrieval problem to the point of diminishing returns and should invest engineering budget elsewhere.

### Failure mode 2: the corpus vocabulary is not in any candidate's training data

Symptom: every dense embedder in your shortlist underperforms BM25 on your corpus. On specific queries you can identify, the models fail because they treat the technical terms as out-of-vocabulary noise.

Cause: your corpus is in a specialized domain (legal filings, drug names, internal codenames, non-Latin scripts, proprietary product taxonomy) whose vocabulary the general models never saw enough of during pretraining and fine-tuning.

**Sometimes fixed by fine-tuning.** But almost always first fixed by:

- Using a **domain-specialized** hosted or open model — `voyage-code-3` for code, `voyage-law-2` for legal, `voyage-finance-2` for financial (chapter 4). Try these first before building a fine-tuning pipeline.
- Adding **BM25 to the retriever** and fusing (mod-106). BM25 handles rare terms natively; a hybrid `BM25 ⊕ dense` often closes the gap without any training.

Only if a domain-specialized model does not exist and hybrid retrieval does not close the gap is fine-tuning the answer. When it is, the hand-off shape below is what you send to the peer track.

### Failure mode 3: real, persistent, corpus-specific quality gap

Symptom: after trying (1) hybrid retrieval, (2) domain-specialized models, (3) better chunking, and (4) a stronger reranker, your domain-fit benchmark still shows a 5+ nDCG-point gap between the best off-the-shelf model and what a hand-analysis says the corpus admits. You have labeled data (or can generate it) and the corpus is stable enough to justify amortizing the cost.

**Fixed by fine-tuning.** This is the case that legitimately triggers the hand-off. It is much rarer in practice than product managers think.

## The three fine-tuning shapes and what the peer track owns

Fine-tuning bi-encoders is not one technique. The peer track covers each — the retrieval engineer should recognize which shape their situation calls for so the handoff request is specific.

### Shape A: supervised fine-tuning with `sentence-transformers`

The default supervised recipe. You provide `(query, positive_doc, [negative_docs...])` triples; the trainer optimizes MultipleNegativesRankingLoss or an in-batch-negatives loss so that positives score higher than negatives under cosine similarity.

- Library: `sentence-transformers` [training docs](https://sbert.net/docs/sentence_transformer/training_overview.html).
- Losses: [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) (the standard); [GISTEmbedLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) (uses a guide model to identify false negatives — helpful when negatives are mined from your corpus and may accidentally contain positives).
- Reference paper: Reimers & Gurevych (2019), *Sentence-BERT*, [arXiv:1908.10084](https://arxiv.org/abs/1908.10084); Henderson et al. (2017) for the in-batch negatives shape.

When to request this: you have >10k labeled (query, positive) pairs on your domain and want the strongest supervised retrieval you can get from your existing base model.

### Shape B: unsupervised warm-up (TSDAE / SimCSE / GPL)

You have a lot of *unlabeled* domain text and few or no query/doc labels. Unsupervised pretraining on the domain corpus (denoising autoencoder — TSDAE; or in-batch positives from dropout — SimCSE; or generated pseudo-queries — GPL) shifts the model's representation to match your domain vocabulary before any labeled fine-tune.

- Papers: Wang et al. (2021), *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning* ([arXiv:2104.06979](https://arxiv.org/abs/2104.06979)); Gao et al. (2021), *SimCSE* ([arXiv:2104.08821](https://arxiv.org/abs/2104.08821)); Wang et al. (2022), *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval* ([arXiv:2112.07577](https://arxiv.org/abs/2112.07577)).

When to request this: your corpus vocabulary is very different from the base model's training distribution and you cannot label thousands of queries.

### Shape C: parameter-efficient fine-tuning (LoRA / adapters / PEFT)

Instead of updating all encoder parameters, insert small trainable adapters and update only those. Dramatically cheaper to train, cheaper to store (per-adapter versions), and reversible. Jina's `jina-embeddings-v3` (chapter 5) is a working example — one base encoder with five task-specific LoRA adapters swapped in at inference.

- Library: Hugging Face [`peft`](https://github.com/huggingface/peft).
- Paper: Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)).

When to request this: you want fast iteration, per-tenant customization, or you cannot afford full-parameter fine-tuning of a large base.

## What the retrieval engineer owns at the boundary — the handoff artifact

The fine-tuning engineer needs three things from you. Each is a real deliverable, not a vague ask.

### 1. Training data

The shape depends on which recipe (above). At minimum:

- **Positive pairs.** `(query, positive_doc_text)` — one line per pair, JSONL. If you have graded relevance, include the grade. If your labels are only doc-level, include the *chunk* text of the relevant chunk within that doc — the training loss operates on the exact text the encoder sees.
- **Hard negatives** (if you have them). Documents/chunks that the *current* retriever ranks highly but that are *not* relevant. These are the most useful negatives for training. Mine them by running your best off-the-shelf embedder over your training queries and pulling top-K hits that are labeled as non-relevant. If you skip this, the trainer will use random in-batch negatives, which is a much weaker signal.

Aim for **10k+ (query, positive) pairs** for a shape-A supervised run. Below that, the shape-B unsupervised warm-up is likely the right first move.

Schema example (JSONL, one training example per line):

```json
{"query": "coverage for laser eye surgery under gold plan",
 "positive": {"chunk_id": "handbook.gold-plan.vision-benefits.3",
              "text": "Refractive procedures including LASIK are covered..."},
 "hard_negatives": [
   {"chunk_id": "handbook.silver-plan.vision-benefits.2",
    "text": "Silver plan covers cataract surgery but excludes refractive..."},
   {"chunk_id": "handbook.gold-plan.dental-benefits.1",
    "text": "Gold plan dental benefits include..."}
 ]}
```

### 2. Eval set — the domain-fit benchmark you built in chapter 7

Reuse it verbatim. The fine-tuning engineer evaluates the new model *against your exact benchmark* — the same corpus, the same queries, the same qrels, the same chunker, the same retriever, the same ANN parameters. That is the only way the "did this training run help?" question has an unambiguous answer.

**Do not label the training and eval sets against the same queries.** Split them cleanly. A model that trained on your eval queries will show inflated scores that do not transfer.

### 3. The target metric and success criterion

"Improve retrieval quality" is not a target. "Beat model X by ≥5 nDCG points on our chapter-7 benchmark at chunker-version v1.2, held-out eval-v1 qrels" is a target. The success criterion should include:

- Primary metric and threshold (nDCG@10, +Δ vs. baseline).
- Secondary metrics and constraints (Recall@100 must not regress; max-tokens must be ≥ our chunk size; inference latency budget).
- Ceiling / stopping rule (if after N training runs the model has not hit the threshold, we stop and revisit).
- Deployment constraint (must be servable on our GPU tier; must fit in our storage budget at our chosen dim).

Put this in a `handoff.md` and version it. The fine-tuning engineer's deliverable back is a model checkpoint + a filled-in results table against the same benchmark.

## What the retrieval engineer owns *after* the handoff comes back

The fine-tuning team ships weights. The rest is yours again:

1. **Re-run the domain-fit benchmark** (chapter 7) with the fine-tuned model as one row in the shortlist. Confirm the target-metric threshold is hit on the held-out eval set. Confirm no regression on secondary metrics.
2. **Compute the cost-latency-quality delta** (chapter 6). A fine-tuned model that beats baseline by 5 nDCG but doubles inference latency needs a business-level trade-off decision. Put the numbers in front of the architect.
3. **Ship as a new index version.** New `embedding_model_id` (`internal-bge-insurance-v1`), new revision, new index in the vector DB (mod-111 covers the dual-index cutover pattern). Do not swap in place; keep the old index alive until the new one is validated in production.
4. **Set up retraining triggers.** Corpora drift; queries drift; a fine-tuned model that beat baseline in Q1 may not in Q4. mod-111 covers drift monitoring and retraining cadence. Every fine-tuned model comes with an implicit maintenance debt — surface it in the design doc.

## What is NOT in this chapter (and where it goes)

- **The training craft itself** — loss functions, negative mining algorithms, LoRA rank, distillation, TSDAE hyperparameters. Explicitly the peer track [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) (level 30, ML Engineering family).
- **Judgment-set methodology, paired significance testing, inter-annotator agreement.** mod-109.
- **Index-version cutover, dual-index deployment, retraining triggers.** mod-111.
- **Product-level "should we fine-tune at all?" business analysis.** Peer track `applied-ai-engineer` at the product surface; this chapter is the retrieval-engineering read of the trigger.

## What you should be able to do after this chapter

- Name the three failure modes from a domain-fit benchmark and state which one triggers a fine-tuning hand-off.
- Name the three fine-tuning shapes (supervised with MNR/GIST losses, unsupervised warm-up TSDAE/SimCSE/GPL, PEFT / LoRA) and pick the right one for a given data-availability situation.
- Describe the three deliverables a fine-tuning engineer needs from a retrieval engineer (training data + hard negatives, eval set = your chapter-7 benchmark, target metric + success criterion + deployment constraint).
- Own the post-fine-tune loop: re-benchmark, compute cost delta, index-version cutover, drift monitoring.
- Explain why fine-tuning is *rarely* the first move — hybrid retrieval, reranker choice, and domain-specialized off-the-shelf models close most gaps first, and are 10–100× cheaper to try.

## Further reading

- `fine-tuning-engineer-learning` (peer track, level 30) — [`ml-engineering-curriculum/fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning). This is the track that owns the training craft this chapter defers to.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
- `sentence-transformers` docs, [*Training Overview*](https://sbert.net/docs/sentence_transformer/training_overview.html), [*Losses*](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).
- Wang, K., Reimers, N., & Gurevych, I. (2021). *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning.* [arXiv:2104.06979](https://arxiv.org/abs/2104.06979).
- Gao, T., Yao, X., & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* [arXiv:2104.08821](https://arxiv.org/abs/2104.08821).
- Wang, K., et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.* [arXiv:2112.07577](https://arxiv.org/abs/2112.07577).
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Hugging Face, [`peft` library](https://github.com/huggingface/peft) — the PEFT / LoRA implementation the fine-tuning track uses.
- Anthropic engineering, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — a public example of a retrieval-quality lift achieved *without* fine-tuning (chunker + BM25 fusion + reranker changes). Useful as a "before you fine-tune, try this" reference.
