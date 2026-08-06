# Exercise 02 — Refusal When No Context: Training-Shot and Eval

**Estimated effort:** ~2 hours. Python, an LLM API, the citation-prompt fragment from exercise 01, a small eval set with **both** answerable and unanswerable queries labeled, and (optionally) a rerank score to calibrate a retriever-side gate.

## Objective

Build the refusal contract from chapter 4 end-to-end: author the refusal template + training-shot, add it to your citation prompt, calibrate a retriever-side confidence gate, and produce a refusal-precision / helpfulness measurement that lets you defend an operating point.

At the end, you have a two-tier refusal architecture (retriever gate + prompt-taught refusal), calibrated numbers, and the refusal fragment the peer track (chapter 7) will inherit.

## Prerequisites

- Chapter 4 read.
- Exercise 01's citation-prompt fragment (you will extend it with the refusal block).
- An eval set of ~40 queries, split:
  - **20 answerable** — the retriever returns chunks that support the answer.
  - **20 unanswerable** — the retriever returns chunks that are either off-topic, related-but-not-answering, or empty.

  If you don't have a pre-labeled unanswerable set, hand-author 20: pick topics your corpus does not cover, pick questions your corpus covers only tangentially (like the "how do I file a warranty claim?" example in chapter 4), and add a few "empty retrieval" cases where you force the retriever to return zero chunks.
- Rerank scores from your reranker (mod-107) for every retrieved chunk. Needed for step 4.

## Steps

**Step 1 — Author the refusal template and prompt block (20 minutes).** In `refusal_prompt.py`:

- Define the exact refusal template as a Python constant (chapter 4's default: `"The provided sources do not contain enough information to answer this question."`). Do not paraphrase.
- Write the refusal rule block (chapter 4's template) and one training-shot showing the "related but not answering" case.
- Compose it into the citation prompt from exercise 01 (`build_generator_prompt` gains a `refusal_block` slot).

**Step 2 — Baseline the LLM without the refusal block (20 minutes).** Run the 40-query eval with the exercise-01 citation prompt (no refusal block). Measure:

- **Over-answer rate on unanswerable** — fraction of unanswerable queries where the LLM produced an answer (any answer, including partial or hedged). This is your before-refusal baseline; it is often 60–95% on smaller / older models.
- **Answer rate on answerable** — fraction of answerable queries where the LLM produced a substantive cited answer. This is what refusal will trade against.

**Step 3 — Add the prompt-taught refusal and re-measure (30 minutes).** Add the refusal block from step 1 to the prompt. Re-run the 40-query eval. Measure the same two metrics plus:

- **Correct-refusal rate on unanswerable** — fraction where the LLM emitted the *exact* refusal template.
- **Incorrect-refusal rate on answerable** — fraction where the LLM refused on a query the answer was in the context for.

Aggregate into the 2×2:

| | Answered (LLM produces cited answer) | Refused (LLM emits template) |
| --- | --- | --- |
| Answerable queries | ✓ helpful | ✗ over-refusal |
| Unanswerable queries | ✗ hallucinated / over-helpful | ✓ correct refusal |

Compute refusal precision (of the refusals emitted, fraction that were correct) and refusal recall (of the queries that should have refused, fraction that did). Both start at 0 with no refusal training; you're calibrating both up.

**Step 4 — Calibrate a retriever-side confidence gate (30 minutes).** For the 40-query eval:

- Log the rerank score of the top-1 chunk for each query.
- Plot (or tabulate) the distribution: rerank score for answerable queries vs. unanswerable queries. There is usually a soft separation.
- Pick a threshold `T` such that "if top-1 rerank score < T, refuse without calling the generator" catches ≥60% of unanswerable queries and refuses <10% of answerable queries. (Numbers vary by model and corpus; if the two distributions overlap heavily, the gate is not useful — note that and skip.)
- Re-run the eval with the two-tier setup (retriever gate + prompt-taught refusal). Measure the joint refusal precision / recall.

If your retriever returns zero chunks on some unanswerable queries, add the empty-retrieval gate (K < 1 → refuse) as the first tier — it is cheap and reliable, always add it.

**Step 5 — Measure the helpfulness trade-off (15 minutes).** With three configurations — (a) no refusal, (b) prompt-taught refusal only, (c) two-tier retriever gate + prompt-taught refusal — plot the Pareto frontier:

- X-axis: over-answer rate on unanswerable (bad; lower is better).
- Y-axis: answer rate on answerable (good; higher is better).

Which configuration is on the frontier? Pick your operating point and defend it based on a stated product context (compliance / support / consumer chat).

**Step 6 — Instrument logging (10 minutes).** Extend the response handler so every refusal logs:

- Query, retrieved chunks (with anchors and rerank scores), the gate that fired (`empty_retrieval` / `threshold` / `prompt_taught`), the exact response string.

This is the chapter 4 logging contract; mod-111 (MLOps) needs it later.

**Step 7 — Write it up (15 minutes).** In `refusal-eval.md`:

- The 2×2 table for each of the three configurations.
- The Pareto plot or a small ASCII summary showing the trade-off.
- One paragraph naming your chosen operating point and the product context.
- One paragraph on failure cases you saw: LLM refusing when it shouldn't (over-refusal), LLM answering when it shouldn't (under-refusal), and the query pattern common to each.
- The refusal fragment file, checked in as the peer-boundary contract deliverable.

## Acceptance criteria

- `refusal_prompt.py` contains the exact refusal template as a constant and the refusal rule block + one training-shot.
- Baseline (no refusal), prompt-taught refusal, and two-tier configurations all measured on ≥ 30 queries with both answerable and unanswerable splits.
- Rerank-score threshold calibrated on the eval data with the two-distribution plot / table.
- Refusal precision, refusal recall, over-answer rate on unanswerable, and answer rate on answerable measured for all three configurations.
- `refusal-eval.md` names an operating point and defends it against a stated product context.
- Every refusal logs the four required fields (query, chunks, gate, response).

## Stretch goals

- **Partial-answer training-shot.** Add the second training-shot from chapter 4 — the LLM answers the supportable half of a question and refuses the unsupportable half. Measure how often it correctly partial-answers vs. refuses entirely. Does the second shot degrade over-refusal?
- **Specificity hint variant.** Author the "grounded refusal" variant: *"The provided sources discuss X and Y but do not specify Z."* Measure whether user-satisfaction proxies (readability, follow-up rate) improve, and count how often the LLM misdescribes what X and Y actually are (new failure surface).
- **Score calibration per query type.** Split the eval by query type (factual / procedural / definitional) and check whether one universal threshold works or whether per-type thresholds beat it. If yes, mod-106 owns the query classifier that routes.
- **Empty-retrieval fallback.** For the empty-retrieval gate, add a "widen the retrieval and try again" fallback (e.g., loosen filters, expand query with a rewrite from mod-106). Measure recovery rate.
- **Cross-model.** Repeat on a smaller model (Claude Haiku, GPT-4o-mini). Chapter 4 predicts smaller models need more training-shot support; verify or refute on your data.

## Notes

- The refusal template must be a byte-exact string. If the LLM occasionally paraphrases it ("Sources do not have enough info..." vs. "The provided sources do not contain enough information..."), the peer track's string-match logging breaks. If paraphrases are frequent, tighten the prompt with the phrase "the following EXACT text" and re-eval.
- Do not use a fuzzy match on the refusal template in your logging. The whole point of the byte-exact template is that string equality is a reliable signal. Fuzzy match hides prompt drift.
- Confidence gate on rerank score works when the reranker and corpus are stable. When either changes, recalibrate. Mod-111 (MLOps) is where drift alerts live.
- The "answerable" and "unanswerable" labels are yours to define — but be consistent. A query where the retriever returned the wrong chunks (retriever miss) is technically answerable-by-the-corpus but unanswerable-with-the-current-context. Decide which label you use; treat that decision as part of the eval-set spec (mod-109 owns the eval-set spec formally).
- The Pareto frontier is domain-specific. A refusal-precision of 90% might be excellent for consumer chat and unacceptable for compliance. Do not chase a universal target; chase the target for your product context.
- Over-refusal is often the more damaging failure once your prompt is tuned — users lose trust in the system's usefulness. Track it as prominently as under-refusal.
