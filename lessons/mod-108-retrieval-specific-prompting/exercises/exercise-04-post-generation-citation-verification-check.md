# Exercise 04 — Post-Generation Citation Verification Check

**Estimated effort:** ~3 hours. Python, an LLM API (for level-3 entailment), a small NLI model from Hugging Face (optional, for level-3b), the prompt fragments from exercises 01–03, and a labeled eval set of ~30 responses with known citation failures (some injected).

## Objective

Build the three-level post-generation citation verifier from chapter 6, wire it into the response path with a `VerificationResult` output, calibrate it on a labeled eval set with intentionally-broken responses, and measure verifier precision and recall.

At the end, you have a runtime verifier that catches unresolved anchors, page mismatches, and unsupported claims, plus a mitigation loop the peer track (chapter 7) integrates into its response handler.

## Prerequisites

- Chapter 6 read.
- Exercises 01–03 in place — you have a working generator prompt (citation + refusal + contradiction) and generated responses to run the verifier over.
- An `AssembledContext` shape with per-chunk `text`, `anchor`, and `source.page` (if you want to test level 2). If your ingest pipeline does not have page numbers, use a mock corpus for the level-2 tests.
- A labeled eval set of ~30 responses:
  - 10 clean responses (real LLM output; probably mostly-clean).
  - 10 responses with **injected level-1 failures** (an unresolved `[7]` when only `[1]`..`[5]` exist; edit the response strings yourself).
  - 5 responses with **injected level-2 failures** (a `[1, p. 42]` when chunk `[1]` is on page 15).
  - 5 responses with **injected level-3 failures** (a claim that flips the sign or a value, keeping the anchor). E.g., "The max temperature is 100 °C [1]" when [1] actually says 85 °C.
- (Optional) A small NLI model — [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base) or similar — for level-3b. Install `sentence-transformers` and download.

## Steps

**Step 1 — Author the level-1 verifier (30 minutes).** In `verifier.py`:

```python
def check_anchor_resolution(answer: str, ctx: AssembledContext) -> list[AnchorViolation]: ...
```

Implementation follows chapter 6's sketch. Parse `[n]` citations from the answer with a regex; check each against `{c.anchor for c in ctx.chunks}`. Return one violation per unresolved citation.

For structured JSON responses (exercise 01 pattern 2), add a second function `check_anchor_resolution_json(response_dict, ctx)` that iterates `response_dict["citations"]` and checks each `anchor`.

Run against the 10 clean + 10 level-1-broken responses. Expected: 100% precision (no false positives on clean); 100% recall (all 10 injected failures caught). Report actual numbers.

**Step 2 — Author the level-2 verifier (30 minutes).** In `verifier.py`, add:

```python
def check_page_match(answer: str, ctx: AssembledContext) -> list[PageViolation]: ...
```

Implementation follows chapter 6's sketch — parse `[n, p. X]` and compare `X` against `chunk.source.page`. Report violations for mismatch and for `p.` in the citation when the chunk has no page.

Run against 5 level-2-broken responses (plus clean and level-1). Expected: 100% recall on level-2 failures; no false positives when the citation has no page.

**Step 3 — Author the level-3a (LLM-based) verifier (45 minutes).** In `verifier.py`:

- Implement `extract_claim_evidence_pairs(answer, ctx)` — split the answer into sentences, attach trailing `[n]` markers to the preceding sentence, resolve each `[n]` to the chunk text.
- Implement `check_entailment_llm(pairs, model)` using the LLM prompt from chapter 6 pattern 3a. One LLM call per (claim, evidence) pair.
- Aggregate into `list[EntailmentViolation]` for pairs where `"supported"` is false.

Run against 5 level-3-broken responses (plus clean). Measure precision and recall on the injected failures.

Track latency: median LLM-call time × number of pairs per response. Chapter 6's budget is 400 ms – 2s per response; is your implementation in range?

**Step 4 — Author the level-3b (NLI-based) verifier (30 minutes).** In `verifier.py`:

- Load a cross-encoder NLI model. Wrapper: `check_entailment_nli(pairs, nli_model, threshold=0.7)`.
- For each (claim, evidence) pair, run the NLI model, take the `entailment` score. If below `threshold`, flag as unsupported.

Calibrate the threshold: on 10 mixed responses (clean + level-3-broken), scan `threshold` from 0.3 to 0.9 in 0.1 steps and plot verifier precision + recall. Pick the elbow.

Compare level-3a vs. level-3b: precision, recall, latency. Which do you deploy?

**Step 5 — Build the `VerificationResult` + mitigation (20 minutes).** In `verifier.py`, expose the composed verifier:

```python
def verify(answer: str, ctx: AssembledContext,
           entailment_checker=None) -> VerificationResult: ...
```

Match chapter 6's contract shape exactly — this is the interface the peer track (chapter 7) consumes.

In `mitigation.py`, implement the four mitigation branches:

- `fail_closed(result, ctx)` → returns the refusal template.
- `retry_stricter(result, ctx, question, generate_fn)` → re-runs generation with an additional "the previous response cited nonexistent anchors" instruction; caps at one retry.
- `redact_sentence(answer, violations)` → strips the sentence containing the violating citation, ships the rest.
- `escalate(result, ctx, answer)` → returns the answer with a warning banner and logs to a review queue mock.

Pick one branch per violation severity — e.g., level-1 violations trigger `retry_stricter`; level-3 violations on high-stakes flows trigger `fail_closed`; etc. Document your choice in `mitigation-policy.md`.

**Step 6 — Wire the verifier into the response path (10 minutes).** In `response_pipeline.py`:

```python
def respond(query: str) -> str:
    ctx = retrieve_and_assemble(query)
    if not ctx.chunks or ctx.chunks[0].rerank_score < REFUSAL_THRESHOLD:
        return REFUSAL_TEMPLATE
    answer = generate(build_generator_prompt(render_context(ctx), query))
    result = verify(answer, ctx, entailment_checker=nli_or_llm_checker)
    if not result.passed:
        return mitigate(result, ctx, query, answer)
    return answer
```

Run on 10 fresh queries end-to-end. Confirm verifier fires (or does not) as expected.

**Step 7 — Verifier-precision / verifier-recall write-up (25 minutes).** In `verifier-eval.md`:

- Per-level precision and recall table (from steps 1–4).
- Latency table (level 1, level 2, level 3a, level 3b) at median and p95.
- One paragraph on which level-3 approach (LLM or NLI) you deploy in production and why.
- One paragraph on the mitigation policy — which branch fires on which severity, and how the peer track should render each.
- The `VerificationResult` fixture your integration test uses (a captured example for regression).

## Acceptance criteria

- `verifier.py` implements level 1 (anchor resolution), level 2 (page match), and level 3 (both LLM-based and NLI-based variants).
- All three levels tested on a labeled eval set of ≥ 30 responses with ≥ 10 injected failures across the three levels.
- Precision and recall measured per level; latency measured at p50/p95.
- `VerificationResult` shape matches chapter 6 exactly and can be consumed by peer-track code.
- `mitigation.py` implements the four mitigation branches (fail-closed, retry, redact, escalate); `mitigation-policy.md` names which fires on which severity.
- End-to-end response pipeline wired: retrieve → assemble → generate → verify → mitigate.
- `verifier-eval.md` reports numbers and defends the level-3 approach chosen for production.

## Stretch goals

- **Sampling strategy.** Deploy level 3 on 5% of responses uniformly instead of on every response. Measure the drop in latency vs. the drop in verifier coverage. This is chapter 6's operational pattern.
- **Answer-with-quote fast path.** For responses generated with pattern 3 (chapter 3), implement the substring-check verifier that skips the LLM/NLI call entirely — the quote is already in the answer. Measure the latency saving; measure whether verifier recall stays at 100% on injected level-3 failures.
- **Retry-with-stricter-prompt effectiveness.** Log every retry: how often does the second attempt pass? A retry that never passes is theater. Chapter 6 suggests capping at one retry; verify that number holds on your data.
- **Self-check baseline.** Implement chapter 6's self-check pattern (same LLM verifies its own answer). Compare verifier precision/recall against level-3a (separate LLM call) and level-3b (NLI). Chapter 6 predicts self-check underperforms; verify or refute.
- **Verifier as SLO signal.** Log verifier-fired rate as a time series. Simulate a prompt regression (e.g., replace your citation fragment with a weaker one) and confirm the verifier-fired rate spikes. This is the mod-111 drift-detection input.
- **Cross-model verifier.** For level-3a, use a *different* LLM (Claude verifying GPT, or vice versa) instead of the same one that generated. Sometimes catches drift that the generating model would miss on self-check.

## Notes

- The injected failures in step 1 (unresolved `[7]`) are the easy case. Real hallucinated citations often use a *valid* anchor for the wrong content — level 1 doesn't catch that; level 3 does. Make sure your eval includes both.
- Level 3 latency is real. On a 5-claim answer with an LLM-based checker at 400 ms per call, you have added ~2s to every response. On the response path, that is often unacceptable. Sampling and answer-with-quote are the two mitigations chapter 6 names; use them.
- The NLI model's threshold is corpus-specific. A threshold that works on formal documents will over-flag on informal / conversational chunks. Recalibrate when the corpus changes.
- Do NOT trust `verifier_fired_rate` as your quality metric. It drifts as the prompt drifts. Use it as a *change signal* (spikes and drops are events to investigate), not as an absolute measure. Mod-109's eval framework is your absolute quality measure.
- The peer track's mitigation policy may differ from yours — chapter 7's clause is that the mitigation choice is peer-side. Publish the `VerificationResult` contract; let the peer configure mitigation.
- Chapter 6's anti-pattern "verifier without mitigation" is common. If your verifier fires and the answer ships anyway, delete the verifier — it is worse than useless (it makes the team think verification is happening when it isn't). Every violation must trigger something, even if it's just "log and warn."
- If your verifier's precision is under 80%, users see false positives — good answers get redacted or refused. That is worse UX than the original hallucination. Tighten the check before shipping.
