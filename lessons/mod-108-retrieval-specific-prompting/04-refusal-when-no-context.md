# Chapter 4 — Refusal When No Context

## Why this chapter exists

Every RAG pipeline eventually retrieves nothing useful. The user's question is out of scope, the corpus does not cover it, the retriever missed, or the chunks that came back are on-topic in embedding-space but off-topic in fact. The default LLM behavior in these cases is to answer anyway — from parametric knowledge, from confident-sounding paraphrase, from whatever fills the gap.

That is the single most common enterprise-RAG failure. A cited-looking answer that is wrong is worse than "I don't know." Refusal is the pattern that makes "I don't know" a first-class, model-emitted output.

The pattern has three moving parts:

1. An explicit **refusal template** the LLM emits verbatim when the context does not support the answer.
2. A **training-shot** — one in-prompt example — that shows the LLM exactly when to emit it.
3. Optional **confidence gating** — a signal from the retriever (score, coverage, or uncertainty) that triggers refusal *before* the LLM even sees the question, when the retriever knows it missed.

The trade-off — every increment of refusal costs some helpfulness — is what this chapter walks through.

## The refusal template

A refusal has two jobs:

1. Tell the user, in words, that the answer is not in the sources.
2. Make itself easy for a downstream check to detect (peer track's UI shows a "no-answer" state; observability logs count refusals; a fallback path may kick in).

A single sentence that does both:

```
The provided sources do not contain enough information to answer this question.
```

Refinements per product:

- **With a scoped suggestion.** *"The provided sources do not contain enough information to answer this question. You could try rephrasing to be more specific, or check the [product-specific escape hatch]."* Good when there is a real escape hatch (human support, a documentation portal, a broader-search retry).
- **With a specificity hint.** *"The provided sources discuss X and Y but do not specify Z."* Named "grounded refusal" or "partial refusal" in some sources. The LLM tells the user what the sources *do* cover, so the user can reformulate. Highest helpfulness / user experience; requires an extra instruction in the prompt (below) and doubles the failure surface (the LLM can now hallucinate "the sources discuss X" incorrectly).

Pick one and stick with it. Consistency at the template level — the same words every time — is what makes the downstream checker easy to write.

## The refusal-instruction prompt (rule + template)

Add this block to the citation prompt from chapter 3:

```
Refusal:
- If the documents do not contain enough information to answer the question,
  respond with EXACTLY:
    "The provided sources do not contain enough information to answer this question."
  Do not add citations to the refusal message. Do not attempt to answer the question
  from your general knowledge.
- A partial answer with a caveat is acceptable ONLY if you can cite at least one
  document for the part you answer. Otherwise, refuse entirely.
```

Two design details:

- **"EXACTLY"** — the template is a fixed string. This lets the peer track detect refusal by string match, count refusals in telemetry, and switch to a fallback UX (see the peer track's product-layer refusal handling).
- **The partial-answer clause** — refusal is not all-or-nothing. If the LLM can answer half the question from the context and half is unsupported, the prompt should say to cite the supported half and refuse the rest, not refuse everything. This is the "helpfulness" side of the trade-off.

## The training-shot

Without an example, models — especially smaller ones — under-refuse. They interpret the rule as "unless it looks impossible" and answer marginal cases. One in-prompt example fixes this:

```
Example (for reference only — not the current question):

<documents>
  <document anchor="[X1]" source="Warranty policy">
    Products purchased before 2020 are covered by a 2-year warranty.
    Products purchased after 2020 are covered by a 1-year warranty.
  </document>
</documents>

Question: How do I file a warranty claim?

Answer: The provided sources do not contain enough information to answer this question.

End of example.
```

The example is deliberately borderline — the retrieved document is *related to* the question (warranty policy) but does not *answer* it (how to file). This trains the LLM to refuse on the failure mode that matters: the retriever brought back relevant-looking chunks that do not resolve the specific question.

If your eval shows over-refusal (the LLM refuses even when the answer is in the context), add a second example showing a valid partial answer:

```
Example 2 (for reference only):

<documents>
  <document anchor="[X1]" source="Warranty policy">
    Products purchased before 2020 are covered by a 2-year warranty.
    Products purchased after 2020 are covered by a 1-year warranty.
  </document>
</documents>

Question: How long is the warranty for a product bought in 2019?

Answer: A product purchased in 2019 is covered by a 2-year warranty [X1].
```

Two examples — one refusal-shaped, one partial-answer-shaped — usually resolves both failure modes.

## Confidence-gated refusal (the retriever-side option)

The prompt-only refusal fires *after* the LLM has read the context. Sometimes you want to refuse *before* — when the retriever's own signal says the corpus does not have the answer. This is confidence gating.

Two signal sources are common:

### 1. Score-threshold gating

The retriever emits a top-1 (or top-K aggregated) similarity or reranker score. If the top score is below a calibrated threshold, skip generation and return the refusal template directly:

```python
def answer(query: str) -> str:
    ctx = retrieve_and_assemble(query)
    if ctx.chunks and ctx.chunks[0].rerank_score < REFUSAL_THRESHOLD:
        return REFUSAL_TEMPLATE
    if not ctx.chunks:
        return REFUSAL_TEMPLATE
    return llm_generate(prompt(query, ctx))
```

The threshold is calibrated on a held-out set — chapter's exercise 02 walks through this. Rerank scores are *not comparable* across queries (mod-107 chapter 1), so calibrating one universal threshold is imperfect. It works when your reranker family and corpus are stable; it drifts when you change models.

### 2. Empty-retrieval gating

If the retriever returns zero chunks (or fewer than some minimum K), refuse without calling the generator. This is the cheapest and most reliable gate; every stack should have it. Set K_min = 1 as a floor and raise it if your eval shows more coverage helps.

### 3. Retrieval-uncertainty gating

More sophisticated retrievers emit an explicit uncertainty signal — coverage of the query terms, agreement across ANN neighbors, or an LLM-based "does the top-K plausibly contain the answer?" classifier. This is expensive (an extra LLM call) but shifts the decision to a purpose-built classifier rather than the generator's implicit judgment.

Confidence gating trades helpfulness for precision on refusal. A miscalibrated threshold refuses too many answerable queries; a permissive threshold lets marginal cases through to the generator, which is the exact case the prompt-only refusal is supposed to catch. Use confidence gating as an **augment** to the prompt-only refusal, not a replacement.

## The two-tier refusal architecture

Most production RAG stacks end up with two refusal layers:

```
  query ──►  retrieve  ──►  gate 1 (empty / score threshold) ─── refuse ──► REFUSAL_TEMPLATE
                                        │
                                        ▼ pass
                                  generator prompt
                                        │
                                        ▼
                                    LLM answer
                                        │
                                        ▼
                                gate 2 (LLM's own refusal, chapter 4)
                                        │
                                        ▼
                                 verifier (chapter 6)
                                        │
                                        ▼
                                   user answer
```

Gate 1 is cheap (retriever-side, no LLM call). Gate 2 is the prompt-taught refusal. Both feed the same telemetry signal ("this query refused"); observability treats them as one metric.

This shape is what the peer track (`applied-ai-engineer`, level 30) composes into its full response-handling logic — fallback UX, retry with a broader search, escalate to human. Chapter 7 marks that boundary.

## The helpfulness / refusal trade-off

There is no free lunch. Every increment of refusal precision (fewer wrong-but-confident answers) costs some helpfulness (more times the user hears "I don't know" when they could have gotten a partial answer).

The knob is your eval. Measure both:

- **Refusal precision** — of the refusals emitted, what fraction were correct refusals (the answer really was not in the context) vs. incorrect (the answer was there and the LLM missed)?
- **Answer faithfulness on non-refusals** — of the answers emitted, what fraction were fully grounded (no hallucinated claims)?

The desirable operating point moves both metrics up together for a while, then hits a Pareto frontier where you have to pick. Product context sets the pick:

- **Compliance, medical, legal** — err strongly toward refusal. A wrong answer is a liability; a refusal is inconvenience.
- **Support, internal search** — permit partial answers. The user is going to iterate anyway.
- **Consumer search / summarisation** — tune to user satisfaction, not to raw refusal precision.

Mod-109 (Retrieval-Quality Evaluation) covers the measurement side. The eval sets used there — with both "answerable" and "unanswerable" queries labeled — are the substrate for tuning the refusal threshold and the partial-answer instruction.

## Cascading refusals in a multi-step pipeline

If the retriever is part of a multi-step agent (mod-110 MCP-exposed retrievers), refusal composes:

- **Tool-level refusal.** The retrieval tool itself returns "no results above threshold" and the calling agent decides what to do.
- **Agent-level refusal.** The calling agent, after several tool calls that all refused, emits a final refusal to the user.
- **User-level refusal message.** The peer track's UI decides how to phrase this to the user — with an "I searched but found nothing" or with a "try a different query" prompt.

The retrieval-specific chapter (this one) covers the first two. The user-facing phrasing is the peer track's — see chapter 7 for the boundary.

## Model-family differences

Refusal training-shot needs vary by model family:

- **Larger frontier models** (Claude Opus/Sonnet, GPT-4 class, Gemini Ultra/Pro) — one refusal example is usually enough; sometimes zero-shot works if the rule is stated firmly.
- **Smaller / instruct-tuned open models** (Llama 3 8B, Mistral, Qwen 7B) — need at least one example, sometimes two. Baseline behavior is more helpful-by-default.
- **Reasoning models** (o-series, Claude "thinking" mode) — sometimes reason themselves into "well, based on general knowledge, ..." even when told not to. Add an explicit "do not reason from general knowledge" instruction; verify with an unanswerable-query eval.

Do not assume the refusal prompt from one model family works on another. Re-run the ablation (chapter 3's rendering ablation shape applies here too) when you swap models.

## What to log

Every refusal should log:

- The query.
- The retrieved chunks (with anchors and scores).
- Which gate fired (retriever-side empty, retriever-side threshold, or prompt-taught).
- The refusal-template text (in case it drifts).

This is the input to your refusal-precision measurement. Without it you cannot distinguish "the retriever missed" (a mod-106 problem) from "the LLM over-refused" (a mod-108 problem).

Mod-111 (Retrieval MLOps) owns the versioning and dashboards for this telemetry. Refusal rate is one of the retrieval SLOs — a sudden spike or drop is a signal that something has changed (corpus, retriever, model, prompt).

## Anti-patterns

### 1. The refusal template is different each time

The prompt says "if you cannot answer, say so" and the LLM produces a different phrasing per query ("I don't have enough information...", "The sources are unclear...", "I'm not sure..."). Downstream detection breaks. Solution: require the *exact string*.

### 2. The refusal cites a fake source

The LLM says "The sources do not answer this question [1]" — a citation on a refusal is nonsense; `[1]` refers to something the sources do cover. Solution: rule "do not add citations to the refusal message" is not optional.

### 3. The retriever-side gate uses raw similarity, uncalibrated

The gate fires at `similarity < 0.7` because "0.7 seemed reasonable" — but the actual distribution on real queries has almost everything above 0.7, or almost nothing above 0.5. Solution: calibrate the threshold on the actual data (exercise 02).

### 4. The confidence gate replaces the prompt refusal

"We have the score threshold, so we don't need to teach refusal in the prompt." When the threshold lets a marginal query through, the LLM answers from parametric knowledge and no other check catches it. Solution: gates are *additive*. Keep both.

### 5. Instructing the LLM to caveat instead of refuse

The prompt says "if unsure, add a caveat like 'I'm not certain, but ...'". The LLM caveats *and* answers — the caveat becomes decorative, and the answer is often wrong. Solution: refusal is a hard mode. Caveats are for genuine partial answers with citations for the partial part.

## What you should be able to do after this chapter

- Author a refusal template (exact string) and the prompt rules that force the LLM to emit it verbatim on empty context.
- Add a training-shot example that trains refusal on the "related but not answering" failure mode; add a second example that permits partial answers when appropriate.
- Design a two-tier refusal architecture (retriever-side gate + prompt-taught refusal) and calibrate the retriever-side gate on real data.
- Reason about the helpfulness / refusal trade-off and pick an operating point for your product context.
- Recognise the five refusal anti-patterns and rewrite the prompt to avoid them.

## Further reading

- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know.* [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) — the reference for "can the model predict its own accuracy?"; useful for confidence-gated refusal design.
- Ren, R., Wang, Y., Qu, Y., et al. (2023). *Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation.* [arXiv:2307.11019](https://arxiv.org/abs/2307.11019) — a systematic study of when RAG helps refusal precision.
- Yu, W., Zhang, H., Pan, X., et al. (2024). *Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models.* [arXiv:2311.09210](https://arxiv.org/abs/2311.09210) — an approach where the LLM writes explicit notes about each retrieved chunk before answering; can improve refusal precision.
- Feng, S., Shi, W., Bai, Y., Balachandran, V., Zhou, T., and Tsvetkov, Y. (2024). *Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration.* ACL. [arXiv:2402.00367](https://arxiv.org/abs/2402.00367) — abstention as a first-class design goal, complementary to this chapter's refusal.
- Anthropic: [*Reduce hallucinations*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/reduce-hallucinations) — vendor guidance covering the "allow the model to say I don't know" instruction pattern.
- Mod-109 of this track (Retrieval-Quality Evaluation) — the eval framework for refusal precision.
