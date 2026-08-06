# Chapter 7 — The Boundary to `llm-application-developer` and `applied-ai-engineer`

## Why this chapter exists

Chapter 1 named two adjacent tracks and pointed at where this module ends. Chapters 2–6 built the retrieval-specific prompt patterns. This chapter is the *contract* those patterns cross when they leave the retrieval engineer's ownership and enter the peer track's — and the *contract* they inherit from the level-20 track underneath.

Two boundaries matter:

1. **Downstream to `applied-ai-engineer` (peer, level 30).** This module's four patterns (citation instructions, refusal, contradiction handling, post-generation verification) are the retrieval-specific slice of the full application prompt. The peer track composes them into its complete prompt, integrates the verifier into its response-handling logic, and owns the product surface.
2. **Below to `llm-application-developer` (level 20).** Baseline prompt engineering — prompt structure, few-shot patterns, output-format instructions, decoding parameters — is the level-20 track's foundation. This module extends that foundation with retrieval-specific patterns; it does not replace it.

Getting either boundary wrong produces the same failure mode: the rag-engineer either over-reaches (writing the full application prompt for the peer, or re-deriving level-20 basics) or under-delivers (handing off primitives without the composition contract). Both waste time.

## Where this module sits in the level-30 division

Mod-107 chapter 9 drew the boundary at `AssembledContext`. This module produces the *prompt-pattern layer* on top of that context — a small set of reusable fragments the peer track composes into its full application prompt:

```
                     rag-engineer (this track)                         applied-ai-engineer (peer, level 30)
                                                                      
   ingest → chunk → embed → index → retrieve → rerank → assemble       compose application prompt →
   ────────────────────────────────────► AssembledContext ────►        generate → parse cites → render →
                                                                       feedback loop
   plus:  citation-instruction fragments,
          refusal instruction + template,       ────► fragments  ────► composed into system + user
          contradiction-handling fragments,                             messages by the peer
          post-generation verifier,             ────► verifier   ────► integrated into peer's response
          the RenderedContextBlock              ────► rendered   ────► inserted verbatim into prompt
                                                        block
   (mod-108: this module)                                              (peer track's mod-103 in the
                                                                        peer curriculum)
```

The rag-engineer ships **fragments + a verifier + a rendered context block**. The peer composes those into the full generator prompt alongside the peer's own material (personality, style, safety, product-specific constraints).

## The prompt-fragment contract (retrieval-specific patterns → peer)

The four fragments this module produces, and what the peer track does with each:

### Fragment 1 — The rendered context block

- **rag-engineer produces:** an already-rendered string (chapter 2) — anchors, delimiters, headers, source metadata. Consumed by the peer's prompt template as a single `{retrieval_context}` slot.
- **peer commits to:** inserting the block verbatim, at the placement the rag-engineer specifies (Placement A/B/C from chapter 2). The peer does not re-shuffle chunks, re-parse and re-render, or truncate the block.

Concrete peer-side shape:

```python
prompt = f"""{peer_system_prompt}

{rag_citation_instructions}
{rag_refusal_block}
{rag_contradiction_block}

<documents>
{rag_rendered_context}
</documents>

Question: {user_question}
"""
```

Where `rag_*` come from this module and `peer_system_prompt` is the peer's own material.

### Fragment 2 — Citation-instruction fragment

- **rag-engineer produces:** the rules block from chapter 3 (pattern 1, 2, or 3) plus its training-shot, as a string. The rag-engineer commits to keeping this fragment consistent with the rendered anchor scheme.
- **peer commits to:** inserting it in the prompt exactly, not paraphrasing the rules "for style." If the peer wants a different citation shape, they open the conversation with the rag-engineer; they do not silently rewrite the rules.

### Fragment 3 — Refusal block

- **rag-engineer produces:** the refusal template (exact string), the rules, and the training-shot from chapter 4.
- **peer commits to:** rendering the refusal template as its own UX state (not as regular text). The peer's response-handler string-matches the template and switches to the appropriate UI (a "no results" state, a follow-up prompt, an escalation).

### Fragment 4 — Contradiction block

- **rag-engineer produces:** the contradiction rules and training-shot from chapter 5. The choice of pattern (surface, prefer authoritative, prefer recent, hand off) is a joint discussion — the retriever provides the metadata (authority, updated_at); the peer provides the product opinion (does the user tolerate "please clarify"?).
- **peer commits to:** rendering the contradiction outputs appropriately. If pattern D is chosen and the LLM emits a structured contradiction record, the peer routes to the resolver (mod-110 tool call, human review, rule engine).

## The verifier contract

Chapter 6's `VerificationResult` is the interface for the runtime verifier:

- **rag-engineer produces:** the verifier code, running at levels 1, 2, and 3 as configured; the `VerificationResult` object; the diagnostic trace.
- **rag-engineer commits to:** the verifier's precision and recall on a labeled eval set. Concrete: "level 1 recall of 100% on unresolved-anchor injection tests; level 3 precision of ≥90% on a labeled entailment eval." Mod-109 owns the eval methodology.
- **peer commits to:** consuming `VerificationResult` in the response handler and choosing a mitigation (fail-closed, retry, redact, escalate) per its product policy. The peer does *not* run its own second verifier over the same claims — that is duplicate work and duplicate calibration surface.

### The mitigation is peer-side

Mitigation choice is a product decision:

- Fail-closed on compliance flows.
- Retry with a stricter prompt on marginal violations.
- Redact-and-ship-partial on support flows.
- Escalate to a human review queue on unusual violations.

The rag-engineer publishes the SLO on the verifier ("verifier catches ≥95% of injected hallucinations, false positive rate ≤5%") and the peer track decides what the application does when the verifier fires. Both sides log to the same telemetry so refusal rate, verification-fired rate, and mitigation choice are traceable end-to-end.

## The four contract clauses

Analogous to mod-107 chapter 9's clauses:

### Clause 1 — Fragment consistency

- **rag-engineer commits to:** the citation-instruction fragment, refusal block, and contradiction block are self-consistent with the rendered anchors and the assembled context format. When the anchor scheme changes, all four fragments change together in one release.
- **peer commits to:** using the fragments as-shipped; not rewording rules, not skipping the training-shots, not translating the refusal template to a different phrase.

### Clause 2 — Verification precision/recall SLO

- **rag-engineer commits to:** verifier precision and recall targets on a versioned eval set (e.g., "level-3 entailment recall ≥ 90%, precision ≥ 85% on eval-set v3"). Mod-109 owns the eval methodology; mod-111 owns the eval-set versioning.
- **peer commits to:** trusting the verifier's verdict for the mitigation decision. The peer does not build a duplicate verifier; if the peer wants a stricter check for a specific flow, they open the conversation with the rag-engineer.

### Clause 3 — Latency contract

- **rag-engineer commits to:** verification latency at levels 1 and 2 is < 10 ms; level 3, when enabled synchronously, has a stated per-response budget (e.g., "≤ 400 ms for a 5-claim answer with NLI checker, ≤ 2s with LLM checker").
- **peer commits to:** budgeting the response time to accommodate the verifier's cost when configured synchronously; using async verification (sampling, offline) when the sync budget is not available.

### Clause 4 — Refusal-template stability

- **rag-engineer commits to:** the exact refusal template does not change without a release. When it does change, both sides update: the rag-engineer changes the template in the fragment; the peer updates its string-match in the response handler.
- **peer commits to:** matching the current template exactly; not doing a fuzzy-match that catches "similar" phrases (fuzzy match tolerates LLM drift and hides bugs).

## What the peer track owns downstream

Once the fragments and the verifier cross the interface, the peer track's level-30 deliverables include:

- **The full application prompt.** System prompt (personality, safety, constraints), the retrieval-specific fragments (this module), few-shot examples for non-retrieval task types, tool descriptions if any.
- **Response-handler integration.** Calling the verifier, choosing a mitigation, matching the refusal template, rendering citations (mapping anchors to the `SourceRef` for UI).
- **Product-facing UX.** How refusals are shown to the user, how contradictions are surfaced, how "verified partially" answers are labelled.
- **RAG-triad eval at the response granularity.** Faithfulness of the whole response, answer relevance, whether the user's actual question was answered. Mod-109 handles the retriever-side; the peer handles the response side.
- **Session-level observability.** Refusal rate, verifier-fired rate, mitigation-choice distribution, click-through-to-citation rate, user feedback signals.
- **Product-side caching, retry, and fallback.** How the app behaves when the verifier keeps firing or the generator keeps refusing.

## What this module owns upstream

Everything from chapter 2 through chapter 6:

- **Context rendering.** Placement, delimiters, per-chunk headers, anchor scheme, `AssembledContext` → prompt-text conversion.
- **Citation-instruction patterns.** Inline, JSON, quote — including training-shots.
- **Refusal patterns.** Template, rule set, training-shots, confidence-gate design.
- **Contradiction-handling patterns.** All four (surface, prefer authoritative, prefer recent, hand off) and their composition rules.
- **The verifier.** Levels 1, 2, 3; the `VerificationResult` contract; the eval-time verifier variant.
- **The retrieval-specific ablations.** Rendering ablation (chapter 2), citation-shape ablation (chapter 3), refusal-precision measurement (chapter 4), contradiction-pattern measurement (chapter 5), verifier-precision measurement (chapter 6).

Everything on that list is a rag-engineer decision. The peer's job is to *compose and consume* the resulting fragments, not to co-author them.

## Where the level-20 boundary sits underneath

`llm-application-developer` (level 20) owns baseline prompt engineering. That includes:

- **Prompt structure basics.** System vs. user vs. assistant roles, message ordering, temperature and top-p, max-token budget arithmetic.
- **Few-shot patterns.** How to author example blocks in general; when to use single-shot vs. multi-shot; the mechanics of putting `<example>` blocks in a prompt.
- **Output format instructions.** JSON output, XML output, delimiter escaping, structured-output APIs at the vendor level.
- **Chain-of-thought and reasoning prompts.** When to ask the LLM to "think step by step"; when to hide the reasoning from the response.
- **Vendor primitives.** Tool use / function calling at the mechanical level, structured outputs, streaming, prompt caching.

This module *uses* all of the above without teaching it. When chapter 3 says "add a training-shot," it assumes you know from level 20 how to author a training-shot; when chapter 2 says "render as JSON with structured-output mode," it assumes you know how to invoke structured-output mode.

If you find yourself unsure about any level-20 material, the `llm-application-developer` track's foundational modules are where it lives. Do not re-derive them here.

## The one-page contract you put on a wiki

The template that ships alongside a mod-108-shaped prompt-pattern set:

> **rag-engineer (level 30, this track) owns:** everything from ingestion through assembly (mod-102–107), **plus the retrieval-specific prompt patterns and post-generation verifier**: rendered context block, citation-instruction fragment, refusal block, contradiction-handling block, post-generation verifier (levels 1–3), and the `VerificationResult` contract.
>
> **applied-ai-engineer (peer, level 30) owns:** full application prompt composition, response-handler integration with the verifier, mitigation choice per product policy, product-facing UX for refusals and contradictions, RAG-triad eval at the response granularity, session-level observability, product-side caching / retry / fallback.
>
> **llm-application-developer (level 20, foundational) owns:** baseline prompt engineering — prompt structure, few-shot mechanics, output-format instructions, chain-of-thought, vendor primitives (tool use, structured outputs, streaming, caching).
>
> **The interface between rag-engineer and applied-ai-engineer:** the four prompt fragments (rendered context, citation instructions, refusal, contradiction) and the `VerificationResult` shape from chapter 6. Fragments inserted verbatim into the peer's prompt; verifier called on every response; mitigation is the peer's choice.
>
> **Shared eval set** with per-query labels at chunk granularity (rag-engineer's precision, citation-verifiability, refusal-precision, verifier-precision) and response granularity (peer's RAG triad). Joint eval reviews on the same cadence as mod-107's boundary.

## Three anti-patterns at this boundary

### 1. Peer track "improving" the citation rules

The peer decides the LLM's answers are wordy, so they replace the citation-instruction rule "cite all supporting sources per claim" with "cite the best source only." The verifier's calibration assumed the original rule; verifier precision drops silently. Solution: fragment changes are a rag-engineer decision. The peer requests changes through the contract, not by editing the fragment.

### 2. Peer track running a second, ad-hoc verifier

The peer, worried about hallucination, adds a second LLM-based sanity check on the response. Now two verifiers fire; they disagree; the mitigation logic gets tangled. Solution: one verifier, owned by the rag-engineer. If the peer needs a stricter level 3, negotiate; do not fork.

### 3. Peer track paraphrasing the refusal template

The peer decides "The provided sources do not contain enough information to answer this question" is stiff, and rewords it in the response handler based on context. The rag-engineer's telemetry that counts refusals (string-matching the template) undercounts. Solution: the peer renders the refusal in its UX however it wants (a "no results" panel, a follow-up prompt), but the template that leaves the LLM is unchanged.

## The two hand-offs this module makes

- **To `applied-ai-engineer` (peer, level 30).** The four fragments and the verifier, per the contract above. The peer composes them into the full application prompt and integrates the verifier into the response handler.
- **To mod-110 (Orchestration and MCP-Exposure).** When the retriever is exposed as an MCP tool, the fragments composed into a tool prompt travel with the tool description. Mod-110 owns the mechanics of that packaging; this module owns the content.

## What you should be able to do after this chapter

- Name the four prompt fragments (rendered context, citation instructions, refusal, contradiction) and the `VerificationResult` that this module hands to the peer track.
- Author the four contract clauses (fragment consistency, verifier SLO, latency, refusal-template stability) with concrete numbers for a real project.
- Distinguish this module's ownership from `llm-application-developer`'s (level 20 baseline prompt engineering, foundational) and from `applied-ai-engineer`'s (peer, level 30, product surface).
- Recognise the three boundary anti-patterns (peer edits fragments, peer runs a second verifier, peer paraphrases the refusal) and correct each.
- Draft the one-page ownership contract naming rag-engineer vs. applied-ai-engineer vs. llm-application-developer deliverables.
- Point to where this module hands off — the peer track for prompt composition + response handling, mod-110 for MCP-exposure of the retrieval-plus-prompt bundle.

## Further reading

- The track's [`CURRICULUM.md`](../../CURRICULUM.md) — the module map and level-ladder positioning.
- The track's [`PREREQUISITES.md`](../../PREREQUISITES.md) — the peer / adjacent-track links, including `llm-application-developer` and `applied-ai-engineer`.
- [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) — the peer track this chapter marks the boundary to.
- [`llm-application-developer-learning`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) — the level-20 track this chapter identifies as the foundation.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the response-granularity eval framework the peer uses; complements this module's rag-engineer-owned checks.
- Mod-107 chapter 9 (this track) — the analogous boundary at the `AssembledContext` seam; this chapter extends the same pattern.
- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — the shape of the contract clauses.
