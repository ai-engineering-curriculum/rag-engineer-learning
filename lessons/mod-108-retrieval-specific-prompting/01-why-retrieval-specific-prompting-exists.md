# Chapter 1 — Why Retrieval-Specific Prompting Exists

## Why this chapter exists

By the time the LLM sees the assembled context, everything that follows depends on the prompt. The retriever built recall; the reranker built precision; the assembler built a citation-anchored, budgeted context. The generator prompt turns those into a grounded, cited answer — or lets all the earlier work leak away.

Three things about *retrieval-augmented* generation make its prompt different from the baseline generator prompt that `llm-application-developer` (level 20) already covers:

1. **The context comes with a data structure, not just text.** Each chunk has an anchor, a source, sometimes a page, section, timestamp. The prompt has to teach the LLM what to do with that structure — how to cite it, when to trust the freshness signal, how to handle two chunks that disagree.
2. **"I don't know" is a first-class output.** When the retriever misses, the LLM must refuse. In a bare LLM call the model was allowed (or even encouraged) to answer from parametric knowledge; in a RAG call, an answer without grounding is a bug, not a feature.
3. **Every claim in the answer maps to a verifiable source.** The point of RAG is that a downstream check — human or automated — can point at a sentence in the answer and follow it back to a chunk. A citation the LLM invents defeats this. A citation the LLM omits defeats this. A citation with the right anchor but the wrong page defeats this in a way that is very hard to spot without post-generation checks.

The pattern set that handles these three properties — citation instructions, refusal contracts, contradiction handling, and post-generation verification — is **retrieval-specific prompting**, and it is what this module covers.

## What this module is not

The generator prompt in a shipped RAG application is a large document. Only a small part of it is retrieval-specific:

```
                                                                  ┌─ personality, tone, style
                                                                  ├─ safety instructions
        ┌─ overall persona & product framing ────────►            ├─ non-retrieval few-shots
        │                                                         └─ product-specific constraints
        │                                                                 (owned by applied-ai-engineer)
Generator                                                                 (peer track, level 30)
prompt ─┤
        │                                                                     ┌─ how to phrase "you don't know"
        │                                                                     ├─ how to break tie between chunks
        └─ retrieval-specific slice ──────────────────────────────────────►    ├─ citation instructions
                                                                              └─ context rendering / anchoring
                                                                                    (owned by rag-engineer)
                                                                                    (this module)
```

Two adjacent tracks own the rest:

- **`llm-application-developer` (level 20)** owns *baseline prompt engineering*: prompt structure (system/user/assistant), few-shot examples for arbitrary tasks, chain-of-thought and reasoning prompts, output-format instructions, temperature and decoding choices. This module assumes you know how to write a prompt at all.
- **`applied-ai-engineer` (level 30, peer)** owns *product-surface prompt shape*: personality, tone, safety, product-specific constraints, non-retrieval few-shots, how the retrieval-specific slice is composed into the full application prompt. Chapter 7 draws that boundary in detail.

This module is *only* the retrieval-specific slice: the joint constraints between the retriever's output shape and the generator's behavior on that shape.

## The four retrieval-specific patterns

Every RAG generator prompt in the wild answers the same four questions:

1. **Rendering and anchoring** (chapter 2). *How is the context inserted into the prompt so that the LLM can (a) attend to it and (b) refer back to it?* Answer: system-vs-user placement, delimiters, per-chunk headers, and an anchor scheme the LLM is told to cite.
2. **Citation instructions** (chapter 3). *What must the LLM's output look like so a downstream check can verify every claim?* Answer: inline bracketed citations, structured JSON citations, or answer-with-quote — three shapes with different trade-offs.
3. **Refusal** (chapter 4). *What does the LLM say when the retrieved context does not contain the answer?* Answer: an explicit refusal template, taught via an in-prompt example, gated by an optional confidence signal.
4. **Contradiction handling** (chapter 5). *What does the LLM do when two retrieved chunks disagree?* Answer: surface the disagreement, prefer the most-recent / most-authoritative source, or hand off to a resolver step.

Post-generation citation verification (chapter 6) is the pattern that catches the LLM's failures at all four when the prompt alone was not enough. Chapter 7 marks the boundary to the peer tracks.

## The four patterns are joint retriever-generator constraints

Each of the four patterns has a piece that lives on the retriever side and a piece that lives on the generator side:

| Pattern | Retriever side (this track, mod-102–107) | Generator side (this module) |
| --- | --- | --- |
| Rendering and anchoring | `AssembledContext.chunks[].anchor` is unique and stable | Prompt renders anchors verbatim; instructs LLM to cite by them |
| Citation instructions | Anchor scheme + `SourceRef.page` / `.section` populated correctly | Prompt requires per-claim citation and a specific citation format |
| Refusal | Retriever knows how to return "no results" and any confidence signal (mod-106) | Prompt trains the LLM to output the refusal template on empty context |
| Contradiction handling | `SourceRef.updated_at` + `metadata.authority` populated on ingest (mod-102) | Prompt tells LLM to prefer newer / more-authoritative or to surface |

Neither side owns the pattern alone. This is why retrieval-specific prompting is on the *retriever's* side of the peer boundary at level 30 rather than the peer track's — the patterns depend intimately on data the retriever's pipeline produces (anchors, freshness, authority signals, confidence). Chapter 7 walks through why.

## Two failures that motivate the discipline

Two well-documented failure classes give this module its urgency.

### Failure 1: hallucinated citations

The LLM writes a fluent answer with `[1]`, `[2]`, `[3]` citations. Two of them point at anchors that exist in the assembled context and support the surrounding claim; one points at an anchor that never appeared in the context and is invented from the prompt distribution. Called out in [OpenAI's own model spec and safety writeups](https://openai.com/index/openai-model-spec/) and studied at length in the "faithfulness" literature: Es et al.'s RAGAS paper ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)) formalises the metric that catches it; the DeepEval / TruLens frameworks operationalise the check. Chapter 6 is the pattern for catching this in production.

### Failure 2: over-helpful answers on empty context

The retriever misses. The LLM, given a prompt that does not train refusal, answers anyway from parametric knowledge. The answer looks confident; some fraction is wrong; the user has no way to tell. This is the failure that the "hallucination" complaints in enterprise RAG bug reports track back to more often than not. Chapter 4 is the refusal contract that stops it.

Both failures survive a good retriever, a good reranker, and a good assembler. They are downstream of the retrieval pipeline and can only be prevented at the prompt-and-verification layer. That layer is this module.

## What "prompt engineering" borrows from — and what it doesn't

Retrieval-specific prompting is prompt engineering with retriever-shaped constraints. It borrows:

- Few-shot examples in the prompt (from mod-102-scoped `llm-application-developer` material).
- Explicit output-format instructions.
- Chain-of-thought reasoning when it helps (rarely useful for citation tasks; occasionally useful for contradiction resolution).
- Model-specific system-message conventions (Claude system prompt, OpenAI system role, etc.).

It does *not* borrow — and in fact deliberately narrows — a few common practices from general prompt engineering:

- "Be helpful when you don't know the answer" — the retrieval prompt inverts this and trains refusal on empty context.
- "Use your prior knowledge to fill in gaps" — retrieval prompts explicitly forbid this.
- "Prefer confident phrasing" — retrieval prompts prefer *cited* phrasing, and are neutral on confidence.

Distinguishing these two lists — what to borrow and what to invert — is the first skill this module builds.

## Where each chapter fits

- **Chapter 2** is the mechanical prerequisite: how the assembled context becomes text that the LLM can read and refer to. Anchors, delimiters, headers, placement.
- **Chapter 3** is the core deliverable: the citation-instruction prompt patterns. Three shapes: inline `[1]`, structured JSON with an evidence list, answer-with-quote (a verbatim quote alongside every claim).
- **Chapter 4** is refusal: how you teach the LLM to say "the context does not contain the answer" without collapsing to unhelpfulness on borderline cases.
- **Chapter 5** is contradiction: two chunks disagree; the LLM must not silently pick one, silently average them, or emit a plausible-sounding blend.
- **Chapter 6** is the checker: post-generation verification that catches unresolved anchors, wrong page numbers, and unsupported claims. This is a rag-engineer deliverable, not the peer track's, because it depends on the data structure the assembler produced.
- **Chapter 7** is the boundary: which of this pattern set the peer track composes into its final product prompt, what the peer track owns that this module does not, and where the level-20 `llm-application-developer` boundary sits underneath.

## What you should be able to do after this chapter

- Name the four retrieval-specific prompt patterns and the property each addresses (rendering/anchoring, citation, refusal, contradiction).
- Explain why each pattern is a joint retriever-generator constraint (rather than a pure generator concern).
- Distinguish the retrieval-specific slice of the generator prompt from the peer track's product-surface prompt and from the level-20 baseline prompt.
- State the two motivating failure classes (hallucinated citations, over-helpful answers on empty context) and which chapter handles each.

## Further reading

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the paper that named RAG; useful for the framing of "grounding" as the point of the technique.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the faithfulness metric that formalises "citation must support claim."
- Bohnet, B., Tran, V. Q., Verga, P., Aharoni, R., Andor, D., Baldini Soares, L., et al. (2022). *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.* [arXiv:2212.08037](https://arxiv.org/abs/2212.08037) — a systematic treatment of attribution/citation for QA answers.
- Anthropic: [*Claude prompt library*](https://docs.claude.com/en/resources/prompt-library) and [*Long context tips*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — vendor guidance on document rendering that chapter 2 builds on.
- OpenAI: [*Prompt engineering guide*](https://platform.openai.com/docs/guides/prompt-engineering) — vendor guidance that shows the level-20 baseline this module extends.
- Menick, J., Trebacz, M., Mikulik, V., Aslanides, J., Song, F., Chadwick, M., et al. (2022). *Teaching Language Models to Support Answers with Verified Quotes.* [arXiv:2203.11147](https://arxiv.org/abs/2203.11147) — the "GopherCite" paper that formalised answer-with-quote and post-generation quote-verification, the pattern chapters 3 and 6 build on.
