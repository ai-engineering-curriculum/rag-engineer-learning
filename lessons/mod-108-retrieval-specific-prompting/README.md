# mod-108-retrieval-specific-prompting: Retrieval-Specific Prompting: Citation, Grounding, and Refusal

**Estimated effort:** 10 hours

## What this module is about

Mod-107 handed you an `AssembledContext` — a token-budgeted, deduped, ordered list of chunks with stable anchors. This module is the *prompt* that sits between that context and the generator. Not baseline prompt engineering (that belongs to `llm-application-developer` at level 20) and not the product-surface prompt (that belongs to `applied-ai-engineer` at level 30) — the narrow retrieval-specific slice: **how the LLM cites the assembled context, when it refuses because the context does not contain the answer, how it handles two retrieved chunks that disagree, and how you catch it when it invents a citation anyway.**

The four patterns in this module — citation, refusal, contradiction handling, post-generation verification — are the parts of the generator prompt that a retrieval engineer must own because they are joint constraints between the retriever's output shape and the generator's behavior. The rest of the prompt (personality, style, product tone, non-retrieval few-shots) belongs to the peer track.

## Learning objectives

- Author citation-instruction prompt patterns that force per-claim source anchoring — inline bracketed citations, structured JSON citations, and answer-with-quote patterns — and know how to validate them post-generation.
- Design refusal-when-no-context patterns — explicit "context does not contain the answer" training-shot, confidence-gated refusal, and the trade-off with helpfulness.
- Handle contradiction between retrieved chunks — surface disagreement to the user, prefer the most-recent / most-authoritative source, or hand off to a resolver step.
- Detect and mitigate hallucinated citations (fabricated source IDs, wrong page numbers) with post-generation citation-verification checks.
- Cite the boundary to `llm-application-developer` (level 20) which owns baseline prompt engineering, and to `applied-ai-engineer` (peer, level 30) which owns product-surface prompt shape. This module covers ONLY the retrieval-specific slice.

## Chapters

- `01-why-retrieval-specific-prompting-exists.md` — Why the retrieval-specific slice of a generator prompt is a distinct discipline and where its edges are.
- `02-context-rendering-and-anchoring.md` — How the `AssembledContext` is rendered into the prompt so the LLM can ground on it: system-vs-user placement, delimiters, per-chunk headers, and the anchor scheme.
- `03-inline-and-structured-citation-patterns.md` — The three citation-instruction shapes (inline bracketed, structured JSON, answer-with-quote), their trade-offs, and how to author each.
- `04-refusal-when-no-context.md` — The refusal contract: explicit "context does not contain the answer" training-shot, confidence-gated refusal, and the helpfulness trade-off.
- `05-contradiction-handling.md` — Contradiction patterns: surface, prefer authoritative, hand off to a resolver step, and where each is appropriate.
- `06-post-generation-citation-verification.md` — Verifying citations after the fact: unresolved anchors, wrong page/section, unsupported spans (NLI-based checks), and the mitigation loop.
- `07-boundary-to-llm-application-developer-and-applied-ai-engineer.md` — Where this module ends and the peer / adjacent tracks pick up.

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
