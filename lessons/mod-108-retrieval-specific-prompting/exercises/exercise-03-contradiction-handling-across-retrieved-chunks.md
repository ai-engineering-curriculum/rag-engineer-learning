# Exercise 03 — Contradiction Handling Across Retrieved Chunks

**Estimated effort:** ~2 hours. Python, an LLM API, the prompt fragment from exercises 01–02, and a hand-authored contradiction eval set of ~15 queries where the retrieved chunks disagree on a fact.

## Objective

Author the four contradiction-handling patterns from chapter 5 — **surface**, **prefer authoritative**, **prefer recent**, **hand off to resolver** — as prompt fragments, run each on a labeled contradiction eval set, and produce a diagnosis that lets you defend one (or a composed fallback chain) as the default for your stack.

At the end, you have four tested contradiction fragments, an ablation showing where each pattern wins, and the composed contradiction fragment that will ship in the peer boundary contract (chapter 7).

## Prerequisites

- Chapter 5 read.
- Chapters 2–4's prompt fragments in place (exercise 01 for citation, exercise 02 for refusal).
- A hand-authored contradiction eval set. Each entry has:
  - `query` — a question that would be answered differently by the two chunks.
  - `chunks[]` — two (or three) `AssembledChunk`-shaped chunks with different values for the same fact. Include the metadata fields patterns B/C depend on: `source.metadata["authority"]` (`official` / `community` / `informal`), `source.updated_at`.
  - `contradiction_type` — one of `factual`, `temporal_supersession`, `scoped`, per chapter 5's flavors.
  - `ground_truth` — the correct final answer under a stated resolution rule (e.g., "if prefer-recent, the 2025 value is correct").

  Aim for a mix: ~5 temporal-supersession queries, ~5 factual (same-date, different-values), ~5 scoped (different-region-or-context, appearing to disagree).

## Steps

**Step 1 — Confirm the retriever-side metadata (10 minutes).** Verify that `SourceRef.updated_at` and `SourceRef.metadata["authority"]` are populated in your assembler output. If not, ingest a small sample corpus with these fields set (mod-102 owns the pipeline; this exercise can use a mock). Chapter 2's rendering must include `updated="..."` and `authority="..."` attributes on the `<document>` tag.

**Step 2 — Author the four contradiction fragments (40 minutes).** In `contradiction_prompts.py`:

- `PATTERN_A_SURFACE` — the rule block from chapter 5 pattern A + the training-shot with the 2022/2025 warranty policy example.
- `PATTERN_B_AUTHORITY` — the rule block from chapter 5 pattern B + the official-vs-community example.
- `PATTERN_C_RECENT` — the rule block from chapter 5 pattern C, the 30-day-window clause, and the firmware-supersedes-spec example.
- `PATTERN_D_HANDOFF` — the rule block that instructs the LLM to emit a structured `{"status": "contradiction", "contradiction": {...}}` object instead of an answer, plus one example.

Each fragment is a self-contained string that can be dropped into the citation prompt from exercise 01 as an additional block.

**Step 3 — Run the eval on each pattern (30 minutes).** For each of the 15 contradiction eval queries, run all four patterns. Score each response:

- **Contradiction detected?** — did the LLM notice the disagreement (yes/no).
- **Behavior matched?** — did the LLM do what the pattern asked (surface / pick authoritative / pick recent / emit structured record)?
- **Ground-truth aligned?** — did the LLM's chosen answer match the labeled ground-truth resolution?
- **Silently-picked wrong?** — the worst outcome: LLM picked one source silently and it was wrong.

Aggregate to a per-pattern-per-contradiction-type table:

| Pattern | Factual | Temporal supersession | Scoped |
| --- | --- | --- | --- |
| A — surface | | | |
| B — prefer authoritative | | | |
| C — prefer recent | | | |
| D — hand off (structured) | | | |

Cells: fraction of queries where the LLM's response was ground-truth-aligned (or, for pattern A, correctly surfaced).

**Step 4 — Compose the fallback chain (20 minutes).** Chapter 5's composed default:

1. Prefer `official` over `community` over `informal` (pattern B).
2. If authority ties, prefer the more recent `updated_at` when it explicitly supersedes (pattern C).
3. Otherwise, surface the disagreement (pattern A).

Author `CONTRADICTION_FALLBACK_CHAIN` — a single prompt block that encodes all three branches. Include *three* training-shots — one per branch (authority-wins, recency-wins, surface).

Run the eval again with the composed fragment. Compare against the individual-pattern numbers.

**Step 5 — Diagnose the failure modes (15 minutes).** For every query where the composed fragment did the wrong thing:

- Was the LLM unable to *detect* the contradiction (the sources look similar and the LLM merged them)?
- Was the LLM able to detect but chose the wrong branch (authority attribute was noisy, or LLM ignored the `updated_at`)?
- Was the ground truth ambiguous (this happens; refine the eval set)?

Write up in `contradiction-diagnosis.md`:

- The per-pattern table.
- The composed-chain table.
- One paragraph per failure category with an example query.
- One paragraph on which of the three contradiction types your prompt handled worst, and why.

**Step 6 — Package the fragment (5 minutes).** In `prompt_fragments.py` (extending exercise 01's), export `CONTRADICTION_FRAGMENT` as your chosen default (composed chain, or a single pattern if the composed didn't clearly win). This is the fragment shape the peer track will inherit per chapter 7.

## Acceptance criteria

- 15-query contradiction eval set with labeled `contradiction_type` and `ground_truth` per entry.
- All four individual patterns (A, B, C, D) + the composed fallback chain implemented as prompt fragments.
- Per-pattern-per-type table filled in with real numbers on the eval set.
- `contradiction-diagnosis.md` names failure modes with example queries and identifies the weakest contradiction type on your prompt.
- Chosen default exported from `prompt_fragments.py` as `CONTRADICTION_FRAGMENT`.
- Retriever-side metadata (`updated_at`, `authority`) verified present in the assembled context; missing-metadata queries handled gracefully (fall back to pattern A).

## Stretch goals

- **Contradiction-detection preprocessing.** Before running the generator, run a small LLM classifier over the assembled context asking "are any two of these chunks contradicting on the same fact?" If yes, route to the contradiction fragment; if no, use the plain answer prompt. Measure whether this preprocessing reduces false positives (the plain prompt does not need to spend tokens on contradiction rules for non-conflicting queries).
- **Resolver step for pattern D.** Implement the resolver: a second LLM call (or a rule-based function) that takes the structured contradiction record from step 2's pattern D and applies the composed fallback chain in code, not in prompt. Compare accuracy of "code-based resolver" vs. "prompt-based fallback chain."
- **Scoped contradiction detection.** Chapter 5 notes that pattern B fails on scoped disagreement (region A vs. region B). Add explicit scope metadata (`scope="region-1"`) and a pattern-B variant that uses scope. Measure whether the scoped variant beats plain pattern B on scoped-type queries.
- **Cross-model stress test.** Run the composed chain on a smaller model. Chapter 5 notes that multi-branch rules are less reliable on smaller models; verify.
- **Contradiction as first-class UX.** For pattern D, mock up the peer track's rendering of the structured contradiction record (a two-column comparison, or an "I found two sources that disagree" panel). This is a joint exercise with an `applied-ai-engineer`-role teammate — it is the boundary this module hands to the peer.

## Notes

- The hand-authored contradiction eval set is a real deliverable; do not skimp on it. 15 queries with clean labels is more useful than 50 queries with fuzzy labels. Mod-109 formalises this eval-set discipline; this exercise is a preview.
- Do not confuse "the LLM emitted a contradiction record" (structured output) with "the LLM handled the contradiction" (semantic correctness). A pattern-D response that emits the record is only useful if a resolver step consumes it. Without the resolver, pattern D is worse than pattern A on user experience.
- Pattern C's 30-day-window heuristic is a starting point; tune per corpus (chapter 5). Firmware release notes need a narrower window; documentation drift needs a wider one.
- If a chunk's `authority` or `updated_at` is missing on your ingest pipeline, patterns B/C/D degrade gracefully to pattern A. This is fine and often the right behavior; log it, do not fail. Fixing the metadata is a mod-102 ingest change.
- On the composed chain, model behavior is much more reliable if you name the three branches explicitly in the training-shots ("Example 1 — authority wins"; "Example 2 — recency wins"; "Example 3 — surface"). Anonymous or interleaved examples confuse the LLM.
- If the LLM systematically ignores your `authority` or `updated_at` attributes (i.e., it applies its own definition of "authoritative"), the header rendering is probably wrong or the attribute name is inconsistent. Chapter 2's "use the attribute exactly, don't infer" rule applies here.
