# Exercise 01 — Inline and Structured Citation Instruction Templates

**Estimated effort:** ~2 hours. Python, an LLM API (Claude, OpenAI, or a local model that supports structured output), an `AssembledContext` from a mod-107 exercise (or a mocked one; see setup), and ~20 eval queries with labeled reference answers.

## Objective

Author the three citation-instruction prompt patterns from chapter 3 — **inline bracketed**, **structured JSON**, and **answer-with-quote** — as reusable prompt fragments, run each over the same 20 eval queries, and produce an ablation that lets you defend one as the default for your stack.

This is the core citation deliverable for the module. At the end, you have three tested prompt templates, an ablation showing the trade-offs on your data, and the citation-instruction fragment the peer track (chapter 7) will inherit.

## Prerequisites

- Chapter 2 read (context rendering — you need the `render_context` function to feed into these prompts).
- Chapter 3 read (all three patterns).
- An `AssembledContext` shape (from mod-107 exercise 04) or an equivalent mock: `chunks[]` each with `anchor`, `text`, `source.title`, and optional `source.page`.
- 20 eval queries — pick from a public dataset ([Natural Questions](https://ai.google.com/research/NaturalQuestions), [HotpotQA](https://hotpotqa.github.io/), or your own domain corpus) or hand-author 20 that exercise varying claim counts (1-claim, multi-claim, multi-source).
- An LLM API. All three patterns need it; pattern 2 works best with structured-output support (Claude, GPT-4, `outlines`, `instructor`, etc.).

## Steps

**Step 1 — Assemble the shared setup (20 minutes).** In `citation_prompts.py`:

- Import (or mock) an `AssembledContext` builder that produces `K=5` chunks with numeric anchors `[1]`..`[5]`.
- Import (or write) a `render_context(ctx)` function that renders the context per chapter 2 (XML delimiters recommended). Same rendering for all three patterns.
- Author a helper `run(prompt, model)` that calls your LLM API and returns the string response.

**Step 2 — Author pattern 1 (inline bracketed) (20 minutes).** In `pattern_inline.py`, write the prompt template from chapter 3 pattern 1: the five rules, one training-shot with anchor `[X1]`, and the `{rendered_context}` + `{question}` slots.

Run it on 3 sample queries. Sanity-check: every claim in the answer has a `[n]` immediately before a period; the anchors resolve to real chunks; no `[X1]` leaks from the training-shot.

**Step 3 — Author pattern 2 (structured JSON) (30 minutes).** In `pattern_json.py`, write the JSON schema (`answer`, `citations[]` with `anchor`, `answer_span`, `quote`) and the prompt from chapter 3 pattern 2. Use your vendor's structured-output mode ([OpenAI JSON schema](https://platform.openai.com/docs/guides/structured-outputs), [Anthropic structured outputs](https://docs.claude.com/en/docs/build-with-claude/structured-outputs), `outlines`, `instructor`) if available; otherwise freeform-generate with a strict schema-shaped instruction and parse.

Run on 3 sample queries. Sanity-check: the JSON parses; every `anchor` is one of the rendered anchors; every `quote` is a substring of the cited chunk; the `answer_span` is a substring of `answer`.

**Step 4 — Author pattern 3 (answer-with-quote) (20 minutes).** In `pattern_quote.py`, write the prompt from chapter 3 pattern 3: the format `<claim>. (From [n]: "<verbatim quote>")`, the rules, and one training-shot.

Run on 3 sample queries. Sanity-check: every quote is verbatim (substring of the cited chunk); the format is consistent per claim; the LLM does not add prefatory "According to..." text.

**Step 5 — The ablation (30 minutes).** Run all 20 eval queries through each of the three prompt patterns. For each response, measure:

- **Citation rate** — fraction of load-bearing claims that carry a citation. Score manually on a sample if you don't have automated claim extraction; a simpler proxy is "number of `[n]` markers ÷ number of sentences."
- **Anchor validity** — fraction of `[n]` in the answer that resolve to a rendered anchor. (This is your future chapter 6 level-1 check; write the simple regex-plus-set-lookup.)
- **Quote fidelity** (patterns 2 and 3 only) — fraction of `quote` / verbatim-quote fields that are exact substrings of their cited chunk.
- **Token cost** — output token count per response.
- **Latency** — end-to-end LLM call time per response.
- **Readability** — subjective 1–5 score from you (rag-engineer) or a teammate on how presentable the raw response is for a user.

Aggregate to a table:

| Metric | Pattern 1 | Pattern 2 | Pattern 3 |
| --- | --- | --- | --- |
| Citation rate | | | |
| Anchor validity | | | |
| Quote fidelity | n/a | | |
| Token cost (median) | | | |
| Latency p50 | | | |
| Readability | | | |

**Step 6 — Choose a default (15 minutes).** In `citation-ablation.md`:

- One paragraph naming which pattern you'd default to for a user-facing chatbot on this stack, with the metric that decided.
- One paragraph naming which pattern you'd choose for a compliance / high-stakes flow, and why.
- One paragraph on the failure mode you saw most often (invented anchors, paraphrased quotes, JSON truncation, format drift on multi-claim answers, etc.), and what you'd do about it in a real deployment.

**Step 7 — Package the fragment (15 minutes).** In `prompt_fragments.py`, expose your chosen default as a reusable fragment:

```python
CITATION_INSTRUCTION_FRAGMENT = """..."""     # the rules block + training-shot

def build_generator_prompt(rendered_context: str, question: str, peer_system_prompt: str = "") -> str:
    return f"""{peer_system_prompt}

{CITATION_INSTRUCTION_FRAGMENT}

<documents>
{rendered_context}
</documents>

Question: {question}

Answer:"""
```

This is the fragment shape from chapter 7's boundary contract. Exercises 02 and 04 (and future exercises in mod-109/110) reuse it.

## Acceptance criteria

- `pattern_inline.py`, `pattern_json.py`, `pattern_quote.py` each contain a working prompt template with the training-shot(s) chapter 3 specifies.
- All three patterns produce well-formed responses on at least 20 eval queries (parses cleanly; no runtime errors).
- Ablation table filled in with real numbers for citation rate, anchor validity, quote fidelity (where applicable), token cost, latency, and readability.
- `citation-ablation.md` names a defaulted pattern with a defensible reason, an alternative for high-stakes flows, and the top failure mode.
- `prompt_fragments.py` exposes the chosen default as a reusable fragment matching chapter 7's contract shape.

## Stretch goals

- **Multi-source-per-claim.** Add a subset of eval queries whose reference answer requires citing 2+ sources per claim. Measure whether each pattern surfaces the multi-source case; add a second training-shot on pattern 1 if it misses.
- **Verbose anchors.** For pattern 1, run a side-by-side against a verbose anchor scheme (`[Source A: Widget Manual, p. 42]`). Measure token cost and readability delta.
- **Cross-model.** Run the same three patterns on two model families (e.g., Claude Sonnet + GPT-4). Compare which pattern each model handles most reliably. Chapter 3 notes model-family differences; measure them.
- **Failure-mode injection.** Manually swap the rendered anchors so the "correct" chunk has anchor `[3]` instead of `[1]`. Measure whether the LLM still cites correctly. This is a robustness check the chapter 6 verifier will catch, but the prompt shouldn't need it to.
- **Cache-friendly prompt shape.** Structure the prompt so the citation-instruction fragment is at the top (cacheable) and the rendered context is at the bottom (dynamic). Verify with your vendor's cache-utilisation metric.

## Notes

- Do NOT couple the citation prompt to a specific model. All three patterns should run against any chat-completion API without model-specific hacks. Model-family notes go in comments; the prompt string is portable.
- Anchor scheme consistency between the rendered context (chapter 2) and the prompt (chapter 3) is the single most common bug. If the rendered anchors are `[1]`, `[2]`, `[3]` and the prompt tells the LLM to cite `[Source 1]`, all three patterns fail. Match byte-for-byte.
- Pattern 3's "verbatim quote" is a substring check on the cited chunk. If the LLM's quote does not match (extra whitespace, a stray quote character), your fidelity metric will show it — that is the point. Do not normalise quotes silently; log every mismatch.
- For pattern 2, if your vendor does not support strict JSON schema mode, the LLM will occasionally produce invalid JSON. Have a retry-with-parse-error path (one retry, max) and log the parse failure rate as its own metric.
- The ablation is on your data. If your corpus is short passages (200-token chunks), pattern 3's quotes will be near-full-chunk; if your corpus is long chunks (800+ tokens), pattern 3 needs the LLM to select a short quote and sometimes picks wrong. This is exactly the trade-off chapter 3 names — measure it, do not guess.
