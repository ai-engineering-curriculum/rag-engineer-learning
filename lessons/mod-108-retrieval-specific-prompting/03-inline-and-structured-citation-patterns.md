# Chapter 3 — Inline and Structured Citation Patterns

## Why this chapter exists

Given a well-rendered context (chapter 2), the LLM will happily produce a fluent answer that ignores your anchors, cites nothing, and paraphrases the sources so thoroughly that no post-hoc check can trace a claim back to a chunk. Citation-instruction prompt patterns are what stop that. They force the LLM to attach a source anchor to every load-bearing claim in a shape a downstream checker can parse.

This chapter is the core deliverable of the module. Three citation shapes cover almost every production RAG stack:

1. **Inline bracketed citations** — the shape most familiar from Wikipedia and Perplexity: `The maximum temperature is 85 °C [1].`
2. **Structured JSON citations** — the shape most amenable to downstream automation: `{"answer": "...", "citations": [{"anchor": "[1]", ...}]}`.
3. **Answer-with-quote** — the shape most amenable to strict grounding evaluation: `The maximum temperature is 85 °C. (Quoted from [1]: "The maximum operating temperature is 85 °C.")`.

Each has a trade-off; each is authored differently. The goal of this chapter is to give you the templates, the model-facing instructions, and the trade-off you accept when you pick one.

## The three shapes in one picture

```
    Pattern 1 — Inline bracketed          Pattern 2 — Structured JSON       Pattern 3 — Answer-with-quote
    ────────────────────────────          ──────────────────────────        ──────────────────────────────

    The maximum operating temperature     {                                 The maximum operating temperature
    is 85 °C [1]. Firmware 4.2              "answer": "The maximum ...",    is 85 °C.
    introduced a limit-override            "citations": [                     Quote from [1]: "The maximum
    setting [2].                             {"anchor": "[1]", ...},           operating temperature is 85 °C."
                                             {"anchor": "[2]", ...}
                                           ]                                  Firmware 4.2 introduced ...
                                         }                                    Quote from [2]: "Firmware version
                                                                              4.2 introduced ..."
```

- Pattern 1 is human-readable, easy for the LLM to produce, harder to verify strictly.
- Pattern 2 is machine-readable, easy to parse, requires the model to hold JSON structure without truncation.
- Pattern 3 is the gold standard for grounding — the answer literally contains the source words — but the most token-heavy and the most awkward for users to read.

The pattern this module treats as the default is **pattern 1 for user-facing answers, augmented by pattern 2 or 3 for high-stakes flows** (compliance, medical, legal, regulated support). Chapters 4, 5, and 6 all assume you have chosen one.

## Pattern 1 — Inline bracketed citations

The pattern: every load-bearing claim in the answer ends with one or more bracketed anchors, matching the anchors the prompt rendered in the context.

### The prompt template

```
You are answering the user's question using ONLY the documents in <documents>.

Rules:
1. Every factual claim in your answer MUST cite at least one document by anchor.
   Use the anchor exactly as shown, e.g. [1], [2], [3].
2. Place the citation immediately after the claim it supports, before punctuation.
   Example: "The maximum temperature is 85 °C [1]."
3. If a single claim is supported by multiple documents, cite them all: "[1][3]".
4. If a claim is NOT supported by any document, do NOT include it in your answer.
5. Answer in complete sentences. Do not add commentary about the documents.

<documents>
  {rendered_context}
</documents>

Question: {question}
```

The five rules cover the failure modes:

- **Rule 1** is the citation contract — no claim without a cite.
- **Rule 2** is the format contract — the anchor position is where the verifier looks.
- **Rule 3** covers the multi-source case; without it, the LLM tends to cite only one source per claim even when several apply.
- **Rule 4** is a soft refusal for claim-level unsupported content. Chapter 4 covers the whole-answer refusal.
- **Rule 5** discourages the LLM from wrapping the answer in "According to the documents, ..." preambles that dilute the citation signal.

### The training-shot (one-shot example)

Add a single example after the rules and before the actual documents:

```
Example (for reference only — not the current question):

<documents>
  <document anchor="[X1]" source="Example Doc" page="1">
    The Foobar 3000 was released on 2024-05-15 and supports up to 10 concurrent users.
  </document>
</documents>

Question: When was the Foobar 3000 released and how many concurrent users can it handle?

Answer: The Foobar 3000 was released on 2024-05-15 [X1] and supports up to 10 concurrent users [X1].

End of example.
```

Two design details:

- The example anchor is `[X1]`, not `[1]`. Using the same anchor style as the real documents but a distinct namespace (`X`) prevents the LLM from mistaking the example for a real document and citing `[X1]` in a real answer.
- The example uses two claims from the same doc to show the LLM that repeated citation is expected.

One example is usually enough. If your eval shows the LLM missing citations on structurally complex answers, add a second example with (a) a "multi-source per claim" case and (b) a "no-support, so omitted" case.

### What this pattern does not solve

Inline bracketed citations do not, on their own, guarantee that the citation *supports* the claim. The LLM can write "The maximum temperature is 100 °C [1]" where `[1]` actually says 85 °C. The verification pattern in chapter 6 catches this; the prompt alone does not. Do not conflate "the LLM emitted a citation" with "the claim is grounded."

## Pattern 2 — Structured JSON citations

The pattern: the LLM produces a JSON object containing both the answer text and an explicit list of citations, each with the anchor it references and (optionally) the span of the answer it supports.

### The output schema

```json
{
  "answer": "The maximum operating temperature is 85 °C. Firmware version 4.2 introduced a temperature-limit override that can raise this to 90 °C for short intervals.",
  "citations": [
    {
      "anchor": "[1]",
      "answer_span": "The maximum operating temperature is 85 °C.",
      "quote": "The maximum operating temperature is 85 °C."
    },
    {
      "anchor": "[2]",
      "answer_span": "Firmware version 4.2 introduced a temperature-limit override ...",
      "quote": "Firmware version 4.2 introduced a temperature-limit override setting ..."
    }
  ]
}
```

Fields:

- `answer` — the human-readable answer with no inline citations. The peer track's UI renders inline `[1]` markers by mapping `answer_span` to `anchor`.
- `citations[].anchor` — one of the rendered anchors from the context.
- `citations[].answer_span` — the substring of `answer` that this citation supports.
- `citations[].quote` — the substring of the cited chunk that supports the span. (Optional; if omitted this collapses to a simpler shape.)

### The prompt template

```
You are answering the user's question using ONLY the documents in <documents>.

Return your answer as a JSON object with this exact schema:
{
  "answer": <string>,
  "citations": [
    {
      "anchor": <string, must be one of the document anchors shown>,
      "answer_span": <string, exact substring of "answer">,
      "quote":       <string, exact substring of the cited document>
    }
  ]
}

Rules:
1. Every substantive claim in "answer" must have at least one entry in "citations"
   whose "answer_span" covers that claim.
2. "anchor" must be an anchor from <documents>. Do not invent anchors.
3. "answer_span" must be an EXACT substring of the "answer" field.
4. "quote" must be an EXACT substring of the document identified by "anchor".
5. If the documents do not contain enough information to answer, return
   {"answer": "The provided sources do not contain enough information to answer this question.",
    "citations": []}.

<documents>
  {rendered_context}
</documents>

Question: {question}

JSON:
```

The four "exact substring" constraints are what make this pattern verifiable. A downstream checker can:

- Confirm every `anchor` matches a rendered anchor (fails on invented `[7]` when only `[1]`, `[2]`, `[3]` were rendered).
- Confirm every `answer_span` is a substring of `answer` (fails on paraphrased spans).
- Confirm every `quote` is a substring of the corresponding chunk (fails on paraphrased quotes and — this is the important one — on wrong-chunk citations, because the quote will not match).

### Model-side considerations

Structured output has three failure modes on the LLM side:

1. **JSON truncation.** Long answers may hit the max-token limit and cut off mid-JSON. Mitigations: use vendor structured-output mode ([OpenAI JSON schema mode](https://platform.openai.com/docs/guides/structured-outputs), Anthropic's [structured outputs guidance](https://docs.claude.com/en/docs/build-with-claude/structured-outputs)), leave headroom in the max-output-tokens budget, and if you use `stop_sequences` do not include `}`.
2. **Escaping errors.** Documents that contain quotes, backslashes, or newlines can trip the LLM's JSON escaping. Mitigations: structured-output modes handle this correctly; freeform generation does not. Prefer the structured mode if the vendor supports it for your target model.
3. **Repetition of `quote` inside `answer`.** Because the LLM sees the `quote` field, it sometimes copies the quote text into `answer` verbatim. This is not wrong; the peer track's UI can render it either way. Note the choice in the prompt or accept the behavior.

### When to use pattern 2

- Compliance, medical, legal, or regulated contexts where every claim must be independently verifiable and the answer is going to be reviewed programmatically.
- Any flow where the LLM's output feeds another LLM or another automated system (mod-110 MCP surfaces often prefer JSON).
- Any flow that runs the chapter 6 verifier strictly — the JSON is the verifier's input format.

## Pattern 3 — Answer-with-quote

The pattern: for every claim, the answer includes a verbatim quote from the cited chunk immediately after the claim. This is the shape Menick et al. formalized in the GopherCite paper ([arXiv:2203.11147](https://arxiv.org/abs/2203.11147)) and that quote-verification pipelines rely on.

### The prompt template

```
You are answering the user's question using ONLY the documents in <documents>.

For every claim, produce:
1. The claim in your own words.
2. Immediately after, in parentheses, the citation and a VERBATIM quote from the
   cited document that supports the claim.

Format:
   <claim>. (From [n]: "<verbatim quote from document [n]>")

Example (for reference only):

   The Foobar 3000 was released on 2024-05-15.
     (From [X1]: "The Foobar 3000 was released on 2024-05-15.")

Rules:
- The quote must be an EXACT substring of the referenced document.
- Do not paraphrase inside the quote. Paraphrasing goes in the claim; the source text
  goes in the quote.
- If a claim cannot be supported by a verbatim quote from any document, omit the claim.

<documents>
  {rendered_context}
</documents>

Question: {question}
```

### Trade-offs

- **Strongest grounding signal.** The verifier's job is a plain substring check; anything that fails is either a fabricated quote or a wrong-source citation.
- **Highest token cost.** Each claim is roughly doubled — the claim in the LLM's words plus the source quote.
- **Awkward for the user.** The reading experience is choppier than inline `[1]`; the peer track's UI usually renders this as a collapsible "show source" element next to each claim.
- **Sensitive to chunk length.** If chunks are long, the LLM has to select a short quote from a long source; sometimes it selects the wrong sentence. Sentence-level or short-chunk retrieval works better with this pattern.

### When to use pattern 3

- The generator is being evaluated with a strict faithfulness / attribution metric (RAGAS attribution, TruLens groundedness) and passing the metric is a hard requirement.
- Regulatory context where "the answer must be traceable to source text" is a compliance constraint.
- Research / knowledge-work products (legal research, medical literature search) where users expect and reward the extra rigor.

## Choosing between the three

| Property | Pattern 1 (inline) | Pattern 2 (JSON) | Pattern 3 (quote) |
| --- | --- | --- | --- |
| User-readable | ✓ | render via UI | choppy |
| Machine-parseable | regex-parseable | direct | direct |
| Verifier strictness | anchor exists only | anchor + span match | anchor + verbatim quote match |
| Token cost | lowest | medium (JSON overhead) | highest (2× claim size) |
| Model robustness | high (freeform) | medium (JSON truncation risk) | medium (quote-selection error) |
| Fits mod-110 MCP surface | ✗ | ✓ | ✓ |

Rule of thumb:

- **User-facing chatbot** → pattern 1, with pattern 2 hidden underneath if the peer track needs machine-readable citations.
- **Tool-in-a-pipeline** (mod-110 MCP tool exposed to another agent) → pattern 2.
- **High-stakes, verifier-first** (compliance, medical, legal) → pattern 3.

## Multi-source citations and citation density

Two questions come up on every citation prompt:

1. **How many sources per claim?** The best-supported prompts tell the LLM to cite *all* supporting sources per claim, not just one. This is rule 3 in pattern 1's template. If the eval shows the LLM only citing one source when two apply, add an explicit example with multi-source citation.

2. **How dense are the citations?** Every sentence? Every clause? Every distinct fact? The industry practice from Perplexity, Bing Chat, and Anthropic examples is roughly **one citation per distinct fact** — not every sentence and not every clause. The prompt phrase "every load-bearing claim" is calibrated for this: the LLM interprets "load-bearing" as "if removed, the answer changes."

If your users complain the answers are cluttered with citations, tighten the prompt to "every distinct factual claim" and add a negative example ("Do not cite obvious transitions like 'Additionally,...'"). If the verifier complains the answers are under-cited, add a positive example that shows multiple citations on a compound sentence.

## Instructing the LLM to use *only* the context

All three patterns share a shared instruction that gets its own paragraph in the prompt:

```
IMPORTANT: Base your answer ONLY on the documents above.
- Do not add facts from general knowledge, even if they are correct.
- Do not fill in gaps from what you know about the topic.
- If the documents contradict your prior knowledge, follow the documents.
```

Without this, the LLM will confidently add plausible facts that are not in the context and cite them anyway — inventing an anchor or, worse, citing a real anchor for a claim the anchor does not support. This instruction is retrieval-specific: no baseline prompt-engineering guide teaches "prefer the context over your knowledge," because non-RAG generators are usually the opposite.

The refusal case — the LLM has no support in the context — is chapter 4. Follow the two chapters together; the "only the context" instruction here and the "refuse when no context" instruction there are the two halves of the grounding contract.

## Anti-patterns

### 1. Prompting for citations without rendering anchors

The prompt says "cite the source" but the rendered context has no anchors. The LLM invents them or paraphrases the source names. Solution: chapter 2's anchor scheme must be present before chapter 3's citation instruction can be enforced.

### 2. Post-processing citations into the answer

The generator emits an uncited answer; a second LLM call is asked to "add citations" to the answer. This second call has no reliable way to attribute claims to sources — it re-guesses. Solution: cite at generation time; verify (chapter 6) after; do not re-cite.

### 3. Citing the retrieval scores

The LLM is given the reranker scores and told to prefer higher-scoring sources. It over-weights the score and under-weights the actual content. Solution: chapter 2's rendering rule — do not put retrieval scores in the prompt.

### 4. Mixing citation patterns in one answer

The LLM is given pattern 1 rules in one part of the prompt and pattern 2 rules in another. It picks one and gets it partially right. Solution: pick one pattern per response shape. If you need both (JSON for the API, inline for the UI), do a single generation in pattern 2 and let the peer track's renderer convert to inline.

## What you should be able to do after this chapter

- Author each of the three citation-instruction patterns (inline bracketed, structured JSON, answer-with-quote) with the appropriate rules, examples, and grounding instruction.
- Choose between the three patterns based on the readability / parseability / verifier-strictness trade-off.
- Instruct the LLM to cite all supporting sources per claim, not only the first one.
- Instruct the LLM to prefer the retrieved context over its prior knowledge and know how the refusal contract in chapter 4 completes this.
- Recognise the four citation anti-patterns (no anchors, post-hoc citing, citing scores, mixed patterns) and rewrite the prompt to avoid them.

## Further reading

- Menick, J., Trebacz, M., Mikulik, V., et al. (2022). *Teaching Language Models to Support Answers with Verified Quotes.* [arXiv:2203.11147](https://arxiv.org/abs/2203.11147) — GopherCite, the reference for pattern 3.
- Bohnet, B., Tran, V. Q., Verga, P., et al. (2022). *Attributed Question Answering.* [arXiv:2212.08037](https://arxiv.org/abs/2212.08037) — a systematic treatment of citation shape and attribution evaluation.
- Gao, T., Yen, H., Yu, J., and Chen, D. (2023). *Enabling Large Language Models to Generate Text with Citations.* EMNLP. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) — the ALCE benchmark, useful for choosing between citation shapes.
- OpenAI: [*Structured Outputs*](https://platform.openai.com/docs/guides/structured-outputs) — the vendor primitive that makes pattern 2 reliable.
- Anthropic: [*Structured outputs / JSON mode*](https://docs.claude.com/en/docs/build-with-claude/structured-outputs) — the Claude equivalent.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the eval framework that scores what these prompts produce.
