# Chapter 2 — Context Rendering and Anchoring

## Why this chapter exists

The assembler produced an `AssembledContext` — an in-memory object with `chunks[]`, per-chunk anchors, source metadata, and a measured token count. The LLM cannot read Python objects. Somewhere between the assembler's output and the model's token stream, the context has to be **rendered** into prompt text: with delimiters, headers, and an anchor scheme the LLM is told to cite.

Every subsequent pattern in this module — citation instructions, refusal, contradiction handling, verification — assumes the rendering is correct. If the LLM cannot tell where chunk 3 ends and chunk 4 begins, no citation prompt will save you. If the anchor in the rendered prompt is not the same string the prompt tells the LLM to cite, refusals will fire spuriously and citations will look hallucinated when they are only formatted differently.

This chapter is the mechanical layer under everything else.

## The rendering problem in one picture

```
    AssembledContext                    Prompt text (what the LLM sees)
    ────────────────                    ────────────────────────────────
    chunks[0].anchor  = "[1]"      ┌─►  "[1] (source: Widget Manual v3, p. 42)
    chunks[0].text    = "..."      │     The maximum operating temperature..."
    chunks[0].source  = ...        │
                                   │    "[2] (source: Support KB #4517, 2025-03-11)
    chunks[1].anchor  = "[2]"      ├─►    Firmware version 4.2 introduced a
    chunks[1].text    = "..."      │     temperature-limit override..."
    chunks[1].source  = ...        │
    ...                            │    "[3] ..."
                                   ▼
                                   ...
```

The rendering step has to answer four questions:

1. **Where in the prompt does the context go?** — system message, user message, a dedicated `<documents>` block, etc.
2. **How is one chunk visually separated from the next?** — delimiter chars, XML tags, whitespace.
3. **What appears in the per-chunk header?** — the anchor, plus whatever `SourceRef` fields help the LLM cite correctly (page, section, date).
4. **What anchor scheme is used?** — `[1]`, `[Source A]`, `<doc id="42">`, etc.

Each choice interacts with the model family, the length of the context, and the citation pattern chosen in chapter 3.

## Placement: system vs. user vs. a dedicated block

Three placement conventions are common in production RAG:

### Placement A — retrieved context in the system message

```
SYSTEM: You are a support assistant. Use ONLY the following documents...
        [1] ...
        [2] ...
        ...
USER:   What is the maximum operating temperature of the widget?
```

Pros: matches the "documents are settings, not conversation" mental model; some model families process the system message with slightly different attention patterns that can help grounding; keeps the user turn short and clean.

Cons: on many hosted APIs the system message is not cached the way a static system prompt is (because the retrieved context is per-query), so you lose the prompt-cache advantage of a stable system message; token accounting and cache eviction become per-query, not per-session. Also awkward for multi-turn conversations where the "documents" change turn-over-turn.

### Placement B — retrieved context in the user message

```
SYSTEM: You are a support assistant. When you answer, cite sources using
        the [n] anchors shown in the DOCUMENTS block.
USER:   <question>What is the maximum operating temperature of the widget?</question>

        <documents>
          [1] (Widget Manual v3, p. 42) The maximum...
          [2] (Support KB #4517, 2025-03-11) Firmware...
        </documents>
```

Pros: the system message stays static and cacheable across queries; each user turn carries its own documents; multi-turn is trivial. Compatible with prompt caching on both Anthropic and OpenAI.

Cons: some model families (especially older instruction-tuned ones) attend less consistently to material inside the user turn; long user messages may crowd out the actual question.

For most stacks this is the recommended default. Vendor guidance from [Anthropic on long-context prompts](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) explicitly recommends putting documents at the top of the user turn and the question after, and reports better attention to the documents in that shape.

### Placement C — a dedicated tool-use / structured message

If your generator is exposed as a tool-use loop (Anthropic tools, OpenAI function calling), a common shape is:

```
SYSTEM: You are a support assistant...
ASSISTANT (tool_use):  retrieve(query="widget temperature")
USER    (tool_result): {"chunks": [{"anchor": "[1]", ...}, ...]}
USER:                  answer the original question, citing anchors from the tool_result.
```

Pros: the retrieved context is naturally scoped to a tool call; multiple retrieval calls per turn compose cleanly; trace / observability is per-call. This is the shape that mod-110 (MCP-exposure) drives toward.

Cons: extra hop; the tool-result shape is model-specific; not all model families expose tool-result rendering that survives compression.

Choose Placement B by default. Move to A only if you measure grounding regressions in the user turn; move to C when the retriever is genuinely a tool call, not a hidden pipeline stage.

## Delimiters: keep them syntactically distinct from prose

The LLM has to find the boundaries between chunks reliably. Three delimiter conventions are common:

### 1. XML-style tags (recommended for Anthropic Claude)

```
<documents>
  <document anchor="[1]" source="Widget Manual v3" page="42">
    The maximum operating temperature is 85 °C ...
  </document>
  <document anchor="[2]" source="Support KB #4517" updated="2025-03-11">
    Firmware version 4.2 introduced a temperature-limit override ...
  </document>
</documents>
```

Anthropic's prompt-engineering docs explicitly recommend XML tags for structured content ([Use XML tags](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)), and Claude models are trained to attend to them. XML has three advantages:

- Unambiguous open/close boundaries.
- Attribute slots (`source=`, `page=`, `updated=`) that keep metadata out of the prose but visible to the LLM.
- Downstream code can regex the same tags for post-generation verification (chapter 6).

### 2. Markdown-style headers

```
## Document [1]  (Widget Manual v3, p. 42)
The maximum operating temperature is 85 °C ...

## Document [2]  (Support KB #4517, 2025-03-11)
Firmware version 4.2 introduced a temperature-limit override ...
```

Fine for most models. Slightly more ambiguous than XML if a chunk itself contains `##` (which happens for any document that includes Markdown-formatted text). Escape or normalize inside chunks before rendering.

### 3. Plain-text separators with an anchor prefix

```
[1] (Widget Manual v3, p. 42)
The maximum operating temperature is 85 °C ...

[2] (Support KB #4517, 2025-03-11)
Firmware version 4.2 introduced a temperature-limit override ...
```

Cheapest in tokens. Works for most models. Vulnerable when chunk content contains `[n]` — the LLM may confuse an in-chunk `[7]` with a boundary marker. Normalize such content on ingest (mod-102) or on assembly.

For any format, **do not omit a delimiter**. A prompt of the form "Documents: {chunk1}{chunk2}...{chunkK}" with no boundary between chunks is a recipe for the LLM to blur chunk contents together and cite the wrong anchor.

## The per-chunk header: what to include and what to leave out

The header for each chunk is the LLM's ground truth for "what is this chunk?" It should carry:

- **The anchor** the LLM will cite (`[1]`, `[Source A]`, `<doc id="42">`) — the string the prompt's citation instructions reference.
- **A short human-readable source label** — title of the doc, section, page. This gives the LLM enough context to distinguish similar chunks (two chunks from the same manual, different pages) and enough context to prefer the more-authoritative source (chapter 5).
- **A freshness signal** if you have one — `updated_at` on documents that change over time. This is what the contradiction-preference rule in chapter 5 uses.

It should *not* carry:

- **The full URL.** URLs are long, tokenize badly, and the LLM does not need them — the peer track's UI resolves the click. Keep URLs in the `SourceRef`, out of the prompt.
- **Free-text descriptions of the source's authority.** ("This is our official documentation." "This is a community forum post.") The LLM will over-weight this. Instead, encode authority as a structured field the contradiction-handling instruction in chapter 5 references (`authority="official"` vs. `authority="community"`).
- **Retrieval scores.** The rerank score, the BM25 score, the fusion rank — none of these belong in the prompt. The LLM does not know what to do with them and they compete for attention with the actual content.

A reasonable default header, in XML style:

```xml
<document anchor="[1]" source="Widget Manual v3" section="§4.2 Operating Envelope"
          page="42" updated="2024-08-15" authority="official">
  The maximum operating temperature is 85 °C ...
</document>
```

The same header in plain-text style:

```
[1] Widget Manual v3, §4.2, p. 42 (updated 2024-08-15, official)
    The maximum operating temperature is 85 °C ...
```

## The anchor scheme

The `AssembledContext.chunks[].anchor` from mod-107 chapter 7 is the scheme. This module inherits the choice; a quick summary of the trade-offs the assembler owns:

- **Numeric — `[1]`, `[2]`, `[3]`.** Compact. Compatible with almost all citation shapes. Vulnerable to LLM ambiguity when the chunk content contains bracketed numbers. Recommended default.
- **Letter — `[A]`, `[B]`, `[C]`.** Compact and less likely to collide with in-chunk content. Awkward when K > 26.
- **Named — `[Source A]`, `[Widget Manual]`.** More human-readable inside the LLM's answer; more tokens per citation; more likely to be paraphrased ("Widget Manual" → "the manual") and thus harder to post-verify.
- **Structured — `<cite doc_id="doc-4517" chunk_id="c-12"/>`.** Best for strict downstream parsing. Verbose. Only use with structured JSON output (chapter 3, pattern 2) so the LLM does not have to emit XML inside prose.

Whichever scheme you pick, **the rendered anchor and the citation-instruction anchor must be byte-identical strings**. If the prompt renders `[1]` and instructs the LLM to cite as `[Source 1]`, the LLM will either invent one, mix both, or drift. Chapter 6's citation-verifier will flag this — but it is cheaper to prevent it at render time.

## Rendering code sketch

A minimal renderer that produces Placement B + XML delimiters, given the `AssembledContext` shape from mod-107 chapter 7:

```python
def render_context(ctx: AssembledContext) -> str:
    lines = ["<documents>"]
    for c in ctx.chunks:
        attrs = [f'anchor="{c.anchor}"',
                 f'source="{_escape(c.source.title)}"']
        if c.source.section: attrs.append(f'section="{_escape(c.source.section)}"')
        if c.source.page:    attrs.append(f'page="{c.source.page}"')
        if c.source.updated_at: attrs.append(f'updated="{c.source.updated_at}"')
        if c.source.metadata.get("authority"):
            attrs.append(f'authority="{c.source.metadata["authority"]}"')
        lines.append(f'  <document {" ".join(attrs)}>')
        lines.append(_indent(c.text, 4))
        lines.append("  </document>")
    lines.append("</documents>")
    return "\n".join(lines)
```

A minimal renderer for Placement B + plain-text delimiters:

```python
def render_context_plain(ctx: AssembledContext) -> str:
    parts = []
    for c in ctx.chunks:
        header = c.anchor + " " + c.source.title
        if c.source.section: header += ", " + c.source.section
        if c.source.page:    header += f", p. {c.source.page}"
        if c.source.updated_at: header += f" (updated {c.source.updated_at})"
        parts.append(header + "\n" + c.text)
    return "\n\n---\n\n".join(parts)
```

Both leave the `SourceRef.url` off the prompt — that field is preserved in the `AssembledContext` for the peer track to render in the UI.

## A rendering ablation you should actually run

The three rendering choices interact with the model, the chunk size, and the K. Rather than guess:

- Take 20 eval queries.
- Render each retrieval result four ways: (i) plain-text delimiters + numeric anchors, (ii) XML delimiters + numeric anchors, (iii) XML delimiters + named anchors, (iv) tool-result placement (if applicable).
- Fix the rest of the prompt (chapter 3 citation instructions, chapter 4 refusal).
- Score citation-verifiability (chapter 6) and answer faithfulness across all four.
- Pick the winner. Do this again when you change model families — the answer moves.

## What "long context" does and does not change

Long-context models (100K, 200K, 1M) do not change any of the rendering choices in this chapter. They change how many chunks fit and how much lost-in-the-middle bites you (mod-107 chapter 8), not how each chunk is rendered.

They do change one thing about **placement**: some vendor docs recommend the question *after* the documents on very long inputs. Anthropic's [Long context tips](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) explicitly say to put documents at the top of the user turn and the question at the bottom, and cite internal experiments showing better attention in that order. Follow the vendor guidance for the model you are using; do not guess from first principles.

## Two anti-patterns to avoid

### 1. Concatenating chunks with no delimiter

The chunks are joined with `"\n\n"` and the LLM has no way to tell them apart. Citations reference `[1]` where the LLM has invented `[1]` because the prompt never said `[1]`; the anchor scheme is either invented by the LLM or absent from the answer entirely. Prevention: render with explicit delimiters, always.

### 2. Rendering a `SourceRef` field the LLM should ignore

The URL, the internal doc ID, the retrieval score, the vector similarity, the fusion rank — none of these belong in the LLM's context. Including them:

- Wastes tokens.
- Gives the LLM information it will over-weight ("this chunk has a higher score, so it must be right" — no).
- Leaks internal identifiers into the LLM's response that a downstream check may not recognize.

Prevention: render only the fields chapters 3, 4, 5 need. Everything else lives in the `SourceRef` for the peer track's UI.

## What you should be able to do after this chapter

- Choose a placement (system / user / tool-result) with defensible reasons, and know when to reconsider.
- Choose a delimiter style (XML / Markdown / plain-text) that matches your model family and chunk content.
- Author a per-chunk header that carries anchor + minimal source metadata + freshness + authority, and *nothing else*.
- Reason about the anchor scheme trade-offs (numeric / letter / named / structured) and defend a default.
- Recognise the two rendering anti-patterns (no delimiter, over-shared metadata) and correct them.

## Further reading

- Anthropic: [*Long context tips*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — placement, ordering, and prompt-cache guidance for document-heavy prompts.
- Anthropic: [*Use XML tags*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) — the model-family recommendation this chapter's XML style follows.
- OpenAI: [*Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching) — the constraint that pushes retrieved context out of the system message on OpenAI.
- Anthropic: [*Prompt caching*](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) — the equivalent constraint on Claude; both push toward "static system message, dynamic user message" as the default shape.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the reference for placement-within-context; complements this chapter's placement-in-prompt discussion.
- Mod-107 chapter 7 (this track) — the `AssembledContext` shape this chapter renders.
