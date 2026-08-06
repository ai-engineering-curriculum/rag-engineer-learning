# Exercise 5 — Wrap the same retriever as OpenAI and Anthropic tool endpoints

**Time:** ~2 hours. Hands-on Python. Requires OpenAI *and* Anthropic API keys (or, if only one is available, Bedrock Converse against Claude on Bedrock as the second vendor).

## What you are practising

Exposing one Python retriever behind two vendor-native tool-calling shapes and building the thin adapter layer that keeps them in sync. This is the pattern for direct-consumer deployment when MCP is overkill.

## Problem statement

Take the same `retrieve(...)` function you used in exercise 04 and expose it as (a) an OpenAI `function` tool and (b) an Anthropic tool. Build a shared JSON-Schema definition that both wrappers derive from, so the tool's schema lives in exactly one place. Verify each vendor's model calls the tool correctly on a small question set, and compare tool-call semantics side-by-side.

## Requirements

1. **`schema.py` — one JSON Schema, one description.** A single Python dict (or a Pydantic model exported to JSON Schema) that captures the tool's argument shape, defaults, ranges, and human-readable description. Both vendor wrappers import from here.

2. **`openai_tool.py`** — a function that:
    - Builds the OpenAI `tools=[...]` value from `schema.py`.
    - Runs a Chat Completions turn (or the Responses API — your choice, document it).
    - Handles the tool-call round-trip: parse `tool_calls[]`, invoke `retrieve(...)`, append the tool result as a `role: "tool"` message, and run a second call to get the final answer.
    - Returns `{"answer": str, "citations": list[dict], "tool_calls_made": int}`.

3. **`anthropic_tool.py`** — the parallel wrapper for Anthropic Messages API:
    - Builds the Anthropic `tools=[...]` value from `schema.py`.
    - Detects `stop_reason == "tool_use"`, invokes `retrieve(...)`, appends a `user` message with a `tool_result` content block, runs the second call.
    - Returns the same shape as `openai_tool.py`.

4. **`bench.py` — a shared harness.** Load ~10 questions (reuse the eval set from any earlier exercise or write fresh ones). Run each through both wrappers. Produce a side-by-side report `results.md` that lists per question:
    - The tool call each vendor made (name, arguments).
    - The number of tool-call round-trips each vendor took.
    - The final answer text from each.
    - Whether the citation URLs in the answer are present in the retrieved passages (a simple substring check is fine).

5. **A short prose comparison** in `results.md` (200–300 words) covering:
    - How the two vendors' tool-call shapes differed in practice.
    - Any question where one vendor called the tool with better arguments than the other.
    - The maintenance cost of running two wrappers against one retriever, and how the shared `schema.py` helped or did not.

## Starter guidance

- The chapter 08 snippets are close to what you need. Do not copy them wholesale — write the shared-schema step first, then derive both vendor payloads from it.
- Cap `k` inside `retrieve`, not in the vendor payloads. Vendors let the model send anything schema-conformant.
- For a fair comparison, use the same LLM tier where possible (e.g., `gpt-4o-mini` and `claude-haiku-4-5-20251001` — both mid-tier). Document your choice.
- Run each question at temperature 0 to reduce vendor-vs-vendor noise in the comparison.
- Log every tool call. The interesting differences are in the arguments the model chose, not in the answer prose.

## Acceptance criteria

The exercise is complete when:

1. `schema.py` defines the tool schema in one place and both `openai_tool.py` and `anthropic_tool.py` derive their payloads from it — grep confirms there are no duplicated field lists.
2. `openai_tool.py` and `anthropic_tool.py` each expose a single top-level function callable with a single question string and return the shared result shape.
3. `bench.py` runs the same question set through both wrappers and writes `results.md`.
4. `results.md` includes:
    - A side-by-side table with at least the tool arguments, tool-call count, and final answer per vendor per question.
    - A basic citation-verification column (present / not present in retrieved passages).
    - The 200–300-word prose comparison.
5. No API key or secret is committed. `.env.example` documents which variables are read.
6. `README.md` states which models you used, which SDK versions, and how to run the benchmark.

## Stretch goals

- **Add Bedrock Converse.** Extend the harness to a third vendor and update the side-by-side table. Note where the Converse shape diverges from OpenAI and Anthropic.
- **Add `tool_choice` forcing.** Re-run the benchmark forcing the tool on every question. How much does that change the tool-call count and answer quality?
- **Add a "no-tool" control.** Run the same questions through each vendor with no tools attached. Compare answer faithfulness (spot-check) — this quantifies how much the tool actually earned.
- **Emit a shared trace.** Log every LLM and tool call with a common schema (OpenTelemetry spans, or plain JSON) so a single dashboard can render both vendors' runs together.
