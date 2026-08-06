# Chapter 8 — Direct tool endpoints: OpenAI, Anthropic, Bedrock Converse

## When you don't need MCP

MCP is the right answer when the consumer of your retriever is a *client host you don't control* — Claude Desktop, an IDE plugin, someone else's agent runtime. But plenty of production RAG systems have exactly one consumer, and that consumer is a specific application talking directly to a specific LLM provider. In that case, MCP adds a hop that buys you nothing; you want the retriever exposed as a *native tool* in the provider's own tool-calling shape.

This chapter surveys the three shapes you will meet most often — OpenAI function calling, Anthropic tool use, and Bedrock Converse `toolConfig` — and how to wrap the same underlying retriever for each.

All three vendors converged on essentially the same idea: pass a JSON-Schema-typed tool spec to the model, receive a structured tool-call in the response, execute it, and feed the result back. The surface differences are cosmetic; the *semantics* are the same. Learn one thoroughly, then the other two are a config change.

## The one retriever, three wrappers

Start with the retrieval function you already have:

```python
def retrieve(query: str, k: int = 6, filters: dict | None = None) -> list[dict]:
    """Search the corpus for passages relevant to the query."""
    results = hybrid_search(query, k=min(max(k, 1), 20), filters=filters or {})
    return [
        {"text": r.text, "source": r.metadata["source_url"], "score": r.score}
        for r in results
    ]
```

Now the three vendor shapes.

## OpenAI — function calling / tools

The OpenAI Chat Completions API (and the newer Responses API) accepts a `tools` list; each tool is `{"type": "function", "function": {name, description, parameters}}` where `parameters` is a JSON Schema. When the model wants to call a tool, the response contains `tool_calls[]` with a `function.name` and JSON-encoded `function.arguments`. See [OpenAI — function calling guide](https://platform.openai.com/docs/guides/function-calling).

```python
from openai import OpenAI
import json

client = OpenAI()

retrieve_tool = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the corpus for passages relevant to a query. "
                       "Returns 1–20 passages, each with text, source URL, and score.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":   {"type": "string",  "description": "The user's question."},
                "k":       {"type": "integer", "default": 6, "minimum": 1, "maximum": 20},
                "filters": {"type": "object",  "description": "Optional metadata filters.",
                            "additionalProperties": True},
            },
            "required": ["query"],
        },
    },
}

messages = [{"role": "user", "content": "When is our office closed for winter break?"}]

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[retrieve_tool],
)
msg = resp.choices[0].message

if msg.tool_calls:
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = retrieve(**args)                             # your retriever
        messages.append(msg)                                  # assistant tool_calls turn
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
    # Second call: model consumes the tool result and drafts an answer.
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[retrieve_tool],
    )
    print(resp2.choices[0].message.content)
```

Points worth naming:

- **Two round-trips.** First call decides *whether* to use the tool; second call consumes the tool result and produces the answer. This is the same shape all three vendors use.
- **`tool_choice`.** Set `tool_choice={"type": "function", "function": {"name": "retrieve"}}` to force the model to call `retrieve`. Useful when you know retrieval is required and want to skip the deliberation.
- **Argument JSON is a string.** `function.arguments` is a JSON-encoded string, not an object. Parse it once, then validate against your own schema — the model is not guaranteed to return schema-conformant JSON, especially with older models.
- **Parallel tool calls.** GPT-4-class models may emit multiple `tool_calls` in a single response; loop over them.

## Anthropic — tool use

Anthropic's Messages API accepts a `tools` list; each tool is `{"name", "description", "input_schema"}` (note `input_schema`, not `parameters`). When the model calls a tool, the response `content` includes a `tool_use` block with `id`, `name`, and pre-parsed `input`. See [Anthropic — tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).

```python
import anthropic

client = anthropic.Anthropic()

retrieve_tool = {
    "name": "retrieve",
    "description": "Search the corpus for passages relevant to a query. "
                   "Returns 1–20 passages, each with text, source URL, and score.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query":   {"type": "string",  "description": "The user's question."},
            "k":       {"type": "integer", "default": 6, "minimum": 1, "maximum": 20},
            "filters": {"type": "object",  "description": "Optional metadata filters."},
        },
        "required": ["query"],
    },
}

messages = [{"role": "user", "content": "When is our office closed for winter break?"}]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=[retrieve_tool],
    messages=messages,
)

if resp.stop_reason == "tool_use":
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    result = retrieve(**tool_use.input)                       # your retriever
    messages.append({"role": "assistant", "content": resp.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(result),
        }],
    })
    resp2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[retrieve_tool],
        messages=messages,
    )
    print("\n".join(b.text for b in resp2.content if b.type == "text"))
```

Differences from OpenAI worth internalising:

- **`input_schema` vs. `parameters`.** Same JSON Schema, different key. Adapter code around a shared schema handles this cleanly.
- **`input` is pre-parsed.** Anthropic returns the tool input as a dict, not a JSON string. One fewer step; still validate.
- **`stop_reason == "tool_use"`.** The signal that the model wants a tool call. If it's `end_turn`, the model chose not to call.
- **Tool results go in a `user` message with `tool_result` content blocks.** This is a modelling difference from OpenAI's `tool` role; the *semantics* are the same.
- **`tool_choice`.** Same idea as OpenAI: `tool_choice={"type": "tool", "name": "retrieve"}` forces the call.

Anthropic's tool-use documentation also covers *parallel tool use*, *tool result caching*, and, for retrieval specifically, the newer built-in *web search* and *code execution* tools. Your custom `retrieve` composes with those.

## Bedrock — the Converse API `toolConfig`

Amazon Bedrock's **Converse** API is a unified, cross-model chat interface — one API for Claude on Bedrock, Llama on Bedrock, Nova, Mistral, and so on. Its tool shape is `toolConfig.tools[].toolSpec` with `{name, description, inputSchema.json}`. See [AWS — Bedrock Converse tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html).

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime")

tool_config = {
    "tools": [{
        "toolSpec": {
            "name": "retrieve",
            "description": "Search the corpus for passages relevant to a query. "
                           "Returns 1–20 passages, each with text, source URL, and score.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query":   {"type": "string",  "description": "The user's question."},
                        "k":       {"type": "integer", "default": 6, "minimum": 1, "maximum": 20},
                        "filters": {"type": "object",  "description": "Optional metadata filters."},
                    },
                    "required": ["query"],
                },
            },
        },
    }],
}

messages = [{"role": "user", "content": [{"text": "When is our office closed for winter break?"}]}]

resp = bedrock.converse(
    modelId="anthropic.claude-sonnet-4-6-v1:0",
    messages=messages,
    toolConfig=tool_config,
)

if resp["stopReason"] == "tool_use":
    tu = next(b["toolUse"] for b in resp["output"]["message"]["content"] if "toolUse" in b)
    result = retrieve(**tu["input"])                          # your retriever
    messages.append(resp["output"]["message"])
    messages.append({
        "role": "user",
        "content": [{
            "toolResult": {
                "toolUseId": tu["toolUseId"],
                "content": [{"json": {"passages": result}}],
                "status": "success",
            },
        }],
    })
    resp2 = bedrock.converse(
        modelId="anthropic.claude-sonnet-4-6-v1:0",
        messages=messages,
        toolConfig=tool_config,
    )
    print(resp2["output"]["message"]["content"][0]["text"])
```

Points worth naming:

- **The API is model-agnostic.** Change `modelId` from Claude to Nova to Llama; the tool-use protocol stays the same. This is Bedrock's strongest sales pitch for tool-heavy workloads.
- **Content is always a block list.** Every message's `content` is `[{type_specific_key: ...}, ...]`. There is no scalar-string shortcut.
- **Tool results carry `status`.** `success` or `error`; the model uses this to decide whether to retry. If your retriever returns empty, decide whether that is `success` with an empty list or `error` with a message — models handle the two differently.
- **`toolChoice`.** `{"auto": {}}` (default), `{"any": {}}` (force *some* tool), or `{"tool": {"name": "retrieve"}}` (force this tool). Same three options, different spelling.

## A cross-vendor comparison

| Concern | OpenAI | Anthropic | Bedrock Converse |
|---|---|---|---|
| Tool list key | `tools` (with `type: "function"`) | `tools` (flat objects) | `toolConfig.tools[].toolSpec` |
| Schema key | `parameters` | `input_schema` | `inputSchema.json` |
| Arguments type | JSON string | dict | dict |
| Tool-call location | `message.tool_calls[]` | `content[]` with `type: "tool_use"` | `content[]` with `toolUse` |
| Result role | `role: "tool"` | `user` message with `tool_result` block | `user` message with `toolResult` block |
| Signal that a tool is wanted | `finish_reason: "tool_calls"` | `stop_reason: "tool_use"` | `stopReason: "tool_use"` |
| Force a tool | `tool_choice=...` | `tool_choice=...` | `toolConfig.toolChoice=...` |
| Parallel calls | Yes (opt-in per model) | Yes | Yes |

The rows read like translation dictionary entries. A thin adapter layer around one shared schema is enough to serve all three.

## When to expose direct tools vs. MCP

Both. The two are not mutually exclusive.

- **MCP** — right when consumers are outside your control (many clients, many teams), when the retriever is one tool of many in a shared toolbox, or when integration parity across hosts is a goal.
- **Direct tool endpoints** — right when the consumer is one app you own, latency budget is tight, you want end-to-end control of the loop, or the deployment target is a serverless runtime and the extra process (an MCP server) is friction.

A common shape in practice: expose the retriever as an MCP server *and* as a direct tool your own app calls, both backed by the same `retrieve(...)` function. The MCP wrapper serves external clients; the direct wrapper serves your own tight-loop app; the retriever below them is one implementation.

## Notes that apply to all three

- **Descriptions matter more than schemas.** The model sees your descriptions. A vague description leads to bad argument choices no matter how strict your schema is. Write descriptions with worked examples of good inputs.
- **Cap `k` server-side, always.** All three vendors let the model pass anything that conforms to the schema. Untrusted input; clamp inside your function.
- **Return citations in the tool result, not just text.** The generation-side model has to cite your passages. Include `source`, `chunk_id`, `score`, and any metadata a downstream citation-verification step needs.
- **Force the tool for retrieval-required flows.** For "answer only from the corpus" apps, `tool_choice` set to your tool is safer than trusting the model to decide. Skip a round-trip, skip a class of "the model chose not to retrieve" bugs.
- **Test the tool loop with a stub retriever first.** Replace `retrieve` with a fixture that returns known passages. Confirm the client-side loop works. Then swap in the real retriever. Debugging both at once is unnecessarily hard.

## Summary

- OpenAI, Anthropic, and Bedrock Converse converge on the same tool-calling semantics with cosmetically different shapes.
- The same retriever function serves all three behind a thin adapter.
- Direct tools are the right choice when you own the consumer; MCP is the right choice when you don't; they compose.
- The description is prompt engineering. The schema is a safety net. Neither substitutes for the other.
- Server-side validation, capped parameters, per-tenant filtering, and citation-rich results apply to all three.

Chapter 09 closes the module by drawing the line between what this module owns and what the next role up owns.
