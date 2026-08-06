# Chapter 6 — The Model Context Protocol contract

## Why a new protocol at all

Every LLM host has had a "tool use" mechanism for two years now: OpenAI function-calling, Anthropic tool-use, Google function-calling, Bedrock Converse `toolConfig`. Each is a *proprietary* shape: the tool schema is passed as a request parameter, the model emits a vendor-specific tool-call structure, and your application dispatches it. If you build one retrieval tool and want to expose it to three hosts, you write three adapters.

The Model Context Protocol (MCP), introduced by Anthropic on 2024-11-25 and now maintained as an open specification, factors out the tool integration into a *server-to-client protocol*. The retrieval tool becomes an **MCP server**; the LLM host — a chat app, an IDE, a coding agent — becomes an **MCP client**. The client already knows how to speak MCP, so the tool integrates once and works everywhere the client runs.

For a retrieval engineer, MCP is the answer to the question "how do I ship my tuned retriever to consumers I don't own?" It is the piece of standardisation that lets a hard-earned retriever get *reused* rather than re-implemented in every downstream app.

See [Anthropic — Introducing MCP](https://www.anthropic.com/news/model-context-protocol) and the [MCP specification](https://modelcontextprotocol.io/specification).

## The mental model

```
   ┌───────────────┐   JSON-RPC 2.0     ┌───────────────┐
   │  MCP client   │◄──────────────────►│  MCP server   │
   │  (LLM host)   │                    │ (your service)│
   └───────────────┘                    └───────────────┘
     Claude Desktop                        retriever
     Claude Code                           GitHub tools
     IDE plugins                           filesystem
     agent runtimes                        …
```

- **Client** — a program that hosts an LLM and can call tools. It can connect to *many* servers at once.
- **Server** — a program that exposes some capability (tools, resources, prompts). It typically wraps one thing well.
- **Transport** — how the two talk. `stdio` (client spawns the server as a subprocess), `Streamable HTTP` (the current network transport), or the legacy `SSE` transport (deprecated in favour of Streamable HTTP).

The wire protocol is JSON-RPC 2.0. Every message is either a `Request` (has an `id`, expects a `Response`), a `Response` (has the same `id`, has `result` or `error`), or a `Notification` (no `id`, no response expected). That's it — MCP is a small vocabulary on top of well-understood plumbing.

## The capability surface

An MCP server can expose four capability families. A retrieval server usually needs only the first two.

### 1. Tools

A **tool** is a named function the model can *call* — it takes structured input, does something (often with side effects or with fresh reads), and returns structured output. Perfect fit for `retrieve(query, k, filters) -> passages`.

Advertised via `tools/list`; invoked via `tools/call`. Each tool has:

```json
{
  "name": "retrieve",
  "description": "Search the corpus for passages relevant to a query.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query":   {"type": "string",  "description": "The user's question."},
      "k":       {"type": "integer", "default": 6,  "minimum": 1, "maximum": 20},
      "filters": {"type": "object",  "description": "Optional metadata filters."}
    },
    "required": ["query"]
  }
}
```

`inputSchema` is JSON Schema. The client passes it to the model verbatim, so tool description and schema field descriptions are *prompt engineering*, not just documentation.

### 2. Resources

A **resource** is a named blob the client can *read*. Resources are addressed by URI (`corpus://document/42`, `file:///…`, `https://…`). The client can list them (`resources/list`), read them (`resources/read`), and subscribe to updates.

For a retriever, resources are useful when you want to expose the *source documents themselves* — not the retriever-selected passages, but the underlying artefacts a passage cites. A client can then fetch the full document on demand.

Tools vs. resources — which shape? Rule of thumb:

- **Tool** — the *action* is retrieval, parameterised by a query. Model chooses when to call.
- **Resource** — the *object* is a document, addressable by ID. Client (or user) chooses when to read.

A retriever usually exposes retrieval as a tool and *may additionally* expose source documents as resources so citations resolve. Most first cuts get away with tools alone.

### 3. Prompts

A **prompt** is a named, parameterised message template the client (or user) can invoke. Advertised via `prompts/list`, retrieved via `prompts/get`. Useful for "here is the standard grounded-answering prompt this server recommends" — a way for the server to teach the client how to use its tools well.

### 4. Sampling

An MCP server can *ask its client* to run an LLM call on its behalf (`sampling/createMessage`). This inverts the normal direction: the server borrows the client's model. Uncommon and often disabled by hosts on security grounds. Retrieval servers almost never need it.

See [MCP specification — server features](https://modelcontextprotocol.io/specification/server/) for the authoritative catalogue.

## The lifecycle

Every session goes through the same steps.

1. **Transport connect** — `stdio` (client spawns server as subprocess) or Streamable HTTP (server is an HTTP endpoint the client POSTs to).
2. **Initialize** — the client sends `initialize` with its supported protocol version and its capabilities. The server responds with its own version + capabilities. This is the version-negotiation handshake.
3. **Discovery** — the client calls `tools/list`, `resources/list`, and/or `prompts/list` to learn what the server offers.
4. **Use** — the client calls `tools/call`, `resources/read`, or `prompts/get` as the model chooses.
5. **Notifications** — either side may send unsolicited notifications: server sends `notifications/tools/list_changed` when the tool inventory changes, `notifications/progress` during long calls, `notifications/logging/message` for logs.
6. **Shutdown** — client-initiated `shutdown`, then transport disconnect.

See [MCP specification — lifecycle](https://modelcontextprotocol.io/specification/basic/lifecycle/).

## The transports, briefly

- **`stdio`** — client spawns your server as a subprocess and speaks JSON-RPC over stdin/stdout. Zero network exposure, zero auth complexity, perfect for a locally-installed tool (a filesystem server, a database probe). Not usable across the network.
- **`Streamable HTTP`** — a single HTTP endpoint that accepts `POST` for requests and streams responses (with `Content-Type: text/event-stream` for streaming, or a single JSON body for one-shot). This replaces the legacy dual-endpoint SSE transport. Use this for any server that must be reachable across the network. Auth is HTTP-standard (`Authorization` header; OAuth 2.1 for hosted servers).
- **`SSE`** (legacy) — a two-endpoint transport (one `GET` for the SSE stream, one `POST` for messages). Deprecated in favour of Streamable HTTP; existing clients still support it.

See [MCP specification — transports](https://modelcontextprotocol.io/specification/basic/transports/).

## The tool-call shape a client sees

When your retrieval server exposes a `retrieve` tool, a client-side conversation with an LLM includes it in the tool list. The model responds with a tool-call; the client dispatches it to your server as:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "retrieve",
    "arguments": {"query": "office winter break 2026", "k": 6}
  }
}
```

Your server returns:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [
      {"type": "text", "text": "[1] 2026-12-22 through 2026-01-02 (HR handbook §7.2)"},
      {"type": "text", "text": "[2] Company-wide closure: 22 Dec 2026 – 2 Jan 2027 (All-hands memo)"}
    ],
    "isError": false
  }
}
```

The `content` array can carry `text`, `image`, `audio`, or `resource` items. For retrieval the common cases are text blobs (with per-passage citation metadata inline) and resource links (so the client can fetch the underlying document on demand).

## What the emerging MCP ecosystem lets an agent consume

MCP's payoff is that any client-hosting environment supports many servers concurrently, and any well-formed server works in any client. As of writing, that ecosystem includes:

- **Claude Desktop and Claude Code** — Anthropic's official reference clients; used heavily for local tool integration.
- **IDE-hosted clients** — VS Code, Cursor, and other coding tools ship built-in MCP support so a project-local server (filesystem, database, docs retriever) auto-attaches.
- **Third-party agent runtimes** — LangGraph, LlamaIndex, and other frameworks now include MCP client adapters so an agent can consume any MCP server without vendor-specific glue.

For a retrieval engineer this means one server, many consumers. The tool you ship works in a code editor, a chat client, and a hosted agent runtime without a rewrite. That is the promise this module is built on.

<!-- needs-research: track the current authoritative registry / catalogue of MCP clients and servers on modelcontextprotocol.io and refresh the "ecosystem" list on the next content pass; naming and coverage will evolve. -->

## Security surface you own as the server

MCP inherits its security model from the transport plus the OS-level trust boundary. Two constraints matter for a retrieval server exposed over the network:

- **Authenticate and authorise every call.** MCP's Streamable HTTP transport uses standard HTTP auth (bearer tokens; OAuth 2.1). Your server has to know which principal is calling and which sub-corpora that principal is allowed to see — do not rely on the client to filter results. This connects directly to `mod-112-enterprise-retrieval-acl-structured-graph-multimodal`.
- **Do not trust tool arguments.** A tool argument comes from an LLM. Validate against your schema, cap `k`, canonicalise filter fields, and reject unknown keys. An untrusted client can hand you any JSON that fits the schema and some that does not.

Chapter 07 puts these constraints into the code.

## Summary

- MCP is an open, JSON-RPC-based protocol for LLM-host ↔ tool-server integration; introduced by Anthropic (2024-11-25) and now widely adopted.
- Servers expose four capability families: tools, resources, prompts, sampling. Retrieval servers typically use tools (and sometimes resources).
- The lifecycle is transport-connect → initialize → list → call → notifications → shutdown.
- Transports are `stdio` (local), Streamable HTTP (network), and legacy SSE.
- The advertised `inputSchema` is prompt engineering — the model sees your descriptions.
- Auth and per-tenant filtering are the server's job, not the client's.

Chapter 07 turns this into code with the official Python SDK.
