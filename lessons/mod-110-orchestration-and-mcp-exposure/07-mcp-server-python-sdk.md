# Chapter 7 — Wrapping a retriever as an MCP server

## What you build in this chapter

By the end of this chapter you have a working Python MCP server that exposes your tuned retriever as an MCP `retrieve` tool, optionally serves source documents as resources, and can be run over either `stdio` (for a local client such as Claude Desktop or Claude Code) or Streamable HTTP (for a networked deployment).

The reference SDK is the official [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk), which ships two APIs in the same package: a low-level `Server` and a high-level `FastMCP` decorator surface modelled on FastAPI. Start with `FastMCP` — it is enough for almost every retrieval server and cuts the boilerplate to almost nothing.

## Install

```bash
pip install "mcp[cli]"
```

The `[cli]` extra installs the `mcp` command-line client — an interactive stdio client you can use to inspect, list, and call your own server without wiring up a real LLM. Use it from the first line of code.

## The minimal server

```python
# retrieval_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("retriever")

@mcp.tool()
def retrieve(query: str, k: int = 6) -> list[dict]:
    """Search the corpus for passages relevant to the query.

    Args:
        query: Natural-language question.
        k: Number of passages to return (1–20).
    """
    results = hybrid_search(query, k=min(max(k, 1), 20))
    return [
        {"text": r.text, "source": r.metadata["source_url"], "score": r.score}
        for r in results
    ]

if __name__ == "__main__":
    mcp.run()   # defaults to stdio transport
```

That's a functioning MCP server. Three details worth noticing:

1. **The docstring becomes the tool description.** `FastMCP` reads the docstring and the type hints and generates the `inputSchema` automatically. The model sees your docstring — write it as if it were the tool's user manual.
2. **Argument validation lives in the type hints.** `k: int` is enforced by the schema; the runtime clamp inside the function is defense in depth (an untrusted client could send `k=1000` if the schema were permissive).
3. **`mcp.run()` picks a transport based on how it's launched.** The default is `stdio`, which is what a locally-installed client (Claude Desktop, Claude Code, an IDE plugin) uses.

Run it from the local client:

```bash
mcp dev retrieval_server.py   # opens the SDK's inspector UI at http://localhost:5173
```

Or wire it into Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "retriever": {
      "command": "python",
      "args": ["/absolute/path/to/retrieval_server.py"]
    }
  }
}
```

## Structured content: what a tool should return

Returning a `list[dict]` (as above) works, but you often want the client to receive *structured content* alongside the text so it can render sources properly. The MCP protocol wraps tool results in a `content` array of typed items:

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("retriever")

@mcp.tool()
def retrieve(query: str, k: int = 6) -> list[TextContent]:
    """Search the corpus and return numbered passages with citations."""
    results = hybrid_search(query, k=min(max(k, 1), 20))
    return [
        TextContent(
            type="text",
            text=f"[{i+1}] {r.text}\n(source: {r.metadata['source_url']}, score={r.score:.3f})",
        )
        for i, r in enumerate(results)
    ]
```

`TextContent` is the common case. `ImageContent`, `AudioContent`, and `EmbeddedResource` (a link to a resource-URI) cover multimodal returns and citation-by-reference. See [MCP specification — content types](https://modelcontextprotocol.io/specification/server/tools/#tool-results).

## Adding resources for source documents

If you want a client to be able to fetch full source documents by ID (e.g., after the LLM has cited passage `[3]` from document 42), expose them as resources:

```python
@mcp.resource("corpus://document/{doc_id}")
def get_document(doc_id: str) -> str:
    """Return the full text of a source document by ID."""
    doc = corpus_store.get(doc_id)
    return doc.text

@mcp.resource("corpus://index")
def list_documents() -> str:
    """A JSON listing of every document ID and its title."""
    import json
    return json.dumps([{"id": d.id, "title": d.title} for d in corpus_store.all()])
```

The `{doc_id}` template makes the URI parameterised — `resources/read` with `corpus://document/42` invokes `get_document("42")`. Resources are content-addressable; the client caches them and can subscribe to change notifications.

Return retrieved *passages* as tool results (they are dynamic to the query); return full *documents* as resources (they are static to a URI). Mixing the two often confuses clients.

## Streamable HTTP transport

For a networked deployment — a shared retrieval service consumed by many MCP clients — swap the transport:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8000)
```

`mcp.streamable_http_app()` returns a Starlette ASGI application implementing the Streamable HTTP transport. Deploy it behind whatever HTTP server and TLS terminator you already use for other services.

Auth is your responsibility. Two common patterns:

- **Bearer token in `Authorization: Bearer …`** — the caller (an MCP client running inside an app) presents a token; you validate it in a Starlette middleware in front of the MCP app. Fine for machine-to-machine.
- **OAuth 2.1 (server-hosted).** The MCP spec references OAuth 2.1 for hosted servers with per-user auth. Your server is an OAuth resource server; the client obtains a token via an authorization-server flow appropriate to the deployment. See [MCP specification — authorization](https://modelcontextprotocol.io/specification/basic/authorization/).

<!-- needs-research: pin the exact MCP-authorization spec revision the curriculum recommends once the OAuth 2.1 profile stabilises upstream. -->

## Per-tenant filtering (the ACL seam)

If your corpus is multi-tenant, the retrieval tool has to *know who is asking*. `FastMCP` exposes a context object that carries request metadata:

```python
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("retriever")

@mcp.tool()
def retrieve(query: str, k: int, ctx: Context) -> list[TextContent]:
    """Search the corpus visible to the caller."""
    principal = principal_from_context(ctx)   # extract from bearer token / OAuth claims
    allowed_scopes = acl_lookup(principal)
    results = hybrid_search(query, k=k, filters={"scope": {"$in": allowed_scopes}})
    return [_to_text(r) for r in results]
```

Two rules from the security-surface section of chapter 06 apply here:

- **Do the ACL filter inside the retriever call**, so the results returned to the client are already scoped. Never return a superset and rely on the client to filter — the client is the LLM host, and the LLM will happily quote from anything you return.
- **Validate arguments.** Cap `k`, sanity-check filter keys, refuse unknown metadata fields. The tool argument comes from an LLM.

The full permission-aware retriever pattern is `mod-112`. This chapter shows only the seam where MCP hands you the principal.

## Streaming, progress, and logging

Retrieval calls are usually fast enough that streaming does not matter. Two cases where it does:

- **Sub-question decomposition.** The tool internally makes several retrievals; stream intermediate passages so the client can start rendering.
- **Long-lived index-side work.** A rebuild-index tool or a fine-grained crawl can send progress notifications while running.

`FastMCP` exposes `ctx.report_progress()` and `ctx.info()/warning()/error()` on the `Context` object:

```python
@mcp.tool()
async def deep_retrieve(query: str, ctx: Context) -> list[TextContent]:
    sub_qs = decompose(query)
    all_results = []
    for i, sub in enumerate(sub_qs):
        await ctx.report_progress(progress=i, total=len(sub_qs), message=f"Retrieving for: {sub}")
        all_results.extend(hybrid_search(sub, k=4))
    await ctx.info(f"Returned {len(all_results)} passages across {len(sub_qs)} sub-questions.")
    return [_to_text(r) for r in all_results]
```

See [MCP specification — progress notifications](https://modelcontextprotocol.io/specification/basic/utilities/progress/) and [logging](https://modelcontextprotocol.io/specification/server/utilities/logging/).

## Testing your server

Two independent surfaces to exercise:

1. **Protocol conformance.** The SDK ships an in-memory client:

    ```python
    from mcp import ClientSession, stdio_client, StdioServerParameters
    import asyncio

    async def main():
        params = StdioServerParameters(command="python", args=["retrieval_server.py"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert any(t.name == "retrieve" for t in tools.tools)
                result = await session.call_tool("retrieve", {"query": "winter break", "k": 3})
                assert not result.isError
                assert len(result.content) >= 1

    asyncio.run(main())
    ```

    This tests the wire protocol end-to-end without any LLM.

2. **Retrieval quality.** Reuse the judgment set and eval harness from mod-109. The MCP wrapper does not change what the retriever returns; you should regression-test the retriever, then separately smoke-test the MCP surface.

Do not test them together. When something breaks you want to know whether the retriever or the wrapper is at fault.

## Deployment shape

For a locally-installed tool (single user, single machine), `stdio` + a config entry in the host's client config is the whole deployment story.

For a shared, networked retrieval server:

- Package the Python code + your vector-DB client into a container.
- Run `uvicorn` (or the equivalent ASGI server) against `mcp.streamable_http_app()`.
- Terminate TLS at the ingress; enforce auth in a middleware before the MCP app.
- Expose metrics (calls per second, per-tool latency, error rate) — `FastMCP` does not do this for you, wire OpenTelemetry or Prometheus in the ASGI layer.
- Version the tool schema. Bumping `inputSchema` in a way that removes fields is a *breaking change* to every consumer; connect this to the versioning discipline you build in `mod-111-retrieval-mlops-versioning-cicd-drift`.

## Debugging notes

- **Use `mcp dev` first.** The inspector UI shows every JSON-RPC message. Nine out of ten "the client can't see my tool" bugs are a wrong module path in the client config, not a protocol issue.
- **Log JSON-RPC IDs.** When a call hangs, correlating client-side and server-side logs by JSON-RPC `id` is the fastest way to see which side is stuck.
- **Return `isError=True` for retrieval failures, not exceptions.** Uncaught exceptions become JSON-RPC errors; well-formed tool errors (an empty result, an invalid filter) should return `isError=True` with a human-readable message so the model can retry.
- **Rate-limit per client.** A misbehaving agent will call your retriever every turn. A middleware token bucket is cheap and prevents budget surprises.

## Summary

- The official MCP Python SDK ships `FastMCP` — decorator-based tool and resource registration modelled on FastAPI.
- Tool docstrings and type hints generate the JSON Schema the client and model see. Write them as documentation the LLM will read.
- Use tools for parameterised retrieval; use resources for content-addressable source documents.
- `stdio` for local single-user servers; Streamable HTTP for networked shared servers. Auth is your job.
- Do per-tenant filtering inside the tool. Validate arguments. Assume the caller is an LLM.
- Test the wire protocol with the SDK's own client; test retrieval quality with mod-109's judgment set. Do not conflate them.

Chapter 08 covers the direct-consumer alternatives — OpenAI, Anthropic, and Bedrock Converse tool endpoints — so you know when to reach for MCP and when a direct tool is enough.
