# Exercise 4 — Wrap a retriever as an MCP server (Python SDK)

**Time:** ~3 hours. Hands-on Python. Requires an MCP-capable client for end-to-end testing (Claude Desktop, Claude Code, or the SDK's inspector — the last is included in `pip install "mcp[cli]"`).

## What you are practising

Taking a tuned retriever and shipping it as a Model Context Protocol server that a real MCP client can discover and call. This is the exercise the module is really about — it is the point where "retriever" stops being a Python function and becomes an interoperable component.

## Problem statement

Build an MCP server that exposes your existing retriever as a `retrieve` tool. Additionally expose source documents as MCP resources so a client can dereference the citations. Support both `stdio` (for local client testing) and Streamable HTTP (for a networked deployment). Verify the server works against a real MCP client and against the SDK's programmatic in-memory client.

## Prerequisites

- A working retriever from any previous exercise in this module or track. Any retriever with a callable interface `retrieve(query, k=..., filters=...) -> list[...]` will do.
- `pip install "mcp[cli]" uvicorn`
- Optionally: Claude Desktop or Claude Code to test end-to-end against a real LLM host. The SDK inspector is enough to complete the exercise.

## Requirements

1. **A `FastMCP` server (`server.py`)** exposing:
    - **Tool `retrieve`** with typed arguments `query: str`, `k: int = 6`, `filters: dict | None = None`, and a docstring the LLM will read. Return a `list[TextContent]` where each item carries the passage text, source URL, and score in a stable format.
    - **Tool `search_summarize`** (optional but recommended) that internally retrieves and returns a concise LLM-side-friendly summary or a JSON structure. Serves as a second, more complex tool for testing multi-tool discovery.
    - **Resource `corpus://document/{doc_id}`** that returns the full text of a source document by ID.
    - **Resource `corpus://index`** that returns a JSON listing of every document ID and title.

2. **Argument validation and server-side clamps.** The `retrieve` tool clamps `k` to `[1, 20]`, rejects unknown filter keys with a clear error message, and returns `isError=True` on bad input (do not raise uncaught exceptions).

3. **A per-caller principal seam.** Even if you do not implement full auth, structure the server so a `principal` (or "None" for anonymous) is threaded through `retrieve` and used to scope `filters`. A commented `principal_from_context(ctx)` stub is acceptable; make it clear where real auth would plug in.

4. **Two transport entry points.**
    - `python server.py` runs stdio (for a local client).
    - `python server.py --http` runs `uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8000)` for a networked deployment.

5. **A programmatic client test (`test_server.py`)** using the SDK's in-memory `ClientSession` + `stdio_client`. The test must:
    - `initialize`.
    - `list_tools` and assert `retrieve` is present.
    - `call_tool("retrieve", {"query": "…", "k": 3})` and assert non-empty, non-error result.
    - `list_resources` and assert `corpus://index` is present.
    - `read_resource("corpus://index")` and parse the JSON.

6. **An end-to-end integration check.** Register the server with Claude Desktop *or* Claude Code (see the SDK docs for the config-file format) *or* run `mcp dev server.py` and interact via the SDK inspector. Take a screenshot or copy the transcript into `demo.md`.

7. **Deployment notes (`README.md`).** Cover:
    - How to install and run.
    - How to register with at least one MCP-capable client.
    - Which auth strategy you would recommend for Streamable HTTP and where in the code it would live (do not implement OAuth 2.1 end-to-end — just point at where it plugs in).
    - Which schema-versioning story you would adopt in production (chapter 09 gives the framing; `mod-111` has the mechanics).

## Starter guidance

- Follow chapter 07's snippets exactly for the initial scaffold, then customise the tool signature to match your retriever.
- Docstrings become tool descriptions the model sees. Write them like documentation — include an example of a good query and note the maximum `k`.
- Use `mcp dev server.py` first. The inspector shows every JSON-RPC message and catches most protocol mistakes before a client sees them.
- Return citation-rich `TextContent` items — the downstream LLM has to cite; give it what it needs.
- Do the per-tenant scoping seam even if you leave auth as a stub. It is easier to add auth to a scoped tool than to retro-fit scoping into an authed one.

## Acceptance criteria

The exercise is complete when:

1. `python server.py` runs a stdio MCP server that starts without errors.
2. `python server.py --http` runs the Streamable HTTP variant on a documented port.
3. `test_server.py` passes with a real invocation of both a tool and a resource read against the running server.
4. `mcp dev server.py` opens the inspector and both `retrieve` (tool) and `corpus://index` (resource) are visible and callable.
5. Sending `k=1000` to `retrieve` returns a well-formed `isError=True` result (or is clamped to 20 — document your choice).
6. An unknown filter key returns a well-formed error, not an uncaught exception.
7. `demo.md` shows at least one end-to-end interaction via a real MCP client or the SDK inspector (transcript or screenshot).
8. `README.md` documents install, run, client-registration, auth-strategy pointer, and versioning stance.

## Stretch goals

- **Real auth on Streamable HTTP.** Add a Starlette middleware that validates a bearer token against a fixture and maps it to a `principal`. Extend `test_server.py` to prove that a token missing the required scope is rejected.
- **Streaming progress.** Convert `retrieve` to `async` and emit `ctx.report_progress(...)` from within a hypothetical multi-step retrieval (sub-question decomposition or multi-hop) so the client sees intermediate progress.
- **Notifications on corpus change.** Emit `notifications/resources/list_changed` when the corpus is updated; subscribe from the test client and observe.
- **Deploy behind a container.** Write a `Dockerfile`, run it, and register the running container with a client. Note the OS-level and network-level surface you now own that stdio hid from you.
