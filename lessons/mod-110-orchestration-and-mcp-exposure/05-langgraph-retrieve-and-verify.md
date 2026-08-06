# Chapter 5 — LangGraph and the retrieve-and-verify loop

## Motivating case

A hybrid retriever returns four passages. A verifier LLM reads the passages, reads the query, and reports "the passages do not answer the question." What should the pipeline do?

An LCEL chain does not have a good answer. It can return "I don't know," but it cannot say "then rewrite the query and try again, up to two times, and stop if the second attempt also fails." That is a *cycle* over shared state, and cycles need a different orchestration primitive.

LangGraph is that primitive. This chapter introduces just enough of it to build the retrieve-and-verify loop that appears in `exercises/exercise-03-…`. See [LangGraph — concepts](https://langchain-ai.github.io/langgraph/concepts/) for the full picture.

## Core primitives in four sentences

- **`State`** is a typed dict (`TypedDict` or Pydantic model) that every node reads from and returns updates for.
- **Nodes** are functions `(state) -> partial_state`; LangGraph merges the returned partial into `State`.
- **Edges** connect nodes. *Normal edges* go always; *conditional edges* choose the next node based on the current `State`.
- **Checkpointers** persist `State` at every step so a graph run can resume, be inspected, or be replayed.

See [LangGraph — low-level API](https://langchain-ai.github.io/langgraph/concepts/low_level/).

## Modelling the retrieve-and-verify loop

The state carries the query, the retrieved passages, the verifier's verdict, and a rewrite counter:

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class RagState(TypedDict):
    query: str
    passages: list[dict]        # {text, source, score}
    verdict: Literal["ok", "insufficient"] | None
    attempts: int
```

Now four nodes:

```python
def retrieve(state: RagState) -> RagState:
    passages = hybrid_search(state["query"], k=6)
    return {"passages": [p.__dict__ for p in passages]}

def verify(state: RagState) -> RagState:
    verdict = verifier_llm.classify(state["query"], state["passages"])  # "ok" | "insufficient"
    return {"verdict": verdict}

def rewrite(state: RagState) -> RagState:
    new_q = rewriter_llm.rewrite(state["query"], state["passages"])
    return {"query": new_q, "attempts": state["attempts"] + 1}

def answer(state: RagState) -> RagState:
    resp = answerer_llm.answer(state["query"], state["passages"])
    return {"answer": resp}
```

And the graph:

```python
def route_after_verify(state: RagState) -> Literal["rewrite", "answer", "give_up"]:
    if state["verdict"] == "ok":
        return "answer"
    if state["attempts"] >= 2:
        return "give_up"
    return "rewrite"

def give_up(state: RagState) -> RagState:
    return {"answer": "I could not find a passage that answers this question."}

g = StateGraph(RagState)
g.add_node("retrieve", retrieve)
g.add_node("verify", verify)
g.add_node("rewrite", rewrite)
g.add_node("answer", answer)
g.add_node("give_up", give_up)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "verify")
g.add_conditional_edges("verify", route_after_verify,
                        {"rewrite": "rewrite", "answer": "answer", "give_up": "give_up"})
g.add_edge("rewrite", "retrieve")     # ← the cycle
g.add_edge("answer", END)
g.add_edge("give_up", END)

app = g.compile()
result = app.invoke({"query": "Which US state has the lowest corporate tax?",
                     "passages": [], "verdict": None, "attempts": 0})
```

The whole point of LangGraph fits in the two lines that create the cycle: `add_conditional_edges("verify", ...)` and `add_edge("rewrite", "retrieve")`. Those two lines are what LCEL cannot say cleanly.

## Checkpointing and streaming

Two features you will want in production, both one-liners:

**Checkpointer.** Persist `State` after every node so you can resume a paused graph, inspect a failed run, or replay from an intermediate step.

```python
from langgraph.checkpoint.memory import MemorySaver         # dev
# from langgraph.checkpoint.sqlite import SqliteSaver       # tests
# from langgraph.checkpoint.postgres import PostgresSaver   # production

app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "abc-123"}}
app.invoke({"query": "..."}, config)
history = list(app.get_state_history(config))               # every step's state
```

**Streaming.** Yield every node's output as it runs — the equivalent of `chain.astream` in LCEL, but with per-node granularity.

```python
for chunk in app.stream({"query": "..."}, stream_mode="updates"):
    print(chunk)   # {"retrieve": {...}} then {"verify": {...}} then ...
```

`stream_mode` also accepts `"values"` (full state after each step) and `"messages"` (LLM token stream). See [LangGraph — streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/).

## Design notes for retrieve-verify

The retrieve-verify pattern looks simple but has several tuning surfaces:

- **Rewrite budget.** The bound on `attempts` matters. Two attempts is a reasonable default; three costs three LLM calls per bad query and rarely helps. Log the empirical distribution of `attempts` per query and revisit.
- **Verifier prompt.** The verifier is another LLM call whose job is to say "the passages either do or do not contain the answer." Prompt-engineer this carefully — a permissive verifier turns the loop off; a strict verifier turns the retry loop on for every ambiguous case.
- **What the rewriter sees.** Feeding the rewriter the *passages that failed* helps it generate a different query rather than a synonym of the same one. Without them, you often get near-identical rewrites and the second retrieve returns the same passages.
- **Termination on "give up" matters.** A production graph should terminate cleanly with "I don't know" rather than looping to a limit and returning a random guess. The `give_up` node in the example encodes this.
- **Cost accounting.** A single retrieve-verify-rewrite-retrieve run is 2 retrievals + 1 verify + 1 rewrite + 1 answer LLM calls. Multiply by expected retry rate. Compare against the alternative of a stronger reranker on a wider first-pass retrieval.

## Where LangGraph earns its keep beyond this pattern

The retrieve-verify loop is a small graph. The same primitives scale to:

- **Self-correcting RAG** — retrieve, generate, check-citations, if any citation is fabricated or unverifiable, re-retrieve with the offending claim as the query, regenerate.
- **Multi-source escalation** — try a cheap corpus first; if the verifier says insufficient, fall back to a slower, more expensive corpus (SQL over a data warehouse, an external web search, a graph traversal).
- **Human-in-the-loop pauses.** LangGraph's `interrupt` primitive pauses the graph until an external event resumes it — useful for approval-gated retrieval or for injecting human labels mid-run. See [LangGraph — human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

Each of these is the same shape: a `State`, a set of node functions, a conditional edge, and a checkpointer. The primitives do not get more complicated; only the graphs do.

## Where LangGraph is the wrong tool

- **Genuinely acyclic pipelines.** If your retrieval is a straight line, LCEL's `|` is simpler and you get the same tracing without the graph vocabulary.
- **State that fits in a single dict is not the real cost.** The real cost is *reasoning about which nodes update which fields*. When your graph has more than ~10 nodes, the `State` schema tends to become a bag-of-anything and correctness moves out of the graph. That is when you should be splitting the graph, not adding another node.
- **The framework is not your agent.** LangGraph provides scaffolding to build agent-like flows, and its `create_react_agent` prebuilt is popular. But *agent design* — tool selection policy, memory, self-reflection strategy — is `agentic-ai-engineer` (level 40) territory. This module owns the retrieval-side loops only.

## Debugging notes

- **Draw the graph.** `app.get_graph().draw_mermaid()` returns Mermaid source you can paste into a rendered diagram. Do this once when you build the graph; refer back when it misbehaves.
- **Replay from a checkpoint.** With a checkpointer attached, `app.update_state(config, values={...}, as_node="verify")` lets you hand-edit state and re-run from a node. Invaluable when debugging a rare bad path.
- **Stream `values`, not `updates`.** During development, `stream_mode="values"` gives you the full state after every step, which is easier to eyeball than the deltas.
- **Time-limit rewrites.** Add a wall-clock cap in addition to the attempt cap. A verifier that keeps saying "insufficient" for legitimately-unanswerable questions will not terminate on `attempts` alone if you ever raise the bound.

## Summary

- LangGraph models a retrieval flow as a state machine: typed `State`, node functions, edges, conditional edges.
- The retrieve-and-verify loop is two lines of graph wiring — a conditional edge from `verify`, and a normal edge from `rewrite` back to `retrieve`.
- Checkpointers make runs resumable, inspectable, and replayable; streaming exposes per-node progress.
- Use LangGraph for cyclic flows, resumable state, or human-in-the-loop pauses. Do not use it for straight-line pipelines — LCEL is simpler.
- Deeper agent design (tool policies, planning, memory) is `agentic-ai-engineer` (level 40) territory, not this module's.

Chapters 06 and 07 leave the framework tour and start on the *exposure* half of the module: MCP and the tool contract.
