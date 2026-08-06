# Exercise 02 — Text-to-SQL Retrieval with Schema-Linking and a Safety Guard

**Estimated effort:** ~3 hours. Python, a local Postgres (or SQLite for the ultra-quick version), the Spider dev set or an equivalent small labeled question set, an LLM you can query (OpenAI / Anthropic / a local Llama via Ollama).

## Objective

Build a text-to-SQL retriever with three components — a schema linker that selects relevant tables per query, an LLM SQL generator (with LlamaIndex's `SQLTableRetrieverQueryEngine` *and* your own bare-bones implementation), and a safety guard that parses generated SQL and rejects unsafe queries. Evaluate execution accuracy on a labeled subset of Spider (or a domain-fit set you author yourself), then compare with-vs-without schema-linking, and with-vs-without few-shot examples.

By the end you have:

- `db/` — a Postgres or SQLite database loaded with a 5–20 table schema (a Spider DB, or a domain schema you author).
- `linker/schema_linker.py` — retrieves the top-N relevant tables + columns for a natural-language query.
- `generator/nlsql_llama_index.py` — LlamaIndex's `SQLTableRetrieverQueryEngine`.
- `generator/nlsql_bare.py` — your bare implementation (prompt → LLM → SQL string).
- `guard/safety.py` — the sqlglot-based safety check.
- `evals/execution_accuracy.py` — the eval loop.
- `RESULTS.md` — the comparison table.

## Prerequisites

- Chapter 4 read.
- Postgres 15+ with `psql` (or SQLite via `sqlite3`).
- Python 3.10+; `pip install sqlglot llama-index llama-index-embeddings-huggingface sqlalchemy` (plus the driver — `psycopg` or `pysqlite3`).
- An LLM you can query. OpenAI/Anthropic API access is easiest; Ollama with a decent-sized coder model (Qwen2.5-Coder-14B or similar) works locally.

## Steps

### Step 1 — Load a database (20 minutes)

Pick one:

- **Spider DB.** Download the [Spider dataset](https://yale-lily.github.io/spider). It ships SQLite databases for each domain. Pick one with 5–15 tables (`concert_singer`, `car_1`, `orchestra` are good).
- **Domain-fit DB.** Author your own: 5–15 tables with foreign keys, ~1,000–100,000 rows total. A synthetic e-commerce schema (customers, orders, products, order_items, categories, addresses) is a solid choice.

Load into Postgres or keep in SQLite. Add column comments (`COMMENT ON COLUMN customers.tier IS '...'`) — the schema linker will use them.

### Step 2 — Build the schema linker (35 minutes)

`linker/schema_linker.py`:

- For each column in the schema, build a text description: `"Table {t.name} column {c.name}: {c.comment}. Type: {c.type}. Sample values: {top_5_distinct_values}."`
- Embed each description with a small local embedder (`all-MiniLM-L6-v2`).
- Store in a vector index (Chroma, FAISS in-memory, or the LlamaIndex default).
- `link(question, top_k=20)` returns the top-K columns; group by table and return the set of tables covered.

Sanity-check by hand: for 5 hand-authored questions, confirm the linker returns the tables you'd expect.

### Step 3 — LlamaIndex `SQLTableRetrieverQueryEngine` (25 minutes)

`generator/nlsql_llama_index.py`: follow the [LlamaIndex SQL guide](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/) shape.

```python
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core import VectorStoreIndex
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://..." )  # or sqlite://
sql_db = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(sql_db)
table_schemas = [SQLTableSchema(table_name=t) for t in sql_db.get_usable_table_names()]
obj_index = ObjectIndex.from_objects(table_schemas, table_node_mapping, VectorStoreIndex)
qe = SQLTableRetrieverQueryEngine(sql_db, obj_index.as_retriever(similarity_top_k=3))
```

Wrap so you can call `qe.query(question)` and capture both the generated SQL (`response.metadata["sql_query"]`) and the result rows.

### Step 4 — Bare implementation (25 minutes)

`generator/nlsql_bare.py`: implement the pipeline yourself so you understand what the framework abstracts:

1. `linker.link(question)` → relevant tables.
2. Build a compact schema snippet — CREATE TABLE statements for those tables plus FK relationships.
3. Assemble a prompt: task instructions + schema snippet + 3–5 few-shot examples + the question. Prompt shape is close to `"Given this schema:\n{schema}\n\nQuestion: {q}\n\nWrite a single Postgres SQL SELECT that answers the question. Reply with just the SQL, no prose."`
4. Call the LLM at `temperature=0`.
5. Return the SQL string.

Compare hand-authored SQL to what your bare version generates for 3–5 questions before proceeding.

### Step 5 — Safety guard (25 minutes)

`guard/safety.py`:

```python
import sqlglot
from sqlglot import exp

ALLOWED_FUNCTIONS = {"count", "sum", "avg", "min", "max", "date_trunc",
                     "extract", "coalesce", "cast", "upper", "lower", "substr", "length"}

def safety_check(sql, dialect, allowed_tables):
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        return f"parse error: {e}"
    if not isinstance(tree, exp.Select):
        return "only SELECT allowed"
    for table in tree.find_all(exp.Table):
        if table.name.lower() not in allowed_tables:
            return f"disallowed table {table.name}"
    for star in tree.find_all(exp.Star):
        return "SELECT * not allowed"
    for func in tree.find_all(exp.Anonymous):
        if func.name.lower() not in ALLOWED_FUNCTIONS:
            return f"disallowed function {func.name}"
    return None
```

Also enforce a `statement_timeout` on the execution side (`SET statement_timeout = '5s'` for Postgres) and wrap execution in `LIMIT 10000`. Verify:

- A `SELECT` that's fine passes.
- A `DROP TABLE` fails at the parser check.
- A `SELECT *` fails at the star check.
- A `SELECT foo FROM system_users` fails at the table-whitelist check.
- A query that would take >5s (e.g., a cartesian product on a big table) is killed by the timeout.

### Step 6 — Author the eval set (25 minutes)

Two options:

- **Use Spider dev.** For your chosen DB, pull the ~30–100 questions Spider dev has for it. Each comes with `gold_sql`.
- **Author your own.** For your domain-fit DB, write 30–50 `(question, gold_sql)` pairs covering: simple filter, aggregate, join, groupby, top-N, time-window, and a couple of tricky-but-realistic cases.

Save as JSONL with `{"question": "...", "gold_sql": "..."}` per line.

### Step 7 — Execution-accuracy evaluation (30 minutes)

`evals/execution_accuracy.py`:

For each eval item, for each generator variant:

1. Generate SQL (via LlamaIndex, or via bare, or via bare-without-linking, or via bare-without-few-shot).
2. Run the safety guard. If it rejects, mark as `guard_reject`.
3. Execute against the DB. Capture rows.
4. Execute `gold_sql` against the DB. Capture gold rows.
5. Compare: sorted equality of the result sets (or order-sensitive if `gold_sql` has `ORDER BY`).

Report per-variant:

- Execution-accuracy rate.
- Guard-reject rate.
- Execution-error rate.

Include the variants:

- LlamaIndex full.
- Bare + linker + few-shot.
- Bare + linker (no few-shot).
- Bare + all-tables (no schema link).
- Bare, minimal prompt (no linker, no few-shot).

### Step 8 — Write up `RESULTS.md` (15 minutes)

Publish the table of variants × metrics. One paragraph on what schema-linking contributed. One paragraph on what few-shot contributed. One paragraph on what the guard rejected (list a few queries and the reason).

## Acceptance criteria

- All five components implemented and callable.
- The safety guard passes all five test cases in step 5 (good SELECT passes; DROP, `SELECT *`, disallowed table, and long-running query all rejected).
- Execution accuracy is measured on ≥ 30 questions.
- `RESULTS.md` has the variant × metric table and one-paragraph summaries per axis (linker, few-shot, guard).
- The bare implementation's prompt and few-shot examples are in the repo (not hidden in a framework config).

## Stretch goals

- **Add a self-correcting loop.** If the generated SQL errors on execution, feed the error message back to the LLM and ask for a fix; cap at 3 retries. Measure the accuracy uplift.
- **Compare LlamaIndex vs. LangChain SQL agent.** Wire the [LangChain SQL agent](https://python.langchain.com/docs/tutorials/sql_qa/) as a sixth variant; note the agentic pattern's cost / latency vs. the single-shot pattern.
- **BIRD subset.** Run against a small BIRD subset ([BIRD leaderboard](https://bird-bench.github.io/)); notice the accuracy drop on realistic schemas.
- **Row-level ACL.** Add a `tenant_id` column to your tables; connect as a DB user whose role has a row-level security policy on `tenant_id`; verify the generated SQL cannot exfiltrate cross-tenant data even if the LLM tries.
- **Column-level FLS.** Grant/deny `SELECT` on specific columns (e.g., `employees.salary`); verify the guard warns / rewrites queries that select them.
- **A domain-fit set.** Author 50 questions against a schema *you* own (a hobby-project DB, your team's dev DB); measure how accuracy differs from Spider.

## Notes

- **The DB grant is the primary defense.** The safety guard is defense in depth. If you connect as `postgres` superuser to iterate faster, refactor before submission to connect as a role with the minimum grants your query set needs.
- **Do not evaluate on the training questions.** If you used any of Spider's `train` set for few-shot examples, do not include them in the eval set — that's data leakage.
- **`sqlglot` normalizes dialects.** If your generator emits SQL Server or MySQL syntax and you're running Postgres, `sqlglot.transpile()` can help; but a better fix is to tell the LLM the target dialect in the prompt.
- **Small models struggle at this.** If you're using Ollama with a 7B–14B model, expect execution-accuracy well below what GPT-4o or Claude Sonnet achieves. Document the model in your `RESULTS.md`; the *comparison across variants* is what matters, not the absolute number.
