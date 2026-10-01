# AI Data Analyst

Upload a CSV or Excel file and get a profiled dataset, an automatic dashboard, and an agent you
can ask questions in plain English. The agent writes and runs its own SQL and pandas, draws
charts, hunts for anomalies and trends, checks its own answer against the evidence, and explains
what it found.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env         # then put your ANTHROPIC_API_KEY in it
python make_sample_data.py   # optional: writes sample_sales.csv to play with
python selftest.py           # optional: checks the data engine, no API key needed

streamlit run app.py
```

Open http://localhost:8501, drop a file in the sidebar, and the dashboard appears immediately —
profiling and charts run locally with no model call. The API key is only needed for the chat.

With `sample_sales.csv` loaded, try:

- *Revenue is down in Q3 2025 — find out why.* (the cause is a pricing fault in one channel)
- *Which customer segment has the best margin, and is the difference real?*
- *Now break that down by region.* — follow-ups resolve against the previous turn
- *Are there delivery times that look wrong?*

---

## What's in the box

| Layer | What it does | Where |
|---|---|---|
| Ingestion | CSV/TSV/Excel/Parquet/JSON, multi-sheet workbooks, column cleaning, type inference (numeric, datetime, categorical) | `core/dataset.py` |
| Profiling | Per-column stats, semantic types, IQR outliers, correlations, ranked quality flags | `core/profiling.py` |
| Dashboard | KPI selection, automatic chart choice from the profile shape | `core/dashboard.py` |
| Charts | Themed, spec-driven Plotly builder shared by the dashboard and the agent | `core/charts.py` |
| Tools | SQL, Python, profiling, charting, anomaly, trend and group-comparison tools | `core/tools.py` |
| Sandbox | Restricted `exec` with a blocklist, safe builtins and a timeout | `core/sandbox.py` |
| Agent | LangGraph state machine: plan → analyse → validate → explain | `core/agent.py` |
| UI | Upload, dashboard, chat, profile and SQL console | `app.py` |
| Self-test | Exercises loading, profiling, DuckDB, charts and the sandbox without a model | `selftest.py` |

---

## How the agent works

```
START ──▶ plan ──▶ analyst ⇄ tools
                      │
                      ▼
                  validate ──(issues found)──▶ repair ──▶ analyst
                      │
                    (pass)
                      ▼
                   explain ──▶ END
```

**plan** turns the question into 2–5 concrete steps naming real columns, and flags the main way
the result could mislead. Small talk and trivial follow-ups skip straight through.

**analyst** is the tool-calling loop. It sees the live schema and profile digest in its system
prompt on every turn, so it never has to guess column names.

**validate** re-reads only the tool outputs and the draft, then answers in JSON: does every number
appear in a tool result, does the aggregation answer the question, are caveats stated, is a
correlation being sold as a cause? A `revise` verdict sends the critique back to the analyst.
After `MAX_REPAIR_ATTEMPTS` the answer ships with the unresolved issues folded in as caveats
rather than looping forever.

**explain** rewrites verified findings for a business reader: answer first, then evidence, then
what would change the conclusion, then one specific next question.

### Tools available to the agent

| Tool | Use |
|---|---|
| `list_tables` | Schema, dtypes and sample rows for everything loaded |
| `describe_columns` | Distribution, missingness, outliers, quality flags |
| `run_sql` | Read-only DuckDB — aggregation, filtering, joins, window functions |
| `run_python` | pandas/numpy for statistics, reshaping, custom logic |
| `create_chart` | Spec-driven Plotly chart rendered straight into the chat |
| `detect_anomalies` | IQR / z-score / quantile outliers, plus what the outlier rows share |
| `analyze_trend` | Resample, fit a slope, quantify growth, volatility, peaks and biggest swings |
| `compare_groups` | Per-group stats, share of total, and an ANOVA significance check |

### Memory

Conversation state lives in a LangGraph `MemorySaver` keyed by `thread_id`, and the tools close
over one `AnalysisContext` holding the tables, the DuckDB connection and the charts. So
*"now split that by region"* works: the agent still has the table, the earlier query results and
the numbers it already reported. **New chat** in the sidebar starts a fresh thread and keeps the
data.

Memory is in-process — restarting the server clears it. For persistence across restarts, swap
`MemorySaver()` in `build_agent` for `SqliteSaver.from_conn_string("checkpoints.db")`.

---

## Configuration

Everything is environment-driven; see `.env.example`. The settings worth knowing:

- `LLM_PROVIDER` / `LLM_MODEL` — `anthropic` (default) or `openai`
- `MAX_TOOL_STEPS` — tool-calling turns before the agent must answer (default 12)
- `ENABLE_PLANNER`, `ENABLE_VALIDATOR` — turn either node off to cut latency and cost roughly in half
- `MAX_ROWS_RETURNED` — how much of a query result the model sees
- `PYTHON_TIMEOUT_S` — per-execution cap on agent-written code

## Extending it

- **New analysis capability**: write a function in `core/tools.py`, give it a docstring the model
  can act on and a pydantic args schema, and append it to the list in `build_tools`. Nothing else
  changes — the agent discovers it.
- **New chart type**: add it to `CHART_KINDS` and a branch in `build_chart`.
- **Different dashboard rules**: `_measure_score` decides which numeric column counts as the
  headline measure, and `build_auto_charts` decides what gets drawn.
- **Bigger data**: DuckDB is already doing the work. For files over a few hundred MB, register a
  Parquet path as a view instead of loading it into pandas.

## Limits worth knowing

- The Python sandbox blocks filesystem, network and interpreter-internals access and caps runtime,
  but `exec` is not a security boundary. Run untrusted uploads in a container you can throw away.
- The timeout uses `SIGALRM`, which is unavailable on Windows and inside Streamlit's worker
  thread, so it degrades to no limit there rather than crashing.
- Everything is held in memory for the session, so concurrent users each get their own copy.
- The validator catches unsupported numbers and overreaching claims. It does not catch a question
  that was well-answered but poorly framed — read the caveats.
