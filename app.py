"""AI Data Analyst — Streamlit front end.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import traceback
import uuid

import pandas as pd
import streamlit as st

from core.agent import build_agent, message_text, stream_steps
from core.charts import register_theme
from core.config import SETTINGS
from core.dashboard import build_auto_charts, build_kpis, suggested_questions
from core.dataset import AnalysisContext, excel_sheet_names, load_dataframe
from core.profiling import profile_dataframe

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="◔",
    layout="wide",
    initial_sidebar_state="expanded",
)
register_theme()

NODE_LABELS = {
    "plan": "Planning the approach",
    "analyst": "Running the analysis",
    "tools": "Querying the data",
    "validate": "Checking the result against the evidence",
    "repair": "Fixing what the check flagged",
    "explain": "Writing the answer",
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink:#0E1419; --panel:#151D25; --line:#26333F;
  --text:#E6EDF3; --muted:#8CA0B3;
  --teal:#3FBFA8; --amber:#E2A03F; --red:#E06C6C;
}
html, body, [class*="css"], .stApp { background:var(--ink); color:var(--text);
  font-family:'IBM Plex Sans', system-ui, sans-serif; }
section[data-testid="stSidebar"] { background:#111920; border-right:1px solid var(--line); }
h1,h2,h3,h4 { font-weight:600; letter-spacing:-0.01em; color:var(--text); }

.masthead { display:flex; align-items:baseline; gap:.9rem; padding:.2rem 0 1.1rem;
  border-bottom:1px solid var(--line); margin-bottom:1.4rem; }
.masthead h1 { font-size:1.45rem; margin:0; }
.masthead span { color:var(--muted); font-size:.87rem; }

.kpi { background:var(--panel); border:1px solid var(--line); border-left:2px solid var(--muted);
  padding:.85rem 1rem; height:100%; }
.kpi.good { border-left-color:var(--teal); }
.kpi.warn { border-left-color:var(--amber); }
.kpi.bad  { border-left-color:var(--red); }
.kpi .label { color:var(--muted); font-size:.74rem; margin-bottom:.35rem; }
.kpi .value { font-family:'IBM Plex Mono', monospace; font-size:1.5rem; font-weight:500;
  line-height:1.15; font-variant-numeric:tabular-nums; }
.kpi .detail { color:var(--muted); font-size:.72rem; margin-top:.3rem; }

.flag { display:flex; gap:.7rem; padding:.55rem .2rem; border-bottom:1px solid var(--line);
  font-size:.86rem; align-items:baseline; }
.flag .sev { font-family:'IBM Plex Mono', monospace; font-size:.68rem; min-width:62px; }
.flag .col { font-family:'IBM Plex Mono', monospace; color:var(--text); min-width:150px; }
.flag .txt { color:var(--muted); }
.sev-high{color:var(--red);} .sev-medium{color:var(--amber);} .sev-low{color:var(--muted);}

.stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background:transparent; padding:.4rem 0; color:var(--muted); }
.stTabs [aria-selected="true"] { color:var(--text); border-bottom:2px solid var(--teal); }

.stChatMessage { background:transparent; border-bottom:1px solid var(--line);
  padding-bottom:.9rem; }
code, pre, .stDataFrame { font-family:'IBM Plex Mono', monospace; }
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono', monospace; }
.stButton button { background:var(--panel); color:var(--text); border:1px solid var(--line);
  font-size:.82rem; }
.stButton button:hover { border-color:var(--teal); color:var(--teal); }
.empty { border:1px dashed var(--line); padding:2.6rem 2rem; text-align:center;
  color:var(--muted); }
.empty strong { color:var(--text); display:block; margin-bottom:.4rem; font-size:1.05rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("ctx", AnalysisContext())
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("chat", [])          # [{role, content, charts:[id], tools:[...]}]
    ss.setdefault("agent", None)
    ss.setdefault("checkpointer", None)
    ss.setdefault("loaded_files", set())
    ss.setdefault("pending_question", None)


init_state()
ctx: AnalysisContext = st.session_state.ctx


def get_agent():
    """Build the graph on demand. The checkpointer outlives the agent, so adding a
    table mid-conversation rebinds the tools without losing the thread."""
    if st.session_state.get("checkpointer") is None:
        from langgraph.checkpoint.memory import MemorySaver

        st.session_state.checkpointer = MemorySaver()
    if st.session_state.agent is None:
        st.session_state.agent = build_agent(ctx, checkpointer=st.session_state.checkpointer)
    return st.session_state.agent


def reset_conversation() -> None:
    """Fresh thread, same data. The old thread's checkpoint is simply orphaned."""
    st.session_state.chat = []
    st.session_state.thread_id = uuid.uuid4().hex


@st.cache_data(show_spinner=False)
def cached_profile(table_name: str, fingerprint: str, _df: pd.DataFrame) -> dict:
    return profile_dataframe(_df)


def profile_for(name: str) -> dict:
    table = ctx.tables[name]
    fingerprint = f"{len(table.df)}-{len(table.df.columns)}-{hash(tuple(table.df.columns))}"
    if not table.profile:
        table.profile = cached_profile(name, fingerprint, table.df)
    return table.profile


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data")
    uploads = st.file_uploader(
        "Add a dataset",
        type=["csv", "tsv", "xlsx", "xls", "xlsm", "parquet", "json"],
        accept_multiple_files=True,
        help="CSV, Excel, Parquet or JSON. Upload several files to query across them.",
    )

    for up in uploads or []:
        key = f"{up.name}:{up.size}"
        if key in st.session_state.loaded_files:
            continue
        raw = up.getvalue()
        if len(raw) > SETTINGS.max_upload_mb * 1_000_000:
            st.error(f"{up.name} is over the {SETTINGS.max_upload_mb} MB limit.")
            continue
        try:
            sheets = excel_sheet_names(raw) if up.name.lower().endswith((".xlsx", ".xls", ".xlsm")) else []
            frames = (
                {s: load_dataframe(raw, up.name, sheet=s) for s in sheets}
                if len(sheets) > 1
                else {None: load_dataframe(raw, up.name)}
            )
            for sheet, frame in frames.items():
                label = f"{up.name}#{sheet}" if sheet else up.name
                table = ctx.add_table(frame, label)
                profile_for(table.name)
            st.session_state.loaded_files.add(key)
            st.session_state.agent = None  # rebind tools to the new context
        except Exception as exc:
            st.error(f"Could not read {up.name}: {exc}")

    if ctx.tables:
        st.markdown("### Tables")
        names = list(ctx.tables)
        active = st.radio(
            "Active table",
            names,
            index=names.index(ctx.active) if ctx.active in names else 0,
            format_func=lambda n: f"{n} · {ctx.tables[n].shape_label}",
            label_visibility="collapsed",
        )
        if active != ctx.active:
            ctx.active = active

        col_a, col_b = st.columns(2)
        if col_a.button("Remove table", use_container_width=True):
            ctx.remove_table(active)
            st.rerun()
        if col_b.button("New chat", use_container_width=True):
            reset_conversation()
            st.rerun()

    st.markdown("---")
    st.markdown("### Agent")
    st.caption(f"{SETTINGS.provider} · {SETTINGS.model}")
    if not SETTINGS.api_key_present():
        st.warning(
            "No API key found. Set GROQ_API_KEY  in your "
            "environment or .env file, then reload. The dashboard works without one."
        )
    st.caption(
        f"Plan {'on' if SETTINGS.enable_planner else 'off'} · "
        f"Validate {'on' if SETTINGS.enable_validator else 'off'} · "
        f"Up to {SETTINGS.max_tool_steps} tool steps"
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
subtitle = (
    f"{ctx.get().source_file} · {ctx.get().shape_label}" if ctx.tables else "No data loaded"
)
st.markdown(
    f'<div class="masthead"><h1>AI Data Analyst</h1><span>{subtitle}</span></div>',
    unsafe_allow_html=True,
)

if not ctx.tables:
    st.markdown(
        '<div class="empty"><strong>Upload a dataset to begin</strong>'
        "Drop a CSV or Excel file in the sidebar. You will get a profile and a dashboard "
        "straight away, then you can ask questions about it in plain English.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

table = ctx.get()
profile = profile_for(table.name)

tab_dash, tab_chat, tab_profile, tab_sql = st.tabs(
    ["Dashboard", "Ask", "Profile", "SQL console"]
)


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
with tab_dash:
    kpis = build_kpis(table.df, profile)
    for row_start in range(0, len(kpis), 4):
        row = kpis[row_start : row_start + 4]
        for col, kpi in zip(st.columns(len(row)), row):
            col.markdown(
                f'<div class="kpi {kpi.tone}"><div class="label">{kpi.label}</div>'
                f'<div class="value">{kpi.value}</div>'
                f'<div class="detail">{kpi.detail}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")

    fig_key = f"{table.name}:{len(table.df)}:{len(table.df.columns)}"
    if st.session_state.get("figs_key") != fig_key:
        st.session_state.figs_key = fig_key
        st.session_state.figs = build_auto_charts(table.df, profile)
    figs = st.session_state.figs
    if not figs:
        st.info("No chartable structure found — the table may be entirely text or identifiers.")
    for i in range(0, len(figs), 2):
        pair = figs[i : i + 2]
        for col, fig in zip(st.columns(len(pair)), pair):
            col.plotly_chart(fig, use_container_width=True, key=f"auto_{i}_{id(fig)}")

    with st.expander(f"Preview rows of {table.name}"):
        st.dataframe(table.df.head(200), use_container_width=True, height=380)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def render_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "🤖"):
        st.markdown(msg["content"])
        for j, chart_id in enumerate(msg.get("charts", [])):
            fig = ctx.charts.get(chart_id)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"m{idx}_{j}_{chart_id}")
        steps = msg.get("tools") or []
        if steps:
            with st.expander(f"How this was worked out — {len(steps)} step(s)"):
                for step in steps:
                    mark = "·" if step["status"] == "ok" else "!"
                    st.markdown(f"`{mark}` **{step['tool']}** — {step['detail']}")


def run_turn(question: str) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    render_message(st.session_state.chat[-1], len(st.session_state.chat) - 1)
    ctx.start_turn()

    with st.chat_message("assistant", avatar="🧠"):
        status = st.status("Reading the question", expanded=True)
        answer = ""
        try:
            agent = get_agent()
            for node, update in stream_steps(agent, question, st.session_state.thread_id):
                status.update(label=NODE_LABELS.get(node, node))
                if node == "plan" and update.get("plan"):
                    status.write(f"**Plan**\n\n{update['plan']}")
                elif node == "tools":
                    for entry in ctx.tool_log[-3:]:
                        status.write(f"`{entry['tool']}` {entry['detail']}")
                elif node == "validate":
                    issues = update.get("issues") or []
                    status.write(
                        "Check passed." if update.get("verdict") == "pass"
                        else "Check flagged: " + "; ".join(issues)
                    )
                elif node == "explain":
                    msgs = update.get("messages") or []
                    if msgs:
                        answer = message_text(msgs[-1])
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            status.write(f"```\n{traceback.format_exc()[-1500:]}\n```")
            answer = (
                f"The analysis could not complete: `{exc}`. "
                "Check that your API key is set and the question refers to columns that exist."
            )

        answer = answer or "No answer was produced. Try rephrasing the question."
        st.markdown(answer)
        for j, chart_id in enumerate(ctx.new_charts):
            fig = ctx.charts.get(chart_id)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"live_{j}_{chart_id}")

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer,
            "charts": list(ctx.new_charts),
            "tools": list(ctx.tool_log),
        }
    )


with tab_chat:
    if not st.session_state.chat:
        st.caption("Ask anything about the data. The agent remembers the table and everything it has already found.")
        cols = st.columns(2)
        for i, q in enumerate(suggested_questions(profile, table.name)):
            if cols[i % 2].button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    for idx, msg in enumerate(st.session_state.chat):
        render_message(msg, idx)

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        run_turn(question)
        st.rerun()


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------
with tab_profile:
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Columns")
        rows = []
        for c in profile["columns"]:
            stats = c.get("stats", {})
            rows.append(
                {
                    "column": c["name"],
                    "type": c["semantic_type"],
                    "dtype": c["dtype"],
                    "missing": f"{c['missing_pct']:.1%}",
                    "distinct": f"{c['distinct']:,}",
                    "summary": (
                        f"{stats.get('min', '')} → {stats.get('max', '')}"
                        if "min" in stats
                        else (stats.get("mode") or "")
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=440, hide_index=True)

    with right:
        st.markdown("#### What to watch")
        if not profile["quality_issues"]:
            st.markdown(
                '<div class="flag"><span class="sev sev-low">clean</span>'
                '<span class="txt">No quality problems detected.</span></div>',
                unsafe_allow_html=True,
            )
        for issue in profile["quality_issues"]:
            st.markdown(
                f'<div class="flag"><span class="sev sev-{issue["severity"]}">{issue["severity"]}</span>'
                f'<span class="col">{issue["column"]}</span>'
                f'<span class="txt">{issue["issue"]}</span></div>',
                unsafe_allow_html=True,
            )

        if profile["correlations"]:
            st.markdown("#### Strongest relationships")
            corr_df = pd.DataFrame(profile["correlations"][:8])
            corr_df["r"] = corr_df["r"].round(3)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

    st.markdown("#### Column detail")
    pick = st.selectbox("Column", [c["name"] for c in profile["columns"]], label_visibility="collapsed")
    detail = next(c for c in profile["columns"] if c["name"] == pick)
    dcol, vcol = st.columns([1, 2])
    dcol.json(detail, expanded=False)
    try:
        from core.charts import build_chart

        kind = "histogram" if detail["semantic_type"] in {"numeric", "categorical-numeric"} else "bar"
        vcol.plotly_chart(
            build_chart(table.df, kind, x=pick, agg="count", top_n=20, orientation="h"),
            use_container_width=True,
        )
    except Exception:
        vcol.caption("No chart available for this column.")


# --------------------------------------------------------------------------
# SQL console
# --------------------------------------------------------------------------
with tab_sql:
    st.caption(
        "Read-only DuckDB over every loaded table. Available: "
        + ", ".join(f"`{n}`" for n in ctx.tables)
    )
    query = st.text_area(
        "Query",
        value=f"SELECT *\nFROM {table.name}\nLIMIT 50",
        height=150,
        label_visibility="collapsed",
    )
    if st.button("Run query", type="primary"):
        try:
            result = ctx.con.execute(query).fetchdf()
            st.success(f"{len(result):,} rows in {len(result.columns)} columns")
            st.dataframe(result, use_container_width=True, height=420)
            st.download_button(
                "Download results as CSV",
                result.to_csv(index=False).encode(),
                file_name="query_results.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Chat input — top level so it stays pinned and works from any tab
# --------------------------------------------------------------------------
asked = st.chat_input(f"Ask about {table.name}…")
if asked:
    st.session_state.pending_question = asked
    st.rerun()
