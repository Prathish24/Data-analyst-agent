"""The agent's toolbelt.

build_tools(ctx) returns LangChain tools closed over one AnalysisContext, so every
tool call in a conversation sees the same tables, the same DuckDB connection, and
writes charts into the same store the UI reads from.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .charts import AGGS, CHART_KINDS, build_chart
from .config import SETTINGS
from .dataset import AnalysisContext
from .profiling import profile_dataframe, profile_summary_text
from .sandbox import run_user_code

WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|export|install|load|pragma)\b", re.I
)


def _frame_to_text(df: pd.DataFrame, limit: int | None = None) -> str:
    limit = limit or SETTINGS.max_rows_returned
    shown = df.head(limit)
    try:
        body = shown.to_markdown(index=False)
    except Exception:
        body = shown.to_string(index=False)
    if len(df) > limit:
        body += f"\n…{len(df) - limit:,} more rows not shown."
    return body


# --------------------------------------------------------------------------
# Tool argument schemas
# --------------------------------------------------------------------------
class SqlArgs(BaseModel):
    query: str = Field(description="A single read-only DuckDB SELECT statement.")
    purpose: str = Field(default="", description="One line on what this query is meant to establish.")


class PythonArgs(BaseModel):
    code: str = Field(
        description=(
            "Python using pandas as pd and numpy as np. Tables are available as DataFrames "
            "by their table name, and the active table is also bound to `df`. "
            "Assign your answer to a variable named `result`."
        )
    )
    purpose: str = Field(default="", description="One line on what this code is meant to establish.")


class DescribeArgs(BaseModel):
    table: Optional[str] = Field(default=None, description="Table name. Defaults to the active table.")
    columns: Optional[list[str]] = Field(default=None, description="Specific columns, or omit for all.")


class ChartArgs(BaseModel):
    kind: str = Field(description=f"One of: {', '.join(CHART_KINDS)}")
    x: Optional[str] = Field(default=None, description="Column for the x axis / grouping / labels.")
    y: Optional[str] = Field(default=None, description="Numeric column to aggregate. Omit to count rows.")
    color: Optional[str] = Field(default=None, description="Optional column to split series by.")
    agg: str = Field(default="sum", description=f"One of: {', '.join(AGGS)}")
    top_n: int = Field(default=15, description="Keep only the top N categories.")
    title: Optional[str] = Field(default=None, description="Chart title written for a business reader.")
    orientation: str = Field(default="v", description="'v' or 'h' — use 'h' for long category labels.")
    table: Optional[str] = Field(default=None, description="Table name. Defaults to the active table.")


class AnomalyArgs(BaseModel):
    column: str = Field(description="Numeric column to scan.")
    method: str = Field(default="iqr", description="'iqr', 'zscore', or 'quantile'.")
    group_by: Optional[str] = Field(default=None, description="Detect anomalies within each group.")
    threshold: float = Field(default=1.5, description="IQR multiplier, or z-score cutoff.")
    table: Optional[str] = Field(default=None)


class TrendArgs(BaseModel):
    date_column: str = Field(description="Datetime column to resample on.")
    value_column: Optional[str] = Field(default=None, description="Numeric column. Omit to count rows.")
    freq: str = Field(default="M", description="Pandas offset alias: D, W, M, Q, or Y.")
    agg: str = Field(default="sum", description=f"One of: {', '.join(AGGS)}")
    segment_by: Optional[str] = Field(default=None, description="Break the trend down by this column.")
    table: Optional[str] = Field(default=None)


class CompareArgs(BaseModel):
    dimension: str = Field(description="Categorical column defining the groups.")
    measure: str = Field(description="Numeric column to compare across groups.")
    table: Optional[str] = Field(default=None)


# --------------------------------------------------------------------------
# Tool factory
# --------------------------------------------------------------------------
def build_tools(ctx: AnalysisContext) -> list[StructuredTool]:
    """Create the tool list bound to this session's data."""

    def log(name: str, detail: str, status: str = "ok") -> None:
        ctx.tool_log.append({"tool": name, "detail": detail, "status": status})

    # ---- schema ----------------------------------------------------------
    def list_tables() -> str:
        """List every loaded table with its columns, dtypes and sample rows. Call this first if unsure what data exists."""
        log("list_tables", f"{len(ctx.tables)} table(s)")
        return ctx.schema_digest(sample_rows=SETTINGS.sample_rows_for_llm)

    # ---- profiling -------------------------------------------------------
    def describe_columns(table: str | None = None, columns: list[str] | None = None) -> str:
        """Get detailed statistics for columns: distribution, missing values, outliers, top values and data quality flags."""
        t = ctx.get(table)
        prof = t.profile or profile_dataframe(t.df)
        t.profile = prof
        cols = prof["columns"]
        if columns:
            wanted = {c.lower() for c in columns}
            cols = [c for c in cols if c["name"].lower() in wanted]
            if not cols:
                return f"None of {columns} exist in {t.name}. Columns: {', '.join(t.df.columns)}"
        payload = {
            "table": t.name,
            "rows": prof["n_rows"],
            "columns": cols,
            "quality_issues": prof["quality_issues"],
            "correlations": prof["correlations"][:8],
        }
        log("describe_columns", f"{t.name}: {len(cols)} column(s)")
        return json.dumps(payload, indent=1, default=str)[:12000]

    # ---- SQL -------------------------------------------------------------
    def run_sql(query: str, purpose: str = "") -> str:
        """Run a single read-only DuckDB SELECT query against the loaded tables. Best tool for aggregation, filtering, joins and window functions. Submit exactly one statement — no semicolon-separated multi-statement queries."""
        if WRITE_SQL.search(query):
            log("run_sql", "rejected write statement", "error")
            return "Rejected: only read-only SELECT queries are allowed."
        statements = [s for s in query.split(";") if s.strip()]
        if len(statements) > 1:
            log("run_sql", "rejected multi-statement query", "error")
            return (
                "Rejected: submit exactly one SELECT statement per call. "
                "Use a single query with a CTE (WITH ...) instead of multiple semicolon-separated statements."
            )
        try:
            with ctx.lock:
                out = ctx.con.execute(query).fetchdf()
        except Exception as exc:
            log("run_sql", f"{purpose or query[:60]} — failed", "error")
            return (
                f"SQL error: {exc}\n\nTables available: {', '.join(ctx.tables)}.\n"
                "Check column names with list_tables and retry."
            )
        if out is None:
            log("run_sql", "query produced no result set", "error")
            return "Query ran successfully but produced no result set. Make sure it's a single SELECT statement."
        log("run_sql", purpose or query.strip().splitlines()[0][:80])
        if out.empty:
            return "Query ran successfully but returned 0 rows. The filter may be too narrow."
        return f"{len(out):,} row(s) returned.\n\n{_frame_to_text(out)}"

    # ---- Python ----------------------------------------------------------
    def run_python(code: str, purpose: str = "") -> str:
        """Run pandas/numpy code for analysis SQL cannot express: statistics, correlations, reshaping, regressions, custom logic. Assign the answer to `result`."""
        namespace: dict[str, Any] = {name: t.df for name, t in ctx.tables.items()}
        if ctx.active:
            namespace["df"] = ctx.tables[ctx.active].df
        namespace["tables"] = {n: t.df for n, t in ctx.tables.items()}

        res = run_user_code(code, namespace, SETTINGS.python_timeout_s)
        if res.figure is not None:
            ctx.store_chart(res.figure, "Analysis chart")
        log("run_python", purpose or code.strip().splitlines()[0][:80], "ok" if res.ok else "error")
        return res.to_text()

    # ---- charting --------------------------------------------------------
    def create_chart(
        kind: str,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        agg: str = "sum",
        top_n: int = 15,
        title: str | None = None,
        orientation: str = "v",
        table: str | None = None,
    ) -> str:
        """Draw a Plotly chart for the user. Use it whenever a comparison, trend or distribution is easier to see than to read."""
        t = ctx.get(table)
        try:
            fig = build_chart(t.df, kind, x, y, color, agg, top_n, title, orientation)
        except Exception as exc:
            log("create_chart", f"{kind} failed", "error")
            return f"Chart error: {exc}"
        chart_id = ctx.store_chart(fig, title or f"{kind} chart")
        log("create_chart", title or f"{kind}: {x or ''} {y or ''}".strip())
        return (
            f"Chart {chart_id} rendered and shown to the user "
            f"({kind}, x={x}, y={y}, agg={agg}). Refer to it in your answer; do not describe its pixels."
        )

    # ---- anomalies -------------------------------------------------------
    def detect_anomalies(
        column: str,
        method: str = "iqr",
        group_by: str | None = None,
        threshold: float = 1.5,
        table: str | None = None,
    ) -> str:
        """Find statistical outliers in a numeric column, optionally within each group, and show what the outlier rows look like."""
        t = ctx.get(table)
        df = t.df
        if column not in df.columns:
            return f"Column {column!r} not found. Columns: {', '.join(df.columns)}"

        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() < 8:
            return f"{column!r} has too few numeric values to test for anomalies."

        def flags(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            if method == "zscore":
                sd = s.std(ddof=0)
                return (s - s.mean()).abs() > (threshold if threshold > 1.5 else 3.0) * sd if sd else s * False
            if method == "quantile":
                lo, hi = s.quantile(0.01), s.quantile(0.99)
                return (s < lo) | (s > hi)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return ((s < q1 - threshold * iqr) | (s > q3 + threshold * iqr)) if iqr else s * False

        if group_by and group_by in df.columns:
            mask = df.groupby(group_by, dropna=False)[column].transform(lambda s: flags(s)).fillna(False).astype(bool)
        else:
            mask = flags(series).fillna(False)

        n = int(mask.sum())
        if n == 0:
            log("detect_anomalies", f"{column}: none found")
            return f"No anomalies in {column!r} using {method} at threshold {threshold}."

        outliers = df[mask]
        lines = [
            f"{n:,} anomalous rows in {column!r} ({n / len(df):.2%} of the table), method={method}.",
            f"Normal range: {series[~mask].min():,.4g} to {series[~mask].max():,.4g}; "
            f"anomalies span {outliers[column].min():,.4g} to {outliers[column].max():,.4g}.",
        ]

        # What do the anomalies have in common?
        for cat in df.columns:
            if cat == column or df[cat].nunique(dropna=True) > 30 or df[cat].nunique(dropna=True) < 2:
                continue
            if not (pd.api.types.is_object_dtype(df[cat]) or str(df[cat].dtype) == "category"
                    or pd.api.types.is_bool_dtype(df[cat])):
                continue
            base = df[cat].value_counts(normalize=True)
            sub = outliers[cat].value_counts(normalize=True)
            lift = (sub / base).dropna()
            over = lift[(lift > 1.8) & (sub > 0.1)]
            if not over.empty:
                top = over.sort_values(ascending=False).head(3)
                detail = "; ".join(f"{k} ({v:.1f}x over-represented)" for k, v in top.items())
                lines.append(f"Concentrated in {cat}: {detail}")

        lines.append("\nSample of anomalous rows:\n" + _frame_to_text(outliers.head(15), 15))
        log("detect_anomalies", f"{column}: {n:,} found")
        return "\n".join(lines)

    # ---- trends ----------------------------------------------------------
    def analyze_trend(
        date_column: str,
        value_column: str | None = None,
        freq: str = "M",
        agg: str = "sum",
        segment_by: str | None = None,
        table: str | None = None,
    ) -> str:
        """Resample a metric over time and quantify direction, growth rate, volatility, peaks and the biggest period-over-period movers."""
        t = ctx.get(table)
        df = t.df.copy()
        if date_column not in df.columns:
            return f"Column {date_column!r} not found. Columns: {', '.join(df.columns)}"

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column])
        if df.empty:
            return f"{date_column!r} could not be parsed as dates."

        freq = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}.get(freq.upper(), freq)
        grouper = pd.Grouper(key=date_column, freq=freq)

        if value_column and value_column in df.columns:
            series = df.groupby(grouper)[value_column].agg(agg)
            label = f"{agg} of {value_column}"
        else:
            series = df.groupby(grouper).size()
            label = "record count"

        series = series.dropna()
        if len(series) < 3:
            return f"Only {len(series)} periods at freq={freq} — too few to call a trend. Try a finer frequency."

        first, last = float(series.iloc[0]), float(series.iloc[-1])
        pct = (last - first) / abs(first) if first else np.nan
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values.astype(float), 1)
        fitted = slope * x + intercept
        ss_res = float(((series.values - fitted) ** 2).sum())
        ss_tot = float(((series.values - series.values.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        cv = float(series.std() / series.mean()) if series.mean() else 0.0
        direction = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")

        changes = series.pct_change().dropna()
        lines = [
            f"{label} by {freq}, {len(series)} periods from {series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d}.",
            f"Trend: {direction}, {slope:,.4g} per period, fit R^2={r2:.2f}.",
            f"First period {first:,.4g} -> last period {last:,.4g} ({pct:+.1%} overall).",
            f"Peak {series.max():,.4g} at {series.idxmax():%Y-%m-%d}; trough {series.min():,.4g} at {series.idxmin():%Y-%m-%d}.",
            f"Volatility: coefficient of variation {cv:.2f}; mean period-over-period change {changes.mean():+.1%}.",
        ]
        if not changes.empty:
            big = changes.reindex(changes.abs().sort_values(ascending=False).index).head(3)
            lines.append(
                "Largest swings: "
                + "; ".join(f"{i:%Y-%m-%d} {v:+.1%}" for i, v in big.items())
            )
        lines.append("\nSeries:\n" + series.round(4).to_string())

        if segment_by and segment_by in df.columns and df[segment_by].nunique() <= 20:
            if value_column and value_column in df.columns:
                seg = df.groupby([grouper, segment_by])[value_column].agg(agg).unstack(fill_value=0)
            else:
                seg = df.groupby([grouper, segment_by]).size().unstack(fill_value=0)
            if len(seg) >= 2:
                delta = (seg.iloc[-1] - seg.iloc[0]).sort_values()
                lines.append(
                    f"\nBy {segment_by}, first vs last period change — "
                    f"grew most: {delta.index[-1]} ({delta.iloc[-1]:+,.4g}); "
                    f"fell most: {delta.index[0]} ({delta.iloc[0]:+,.4g})."
                )

        log("analyze_trend", f"{label} by {freq}: {direction}")
        return "\n".join(lines)

    # ---- group comparison ------------------------------------------------
    def compare_groups(dimension: str, measure: str, table: str | None = None) -> str:
        """Compare a numeric measure across the values of a categorical column, with per-group stats and a significance test."""
        t = ctx.get(table)
        df = t.df
        for col in (dimension, measure):
            if col not in df.columns:
                return f"Column {col!r} not found. Columns: {', '.join(df.columns)}"

        work = df[[dimension, measure]].dropna()
        work[measure] = pd.to_numeric(work[measure], errors="coerce")
        work = work.dropna()
        if work.empty:
            return f"No overlapping non-null values for {dimension} and {measure}."

        stats = (
            work.groupby(dimension)[measure]
            .agg(["count", "mean", "median", "std", "sum"])
            .sort_values("sum", ascending=False)
            .head(25)
        )
        total = work[measure].sum()
        stats["share_of_total"] = (stats["sum"] / total) if total else np.nan

        lines = [
            f"{measure} across {work[dimension].nunique()} values of {dimension} "
            f"(overall mean {work[measure].mean():,.4g}):",
            _frame_to_text(stats.round(4).reset_index(), 25),
        ]

        top = stats.head(3)["sum"].sum()
        if total:
            lines.append(f"Top 3 groups hold {top / total:.1%} of the total {measure}.")

        groups = [g[measure].values for _, g in work.groupby(dimension) if len(g) >= 5]
        if 2 <= len(groups) <= 30:
            try:
                from scipy import stats as sps

                f_stat, p = sps.f_oneway(*groups)
                verdict = "differences are unlikely to be noise" if p < 0.05 else "differences are within noise"
                lines.append(f"One-way ANOVA: F={f_stat:.2f}, p={p:.4g} — {verdict}.")
            except Exception:
                spread = stats["mean"].max() - stats["mean"].min()
                lines.append(f"Range between group means: {spread:,.4g} (install scipy for a significance test).")

        log("compare_groups", f"{measure} by {dimension}")
        return "\n".join(lines)

    # ---- assemble --------------------------------------------------------
    return [
        StructuredTool.from_function(func=list_tables, name="list_tables", description=list_tables.__doc__),
        StructuredTool.from_function(func=describe_columns, name="describe_columns",
                                     description=describe_columns.__doc__, args_schema=DescribeArgs),
        StructuredTool.from_function(func=run_sql, name="run_sql", description=run_sql.__doc__,
                                     args_schema=SqlArgs),
        StructuredTool.from_function(func=run_python, name="run_python", description=run_python.__doc__,
                                     args_schema=PythonArgs),
        StructuredTool.from_function(func=create_chart, name="create_chart", description=create_chart.__doc__,
                                     args_schema=ChartArgs),
        StructuredTool.from_function(func=detect_anomalies, name="detect_anomalies",
                                     description=detect_anomalies.__doc__, args_schema=AnomalyArgs),
        StructuredTool.from_function(func=analyze_trend, name="analyze_trend",
                                     description=analyze_trend.__doc__, args_schema=TrendArgs),
        StructuredTool.from_function(func=compare_groups, name="compare_groups",
                                     description=compare_groups.__doc__, args_schema=CompareArgs),
    ]


def summarize_table_for_prompt(ctx: AnalysisContext) -> str:
    """Schema plus profile highlights for the active table."""
    if not ctx.tables:
        return "No datasets are loaded yet."
    parts = [ctx.schema_digest(sample_rows=SETTINGS.sample_rows_for_llm)]
    if ctx.active:
        t = ctx.get()
        if not t.profile:
            t.profile = profile_dataframe(t.df)
        parts.append(profile_summary_text(t.profile, t.name))
    return "\n\n".join(parts)