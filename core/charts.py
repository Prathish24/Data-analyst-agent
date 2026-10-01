"""Plotly theming and a spec-driven chart builder.

The agent never writes raw Plotly code. It calls create_chart with a small spec
(kind, x, y, aggregation), which keeps charts consistent and impossible to break.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from .config import SETTINGS

INK = "#0E1419"
PANEL = "#151D25"
LINE = "#26333F"
TEXT = "#E6EDF3"
MUTED = "#8CA0B3"

CHART_KINDS = [
    "bar", "line", "area", "scatter", "histogram",
    "box", "pie", "heatmap", "treemap",
]
AGGS = ["sum", "mean", "median", "count", "min", "max", "nunique"]


def register_theme() -> None:
    """Install a dark analyst theme as the Plotly default."""
    pio.templates["analyst"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, Inter, system-ui, sans-serif", size=13, color=TEXT),
            title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left", pad=dict(b=12)),
            colorway=SETTINGS.palette,
            margin=dict(l=8, r=8, t=48, b=8),
            xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED)),
            yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED)),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                        y=-0.18, x=0),
            hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(color=TEXT)),
            colorscale=dict(sequential=[[0, "#14313A"], [0.5, "#2C8E86"], [1, "#6FE3C6"]]),
        )
    )
    pio.templates.default = "analyst"


def _aggregate(
    df: pd.DataFrame, x: str, y: str | None, agg: str, color: str | None, top_n: int | None
) -> pd.DataFrame:
    if y is None or agg == "count":
        grouped = df.groupby([c for c in [x, color] if c], dropna=False).size().reset_index(name="count")
        value_col = "count"
    else:
        grouped = (
            df.groupby([c for c in [x, color] if c], dropna=False)[y]
            .agg(agg)
            .reset_index()
        )
        value_col = y

    if top_n and not pd.api.types.is_datetime64_any_dtype(grouped[x]):
        keep = grouped.groupby(x)[value_col].sum().nlargest(top_n).index
        grouped = grouped[grouped[x].isin(keep)]
        grouped = grouped.sort_values(value_col, ascending=False)
    elif pd.api.types.is_datetime64_any_dtype(grouped[x]):
        grouped = grouped.sort_values(x)

    return grouped


def build_chart(
    df: pd.DataFrame,
    kind: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    agg: str = "sum",
    top_n: int | None = 15,
    title: str | None = None,
    orientation: str = "v",
) -> go.Figure:
    """Build one Plotly figure from a validated spec."""
    kind = kind.lower().strip()
    if kind not in CHART_KINDS:
        raise ValueError(f"Unsupported chart kind {kind!r}. Choose from: {', '.join(CHART_KINDS)}.")

    for col in [c for c in (x, y, color) if c]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} is not in the table. Columns: {', '.join(df.columns)}")

    if kind == "histogram":
        fig = px.histogram(df, x=x, color=color, nbins=40, marginal="box")
    elif kind == "box":
        fig = px.box(df, x=x, y=y, color=color, points="outliers")
    elif kind == "scatter":
        try:  # trendline needs statsmodels; the plot is still useful without it
            fig = px.scatter(
                df, x=x, y=y, color=color, opacity=0.65,
                trendline="ols" if len(df) <= 20000 else None,
                trendline_color_override="#E2A03F",
            )
        except Exception:
            fig = px.scatter(df, x=x, y=y, color=color, opacity=0.65)
    elif kind == "heatmap":
        numeric = df.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("A heatmap needs at least two numeric columns.")
        corr = numeric.corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Teal",
                        zmin=-1, zmax=1)
    elif kind == "pie":
        data = _aggregate(df, x, y, agg, None, top_n or 10)
        value_col = "count" if (y is None or agg == "count") else y
        fig = px.pie(data, names=x, values=value_col, hole=0.55)
    elif kind == "treemap":
        data = _aggregate(df, x, y, agg, color, top_n or 20)
        value_col = "count" if (y is None or agg == "count") else y
        fig = px.treemap(data, path=[c for c in [color, x] if c], values=value_col)
    else:
        data = _aggregate(df, x, y, agg, color, top_n)
        value_col = "count" if (y is None or agg == "count") else y
        if kind == "bar":
            if orientation == "h":
                fig = px.bar(data, x=value_col, y=x, color=color, orientation="h")
                fig.update_yaxes(categoryorder="total ascending")
            else:
                fig = px.bar(data, x=x, y=value_col, color=color)
        elif kind == "line":
            fig = px.line(data, x=x, y=value_col, color=color, markers=len(data) <= 60)
        else:
            fig = px.area(data, x=x, y=value_col, color=color)

    label = title or _auto_title(kind, x, y, agg, color)
    fig.update_layout(title=label, height=420)
    return fig


def _auto_title(kind: str, x: str | None, y: str | None, agg: str, color: str | None) -> str:
    pretty = lambda c: str(c).replace("_", " ") if c else ""
    if kind == "heatmap":
        return "Correlation between numeric columns"
    if kind == "histogram":
        return f"Distribution of {pretty(x)}"
    if kind == "scatter":
        return f"{pretty(y)} against {pretty(x)}"
    metric = f"{agg} of {pretty(y)}" if y else "row count"
    base = f"{metric} by {pretty(x)}"
    return f"{base}, split by {pretty(color)}" if color else base


def spec_from_dict(df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
    return build_chart(
        df,
        kind=spec.get("kind", "bar"),
        x=spec.get("x"),
        y=spec.get("y"),
        color=spec.get("color"),
        agg=spec.get("agg", "sum"),
        top_n=spec.get("top_n", 15),
        title=spec.get("title"),
        orientation=spec.get("orientation", "v"),
    )
