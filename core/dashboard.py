"""Automatic dashboard: which KPIs matter and which charts are worth drawing.

The rules here are deliberately deterministic. The dashboard appears the moment a
file is uploaded, before any LLM call, so the user sees their data immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .charts import build_chart
from .config import SETTINGS


@dataclass
class Kpi:
    label: str
    value: str
    detail: str = ""
    tone: str = "neutral"  # neutral | good | warn | bad


def _fmt(n: float) -> str:
    """Human-readable number formatting for KPI tiles."""
    if n is None or pd.isna(n):
        return "—"
    a = abs(n)
    if a >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if a >= 10_000:
        return f"{n / 1_000:.1f}K"
    if a >= 100:
        return f"{n:,.0f}"
    if a >= 1:
        return f"{n:,.2f}"
    return f"{n:.3g}"


def _measure_score(name: str, col: dict[str, Any]) -> float:
    """Rank numeric columns by how likely they are to be a business measure."""
    lowered = name.lower()
    score = 0.0
    for kw, w in {
        "revenue": 5, "sales": 5, "amount": 4, "total": 4, "price": 3.5, "cost": 3.5,
        "profit": 4.5, "margin": 3.5, "spend": 3.5, "value": 3, "qty": 3,
        "quantity": 3, "count": 2, "score": 2, "rating": 2, "duration": 1.5,
    }.items():
        if kw in lowered:
            score += w
    if any(k in lowered for k in ("id", "index", "zip", "code", "year")):
        score -= 4
    stats = col.get("stats", {})
    if stats.get("negatives", 0) == 0 and stats.get("mean", 0) > 0:
        score += 0.5
    if col["distinct"] > 20:
        score += 0.5
    return score


def pick_measures(profile: dict[str, Any], limit: int = 3) -> list[str]:
    by_name = {c["name"]: c for c in profile["columns"]}
    candidates = [
        (n, _measure_score(n, by_name[n]))
        for n in profile["numeric_columns"]
        if by_name[n]["semantic_type"] != "identifier"
    ]
    candidates.sort(key=lambda p: p[1], reverse=True)
    return [n for n, _ in candidates[:limit]]


def build_kpis(df: pd.DataFrame, profile: dict[str, Any]) -> list[Kpi]:
    kpis = [Kpi("Rows", f"{profile['n_rows']:,}", f"{profile['n_cols']} columns")]

    miss = profile["missing_cells_pct"]
    kpis.append(
        Kpi(
            "Complete values",
            f"{1 - miss:.1%}",
            "No gaps" if miss == 0 else f"{miss:.1%} of cells empty",
            tone="good" if miss < 0.02 else ("warn" if miss < 0.15 else "bad"),
        )
    )

    dup = profile["duplicate_pct"]
    kpis.append(
        Kpi(
            "Unique rows",
            f"{1 - dup:.1%}",
            "No duplicates" if dup == 0 else f"{profile['duplicate_rows']:,} repeated",
            tone="good" if dup == 0 else ("warn" if dup < 0.02 else "bad"),
        )
    )

    for col in pick_measures(profile, 3):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        label = col.replace("_", " ")
        kpis.append(
            Kpi(
                f"Total {label}",
                _fmt(float(s.sum())),
                f"avg {_fmt(float(s.mean()))} · median {_fmt(float(s.median()))}",
            )
        )

    if profile["datetime_columns"]:
        dcol = profile["datetime_columns"][0]
        d = pd.to_datetime(df[dcol], errors="coerce").dropna()
        if not d.empty:
            kpis.append(
                Kpi(
                    "Time covered",
                    f"{(d.max() - d.min()).days:,} days",
                    f"{d.min():%d %b %Y} to {d.max():%d %b %Y}",
                )
            )

    highs = sum(1 for i in profile["quality_issues"] if i["severity"] == "high")
    if highs:
        kpis.append(Kpi("Quality flags", str(highs), "Needs review before modelling", tone="bad"))

    return kpis[:7]


def build_auto_charts(df: pd.DataFrame, profile: dict[str, Any]) -> list[go.Figure]:
    """Pick charts that answer the questions a first look usually raises."""
    figs: list[go.Figure] = []
    measures = pick_measures(profile, 2)
    primary = measures[0] if measures else None
    dates = profile["datetime_columns"]
    by_name = {c["name"]: c for c in profile["columns"]}

    cats = [
        c for c in profile["categorical_columns"]
        if 1 < by_name[c]["distinct"] <= SETTINGS.high_cardinality_cutoff
    ]
    cats.sort(key=lambda c: by_name[c]["distinct"])

    def safe(fn):
        try:
            figs.append(fn())
        except Exception:
            pass

    # 1. Trend over time
    if dates and primary:
        d = dates[0]
        raw_freq = by_name[d].get("stats", {}).get("suggested_freq", "M")
        freq = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}.get(raw_freq, "ME")
        tmp = df[[d, primary]].dropna().copy()
        tmp[d] = pd.to_datetime(tmp[d], errors="coerce")
        tmp = tmp.dropna().set_index(d).resample(freq)[primary].sum().reset_index()
        if len(tmp) > 1:
            safe(lambda: build_chart(
                tmp, "line", x=d, y=primary, agg="sum", top_n=None,
                title=f"{primary.replace('_', ' ')} over time",
            ))
    elif dates:
        d = dates[0]
        tmp = df[[d]].dropna().copy()
        tmp[d] = pd.to_datetime(tmp[d], errors="coerce")
        tmp = tmp.dropna().set_index(d).resample("ME").size().reset_index(name="records")
        if len(tmp) > 1:
            safe(lambda: build_chart(tmp, "line", x=d, y="records", agg="sum", top_n=None,
                                     title="Records over time"))

    # 2-3. Measure by the most useful categories
    for cat in cats[:2]:
        if primary:
            safe(lambda c=cat: build_chart(df, "bar", x=c, y=primary, agg="sum", top_n=12,
                                           orientation="h"))
        else:
            safe(lambda c=cat: build_chart(df, "bar", x=c, agg="count", top_n=12, orientation="h"))

    # 4. Distribution of the primary measure
    if primary:
        safe(lambda: build_chart(df, "histogram", x=primary))

    # 5. Strongest relationship
    corrs = [c for c in profile["correlations"] if abs(c["r"]) >= 0.35]
    if corrs:
        top = corrs[0]
        safe(lambda: build_chart(
            df.sample(min(len(df), 5000), random_state=0), "scatter", x=top["a"], y=top["b"],
            title=f"{top['b'].replace('_', ' ')} vs {top['a'].replace('_', ' ')} (r = {top['r']:.2f})",
        ))

    # 6. Spread of the second measure across a category
    if len(measures) > 1 and cats:
        safe(lambda: build_chart(df, "box", x=cats[0], y=measures[1],
                                 title=f"Spread of {measures[1].replace('_', ' ')} by {cats[0].replace('_', ' ')}"))

    # 7. Correlation matrix
    if len(profile["numeric_columns"]) >= 3:
        safe(lambda: build_chart(df[profile["numeric_columns"]], "heatmap"))

    return figs[: SETTINGS.max_auto_charts]


def suggested_questions(profile: dict[str, Any], table: str) -> list[str]:
    """Starter prompts grounded in the actual columns present."""
    qs: list[str] = []
    measures = pick_measures(profile, 2)
    dates = profile["datetime_columns"]
    cats = profile["categorical_columns"]

    if measures and dates:
        qs.append(f"How has {measures[0].replace('_', ' ')} trended over {dates[0].replace('_', ' ')}, and what drove the change?")
    if measures and cats:
        qs.append(f"Which {cats[0].replace('_', ' ')} values contribute most to {measures[0].replace('_', ' ')}?")
    if measures:
        qs.append(f"Find anomalies in {measures[0].replace('_', ' ')} and explain what they have in common.")
    if profile["quality_issues"]:
        qs.append("What data quality problems would break an analysis of this table?")
    qs.append(f"Give me the three most surprising findings in {table}.")
    return qs[:4]
