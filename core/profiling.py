"""Automatic dataset profiling.

profile_dataframe() returns a plain dict so it can be cached, serialised into the
agent prompt, and rendered in the UI without recomputation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SETTINGS


def _semantic_type(s: pd.Series, n_rows: int) -> str:
    """Label a column by how an analyst would actually use it."""
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    nunique = non_null.nunique()

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s) or (nunique == 2 and not pd.api.types.is_float_dtype(s)):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        if nunique == n_rows and pd.api.types.is_integer_dtype(s):
            return "identifier"
        if nunique <= 12 and pd.api.types.is_integer_dtype(s):
            return "categorical-numeric"
        return "numeric"
    if nunique == n_rows and n_rows > 20:
        return "identifier"
    avg_len = non_null.astype(str).str.len().mean()
    if nunique > SETTINGS.high_cardinality_cutoff and avg_len > 30:
        return "text"
    return "categorical"


def _outlier_stats(s: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(clean) < 8:
        return {"count": 0, "share": 0.0, "lower": None, "upper": None}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"count": 0, "share": 0.0, "lower": None, "upper": None}
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (clean < lo) | (clean > hi)
    return {
        "count": int(mask.sum()),
        "share": float(mask.mean()),
        "lower": float(lo),
        "upper": float(hi),
    }


def profile_column(s: pd.Series, n_rows: int) -> dict[str, Any]:
    kind = _semantic_type(s, n_rows)
    non_null = s.dropna()
    out: dict[str, Any] = {
        "name": s.name,
        "dtype": str(s.dtype),
        "semantic_type": kind,
        "missing": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean()),
        "distinct": int(non_null.nunique()),
        "distinct_pct": float(non_null.nunique() / n_rows) if n_rows else 0.0,
    }

    if kind in {"numeric", "categorical-numeric"} and not non_null.empty:
        desc = non_null.describe()
        out["stats"] = {
            "mean": float(desc.get("mean", np.nan)),
            "std": float(desc.get("std", np.nan)) if not pd.isna(desc.get("std", np.nan)) else 0.0,
            "min": float(desc.get("min", np.nan)),
            "p25": float(desc.get("25%", np.nan)),
            "median": float(desc.get("50%", np.nan)),
            "p75": float(desc.get("75%", np.nan)),
            "max": float(desc.get("max", np.nan)),
            "sum": float(non_null.sum()),
            "zeros": int((non_null == 0).sum()),
            "negatives": int((non_null < 0).sum()),
            "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
        }
        out["outliers"] = _outlier_stats(non_null)

    elif kind == "datetime" and not non_null.empty:
        span_days = (non_null.max() - non_null.min()).days
        out["stats"] = {
            "min": str(non_null.min()),
            "max": str(non_null.max()),
            "span_days": int(span_days),
            "suggested_freq": "D" if span_days <= 180 else ("W" if span_days <= 730 else "ME"),
        }

    else:
        counts = non_null.astype(str).value_counts().head(8)
        out["top_values"] = [
            {"value": str(v), "count": int(c), "share": float(c / n_rows)}
            for v, c in counts.items()
        ]
        if not non_null.empty:
            out["stats"] = {
                "mode": str(counts.index[0]) if len(counts) else None,
                "mode_share": float(counts.iloc[0] / n_rows) if len(counts) else 0.0,
            }
    return out


def _quality_issues(df: pd.DataFrame, cols: list[dict[str, Any]], dupes: int) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    n = len(df)

    if dupes:
        issues.append(
            {
                "severity": "high" if dupes / max(n, 1) > 0.02 else "medium",
                "column": "(table)",
                "issue": f"{dupes:,} fully duplicated rows ({dupes / max(n, 1):.1%} of the table)",
            }
        )

    for c in cols:
        if c["missing_pct"] > 0.4:
            issues.append(
                {
                    "severity": "high",
                    "column": c["name"],
                    "issue": f"{c['missing_pct']:.0%} of values are missing",
                }
            )
        elif c["missing_pct"] > 0.05:
            issues.append(
                {
                    "severity": "medium",
                    "column": c["name"],
                    "issue": f"{c['missing_pct']:.0%} of values are missing",
                }
            )

        if c["distinct"] == 1:
            issues.append(
                {"severity": "low", "column": c["name"], "issue": "Only one value — no signal"}
            )

        out = c.get("outliers", {})
        if out.get("share", 0) > 0.05:
            issues.append(
                {
                    "severity": "medium",
                    "column": c["name"],
                    "issue": f"{out['count']:,} values fall outside 1.5x IQR ({out['share']:.1%})",
                }
            )

        st = c.get("stats", {})
        if c["semantic_type"] == "numeric" and abs(st.get("skew", 0)) > 3:
            issues.append(
                {
                    "severity": "low",
                    "column": c["name"],
                    "issue": f"Heavily skewed (skew {st['skew']:.1f}) — medians beat means here",
                }
            )
        if c["semantic_type"] == "categorical" and st.get("mode_share", 0) > 0.95:
            issues.append(
                {
                    "severity": "low",
                    "column": c["name"],
                    "issue": f"{st['mode_share']:.0%} of rows share one value",
                }
            )

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: order[i["severity"]])[:25]


def _correlations(df: pd.DataFrame, numeric: list[str]) -> list[dict[str, Any]]:
    if len(numeric) < 2:
        return []
    corr = df[numeric].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append({"a": a, "b": b, "r": float(r)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pairs[:12]


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Full profile: shape, per-column detail, quality issues, correlations."""
    n_rows = len(df)
    cols = [profile_column(df[c], n_rows) for c in df.columns]
    dupes = int(df.duplicated().sum())

    by_type: dict[str, list[str]] = {}
    for c in cols:
        by_type.setdefault(c["semantic_type"], []).append(c["name"])

    numeric = by_type.get("numeric", []) + by_type.get("categorical-numeric", [])
    memory_mb = float(df.memory_usage(deep=True).sum()) / 1e6
    cells = n_rows * max(len(df.columns), 1)

    return {
        "n_rows": n_rows,
        "n_cols": len(df.columns),
        "memory_mb": memory_mb,
        "duplicate_rows": dupes,
        "duplicate_pct": float(dupes / n_rows) if n_rows else 0.0,
        "missing_cells_pct": float(df.isna().sum().sum() / cells) if cells else 0.0,
        "columns": cols,
        "by_type": by_type,
        "numeric_columns": numeric,
        "datetime_columns": by_type.get("datetime", []),
        "categorical_columns": by_type.get("categorical", []) + by_type.get("boolean", []),
        "correlations": _correlations(df, numeric),
        "quality_issues": _quality_issues(df, cols, dupes),
    }


def profile_summary_text(profile: dict[str, Any], table_name: str) -> str:
    """Condensed profile for the agent prompt — a few hundred tokens, not thousands."""
    lines = [
        f"Profile of `{table_name}`: {profile['n_rows']:,} rows x {profile['n_cols']} columns, "
        f"{profile['missing_cells_pct']:.1%} of cells missing, "
        f"{profile['duplicate_rows']:,} duplicate rows."
    ]
    for kind, names in profile["by_type"].items():
        lines.append(f"{kind}: {', '.join(names[:15])}")

    if profile["correlations"]:
        top = "; ".join(
            f"{c['a']}~{c['b']} r={c['r']:.2f}" for c in profile["correlations"][:5]
        )
        lines.append(f"Strongest correlations: {top}")

    if profile["quality_issues"]:
        issues = "; ".join(
            f"{i['column']}: {i['issue']}" for i in profile["quality_issues"][:6]
        )
        lines.append(f"Quality flags: {issues}")
    return "\n".join(lines)
