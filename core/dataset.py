"""Loading uploaded files and holding the live analysis context.

An AnalysisContext owns every table the user has uploaded, one DuckDB connection
with those tables registered, and the charts produced during a session. The agent
tools close over a single context instance, which is what lets the chat remember
the dataset across turns.
"""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

RESERVED = {"select", "from", "where", "table", "order", "group", "join", "case", "when"}


def slugify_table(name: str) -> str:
    """Turn a filename into a safe SQL identifier."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower() or "dataset"
    if slug[0].isdigit() or slug in RESERVED:
        slug = f"t_{slug}"
    return slug[:48]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names without destroying their meaning."""
    seen: dict[str, int] = {}
    out = []
    for col in df.columns:
        name = re.sub(r"\s+", "_", str(col).strip())
        name = re.sub(r"[^0-9a-zA-Z_]", "", name)
        name = re.sub(r"_+", "_", name).strip("_") or "col"
        if name[0].isdigit():
            name = f"c_{name}"
        key = name.lower()
        if key in seen:
            seen[key] += 1
            name = f"{name}_{seen[key]}"
        else:
            seen[key] = 0
        out.append(name)
    df = df.copy()
    df.columns = out
    return df


def coerce_types(df: pd.DataFrame, datetime_threshold: float = 0.8) -> pd.DataFrame:
    """Best-effort conversion of object columns to numeric / datetime / category."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue

        stripped = non_null.astype(str).str.replace(r"[,$%\s]", "", regex=True)
        numeric = pd.to_numeric(stripped, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            df[col] = pd.to_numeric(
                s.astype(str).str.replace(r"[,$%\s]", "", regex=True), errors="coerce"
            )
            continue

        looks_dateish = bool(
            re.search(r"date|time|day|month|year|_at$|_on$", str(col), re.I)
        ) or bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", str(non_null.iloc[0])))
        if looks_dateish:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            if parsed.notna().mean() >= datetime_threshold:
                df[col] = pd.to_datetime(s, errors="coerce", format="mixed")
                continue

        if non_null.nunique() <= max(20, int(len(non_null) * 0.05)):
            df[col] = s.astype("category")
    return df


def load_dataframe(file_bytes: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel / Parquet / JSON bytes into a tidy DataFrame."""
    lower = filename.lower()
    buf = io.BytesIO(file_bytes)

    if lower.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(buf, sheet_name=sheet or 0)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(buf)
    elif lower.endswith(".json"):
        df = pd.read_json(buf)
    elif lower.endswith((".tsv", ".tab")):
        df = pd.read_csv(buf, sep="\t")
    else:
        try:
            df = pd.read_csv(buf, sep=None, engine="python")
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf, encoding="latin-1", on_bad_lines="skip")

    if isinstance(df, dict):  # multi-sheet Excel fallback
        df = next(iter(df.values()))

    df = clean_columns(df)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return coerce_types(df)


def excel_sheet_names(file_bytes: bytes) -> list[str]:
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    except Exception:
        return []


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    source_file: str
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def shape_label(self) -> str:
        return f"{len(self.df):,} rows x {len(self.df.columns)} columns"


class AnalysisContext:
    """Session-scoped state shared by the UI and every agent tool."""

    def __init__(self) -> None:
        self.con = duckdb.connect(database=":memory:")
        self.tables: dict[str, Table] = {}
        self.active: str | None = None
        self.charts: dict[str, Any] = {}          # chart_id -> plotly Figure
        self.new_charts: list[str] = []           # ids produced in the current turn
        self.tool_log: list[dict[str, Any]] = []  # audit trail for the UI

    # ---- tables -------------------------------------------------------
    def add_table(self, df: pd.DataFrame, source_file: str) -> Table:
        base = slugify_table(source_file)
        name, i = base, 2
        while name in self.tables:
            name, i = f"{base}_{i}", i + 1

        table = Table(name=name, df=df, source_file=source_file)
        self.tables[name] = table
        self.con.register(name, df)
        self.active = self.active or name
        return table

    def remove_table(self, name: str) -> None:
        self.tables.pop(name, None)
        try:
            with self.lock:
                self.con.unregister(name)
        except Exception:
            pass
        if self.active == name:
            self.active = next(iter(self.tables), None)

    def get(self, name: str | None = None) -> Table:
        key = name or self.active
        if key is None:
            raise ValueError("No dataset has been uploaded yet.")
        if key not in self.tables:
            known = ", ".join(self.tables) or "none"
            raise ValueError(f"Unknown table {key!r}. Available tables: {known}.")
        return self.tables[key]

    def refresh_view(self, name: str) -> None:
        """Re-register a table after its DataFrame was replaced."""
        with self.lock:
            try:
                self.con.unregister(name)
            except Exception:
                pass
            self.con.register(name, self.tables[name].df)

    # ---- charts -------------------------------------------------------
    def store_chart(self, fig: Any, title: str) -> str:
        chart_id = f"chart_{uuid.uuid4().hex[:8]}"
        fig.update_layout(title=title)
        self.charts[chart_id] = fig
        self.new_charts.append(chart_id)
        return chart_id

    def start_turn(self) -> None:
        self.new_charts = []
        self.tool_log = []

    # ---- prompt context ----------------------------------------------
    def schema_digest(self, sample_rows: int = 3) -> str:
        """Compact schema text injected into the agent's system prompt."""
        if not self.tables:
            return "No datasets are loaded."
        blocks = []
        for t in self.tables.values():
            cols = [
                f"  - {c}: {t.df[c].dtype} "
                f"({t.df[c].nunique(dropna=True)} distinct, "
                f"{t.df[c].isna().mean():.0%} missing)"
                for c in t.df.columns
            ]
            head = t.df.head(sample_rows).to_csv(index=False).strip()
            marker = "  <- active" if t.name == self.active else ""
            blocks.append(
                f"TABLE {t.name}{marker} — from {t.source_file} — {t.shape_label}\n"
                + "\n".join(cols)
                + f"\nSample rows:\n{head}"
            )
        return "\n\n".join(blocks)