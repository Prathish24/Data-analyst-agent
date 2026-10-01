"""Check the non-LLM half of the app: loading, profiling, DuckDB, charts, sandbox.

    python make_sample_data.py && python selftest.py

No API key required. If this passes, any failure you see in the app is in the
agent layer (model, key, or network), not in the data engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from core.charts import build_chart, register_theme
from core.dashboard import build_auto_charts, build_kpis, suggested_questions
from core.dataset import AnalysisContext, load_dataframe
from core.profiling import profile_dataframe, profile_summary_text
from core.sandbox import run_user_code

CSV = Path("sample_sales.csv")


def main() -> int:
    if not CSV.exists():
        print("sample_sales.csv missing — run: python make_sample_data.py")
        return 1

    register_theme()
    ctx = AnalysisContext()
    df = load_dataframe(CSV.read_bytes(), CSV.name)
    table = ctx.add_table(df, CSV.name)
    print(f"loaded    {table.name}: {table.shape_label}")

    profile = profile_dataframe(df)
    table.profile = profile
    print(f"profiled  {len(profile['columns'])} columns, {len(profile['quality_issues'])} quality flags")
    print("          " + profile_summary_text(profile, table.name).splitlines()[0])

    out = ctx.con.execute(
        f"SELECT region, ROUND(SUM(revenue), 0) AS revenue FROM {table.name} "
        "GROUP BY region ORDER BY revenue DESC"
    ).fetchdf()
    assert not out.empty, "DuckDB returned nothing"
    print(f"duckdb    {len(out)} rows: {out.iloc[0, 0]} leads with {out.iloc[0, 1]:,.0f}")

    kpis = build_kpis(df, profile)
    print(f"kpis      {', '.join(k.label + '=' + k.value for k in kpis[:4])}")

    figs = build_auto_charts(df, profile)
    assert figs, "no dashboard charts were generated"
    print(f"charts    {len(figs)} auto charts: {', '.join(f.layout.title.text[:28] for f in figs[:3])}…")

    fig = build_chart(df, "bar", x="region", y="revenue", agg="sum", orientation="h")
    ctx.store_chart(fig, "Revenue by region")
    assert len(ctx.charts) == 1

    res = run_user_code(
        "result = df.groupby('region', observed=True)['revenue'].sum().round(0)",
        {"df": df},
    )
    assert res.ok, res.error
    print(f"sandbox   ok — {res.result_repr.splitlines()[1].strip()}")

    blocked = run_user_code("import os\nresult = 1", {"df": df})
    assert not blocked.ok, "sandbox failed to block an os import"
    print("sandbox   blocked an os import as expected")

    print("questions " + " | ".join(suggested_questions(profile, table.name))[:140])
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
