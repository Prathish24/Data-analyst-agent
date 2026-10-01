"""A restricted exec() environment for agent-written pandas code.

This is a guard rail, not a security boundary. It blocks the obvious foot-guns
(filesystem, network, subprocess, dunder escapes) and caps runtime. Run the app
on untrusted input only inside a container you are willing to lose.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import re
import signal
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BLOCKED_PATTERNS = [
    r"\bimport\s+(os|sys|subprocess|shutil|socket|requests|urllib|pathlib|ctypes|pickle)\b",
    r"\bfrom\s+(os|sys|subprocess|shutil|socket|requests|urllib|pathlib|ctypes|pickle)\b",
    r"\b__(import|builtins|globals|subclasses|class|bases|mro)__\b",
    r"\b(eval|exec|compile|open|input|breakpoint)\s*\(",
    r"\bgetattr\s*\(\s*[^,]+,\s*['\"]__",
    r"\.(to_csv|to_excel|to_parquet|to_pickle|to_json)\s*\(\s*['\"]",
]

SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in (
        "abs aint all any bool dict divmod enumerate filter float format frozenset "
        "hasattr hash int isinstance issubclass len list map max min next object "
        "pow print range repr reversed round set setattr slice sorted str sum tuple "
        "type zip True False None ValueError TypeError KeyError IndexError Exception"
    ).split()
    if hasattr(builtins, k)
}


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    result_repr: str
    error: str = ""
    figure: Any = None
    frame: pd.DataFrame | None = None

    def to_text(self, max_chars: int = 6000) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        parts = []
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.result_repr.strip():
            parts.append(f"result:\n{self.result_repr.strip()}")
        if self.figure is not None:
            parts.append("A Plotly figure was produced and shown to the user.")
        text = "\n\n".join(parts) or "Code ran without producing output. Assign to `result` or print()."
        return text[:max_chars] + ("\n…(truncated)" if len(text) > max_chars else "")


def _check_source(code: str) -> str | None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return (
                "Blocked: this code touches the filesystem, network, or interpreter internals. "
                "Use the provided DataFrames and return results in memory."
            )
    return None


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM timeout. No-op on platforms without it (Windows)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Code exceeded the {seconds}s time limit.")

    try:
        old = signal.signal(signal.SIGALRM, handler)
    except ValueError:
        # Streamlit runs scripts off the main thread, where signals are unavailable.
        yield
        return

    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_user_code(code: str, namespace: dict[str, Any], timeout_s: int = 30) -> ExecResult:
    """Execute code with pandas/numpy/plotly available and capture `result`."""
    blocked = _check_source(code)
    if blocked:
        return ExecResult(False, "", "", blocked)

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        figure_type: Any = go.Figure
    except ImportError:  # charts unavailable, analysis still works
        px = go = None
        figure_type = ()

    env: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        **namespace,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), _time_limit(timeout_s):
            exec(code, env)  # noqa: S102 - intentional, sandboxed above
    except Exception as exc:  # surface the error to the agent so it can self-correct
        return ExecResult(False, buf.getvalue(), "", f"{type(exc).__name__}: {exc}")

    value = env.get("result")
    figure = value if figure_type and isinstance(value, figure_type) else None
    frame = value if isinstance(value, (pd.DataFrame, pd.Series)) else None

    if figure is not None:
        rendered = ""
    elif isinstance(value, pd.DataFrame):
        rendered = value.head(50).to_markdown(index=False)
        if len(value) > 50:
            rendered += f"\n…{len(value) - 50:,} more rows"
    elif isinstance(value, pd.Series):
        rendered = value.head(50).to_string()
    elif value is None:
        rendered = ""
    else:
        rendered = repr(value)

    return ExecResult(
        True,
        buf.getvalue(),
        rendered,
        figure=figure,
        frame=frame.to_frame() if isinstance(frame, pd.Series) else frame,
    )
