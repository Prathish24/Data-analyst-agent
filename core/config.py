"""Runtime settings and the LLM factory.

Everything configurable lives here so the rest of the code never reads os.environ
directly. Copy .env.example to .env and fill in a key before running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_groq import ChatGroq

try:  # optional, only needed for local .env files
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """All tunable knobs for the app."""

    provider: str = os.getenv("LLM_PROVIDER", "groq")  # anthropic | openai
    model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Agent behaviour
    max_tool_steps: int = _int("MAX_TOOL_STEPS", 12)
    max_repair_attempts: int = _int("MAX_REPAIR_ATTEMPTS", 2)
    enable_planner: bool = os.getenv("ENABLE_PLANNER", "1") == "1"
    enable_validator: bool = os.getenv("ENABLE_VALIDATOR", "1") == "1"

    # Data limits
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)
    sample_rows_for_llm: int = _int("SAMPLE_ROWS_FOR_LLM", 5)
    max_rows_returned: int = _int("MAX_ROWS_RETURNED", 200)
    python_timeout_s: int = _int("PYTHON_TIMEOUT_S", 30)

    # Dashboard
    max_auto_charts: int = _int("MAX_AUTO_CHARTS", 8)
    high_cardinality_cutoff: int = _int("HIGH_CARDINALITY_CUTOFF", 50)

    palette: list[str] = field(
        default_factory=lambda: [
            "#3FBFA8",
            "#E2A03F",
            "#6F9BD1",
            "#C77DBB",
            "#E06C6C",
            "#7FB069",
            "#B8A1E0",
            "#D68F5C",
        ]
    )

    def api_key_present(self) -> bool:
        key = {
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(self.provider, "GROQ_API_KEY")  # default to GROQ_API_KEY if provider is unknown
        return bool(os.getenv(key))


SETTINGS = Settings()


def build_llm(streaming: bool = False):
    """Return a LangChain chat model for the configured provider."""
    s = SETTINGS
    if s.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=s.model,
            temperature=s.temperature,
            max_tokens=4096,
            streaming=streaming,
            timeout=120,
            max_retries=2,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {s.provider!r}. Use 'anthropic' or 'openai'.")
