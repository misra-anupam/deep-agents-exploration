"""Paths and environment-driven settings."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fixtures_dir() -> Path:
    return Path(os.environ.get("EQUITY_DESK_FIXTURES", ROOT / "fixtures"))


def cache_dir() -> Path:
    """Where downloaded filings are cached; routed to /filings/ in the agent."""
    path = Path(os.environ.get("EQUITY_DESK_CACHE", ROOT / "data" / "filings_cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def skills_dir() -> Path:
    return ROOT / "skills"


def data_mode() -> str:
    """'fixture' (default, offline) or 'edgar' (live SEC data)."""
    return os.environ.get("EQUITY_DESK_DATA", "fixture")


DEFAULT_MODEL = "openrouter:anthropic/claude-sonnet-5.5"

_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


def model_name() -> str:
    """Lead model, as 'provider:model' (OpenRouter ids look like 'openrouter:anthropic/claude-sonnet-5.5')."""
    return os.environ.get("EQUITY_DESK_MODEL", DEFAULT_MODEL)


def redteam_model() -> str | None:
    """Optional separate model for the red team; a different family reduces correlated blind spots."""
    return os.environ.get("EQUITY_DESK_REDTEAM_MODEL") or None


def subagent_model() -> str | None:
    """Optional model for the other subagents (e.g. a cheaper one); None = inherit the lead's."""
    return os.environ.get("EQUITY_DESK_SUBAGENT_MODEL") or None


def missing_api_keys(*models: str | None) -> list[str]:
    """Env vars that must be set for the given models (default: every configured model)."""
    models = models or (model_name(), redteam_model(), subagent_model())
    needed = {_KEY_ENV[m.split(":", 1)[0]] for m in models if m and ":" in m and m.split(":", 1)[0] in _KEY_ENV}
    return sorted(k for k in needed if not os.environ.get(k))


def edgar_user_agent() -> str:
    """SEC requires a descriptive User-Agent with contact info."""
    ua = os.environ.get("EDGAR_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "EDGAR_USER_AGENT is required for live SEC access, "
            "e.g. 'Your Name your@email.com'."
        )
    return ua


def broker_path() -> Path:
    return Path(os.environ.get("EQUITY_DESK_BROKER", ROOT / "data" / "broker.sqlite"))


def audit_path() -> Path:
    return Path(os.environ.get("EQUITY_DESK_AUDIT", ROOT / "data" / "audit.jsonl"))
