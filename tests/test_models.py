import pytest

from equity_desk import config
from equity_desk.agent import build_agent, default_subagent_models
from equity_desk.subagents import build_subagents


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("EQUITY_DESK_MODEL", "EQUITY_DESK_REDTEAM_MODEL", "EQUITY_DESK_SUBAGENT_MODEL",
              "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_an_openrouter_model_needing_only_the_openrouter_key():
    assert config.model_name().startswith("openrouter:")
    assert config.missing_api_keys() == ["OPENROUTER_API_KEY"]


def test_key_requirements_follow_the_providers_in_use(monkeypatch):
    monkeypatch.setenv("EQUITY_DESK_REDTEAM_MODEL", "openai:gpt-x")
    monkeypatch.setenv("EQUITY_DESK_SUBAGENT_MODEL", "anthropic:claude-x")
    assert config.missing_api_keys() == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert config.missing_api_keys() == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    assert config.missing_api_keys("openrouter:foo/bar") == []  # explicit models only


def test_role_models_default_to_inheriting_the_lead():
    assert default_subagent_models() == {}
    assert all("model" not in s for s in build_subagents(default_subagent_models()))


def test_redteam_and_subagent_overrides_apply_per_role(monkeypatch):
    monkeypatch.setenv("EQUITY_DESK_SUBAGENT_MODEL", "openrouter:anthropic/claude-haiku-5.5")
    monkeypatch.setenv("EQUITY_DESK_REDTEAM_MODEL", "openrouter:openai/gpt-5.6-terra")
    by_name = {s["name"]: s.get("model") for s in build_subagents(default_subagent_models())}
    assert by_name == {
        "filings-reader": "openrouter:anthropic/claude-haiku-5.5",
        "quant-analyst": "openrouter:anthropic/claude-haiku-5.5",
        "news-sentiment": "openrouter:anthropic/claude-haiku-5.5",
        "red-team": "openrouter:openai/gpt-5.6-terra",
    }


def test_openrouter_model_string_builds_a_real_graph(monkeypatch):
    """No network: constructs ChatOpenRouter objects and compiles the full agent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EQUITY_DESK_REDTEAM_MODEL", "openrouter:openai/gpt-5.6-terra")
    agent = build_agent()
    assert agent is not None
