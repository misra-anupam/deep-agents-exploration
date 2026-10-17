import pytest


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    """Every test: fixture data, isolated filing cache, no live keys needed."""
    monkeypatch.setenv("EQUITY_DESK_DATA", "fixture")
    monkeypatch.setenv("EQUITY_DESK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("EQUITY_DESK_BROKER", str(tmp_path / "broker.sqlite"))
    monkeypatch.setenv("EQUITY_DESK_AUDIT", str(tmp_path / "audit.jsonl"))
