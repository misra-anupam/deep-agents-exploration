"""Phase 6: the eval harness itself must be trustworthy before we trust its scores."""

import pytest
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from equity_desk.evals.run import EVALS_DIR, main, run_event
from equity_desk.evals.scoring import load_golden, score_note
from equity_desk.testing import ScriptedChatModel, call, turn
from equity_desk.tracing import UsageTracker, run_config, tracing_enabled, tracing_status

GOLDEN = load_golden(EVALS_DIR / "golden" / "acme_fy2024.json")
GOOD = (EVALS_DIR / "samples" / "good_note.md").read_text()
BAD = (EVALS_DIR / "samples" / "bad_note.md").read_text()
ACCN = "0009999999-25-000001"


def test_scorer_separates_good_from_bad_samples():
    assert main(["samples"]) == 0
    assert score_note(GOOD, GOLDEN).passed and not score_note(BAD, GOLDEN).passed


def failed_metrics(card):
    return {f.split()[0] for f in card.failures}


@pytest.mark.parametrize(
    "mutate,expected",
    [
        # Each mutation breaks exactly one property; only the matching metric may fire.
        (lambda t: t.replace("$1,008.0 million", "$1,100.0 million"), {"numeric_grounding", "required_figures"}),
        (lambda t: t.replace("12.0% to $1,008.0 million", "37% to $1,008.0 million"), {"numeric_grounding"}),
        (lambda t: t.replace(" [XBRL Revenues FY2024, accn 0009999999-25-000001]", ""), {"citation_coverage", "violations"}),
        (lambda t: t.replace("Research aid only. Not investment advice.", ""), {"violations"}),
        (lambda t: t.replace("Stance: Neutral.", "Returns are guaranteed."), {"violations"}),
        (lambda t: t.replace("- /filings/ACME/0009999999-25-000001.txt", "- 0000000000-00-000000"), {"violations"}),
        (lambda t: t.replace("Receivables", "Debtors").replace("receivables", "debtors"), {"flag_recall"}),
    ],
)
def test_single_fault_mutations_are_caught_by_the_right_metric(mutate, expected):
    mutated = mutate(GOOD)
    assert mutated != GOOD, "mutation did not change the note"
    assert failed_metrics(score_note(mutated, GOLDEN)) == expected


def test_no_note_is_a_failure_not_a_crash():
    card = score_note(None, GOLDEN)
    assert not card.passed and card.failures == ["no note produced"]


# ---------- end to end ----------
def scripted_event(note, finish_todos=True):
    done = "completed" if finish_todos else "pending"
    lead = ScriptedChatModel(script=[
        turn(call("write_todos", todos=[{"content": s, "status": "in_progress"} for s in ("draft", "review", "ship")])),
        turn(call("write_file", file_path="/reports/ACME-FY2024.md", content=note)),
        turn(call("task", description="Review /reports/ACME-FY2024.md", subagent_type="red-team")),
        turn(call("write_todos", todos=[{"content": s, "status": done} for s in ("draft", "review", "ship")])),
        turn(
            call("update_thesis", ticker="ACME", period="FY2024", stance="Neutral", thesis="Watch DSO. [10-K x, Item 7]"),
            call("publish_note", report_name="ACME-FY2024.md"),
            call("place_paper_trade", ticker="ACME", side="buy", quantity=10, rationale="r"),
        ),
        turn(text="done"),
    ])
    red = ScriptedChatModel(script=[turn(text="No HIGH findings.")])
    return lead, {"red-team": red}


def test_end_to_end_good_run_passes_and_reports_efficiency():
    lead, subs = scripted_event(GOOD)
    card = run_event(GOLDEN, model=lead, subagent_models=subs)
    assert card.passed, card.render()
    assert card.metrics["plan_completion"] == 1.0
    assert card.efficiency["llm_calls"] == 7  # six lead turns + one red-team turn: subagent calls are counted
    # Executed tools only: 2x write_todos, write_file, task, update_thesis, publish_note.
    # The rejected place_paper_trade never ran, so it is not counted.
    assert card.efficiency["tool_calls"] == 6


def test_eval_mode_refuses_trades():
    from equity_desk.broker import PaperBroker

    lead, subs = scripted_event(GOOD)
    run_event(GOLDEN, model=lead, subagent_models=subs)
    assert PaperBroker().orders() == []


def test_unfinished_plan_fails_the_run():
    lead, subs = scripted_event(GOOD, finish_todos=False)
    card = run_event(GOLDEN, model=lead, subagent_models=subs)
    assert failed_metrics(card) == {"plan_completion"}


def test_bad_note_run_fails_and_is_never_published():
    lead, subs = scripted_event(BAD)
    card = run_event(GOLDEN, model=lead, subagent_models=subs)
    assert not card.passed and "note was never published" in card.failures


def test_live_eval_refuses_to_run_without_a_key(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("EQUITY_DESK_MODEL", raising=False)
    assert main(["live"]) == 2


# ---------- tracing / usage ----------
def test_usage_tracker_sums_tokens_and_counts_tools():
    t = UsageTracker()
    for i, o in ((100, 20), (50, 5)):
        msg = AIMessage(content="x", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})
        t.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    t.on_tool_start({"name": "grep"}, "")
    t.on_tool_start({"name": "grep"}, "")
    t.on_tool_error(RuntimeError("x"))
    assert t.summary() == {"llm_calls": 2, "tool_calls": 2, "tool_errors": 1, "input_tokens": 150, "output_tokens": 25}


def test_tracing_status_reflects_environment(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not tracing_enabled() and "off" in tracing_status()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert not tracing_enabled()  # flag without a key is not enough
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-x")
    monkeypatch.setenv("LANGSMITH_PROJECT", "desk")
    assert tracing_enabled() and "desk" in tracing_status()


def test_run_config_carries_filterable_metadata():
    cfg = run_config("th", "ACME", "FY2024")
    assert cfg["configurable"]["thread_id"] == "th" and "ticker:ACME" in cfg["tags"]
    assert cfg["metadata"]["period"] == "FY2024"


def test_cli_names_the_missing_provider_key(monkeypatch, capsys):
    from equity_desk import cli

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("EQUITY_DESK_MODEL", raising=False)  # default is OpenRouter
    assert cli.main(["run", "ACME"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
