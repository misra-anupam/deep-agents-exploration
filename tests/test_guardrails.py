"""Phase 6: compliance/citation checks, publish gates, audit chain, retries."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from equity_desk import config
from equity_desk.agent import build_agent
from equity_desk.audit import AuditLog
from equity_desk.guardrails import (
    DISCLAIMER, check_note, citation_coverage, known_accessions, red_team_done,
)
from equity_desk.memory import report_backend, seed_house_memory
from equity_desk.review import approve_all, drive
from equity_desk.testing import ScriptedChatModel, call, turn

ACCN = "0009999999-25-000001"
GOOD = f"""# ACME FY2024
Revenue grew 12.0% to $1,008.0 million [XBRL Revenues FY2024, accn {ACCN}].
Receivables rose to $182.0M [10-K {ACCN}, Item 7].

## Sources
- /filings/ACME/{ACCN}.txt (total $9.9M not cited here, and that is fine)

{DISCLAIMER}
"""


def codes(text, known=None):
    return [v.code for v in check_note(text, known)]


# ---------- pure checks ----------
def test_good_note_is_clean():
    assert check_note(GOOD, {ACCN}) == []
    assert citation_coverage(GOOD) == 1.0


@pytest.mark.parametrize(
    "phrase",
    ["Returns are guaranteed.", "A risk-free setup.", "Margins will keep expanding.",
     "Demand will definitely rise.", "This can't lose.", "Buy now.", "Total no-brainer."],
)
def test_banned_language_is_caught(phrase):
    assert "banned_language" in codes(GOOD + phrase + "\n")


def test_neutral_phrasing_is_allowed():
    assert "banned_language" not in codes(GOOD + "Management guides to 8% to 10% growth [10-K x1, Item 7].\n")


def test_missing_disclaimer_and_uncited_figures_and_unknown_accession():
    bad = f"# T\nRevenue rose 12% and margin hit $3.2M.\nSee [10-K 0000000000-00-000000, Item 7] text.\n"
    found = codes(bad, {ACCN})
    assert found.count("uncited_figure") == 1 and "missing_disclaimer" in found and "unknown_accession" in found


def test_years_and_headings_are_not_figures():
    assert "uncited_figure" not in codes(f"# FY2024 Results 2024\nFiscal 2024 compared with 2023.\n{DISCLAIMER}\n")


def test_citation_coverage_is_fractional():
    text = f"Rev $1.0M [10-K x, Item 7].\nCosts $2.0M.\n{DISCLAIMER}\n"
    assert citation_coverage(text) == 0.5


def test_known_accessions_come_from_filings_and_xbrl():
    assert ACCN in known_accessions("ACME") and "0009999999-24-000001" in known_accessions("ACME")


def _ai(*calls):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in calls])


def _tool(i, text="ok", status="success"):
    return ToolMessage(content=text, tool_call_id=i, status=status)


def test_red_team_done_requires_review_after_the_draft_and_a_real_result():
    write = _ai(("write_file", {"file_path": "/reports/A.md"}, "w"))
    red = _ai(("task", {"subagent_type": "red-team"}, "r"))
    assert red_team_done([write, _tool("w"), red, _tool("r", "1. MED ...")], "A.md")
    assert not red_team_done([red, _tool("r", "ok"), write, _tool("w")], "A.md")  # reviewed an older draft
    assert not red_team_done([write, _tool("w"), red, _tool("r", "boom", "error")], "A.md")
    assert not red_team_done([write, _tool("w"), red], "A.md")  # never returned
    assert not red_team_done([write, _tool("w")], "A.md")


# ---------- publish gates in the real graph ----------
def publishing_agent(draft, with_red_team):
    store = InMemoryStore()
    seed_house_memory(store)
    script = [turn(call("write_file", file_path="/reports/ACME-FY2024.md", content=draft))]
    if with_red_team:
        script.append(turn(call("task", description="Review /reports/ACME-FY2024.md", subagent_type="red-team")))
    script += [turn(call("publish_note", report_name="ACME-FY2024.md")), turn(text="done")]
    subs = {"red-team": ScriptedChatModel(script=[turn(text="No HIGH findings.")])}
    agent = build_agent(ScriptedChatModel(script=script), subagent_models=subs, store=store,
                        checkpointer=InMemorySaver())
    return agent, store


CFG = {"configurable": {"thread_id": "g"}}
START = {"messages": [{"role": "user", "content": "go"}]}


def run_and_approve(agent):
    paused = agent.invoke(START, CFG)
    description = paused["__interrupt__"][0].value["action_requests"][0]["description"]
    result = drive(agent, Command(resume={"decisions": [{"type": "approve"}]}), CFG, approve_all)
    return description, [m.text for m in result["messages"] if m.type == "tool"][-1]


def test_publish_without_red_team_is_refused_even_when_a_human_approves():
    agent, store = publishing_agent(GOOD, with_red_team=False)
    description, outcome = run_and_approve(agent)
    assert "Red-team review: NOT DONE" in description and "Compliance check: PASS" in description
    assert "red-team has not reviewed" in outcome
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data is None


def test_noncompliant_note_is_refused_even_when_reviewed_and_approved():
    bad = f"# ACME\nReturns are guaranteed at 40% [10-K {ACCN}, Item 7]. Fake ref 0000000000-00-000000.\n"
    agent, store = publishing_agent(bad, with_red_team=True)
    description, outcome = run_and_approve(agent)
    assert "Red-team review: done" in description and "Compliance check: FAIL" in description
    assert "compliance check failed" in outcome and "banned_language" in outcome
    assert "unknown_accession" in outcome and "missing_disclaimer" in outcome
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data is None


def test_clean_reviewed_note_publishes():
    agent, store = publishing_agent(GOOD, with_red_team=True)
    description, outcome = run_and_approve(agent)
    assert "Compliance check: PASS" in description and "Red-team review: done" in description
    assert outcome == "Published /reports/published/ACME-FY2024.md"
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data["content"] == GOOD


# ---------- audit log ----------
def test_audit_chain_detects_edits_and_deletions(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(4):
        log.append({"event": "tool_call", "tool": f"t{i}"})
    assert log.verify() is None
    lines = log.path.read_text().splitlines()

    log.path.write_text("\n".join([lines[0], lines[1].replace("t1", "tX"), *lines[2:]]) + "\n")
    assert log.verify() == 3  # the line AFTER the edited one no longer chains

    log.path.write_text("\n".join(lines[1:]) + "\n")  # drop the first entry
    assert log.verify() == 1
    log.path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")  # drop a middle entry
    assert log.verify() == 2


def test_audit_redacts_secrets_and_truncates(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append({"event": "tool_call", "args": {"api_key": "sk-secret", "Authorization": "Bearer x", "body": "z" * 900}})
    (entry,) = log.entries()
    assert entry["args"]["api_key"] == "[REDACTED]" and entry["args"]["Authorization"] == "[REDACTED]"
    assert "sk-secret" not in log.path.read_text() and "chars]" in entry["args"]["body"]


def test_every_agent_and_human_decision_lands_in_one_intact_chain():
    store = InMemoryStore()
    seed_house_memory(store)
    lead = ScriptedChatModel(script=[
        turn(call("task", description="QoE", subagent_type="quant-analyst")),
        turn(call("place_paper_trade", ticker="ACME", side="buy", quantity=10, rationale="r")),
        turn(text="done"),
    ])
    quant = ScriptedChatModel(script=[turn(call("compute_ratios", ticker="ACME")), turn(text="flags")])
    agent = build_agent(lead, subagent_models={"quant-analyst": quant}, store=store, checkpointer=InMemorySaver())
    log = AuditLog(config.audit_path())
    drive(agent, START, CFG, lambda reqs, cfgs: [{"type": "reject", "message": "no"} for _ in reqs], audit=log)

    rows = [(e["event"], e.get("agent"), e["tool"]) for e in log.entries()]
    assert ("tool_call", "quant-analyst", "compute_ratios") in rows
    assert ("tool_call", "lead", "task") in rows
    decision = next(e for e in log.entries() if e["event"] == "human_decision")
    assert (decision["tool"], decision["decision"], decision["detail"]) == ("place_paper_trade", "reject", "no")
    assert log.verify() is None


# ---------- retries ----------
def test_transient_network_errors_are_retried_then_succeed(monkeypatch):
    from equity_desk.data import FixtureProvider

    class Flaky(FixtureProvider):
        calls = 0

        def get_company_facts(self, ticker):
            Flaky.calls += 1
            if Flaky.calls <= 2:
                raise httpx.ConnectError("blip")
            return super().get_company_facts(ticker)

    monkeypatch.setattr("equity_desk.tools.quant.get_provider", lambda: Flaky())
    lead = ScriptedChatModel(script=[
        turn(call("task", description="QoE", subagent_type="quant-analyst")), turn(text="done")])
    quant = ScriptedChatModel(script=[turn(call("compute_ratios", ticker="ACME")), turn(text="ok")])
    agent = build_agent(lead, subagent_models={"quant-analyst": quant}, store=InMemoryStore())
    agent.invoke(START)
    assert Flaky.calls == 3  # two failures retried, third succeeded
    assert "receivables_outpacing_revenue" in " ".join(m.text for m in quant.seen[-1] if m.type == "tool")
