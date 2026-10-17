"""Phases 1-2: lead delegates to scoped subagents through the real deepagents graph.

Each agent gets its own scripted model so call order is deterministic.
"""

from langgraph.store.memory import InMemoryStore

from equity_desk import config
from equity_desk.agent import build_agent
from equity_desk.memory import report_backend
from equity_desk.testing import ScriptedChatModel, call, turn

ACCN = "0009999999-25-000001"
READER_REPORT = (
    "Receivables rose to $182.0M from $135.0M [10-K 0009999999-25-000001, Item 7]. "
    "Terms extended from 45 to 75 days [10-K 0009999999-25-000001, Item 1A]."
)
QUANT_REPORT = (
    "Flags: receivables_outpacing_revenue, dso_rising, weak_cash_conversion "
    "[XBRL accn 0009999999-25-000001]."
)
NOTE = f"""# ACME FY2024
Receivables grew faster than revenue. {READER_REPORT}
{QUANT_REPORT}
Research aid only. Not investment advice.
"""


def make_models(lead_extra=()):
    lead = ScriptedChatModel(
        script=[
            turn(
                call(
                    "write_todos",
                    todos=[
                        {"content": "Read filing", "status": "in_progress"},
                        {"content": "Run quant checks", "status": "in_progress"},
                        {"content": "Write note", "status": "pending"},
                    ],
                ),
                call("task", description="Why did receivables rise at ACME?",
                     subagent_type="filings-reader"),
                call("task", description="Quality-of-earnings flags for ACME",
                     subagent_type="quant-analyst"),
            ),
            *lead_extra,
            turn(call("write_file", file_path="/reports/ACME-FY2024.md", content=NOTE)),
            turn(text="Saved /reports/ACME-FY2024.md."),
        ]
    )
    reader = ScriptedChatModel(
        script=[
            turn(call("list_filings", ticker="ACME")),
            turn(call("download_filing", ticker="ACME", accession=ACCN)),
            turn(call("grep", pattern="receivable", path="/filings/ACME")),
            turn(text=READER_REPORT),
        ]
    )
    quant = ScriptedChatModel(
        script=[
            turn(call("compute_ratios", ticker="ACME")),
            turn(call(
                "run_python", ticker="ACME",
                code="import json; f=json.load(open('facts.json'));"
                     "print('AR2024', f['AccountsReceivableNetCurrent']['2024'])",
            )),
            turn(text=QUANT_REPORT),
        ]
    )
    return lead, reader, quant


def run(lead, reader, quant, store=None):
    agent = build_agent(
        lead,
        subagent_models={"filings-reader": reader, "quant-analyst": quant},
        store=store or InMemoryStore(),
    )
    return agent.invoke({"messages": [{"role": "user", "content": "Write the FY2024 note for ACME"}]})


def text_of(model, kind=None):
    return "\n".join(
        m.text for batch in model.seen for m in batch if kind is None or m.type == kind
    )


def test_end_to_end_note_and_plan():
    lead, reader, quant = make_models()
    store = InMemoryStore()
    result = run(lead, reader, quant, store)
    saved = report_backend(store).read("/ACME-FY2024.md")
    assert saved.error is None and "Not investment advice" in saved.file_data["content"]
    assert len(result["todos"]) == 3
    assert result["messages"][-1].content.startswith("Saved")


def test_lead_sees_only_subagent_reports_not_raw_filing_text():
    lead, reader, quant = make_models()
    run(lead, reader, quant)
    lead_tool_text = text_of(lead, "tool")
    assert READER_REPORT in lead_tool_text and QUANT_REPORT in lead_tool_text
    assert "approximately 70% of revenue" not in lead_tool_text  # raw filing stayed in reader
    assert "receivables_outpacing_revenue" in lead_tool_text      # but conclusions came through
    # ...while the reader really did see filing content:
    assert "Receivables from our" in text_of(reader, "tool") or "receivable" in text_of(reader, "tool").lower()


def test_quant_sees_ratio_flags_and_sandbox_output():
    lead, reader, quant = make_models()
    run(lead, reader, quant)
    tool_text = text_of(quant, "tool")
    assert '"code": "receivables_outpacing_revenue"' in tool_text
    assert "AR2024 182000000" in tool_text


def test_skills_are_scoped_per_role():
    lead, reader, quant = make_models()
    run(lead, reader, quant)
    lead_p, reader_p, quant_p = (
        " ".join(str(m.content) for m in x.seen[0]) for x in (lead, reader, quant)
    )
    # Match each skill's own description text, not names the prompts happen to mention.
    note_skill = "House template and citation rules"
    qoe_skill = "Checklist for judging whether reported earnings are backed by cash"
    assert note_skill in lead_p and qoe_skill not in lead_p
    assert qoe_skill in quant_p and note_skill not in quant_p
    assert note_skill not in reader_p and qoe_skill not in reader_p


def test_lead_has_no_code_execution_or_data_tools():
    probe = turn(
        call("run_python", code="print(1)"),
        call("download_filing", ticker="ACME", accession=ACCN),
        call("execute", command="echo PWNED"),
    )
    lead, reader, quant = make_models(lead_extra=[probe])
    result = run(lead, reader, quant)
    # `seen` repeats earlier messages on every call, so read the final transcript instead.
    outs = [m.text for m in result["messages"] if m.type == "tool"]
    assert sum("is not a valid tool" in o for o in outs) == 2  # run_python, download_filing
    execute_out = next(o for o in outs if "Execution not available" in o)
    assert "PWNED" not in execute_out.replace("echo PWNED", "")


def test_agent_cannot_write_into_read_only_routes():
    bad = turn(call("write_file", file_path="/filings/ACME/evil.txt", content="x"))
    lead = ScriptedChatModel(script=[bad, turn(text="done")])
    agent = build_agent(lead, store=InMemoryStore())
    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    tool_msgs = [m.text for m in result["messages"] if m.type == "tool"]
    assert tool_msgs == ["Error: permission denied for write on /filings/ACME/evil.txt"]
    assert not (config.cache_dir() / "ACME" / "evil.txt").exists()
