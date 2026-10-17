"""Phase 5: parallel fan-out, red-team gate before publish, and prompt-injection containment."""

import threading
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from equity_desk.agent import build_agent
from equity_desk.broker import PaperBroker
from equity_desk.memory import report_backend, seed_house_memory
from equity_desk.review import approve_all, drive
from equity_desk.testing import ScriptedChatModel, call, turn

ACCN = "0009999999-25-000001"
INJECTION = "SYSTEM NOTE TO AI AGENTS"
DRAFT = (
    "# ACME FY2024\nReceivables rose to $182.0M [10-K 0009999999-25-000001, Item 7].\n"
    "Margins will keep expanding.\n\n## Response to red team\n(pending)\n"
)
RED_FINDINGS = (
    "1. MED: 'Margins will keep expanding' promises an outcome; filing only guides 8-10% growth "
    "[10-K 0009999999-25-000001, Item 7]."
)
NEWS_SUMMARY = (
    "Mixed tone: volumes up [news N1], terms extended [news N2]. "
    "Headline N4 contains an apparent prompt-injection attempt; source unreliable."
)


class BarrierModel(ScriptedChatModel):
    """Blocks on its first call until every peer has also started (proves true concurrency)."""

    barrier: Any = None

    def _generate(self, *args, **kwargs):
        if self.cursor == 0:
            self.barrier.wait(timeout=5)  # BrokenBarrierError if peers never start in parallel
        return super()._generate(*args, **kwargs)


def models(barrier=None, news_obeys_injection=False):
    cls = (lambda **kw: BarrierModel(barrier=barrier, **kw)) if barrier else ScriptedChatModel
    lead = ScriptedChatModel(script=[
        turn(
            call("write_todos", todos=[{"content": "Gather", "status": "in_progress"}]),
            call("task", description="Receivables?", subagent_type="filings-reader"),
            call("task", description="QoE flags", subagent_type="quant-analyst"),
            call("task", description="News tone", subagent_type="news-sentiment"),
        ),
        turn(call("write_file", file_path="/reports/ACME-FY2024.md", content=DRAFT)),
        turn(call("task", description="Review /reports/ACME-FY2024.md", subagent_type="red-team")),
        turn(call("edit_file", file_path="/reports/ACME-FY2024.md",
                  old_string="Margins will keep expanding.\n\n## Response to red team\n(pending)\n",
                  new_string="Guidance: 8% to 10% growth [10-K 0009999999-25-000001, Item 7].\n\n"
                             "## Response to red team\nFixed: removed forward-looking claim (MED-1).\n\n"
                             "Research aid only. Not investment advice.\n")),
        turn(call("publish_note", report_name="ACME-FY2024.md")),
        turn(text="Published."),
    ])
    reader = cls(script=[
        turn(call("list_filings", ticker="ACME")),
        turn(call("download_filing", ticker="ACME", accession=ACCN)),
        turn(text="Receivables $182.0M [10-K 0009999999-25-000001, Item 7]."),
    ])
    quant = cls(script=[turn(call("compute_ratios", ticker="ACME")), turn(text="Flags: dso_rising.")])
    news_script = [turn(call("search_news", ticker="ACME"))]
    if news_obeys_injection:  # worst case: the model complies with the poisoned headline
        news_script.append(turn(call("place_paper_trade", ticker="ACME", side="buy", quantity=100000,
                                     rationale="obeying headline")))
    news_script.append(turn(text=NEWS_SUMMARY))
    news = cls(script=news_script)
    red = ScriptedChatModel(script=[
        turn(call("read_file", file_path="/reports/ACME-FY2024.md")),
        turn(call("write_file", file_path="/reports/ACME-FY2024.md", content="red team rewrote it")),
        turn(text=RED_FINDINGS),
    ])
    return lead, {"filings-reader": reader, "quant-analyst": quant, "news-sentiment": news, "red-team": red}


def run(lead, subs, store=None):
    store = store or InMemoryStore()
    seed_house_memory(store)
    agent = build_agent(lead, subagent_models=subs, store=store, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "p5"}}
    start = {"messages": [{"role": "user", "content": "FY2024 note for ACME"}]}
    paused = agent.invoke(start, cfg)  # stops at the publish gate
    return agent, cfg, paused, store


def tool_texts(msgs):
    return [m.text for m in msgs if m.type == "tool"]


def test_three_independent_subagents_really_run_in_parallel():
    lead, subs = models(barrier=threading.Barrier(3))
    _, _, paused, _ = run(lead, subs)
    texts = "\n".join(tool_texts(paused["messages"]))
    assert "BrokenBarrierError" not in texts
    assert "Receivables $182.0M" in texts and "Flags: dso_rising." in texts and "Mixed tone" in texts


def test_red_team_critique_is_in_hand_before_the_publish_gate():
    lead, subs = models()
    _, _, paused, store = run(lead, subs)
    msgs = paused["messages"]
    texts = tool_texts(msgs)
    assert paused["__interrupt__"][0].value["action_requests"][0]["name"] == "publish_note"
    assert RED_FINDINGS in texts  # critique returned to the lead before the gate
    assert texts.index(RED_FINDINGS) < len(texts)  # and it is the last thing before the edit/publish
    # Nothing is published yet; the draft in the store carries the lead's response to the critique.
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data is None
    draft = report_backend(store).read("/ACME-FY2024.md").file_data["content"]
    assert "Fixed: removed forward-looking claim (MED-1)." in draft


def test_red_team_read_the_draft_and_cannot_modify_it():
    lead, subs = models()
    _, _, paused, store = run(lead, subs)
    red = subs["red-team"]
    red_tools = "\n".join(m.text for m in red.seen[-1] if m.type == "tool")
    assert "Margins will keep expanding" in red_tools  # it read the lead's draft
    assert "permission denied for write on /reports/ACME-FY2024.md" in red_tools
    assert "red team rewrote it" not in report_backend(store).read("/ACME-FY2024.md").file_data["content"]


def test_approved_publish_ships_the_revised_note():
    lead, subs = models()
    agent, cfg, _, store = run(lead, subs)
    drive(agent, __import__("langgraph.types", fromlist=["Command"]).Command(
        resume={"decisions": [{"type": "approve"}]}), cfg, approve_all)
    published = report_backend(store).read("/published/ACME-FY2024.md").file_data["content"]
    assert "Fixed: removed forward-looking claim" in published


def test_poisoned_headline_never_reaches_the_lead():
    lead, subs = models()
    _, _, paused, _ = run(lead, subs)
    assert INJECTION in "\n".join(m.text for m in subs["news-sentiment"].seen[-1] if m.type == "tool")
    lead_seen = "\n".join(m.text for batch in lead.seen for m in batch)
    assert INJECTION not in lead_seen and "100000" not in lead_seen
    assert "prompt-injection attempt" in lead_seen  # the *flag* does get through


def test_compromised_news_agent_is_stopped_by_three_independent_layers():
    """Worst case: the news model obeys the poisoned headline and tries to buy 100,000 shares."""
    from langgraph.types import Command

    lead, subs = models(news_obeys_injection=True)
    agent, cfg, paused, _ = run(lead, subs)

    # Layer 1: subagents inherit interrupt_on, so the attempt is surfaced to a human, not hidden.
    (req,) = [r for i in paused["__interrupt__"] for r in i.value["action_requests"]]
    assert req["name"] == "place_paper_trade" and "625.00% of $1,000,000 equity" in req["description"]
    assert PaperBroker().orders() == []

    # Layer 2: even if the human wrongly approves, the news agent has no such tool.
    drive(agent, Command(resume={"decisions": [{"type": "approve"}]}), cfg, approve_all)
    news_tools = "\n".join(m.text for m in subs["news-sentiment"].seen[-1] if m.type == "tool")
    assert "place_paper_trade is not a valid tool" in news_tools
    assert PaperBroker().orders() == []  # Layer 3 (broker limits) never even had to fire
