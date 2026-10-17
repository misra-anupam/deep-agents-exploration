"""Phase 4: approval gates, decisions, and durability."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from equity_desk.agent import build_agent
from equity_desk.broker import PaperBroker
from equity_desk.memory import memory_backend, open_persistence, report_backend, seed_house_memory
from equity_desk.review import approve_all, drive, reject_all
from equity_desk.testing import ScriptedChatModel, call, turn

TRADE = dict(ticker="ACME", side="buy", quantity=100, rationale="Thesis: Neutral; valuation fair.")
CFG = {"configurable": {"thread_id": "t1"}}
START = {"messages": [{"role": "user", "content": "go"}]}


GOOD_NOTE = (
    "# ACME FY2024\nReceivables rose to $182.0M [10-K 0009999999-25-000001, Item 7].\n"
    "Research aid only. Not investment advice.\n"
)


def agent_with(script, store=None, saver=None, subs=None):
    store = store or InMemoryStore()
    seed_house_memory(store)
    model = ScriptedChatModel(script=script)
    agent = build_agent(model, store=store, checkpointer=saver or InMemorySaver(), subagent_models=subs)
    return agent, model, store


def tool_texts(result):
    return [m.text for m in result["messages"] if m.type == "tool"]


def test_trade_pauses_before_executing_and_shows_portfolio_impact():
    agent, _, _ = agent_with([turn(call("place_paper_trade", **TRADE)), turn(text="done")])
    result = agent.invoke(START, CFG)
    (intr,) = result["__interrupt__"]
    req = intr.value["action_requests"][0]
    assert req["name"] == "place_paper_trade"
    assert "0.62% of $1,000,000 equity" in req["description"]
    assert PaperBroker().orders() == []  # nothing happened yet


def test_approve_executes():
    agent, _, _ = agent_with([turn(call("place_paper_trade", **TRADE)), turn(text="done")])
    result = drive(agent, START, CFG, approve_all)
    assert PaperBroker().snapshot()["positions"][0]["quantity"] == 100
    assert "FILLED order #1: buy 100 ACME" in tool_texts(result)[0]


def test_reject_blocks_trade_and_tells_the_agent_why():
    agent, model, _ = agent_with([turn(call("place_paper_trade", **TRADE)), turn(text="ok, no trade")])
    result = drive(agent, START, CFG, reject_all("Too early; wait for 10-Q"))
    assert PaperBroker().orders() == []
    assert "Too early; wait for 10-Q" in tool_texts(result)[0]
    assert "Too early; wait for 10-Q" in " ".join(m.text for m in model.seen[-1] if m.type == "tool")


def test_edit_changes_what_executes():
    agent, _, _ = agent_with([turn(call("place_paper_trade", **TRADE)), turn(text="done")])

    def shrink(requests, configs):
        (req,) = requests
        return [{"type": "edit", "edited_action": {"name": req["name"], "args": {**req["args"], "quantity": 40}}}]

    drive(agent, START, CFG, shrink)
    (order,) = PaperBroker().orders()
    assert order["quantity"] == 40


@pytest.mark.parametrize("qty", [10_000, 801])  # over order-size limit; over 5% position limit
def test_broker_limits_hold_even_after_human_approval(qty):
    agent, _, _ = agent_with([turn(call("place_paper_trade", **{**TRADE, "quantity": qty})), turn(text="done")])
    result = drive(agent, START, CFG, approve_all)
    assert tool_texts(result)[0].startswith("REJECTED by broker:")
    assert PaperBroker().orders() == []


def test_human_edit_cannot_breach_limits_either():
    agent, _, _ = agent_with([turn(call("place_paper_trade", **TRADE)), turn(text="done")])

    def fat_finger(requests, configs):
        (req,) = requests
        return [{"type": "edit", "edited_action": {"name": req["name"], "args": {**req["args"], "quantity": 100_000}}}]

    result = drive(agent, START, CFG, fat_finger)
    assert "REJECTED by broker" in tool_texts(result)[0] and PaperBroker().orders() == []


def test_read_only_tools_do_not_interrupt():
    agent, _, _ = agent_with([turn(call("get_portfolio")), turn(text="done")])
    result = agent.invoke(START, CFG)
    assert "__interrupt__" not in result and '"cash": 1000000.0' in tool_texts(result)[0]


def test_thesis_and_publish_are_gated_until_approved():
    thesis = dict(ticker="ACME", period="FY2024", stance="Neutral", thesis="Watch DSO. [10-K, Item 7]")
    store = InMemoryStore()
    agent, _, _ = agent_with(
        [
            turn(call("write_file", file_path="/reports/ACME-FY2024.md", content=GOOD_NOTE)),
            turn(call("task", description="Review /reports/ACME-FY2024.md", subagent_type="red-team")),
            turn(call("update_thesis", **thesis), call("publish_note", report_name="ACME-FY2024.md")),
            turn(text="done"),
        ],
        store=store,
        subs={"red-team": ScriptedChatModel(script=[turn(text="No HIGH findings.")])},
    )
    result = agent.invoke(START, CFG)
    assert result["__interrupt__"]
    assert memory_backend(store).read("/ACME/thesis.md").file_data is None  # not written yet
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data is None

    drive(agent, Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), CFG, approve_all)
    assert memory_backend(store).read("/ACME/thesis.md").file_data is not None
    assert report_backend(store).read("/published/ACME-FY2024.md").file_data["content"] == GOOD_NOTE


def test_rejected_thesis_is_never_written():
    thesis = dict(ticker="ACME", period="FY2024", stance="Positive", thesis="Bullish.")
    store = InMemoryStore()
    agent, _, _ = agent_with([turn(call("update_thesis", **thesis)), turn(text="done")], store=store)
    drive(agent, START, CFG, reject_all("unsupported"))
    assert memory_backend(store).read("/ACME/thesis.md").file_data is None


def test_pending_approval_survives_a_process_restart(tmp_path):
    with open_persistence(tmp_path) as (saver, store):
        agent, _, _ = agent_with([turn(call("place_paper_trade", **TRADE))], store=store, saver=saver)
        assert agent.invoke(START, CFG)["__interrupt__"]
    # --- process "dies"; a brand-new agent, saver and store reopen the same files ---
    with open_persistence(tmp_path) as (saver2, store2):
        agent2, _, _ = agent_with([turn(text="filled, done")], store=store2, saver=saver2)
        result = agent2.invoke(Command(resume={"decisions": [{"type": "approve"}]}), CFG)
    assert "FILLED" in tool_texts(result)[0]
    assert PaperBroker().snapshot()["positions"][0]["quantity"] == 100
