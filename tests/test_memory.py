"""Phase 3: durable memory across threads and restarts."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from equity_desk.agent import build_agent
from equity_desk.review import approve_all, drive
from equity_desk.memory import memory_backend, open_persistence, report_backend, seed_house_memory
from equity_desk.testing import ScriptedChatModel, call, turn
from equity_desk.tools.memory import update_thesis

THESIS = "Receivables outpacing revenue (+35% vs +12%). Watch DSO. [10-K 0009999999-25-000001, Item 7]"


def invoke(agent, thread, text="go"):
    cfg = {"configurable": {"thread_id": thread}}
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, cfg)


def tool_texts(result):
    return [m.text for m in result["messages"] if m.type == "tool"]


def test_seed_house_memory_is_idempotent_and_preserves_edits():
    store = InMemoryStore()
    assert seed_house_memory(store) is True
    memory_backend(store).write("/AGENTS.md", "edited by a human")
    assert seed_house_memory(store) is False
    assert memory_backend(store).read("/AGENTS.md").file_data["content"] == "edited by a human"


def test_update_thesis_validates_and_archives():
    store = InMemoryStore()
    ok = update_thesis.invoke(
        {"ticker": "acme", "period": "FY2024", "stance": "Neutral", "thesis": THESIS}
        | {"store": store}
    )
    assert "updated" in ok
    mem = memory_backend(store)
    current = mem.read("/ACME/thesis.md").file_data["content"]
    assert current == mem.read("/ACME/history/FY2024.md").file_data["content"]
    assert "stance: Neutral" in current and THESIS in current

    for bad in (
        {"ticker": "ACME", "period": "2024", "stance": "Neutral", "thesis": THESIS},
        {"ticker": "ACME", "period": "FY2024", "stance": "Buy", "thesis": THESIS},
        {"ticker": "ACME", "period": "FY2024", "stance": "Neutral", "thesis": "  "},
        {"ticker": "AC ME", "period": "FY2024", "stance": "Neutral", "thesis": THESIS},
    ):
        assert update_thesis.invoke(bad | {"store": store}).startswith("Error:")


def test_second_thread_sees_prior_thesis_report_and_house_memory():
    store = InMemoryStore()
    seed_house_memory(store)
    saver = InMemorySaver()

    run1 = ScriptedChatModel(script=[
        turn(call("write_file", file_path="/reports/ACME-FY2024.md", content="note v1")),
        turn(call("update_thesis", ticker="ACME", period="FY2024", stance="Neutral", thesis=THESIS)),
        turn(text="done"),
    ])
    # update_thesis is approval-gated (Phase 4), so a reviewer must approve it.
    drive(
        build_agent(run1, store=store, checkpointer=saver),
        {"messages": [{"role": "user", "content": "go"}]},
        {"configurable": {"thread_id": "thread-1"}},
        approve_all,
    )

    run2 = ScriptedChatModel(script=[
        turn(call("read_file", file_path="/memories/ACME/thesis.md"), call("ls", path="/reports/")),
        turn(text="Thesis changed: ..."),
    ])
    result = invoke(build_agent(run2, store=store, checkpointer=saver), "thread-2")  # new thread

    assert "Desk conventions" in " ".join(str(m.content) for m in run2.seen[0])  # AGENTS.md loaded
    outs = "\n".join(tool_texts(result))
    assert THESIS in outs and "stance: Neutral" in outs
    assert "/reports/ACME-FY2024.md" in outs


def test_threads_do_not_share_scratch_state():
    store = InMemoryStore()
    seed_house_memory(store)
    saver = InMemorySaver()
    a = ScriptedChatModel(script=[turn(call("write_file", file_path="/scratch/n.txt", content="x")), turn(text="ok")])
    invoke(build_agent(a, store=store, checkpointer=saver), "t1")
    b = ScriptedChatModel(script=[turn(call("read_file", file_path="/scratch/n.txt")), turn(text="ok")])
    result = invoke(build_agent(b, store=store, checkpointer=saver), "t2")
    assert tool_texts(result) == ["Error: File '/scratch/n.txt' not found"]


def test_agent_cannot_bypass_update_thesis_but_can_edit_house_memory():
    store = InMemoryStore()
    seed_house_memory(store)
    model = ScriptedChatModel(script=[
        turn(
            call("write_file", file_path="/memories/ACME/thesis.md", content="forged"),
            call("write_file", file_path="/memories/ACME/history/FY2024.md", content="forged"),
            call("edit_file", file_path="/memories/AGENTS.md",
                 old_string="(none yet; add durable lessons here when a human reviewer corrects the desk)",
                 new_string="- Always check DSO terms in Item 1A."),
        ),
        turn(text="done"),
    ])
    result = invoke(build_agent(model, store=store, checkpointer=InMemorySaver()), "t")
    outs = tool_texts(result)
    assert sum("permission denied" in o for o in outs) == 2
    assert memory_backend(store).read("/ACME/thesis.md").file_data is None
    assert "Always check DSO terms" in memory_backend(store).read("/AGENTS.md").file_data["content"]


def test_memory_survives_process_restart(tmp_path):
    with open_persistence(tmp_path) as (_, store):
        update_thesis.invoke(
            {"ticker": "ACME", "period": "FY2024", "stance": "Negative", "thesis": THESIS} | {"store": store}
        )
        report_backend(store).write("/ACME-FY2024.md", "note")
    with open_persistence(tmp_path) as (_, store2):  # simulates a fresh process
        assert "stance: Negative" in memory_backend(store2).read("/ACME/thesis.md").file_data["content"]
        assert report_backend(store2).read("/ACME-FY2024.md").file_data["content"] == "note"
        assert "Desk conventions" in memory_backend(store2).read("/AGENTS.md").file_data["content"]
