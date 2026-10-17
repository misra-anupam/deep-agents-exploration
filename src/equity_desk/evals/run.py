"""Run the desk on a golden event (or score sample notes) and print a scorecard."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .. import config
from ..agent import build_agent
from ..memory import report_backend, seed_house_memory
from ..review import drive
from ..tracing import UsageTracker, run_config, tracing_status
from .scoring import Scorecard, load_golden, score_note

EVALS_DIR = config.ROOT / "evals"


def eval_policy(requests: list[dict], configs: list[dict]) -> list[dict]:
    """Stand-in reviewer: approve thesis/publish, refuse every trade (evals never trade)."""
    return [
        {"type": "reject", "message": "Eval mode: no trades."} if r["name"] == "place_paper_trade"
        else {"type": "approve"}
        for r in requests
    ]


def run_event(golden: dict, model=None, subagent_models=None) -> Scorecard:
    """Execute one golden event end to end and score what got published (falling back to the draft)."""
    store = InMemoryStore()
    seed_house_memory(store)
    agent = build_agent(model, subagent_models=subagent_models, store=store, checkpointer=InMemorySaver())
    tracker = UsageTracker()
    cfg = run_config(f"eval-{uuid.uuid4().hex[:8]}", golden["ticker"], golden["period"], callbacks=[tracker])
    prompt = (
        f"Write the {golden['period']} earnings note for {golden['ticker']} as "
        f"{golden['report_name']}, red-team it, update the thesis, and publish it."
    )
    result = drive(agent, {"messages": [{"role": "user", "content": prompt}]}, cfg, eval_policy)

    backend = report_backend(store)
    note = None
    for path in (f"/published/{golden['report_name']}", f"/{golden['report_name']}"):
        read = backend.read(path)
        if read.file_data:
            note = read.file_data["content"]
            break
    card = score_note(note, golden, todos=result.get("todos"), efficiency=tracker.summary())
    if note and not backend.read(f"/published/{golden['report_name']}").file_data:
        card.failures.append("note was never published")
    return card


def cmd_samples(_: argparse.Namespace) -> int:
    """Offline self-check: the scorer must pass the good sample and fail the bad one."""
    golden = load_golden(EVALS_DIR / "golden" / "acme_fy2024.json")
    ok = True
    for name, should_pass in (("good_note.md", True), ("bad_note.md", False)):
        card = score_note((EVALS_DIR / "samples" / name).read_text(), golden)
        verdict = card.passed == should_pass
        ok &= verdict
        print(f"\n== {name} (expected {'PASS' if should_pass else 'FAIL'}) -> "
              f"{'as expected' if verdict else 'SCORER IS WRONG'}\n{card.render()}")
    return 0 if ok else 1


def cmd_live(args: argparse.Namespace) -> int:
    models = (args.model,) if args.model else ()
    if missing := config.missing_api_keys(*models):
        print(f"Missing environment variable(s) for a live eval: {', '.join(missing)}", file=sys.stderr)
        return 2
    print(tracing_status())
    golden = load_golden(args.golden)
    card = run_event(golden, model=args.model)
    print(card.render())
    return 0 if card.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m equity_desk.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("samples", help="offline: verify the scorer separates good from bad notes").set_defaults(fn=cmd_samples)
    live = sub.add_parser("live", help="run the real agent on a golden event (needs an API key)")
    live.add_argument("--golden", default=str(EVALS_DIR / "golden" / "acme_fy2024.json"))
    live.add_argument("--model", help="override lead model, e.g. openrouter:anthropic/claude-sonnet-5.5")
    live.set_defaults(fn=cmd_live)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
