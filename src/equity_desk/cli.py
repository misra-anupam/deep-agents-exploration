"""Command line: run the desk on a ticker with an interactive approval prompt."""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from . import config
from .agent import build_agent
from .audit import AuditLog
from .broker import PaperBroker
from .memory import open_persistence
from .review import drive
from .tracing import UsageTracker, run_config, tracing_status


def prompt_decisions(requests: list[dict], configs: list[dict]) -> list[dict]:
    decisions = []
    for req, cfg in zip(requests, configs):
        print("\n" + "=" * 70)
        print(f"APPROVAL NEEDED: {req['name']}")
        print(req.get("description") or json.dumps(req["args"], indent=2))
        allowed = cfg["allowed_decisions"]
        while True:
            choice = input(f"[{'/'.join(a[0] for a in allowed)}] "
                           f"({', '.join(allowed)}) > ").strip().lower()[:1]
            if choice == "a" and "approve" in allowed:
                decisions.append({"type": "approve"})
            elif choice == "r" and "reject" in allowed:
                reason = input("reason (shown to the agent) > ").strip() or "Rejected by reviewer"
                decisions.append({"type": "reject", "message": reason})
            elif choice == "e" and "edit" in allowed:
                print("Edit args as JSON (blank = keep):")
                raw = input("> ").strip()
                args = {**req["args"], **(json.loads(raw) if raw else {})}
                decisions.append({"type": "edit", "edited_action": {"name": req["name"], "args": args}})
            else:
                continue
            break
    return decisions


def cmd_run(args: argparse.Namespace) -> int:
    if missing := config.missing_api_keys():
        print(f"Missing environment variable(s) for the configured models: {', '.join(missing)}", file=sys.stderr)
        return 2
    with open_persistence() as (checkpointer, store):
        audit, tracker = AuditLog(config.audit_path()), UsageTracker()  # one instance: one lock
        agent = build_agent(checkpointer=checkpointer, store=store, audit=audit)
        thread = args.thread or f"{args.ticker}-{uuid.uuid4().hex[:8]}"
        print(f"thread: {thread}\n{tracing_status()}")
        prompt = (
            f"Write the {args.period} earnings note for {args.ticker}, update the thesis, "
            "publish it, and propose a paper trade only if clearly warranted."
        )
        result = drive(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            run_config(thread, args.ticker, args.period, callbacks=[tracker]),
            prompt_decisions,
            audit=audit,
        )
        print("\n" + result["messages"][-1].text)
        print(f"\nusage: {tracker.summary()}\naudit log: {audit.path} (chain "
              f"{'intact' if audit.verify() is None else 'BROKEN'})")
    return 0


def cmd_portfolio(_: argparse.Namespace) -> int:
    print(json.dumps(PaperBroker().snapshot(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="equity-desk", description="Equity research desk (paper trading only).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="analyse a ticker")
    run.add_argument("ticker")
    run.add_argument("--period", default="FY2024")
    run.add_argument("--thread", help="resume an existing thread id")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("portfolio", help="show paper portfolio").set_defaults(fn=cmd_portfolio)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
