"""Human-in-the-loop policy: which tools need approval, and what the reviewer sees."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

from .broker import PaperBroker, last_price
from .guardrails import red_team_done

ALL_DECISIONS = ["approve", "edit", "reject"]


def _trade_description(tool_call: dict, state: Any, runtime: Any) -> str:
    a = tool_call["args"]
    try:
        px = last_price(a["ticker"])
        snap = PaperBroker().snapshot()
        notional = a["quantity"] * px
        impact = f"${notional:,.0f} = {notional / snap['equity']:.2%} of ${snap['equity']:,.0f} equity"
    except Exception as exc:  # never let a description failure block the review itself
        impact = f"impact unavailable ({exc})"
    return (
        f"PAPER TRADE: {a['side']} {a['quantity']} {a['ticker']}\n{impact}\n"
        f"Rationale: {a['rationale']}"
    )


def _thesis_description(tool_call: dict, state: Any, runtime: Any) -> str:
    a = tool_call["args"]
    return f"THESIS CHANGE: {a['ticker']} -> {a['stance']} ({a['period']})\n{a['thesis']}"


def _publish_description(tool_call: dict, state: Any, runtime: Any) -> str:
    from .tools.trading import screen_report  # local: avoids an import cycle with the tools

    name = tool_call["args"]["report_name"]
    try:
        problems = screen_report(name, runtime.store)
        compliance = "PASS" if not problems else f"FAIL ({len(problems)}): " + "; ".join(problems[:3])
    except Exception as exc:
        compliance = f"unavailable ({exc})"
    reviewed = "done" if red_team_done(state["messages"], name) else "NOT DONE"
    return f"PUBLISH NOTE: {name}\nCompliance check: {compliance}\nRed-team review: {reviewed}"


def interrupt_policy() -> dict[str, InterruptOnConfig]:
    return {
        "place_paper_trade": {"allowed_decisions": ALL_DECISIONS, "description": _trade_description},
        "update_thesis": {"allowed_decisions": ALL_DECISIONS, "description": _thesis_description},
        "publish_note": {"allowed_decisions": ALL_DECISIONS, "description": _publish_description},
    }
