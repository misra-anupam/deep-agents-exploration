"""Trading and publishing tools. The consequential ones are gated by interrupt_on."""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore

from ..broker import OrderRejected, PaperBroker
from ..guardrails import check_note, known_accessions
from ..memory import report_backend


@tool
def get_portfolio() -> str:
    """Current paper portfolio: cash, equity, and positions. Call before sizing any trade."""
    return json.dumps(PaperBroker().snapshot(), indent=1)


@tool
def place_paper_trade(ticker: str, side: str, quantity: int, rationale: str) -> str:
    """Propose a PAPER trade (simulated, no real money). Requires human approval.

    Args:
        ticker: Upper-case symbol.
        side: "buy" or "sell". Shorting is not allowed.
        quantity: Whole shares, positive.
        rationale: Why, citing the thesis and sources. Required.

    The broker enforces limits (order size, 5% max position, cash, no shorting) even
    after approval; a rejection is returned as text so you can resize.
    """
    try:
        fill = PaperBroker().place_order(ticker, side, quantity, rationale)
    except OrderRejected as exc:
        return f"REJECTED by broker: {exc}"
    return (
        f"FILLED order #{fill.order_id}: {fill.side} {fill.quantity} {fill.ticker} "
        f"@ {fill.price:.2f} (notional ${fill.notional:,.0f}) [paper]"
    )


@tool
def publish_note(report_name: str, store: Annotated[Any, InjectedStore()]) -> str:
    """Publish a finished note from /reports/<report_name> to /reports/published/. Requires approval.

    Args:
        report_name: File name only, e.g. ACME-FY2024.md.
    """
    if "/" in report_name or not report_name.endswith(".md"):
        return "Error: report_name must be a bare file name ending in .md"
    backend = report_backend(store)
    draft = backend.read(f"/{report_name}")
    if draft.error or draft.file_data is None:
        return f"Error: no draft at /reports/{report_name}"
    violations = screen_report(report_name, store)
    if violations:  # enforced even after human approval: the last line of defence
        return "Error: cannot publish, compliance check failed:\n" + "\n".join(violations)
    result = backend.write(f"/published/{report_name}", draft.file_data["content"])
    if result.error:
        return f"Error: {result.error}"
    return f"Published /reports/published/{report_name}"


def screen_report(report_name: str, store) -> list[str]:
    """Guardrail violations for a draft, as readable strings. Shared with the approval prompt."""
    draft = report_backend(store).read(f"/{report_name}")
    if draft.error or draft.file_data is None:
        return [f"no draft at /reports/{report_name}"]
    ticker = report_name.split("-")[0].upper()
    try:
        known = known_accessions(ticker)
    except KeyError:
        known = None  # unknown ticker: skip the accession cross-check, keep the rest
    return [str(v) for v in check_note(draft.file_data["content"], known)]


TRADING_TOOLS = [get_portfolio, place_paper_trade, publish_note]
