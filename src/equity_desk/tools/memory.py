"""Thesis memory: the only sanctioned way to change a ticker's standing thesis."""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore

from ..memory import memory_backend

STANCES = ("Positive", "Neutral", "Negative")
_TICKER = re.compile(r"^[A-Z]{1,6}$")
_PERIOD = re.compile(r"^(FY\d{4}|Q[1-4]FY\d{4})$")


def thesis_path(ticker: str) -> str:
    return f"/{ticker}/thesis.md"


def render_thesis(ticker: str, period: str, stance: str, body: str) -> str:
    return f"---\nticker: {ticker}\nperiod: {period}\nstance: {stance}\n---\n{body.strip()}\n"


@tool
def update_thesis(
    ticker: str,
    period: str,
    stance: str,
    thesis: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Replace the standing thesis for a ticker and archive it under history/<period>.md.

    Args:
        ticker: Upper-case symbol, e.g. ACME.
        period: Fiscal period the thesis is based on, e.g. FY2024 or Q3FY2024.
        stance: Positive, Neutral, or Negative.
        thesis: Markdown body: what we believe, why, and what would change our mind.
            Cite sources. Must state what changed vs. the previous thesis if one exists.
    """
    ticker = ticker.upper()
    if not _TICKER.match(ticker):
        return f"Error: invalid ticker {ticker!r}"
    if not _PERIOD.match(period):
        return f"Error: invalid period {period!r}; use FY2024 or Q3FY2024"
    if stance not in STANCES:
        return f"Error: stance must be one of {STANCES}"
    if not thesis.strip():
        return "Error: thesis body is empty"

    backend = memory_backend(store)
    content = render_thesis(ticker, period, stance, thesis)
    for path in (thesis_path(ticker), f"/{ticker}/history/{period}.md"):
        result = backend.write(path, content)
        if result.error:
            return f"Error: {result.error}"
    return f"Thesis for {ticker} updated ({stance}, {period}); archived to history/{period}.md"


MEMORY_TOOLS = [update_thesis]
