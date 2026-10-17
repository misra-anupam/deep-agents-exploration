"""Deterministic filing/XBRL tools. The agent decides what to look at; these fetch it.

Large filings are written to the cache (routed to /filings/ in the agent's
filesystem) and only a small outline is returned, keeping the context window clean.
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from .. import config
from ..data import get_provider

_ITEM = re.compile(r"^(Item\s+\d+[A-C]?\.[^\n]*|Note\s+\d+\.[^\n]*)", re.IGNORECASE)


def _outline(text: str) -> list[dict]:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        m = _ITEM.match(line.strip())
        if m:
            out.append({"line": n, "heading": m.group(1)[:80]})
    return out


@tool
def list_filings(ticker: str, form: str = "10-K") -> str:
    """List available SEC filings for a ticker (accession, form, filed date, period).

    Use this first to find the accession number to download.
    """
    rows = get_provider().list_filings(ticker, form)
    return json.dumps(rows, indent=1)


@tool
def download_filing(ticker: str, accession: str) -> str:
    """Download a filing and cache it under /filings/<TICKER>/<accession>.txt.

    Returns the virtual path, line count, and a section outline with line numbers.
    The full text is NOT returned: use grep and read_file(offset, limit) on the path.
    """
    text = get_provider().get_filing_text(ticker, accession)
    path = config.cache_dir() / ticker.upper() / f"{accession}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return json.dumps(
        {
            "path": f"/filings/{ticker.upper()}/{accession}.txt",
            "lines": text.count("\n") + 1,
            "chars": len(text),
            "outline": _outline(text),
        },
        indent=1,
    )


@tool
def get_xbrl_facts(ticker: str, concepts: list[str], periods: int = 3) -> str:
    """Get annual (FY) XBRL values for us-gaap concepts, newest first.

    Example concepts: Revenues, NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities,
    AccountsReceivableNetCurrent. Values are in USD. Each row cites its accession.
    """
    facts = get_provider().get_company_facts(ticker)["facts"].get("us-gaap", {})
    result: dict[str, list[dict] | str] = {}
    for concept in concepts:
        entry = facts.get(concept)
        if not entry:
            result[concept] = "not reported"
            continue
        rows = [r for r in entry["units"].get("USD", []) if r.get("fp") == "FY"]
        rows.sort(key=lambda r: r["end"], reverse=True)
        result[concept] = [
            {"end": r["end"], "value": r["val"], "form": r["form"], "accn": r["accn"]}
            for r in rows[:periods]
        ]
    return json.dumps(result, indent=1)


FILING_TOOLS = [list_filings, download_filing, get_xbrl_facts]
