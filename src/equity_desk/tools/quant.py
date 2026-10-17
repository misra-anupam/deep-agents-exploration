"""Quant tools: deterministic ratios plus sandboxed free-form Python."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .. import ratios, sandbox
from ..data import get_provider

_CONCEPTS = {
    "revenue": "Revenues",
    "cost_of_revenue": "CostOfRevenue",
    "net_income": "NetIncomeLoss",
    "cfo": "NetCashProvidedByUsedInOperatingActivities",
    "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
    "receivables": "AccountsReceivableNetCurrent",
    "assets": "Assets",
    "depreciation": "DepreciationDepletionAndAmortization",
}


def annual_facts(ticker: str) -> dict[str, dict[int, dict]]:
    """concept -> {fiscal_year: {"val", "accn"}} using FY rows only."""
    gaap = get_provider().get_company_facts(ticker)["facts"].get("us-gaap", {})
    out: dict[str, dict[int, dict]] = {}
    for concept, entry in gaap.items():
        rows = {
            r["fy"]: {"val": r["val"], "accn": r["accn"]}
            for r in entry["units"].get("USD", [])
            if r.get("fp") == "FY"
        }
        if rows:
            out[concept] = rows
    return out


@tool
def compute_ratios(ticker: str) -> str:
    """Compute quality-of-earnings metrics and red flags from XBRL annual data.

    Returns growth, margins, DSO, cash conversion, accruals ratio, FCF, capex/D&A and a
    list of flags with thresholds applied. Deterministic: prefer this to hand-rolled math.
    """
    facts = annual_facts(ticker)
    missing = [c for c in _CONCEPTS.values() if c not in facts]
    if missing:
        return json.dumps({"error": f"missing XBRL concepts: {missing}"})
    years = sorted(set.intersection(*(set(facts[c]) for c in _CONCEPTS.values())))
    if len(years) < 2:
        return json.dumps({"error": "need at least two fiscal years of data"})
    series = {k: [float(facts[c][y]["val"]) for y in years] for k, c in _CONCEPTS.items()}
    report = ratios.quality_of_earnings(years=years, **series).as_dict()
    report["sources"] = sorted({facts[c][y]["accn"] for c in _CONCEPTS.values() for y in years})
    return json.dumps(report, indent=1)


@tool
def run_python(code: str, ticker: str = "") -> str:
    """Run Python in a locked-down sandbox (no network, writes confined, 20s timeout).

    If `ticker` is given, `facts.json` in the working directory holds annual XBRL values
    as {concept: {fiscal_year: value}}. Print results to stdout; stdout is returned.
    """
    files = {}
    if ticker:
        facts = annual_facts(ticker)
        files["facts.json"] = json.dumps(
            {c: {str(y): v["val"] for y, v in rows.items()} for c, rows in facts.items()}
        )
    try:
        return sandbox.run_python(code, files=files).render()
    except sandbox.SandboxUnavailable as exc:
        return f"Error: {exc}"


QUANT_TOOLS = [compute_ratios, run_python]
