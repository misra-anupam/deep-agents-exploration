"""Scorers for a research note against a golden event. Deterministic; no LLM judge."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..data import get_provider
from ..guardrails import check_note, citation_coverage, known_accessions
from ..tools.quant import annual_facts, compute_ratios

_USD = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?(billion|million|thousand|bn|mm|[BMK])?\b", re.IGNORECASE)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_MULT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}

USD_REL_TOL = 0.005   # 0.5%: covers rounding like $1,008.0 million
PCT_ABS_TOL = 0.5     # percentage points: covers "35%" for 34.8%


def usd_figures(text: str) -> list[float]:
    return [float(n.replace(",", "")) * _MULT.get((u or "").lower(), 1.0) for n, u in _USD.findall(text)]


def pct_figures(text: str) -> list[float]:
    return [float(n) for n in _PCT.findall(text)]


@dataclass
class Grounding:
    usd: list[float] = field(default_factory=list)
    pct: list[float] = field(default_factory=list)

    def has_usd(self, v: float) -> bool:
        return any(abs(v - g) <= USD_REL_TOL * max(abs(g), 1.0) for g in self.usd)

    def has_pct(self, v: float) -> bool:
        return any(abs(v - g) <= PCT_ABS_TOL for g in self.pct)


def build_grounding(ticker: str) -> Grounding:
    """Everything a note may legitimately quote: XBRL, filing text, and the ratio engine."""
    g = Grounding()
    for rows in annual_facts(ticker).values():
        g.usd.extend(float(r["val"]) for r in rows.values())
    provider = get_provider()
    for filing in provider.list_filings(ticker):
        text = provider.get_filing_text(ticker, filing["accession"])
        g.usd.extend(usd_figures(text))
        g.pct.extend(pct_figures(text))
    report = json.loads(compute_ratios.invoke({"ticker": ticker}))
    for name in ("revenue_growth", "gross_margin", "net_margin", "receivables_growth", "accruals_ratio"):
        g.pct.extend(v * 100 for v in report["metrics"][name] if v is not None)
    g.usd.extend(v for v in report["metrics"]["free_cash_flow"])
    return g


def numeric_grounding(note: str, g: Grounding) -> tuple[float, list[str]]:
    ungrounded = [f"${v:,.0f}" for v in usd_figures(note) if not g.has_usd(v)]
    ungrounded += [f"{v}%" for v in pct_figures(note) if not g.has_pct(v)]
    total = len(usd_figures(note)) + len(pct_figures(note))
    return (1.0 if total == 0 else 1 - len(ungrounded) / total), ungrounded


def flag_recall(note: str, golden: dict) -> tuple[float, list[str]]:
    # Citations like [XBRL AccountsReceivableNetCurrent ...] must not count as discussing a flag.
    low = re.sub(r"\[[^\]\n]*\]", " ", note).lower()
    missed = [f["code"] for f in golden["required_flags"] if not any(k.lower() in low for k in f["keywords"])]
    n = len(golden["required_flags"])
    return (1.0 if n == 0 else 1 - len(missed) / n), missed


def required_figures(note: str, golden: dict, ticker: str) -> tuple[float, list[str]]:
    facts, usd = annual_facts(ticker), usd_figures(note)
    missing = []
    for req in golden["required_figures"]:
        value = float(facts[req["concept"]][req["fy"]]["val"])
        if not any(abs(u - value) <= USD_REL_TOL * value for u in usd):
            missing.append(f"{req['concept']} FY{req['fy']} (${value:,.0f})")
    n = len(golden["required_figures"])
    return (1.0 if n == 0 else 1 - len(missing) / n), missing


def plan_completion(todos: list[dict] | None) -> float:
    if not todos:
        return 0.0
    return sum(1 for t in todos if t.get("status") == "completed") / len(todos)


@dataclass
class Scorecard:
    metrics: dict[str, float]
    details: dict[str, list[str]]
    failures: list[str]
    efficiency: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"{'metric':<22}{'value':>10}"]
        lines += [f"{k:<22}{v:>10.2f}" if isinstance(v, float) else f"{k:<22}{v:>10}" for k, v in self.metrics.items()]
        for k, v in self.efficiency.items():
            lines.append(f"{k:<22}{v:>10}")
        for name, items in self.details.items():
            if items:
                lines.append(f"  {name}: " + "; ".join(items[:6]))
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL -> " + ", ".join(self.failures)))
        return "\n".join(lines)


def load_golden(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def score_note(note: str | None, golden: dict, todos: list[dict] | None = None,
               efficiency: dict[str, int] | None = None) -> Scorecard:
    th = golden["thresholds"]
    if not note:
        return Scorecard({}, {}, ["no note produced"], efficiency or {})
    ticker = golden["ticker"]
    grounding, ungrounded = numeric_grounding(note, build_grounding(ticker))
    recall, missed = flag_recall(note, golden)
    figs, missing_figs = required_figures(note, golden, ticker)
    violations = check_note(note, known_accessions(ticker))
    metrics = {
        "numeric_grounding": grounding,
        "citation_coverage": citation_coverage(note),
        "flag_recall": recall,
        "required_figures": figs,
        "violations": len(violations),
    }
    if todos is not None:
        metrics["plan_completion"] = plan_completion(todos)
    failures = [
        f"{k} {metrics[k]:.2f} < {th[k]}" for k in
        ("numeric_grounding", "citation_coverage", "flag_recall", "required_figures", "plan_completion")
        if k in metrics and metrics[k] < th[k]
    ]
    if metrics["violations"] > th["violations"]:
        failures.append(f"violations {metrics['violations']} > {th['violations']}")
    details = {
        "ungrounded figures": ungrounded,
        "missed flags": missed,
        "missing figures": missing_figs,
        "violations": [str(v) for v in violations],
    }
    return Scorecard(metrics, details, failures, efficiency or {})
