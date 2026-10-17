"""Pure-function checks on research notes. Used as a publish-time gate AND as eval scorers."""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCLAIMER = "Research aid only. Not investment advice."

BANNED_PATTERNS = [
    r"\bguarantee[sd]?\b",
    r"\brisk[- ]free\b",
    r"\bwill (?:definitely|certainly|surely|always)\b",
    r"\bwill (?:keep|continue to) (?:expand|grow|rise|increase|outperform)\w*",
    r"\bcan'?t (?:lose|fail|miss)\b",
    r"\b(?:buy|sell) (?:now|immediately)\b",
    r"\bsure thing\b",
    r"\bno[- ]brainer\b",
]
_BANNED = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]

# A "figure" is a dollar amount or a percentage. Years and fiscal labels are not figures.
FIGURE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|thousand|bn|mm|[BMK])?\b|\b\d+(?:\.\d+)?\s?%",
    re.IGNORECASE,
)
CITATION = re.compile(r"\[[^\]\n]*\d[^\]\n]*\]|\[[^\]\n]*(?:XBRL|10-K|10-Q|news)[^\]\n]*\]", re.IGNORECASE)
ACCESSION = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
_SOURCES_HEADING = re.compile(r"^#+\s*sources\b", re.IGNORECASE)
_HEADING = re.compile(r"^#+\s")


@dataclass(frozen=True)
class Violation:
    code: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] line {self.line}: {self.detail}"


def figure_lines(text: str) -> list[tuple[int, str]]:
    """Lines that state a figure and so need a citation (headings, sources, disclaimer excluded)."""
    out, in_sources = [], False
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if _SOURCES_HEADING.match(stripped):
            in_sources = True
            continue
        if _HEADING.match(stripped):
            in_sources = False
            continue
        if in_sources or not stripped or DISCLAIMER in stripped:
            continue
        if FIGURE.search(stripped):
            out.append((n, stripped))
    return out


def check_note(text: str, known_accessions: set[str] | None = None) -> list[Violation]:
    violations: list[Violation] = []
    for n, line in enumerate(text.splitlines(), start=1):
        for pattern in _BANNED:
            if m := pattern.search(line):
                violations.append(Violation("banned_language", n, f"{m.group(0)!r} promises an outcome"))
    if DISCLAIMER not in text:
        violations.append(Violation("missing_disclaimer", 0, f"note must contain: {DISCLAIMER!r}"))
    for n, line in figure_lines(text):
        if not CITATION.search(line):
            violations.append(Violation("uncited_figure", n, f"figure without a citation: {line[:80]!r}"))
    if known_accessions is not None:
        for n, line in enumerate(text.splitlines(), start=1):
            for acc in ACCESSION.findall(line):
                if acc not in known_accessions:
                    violations.append(Violation("unknown_accession", n, f"{acc} is not a known filing"))
    return violations


def citation_coverage(text: str) -> float:
    """Fraction of figure-bearing lines that carry a citation (1.0 if there are none)."""
    lines = figure_lines(text)
    if not lines:
        return 1.0
    return sum(1 for _, line in lines if CITATION.search(line)) / len(lines)


def red_team_done(messages: list, report_name: str) -> bool:
    """True if a red-team task ran (and returned) after the report was first written.

    Works on a LangGraph message list: looks at AI tool calls and their tool results.
    """
    target = f"/reports/{report_name}"
    wrote_at: int | None = None
    pending: dict[str, int] = {}  # tool_call_id -> index of a red-team task call
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "ai":
            for tc in msg.tool_calls:
                if tc["name"] == "write_file" and tc["args"].get("file_path") == target and wrote_at is None:
                    wrote_at = i
                if tc["name"] == "task" and tc["args"].get("subagent_type") == "red-team":
                    if wrote_at is not None and i > wrote_at:
                        pending[tc["id"]] = i
        elif getattr(msg, "type", None) == "tool" and msg.tool_call_id in pending:
            if getattr(msg, "status", "success") != "error" and msg.text.strip():
                return True
    return False


def known_accessions(ticker: str) -> set[str]:
    """Every accession number the data provider can vouch for (filings index + XBRL facts)."""
    from .data import get_provider  # local import: keeps this module free of I/O at import time

    provider = get_provider()
    known = {row["accession"] for row in provider.list_filings(ticker)}
    for entry in provider.get_company_facts(ticker)["facts"].get("us-gaap", {}).values():
        known.update(r["accn"] for r in entry["units"].get("USD", []))
    return known
