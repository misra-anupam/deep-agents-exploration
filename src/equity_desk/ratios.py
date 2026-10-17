"""Deterministic quality-of-earnings math. Pure functions, no LLM, fully testable.

All inputs are annual USD series ordered oldest -> newest. Balance-sheet items
(receivables, assets) are year-end values, so DSO uses year-end receivables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Flag thresholds. Deliberately simple and explicit; tune per sector later.
AR_VS_REVENUE_GAP = 0.10  # AR growth exceeding revenue growth by more than 10pp
DSO_RISE_DAYS = 5.0
CASH_CONVERSION_FLOOR = 0.90  # CFO / net income
ACCRUALS_CEILING = 0.05  # (NI - CFO) / average assets


def growth(series: list[float]) -> list[float | None]:
    """Year-over-year growth; first element has no prior year."""
    return [None] + [
        (cur / prev - 1) if prev else None for prev, cur in zip(series, series[1:])
    ]


def dso(receivables: list[float], revenue: list[float], days: int = 365) -> list[float]:
    return [ar / rev * days for ar, rev in zip(receivables, revenue)]


@dataclass
class Flag:
    code: str
    detail: str


@dataclass
class QualityReport:
    years: list[int]
    metrics: dict[str, list[float | None]]
    flags: list[Flag] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "years": self.years,
            "metrics": {
                k: [None if v is None else round(v, 4) for v in vs]
                for k, vs in self.metrics.items()
            },
            "flags": [{"code": f.code, "detail": f.detail} for f in self.flags],
        }


def quality_of_earnings(
    years: list[int],
    revenue: list[float],
    cost_of_revenue: list[float],
    net_income: list[float],
    cfo: list[float],
    capex: list[float],
    receivables: list[float],
    assets: list[float],
    depreciation: list[float],
) -> QualityReport:
    n = len(years)
    if not all(
        len(s) == n
        for s in (revenue, cost_of_revenue, net_income, cfo, capex, receivables, assets, depreciation)
    ):
        raise ValueError("All series must have the same length as `years`")
    if n < 2:
        raise ValueError("Need at least two fiscal years")

    avg_assets = [None] + [(a + b) / 2 for a, b in zip(assets, assets[1:])]
    metrics: dict[str, list[float | None]] = {
        "revenue_growth": growth(revenue),
        "gross_margin": [1 - c / r for c, r in zip(cost_of_revenue, revenue)],
        "net_margin": [ni / r for ni, r in zip(net_income, revenue)],
        "receivables_growth": growth(receivables),
        "dso_days": dso(receivables, revenue),
        "cash_conversion": [c / ni if ni else None for c, ni in zip(cfo, net_income)],
        "accruals_ratio": [None]
        + [(ni - c) / aa for ni, c, aa in zip(net_income[1:], cfo[1:], avg_assets[1:])],
        "free_cash_flow": [c - x for c, x in zip(cfo, capex)],
        "capex_to_depreciation": [x / d if d else None for x, d in zip(capex, depreciation)],
    }

    flags: list[Flag] = []
    y = years[-1]
    rg, ag = metrics["revenue_growth"][-1], metrics["receivables_growth"][-1]
    if rg is not None and ag is not None and ag - rg > AR_VS_REVENUE_GAP:
        flags.append(
            Flag(
                "receivables_outpacing_revenue",
                f"FY{y}: receivables grew {ag:.1%} vs revenue {rg:.1%}",
            )
        )
    d_now, d_prev = metrics["dso_days"][-1], metrics["dso_days"][-2]
    if d_now - d_prev > DSO_RISE_DAYS:
        flags.append(Flag("dso_rising", f"FY{y}: DSO {d_prev:.1f} -> {d_now:.1f} days"))
    cc = metrics["cash_conversion"][-1]
    if cc is not None and cc < CASH_CONVERSION_FLOOR:
        flags.append(Flag("weak_cash_conversion", f"FY{y}: CFO/net income = {cc:.2f}"))
    ar = metrics["accruals_ratio"][-1]
    if ar is not None and ar > ACCRUALS_CEILING:
        flags.append(Flag("high_accruals", f"FY{y}: accruals ratio {ar:.1%} of average assets"))
    return QualityReport(years=years, metrics=metrics, flags=flags)
