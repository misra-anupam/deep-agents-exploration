import pytest

from equity_desk import ratios


def acme(**override):
    base = dict(
        years=[2022, 2023, 2024],
        revenue=[800e6, 900e6, 1008e6],
        cost_of_revenue=[480e6, 531e6, 594.7e6],
        net_income=[80e6, 96e6, 113e6],
        cfo=[85e6, 88e6, 87e6],
        capex=[30e6, 32e6, 40e6],
        receivables=[100e6, 135e6, 182e6],
        assets=[1200e6, 1300e6, 1450e6],
        depreciation=[28e6, 30e6, 31e6],
    )
    base.update(override)
    return ratios.quality_of_earnings(**base)


def test_growth_and_dso_values():
    assert ratios.growth([100, 110]) == [None, pytest.approx(0.10)]
    assert ratios.dso([100e6], [800e6]) == [pytest.approx(45.625)]


def test_acme_metrics():
    m = acme().metrics
    assert m["revenue_growth"][-1] == pytest.approx(0.12)
    assert m["receivables_growth"][-1] == pytest.approx(182 / 135 - 1)
    assert m["gross_margin"][-1] == pytest.approx(0.41, abs=1e-3)
    assert m["cash_conversion"][-1] == pytest.approx(87 / 113)
    assert m["dso_days"][-1] == pytest.approx(182 / 1008 * 365)
    assert m["free_cash_flow"][-1] == pytest.approx(47e6)


def test_acme_flags_the_planted_problem():
    codes = {f.code for f in acme().flags}
    assert codes == {"receivables_outpacing_revenue", "dso_rising", "weak_cash_conversion"}


def test_clean_company_has_no_flags():
    clean = acme(
        cfo=[85e6, 100e6, 118e6],
        receivables=[100e6, 112e6, 126e6],
    )
    assert clean.flags == []


def test_high_accruals_flag():
    r = acme(cfo=[85e6, 40e6, 20e6], net_income=[80e6, 96e6, 160e6])
    assert "high_accruals" in {f.code for f in r.flags}


def test_validates_input_shape():
    with pytest.raises(ValueError):
        acme(revenue=[1, 2])
    with pytest.raises(ValueError):
        ratios.quality_of_earnings(
            years=[2024], revenue=[1], cost_of_revenue=[1], net_income=[1], cfo=[1],
            capex=[1], receivables=[1], assets=[1], depreciation=[1],
        )


def test_zero_net_income_does_not_divide_by_zero():
    r = acme(net_income=[80e6, 96e6, 0.0])
    assert r.metrics["cash_conversion"][-1] is None
