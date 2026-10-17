import pytest

from equity_desk.broker import MAX_ORDER_NOTIONAL, OrderRejected, PaperBroker


def test_buy_updates_cash_and_position_with_average_cost():
    b = PaperBroker()
    b.place_order("ACME", "buy", 100, "thesis")
    b.place_order("acme", "BUY", 100, "add")
    snap = b.snapshot()
    assert snap["cash"] == 1_000_000 - 200 * 62.5
    assert snap["positions"] == [
        {"ticker": "ACME", "quantity": 200, "avg_cost": 62.5, "last": 62.5, "value": 12500.0}
    ]
    assert snap["equity"] == 1_000_000.0  # marking at the same price: no P&L


def test_sell_reduces_then_closes_position():
    b = PaperBroker()
    b.place_order("ACME", "buy", 100, "x")
    b.place_order("ACME", "sell", 40, "trim")
    assert b.snapshot()["positions"][0]["quantity"] == 60
    b.place_order("ACME", "sell", 60, "exit")
    assert b.snapshot()["positions"] == []


@pytest.mark.parametrize(
    "side,qty,rationale,match",
    [
        ("short", 10, "x", "buy or sell"),
        ("buy", 0, "x", "positive integer"),
        ("buy", -5, "x", "positive integer"),
        ("buy", 1.5, "x", "positive integer"),
        ("buy", True, "x", "positive integer"),
        ("buy", 10, "  ", "rationale"),
        ("sell", 10, "x", "hold 0"),  # no shorting
        ("buy", int(MAX_ORDER_NOTIONAL / 62.5) + 1, "x", "exceeds limit"),
    ],
)
def test_invalid_orders_are_rejected_and_change_nothing(side, qty, rationale, match):
    b = PaperBroker()
    with pytest.raises(OrderRejected, match=match):
        b.place_order("ACME", side, qty, rationale)
    assert b.orders() == [] and b.snapshot()["cash"] == 1_000_000


def test_position_limit_is_enforced_cumulatively():
    b = PaperBroker()  # 5% of $1M = $50k = 800 shares at 62.5
    b.place_order("ACME", "buy", 700, "x")
    with pytest.raises(OrderRejected, match="5% of equity"):
        b.place_order("ACME", "buy", 101, "x")
    b.place_order("ACME", "buy", 100, "x")  # exactly at the limit is fine


def test_insufficient_cash_and_unknown_ticker():
    b = PaperBroker(starting_cash=1_000)
    with pytest.raises(OrderRejected, match="insufficient cash"):
        b.place_order("ACME", "buy", 100, "x")
    with pytest.raises(OrderRejected, match="No price"):
        b.place_order("ZZZZ", "buy", 1, "x")


def test_state_persists_across_instances(tmp_path):
    path = tmp_path / "b.sqlite"
    PaperBroker(path).place_order("ACME", "buy", 10, "x")
    again = PaperBroker(path)
    assert again.snapshot()["positions"][0]["quantity"] == 10
    assert len(again.orders()) == 1
