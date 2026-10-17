"""Paper-trading broker backed by SQLite. No real money, no real orders.

Validation lives here, not in the prompt: even an approved order is re-checked, so a
human fat-finger or an `edit` decision cannot breach limits.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config

STARTING_CASH = 1_000_000.0
MAX_POSITION_PCT = 0.05  # one name may not exceed 5% of equity after the trade
MAX_ORDER_NOTIONAL = 50_000.0


class OrderRejected(ValueError):
    pass


@dataclass(frozen=True)
class Fill:
    order_id: int
    ticker: str
    side: str
    quantity: int
    price: float

    @property
    def notional(self) -> float:
        return self.quantity * self.price


def last_price(ticker: str) -> float:
    """Fixture price for offline runs. A live quote source plugs in here."""
    path = config.fixtures_dir() / ticker.upper() / "prices.json"
    if not path.exists():
        raise OrderRejected(f"No price available for {ticker!r}")
    return float(json.loads(path.read_text())["last"])


class PaperBroker:
    def __init__(self, db_path: Path | str | None = None, starting_cash: float = STARTING_CASH):
        self.db_path = Path(db_path or config.broker_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK (id = 1), cash REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, quantity INTEGER NOT NULL, avg_cost REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, side TEXT,
                    quantity INTEGER, price REAL, rationale TEXT
                );
                """
            )
            c.execute("INSERT OR IGNORE INTO account (id, cash) VALUES (1, ?)", (starting_cash,))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def snapshot(self, prices: dict[str, float] | None = None) -> dict:
        with self._conn() as c:
            cash = c.execute("SELECT cash FROM account").fetchone()["cash"]
            rows = c.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
        positions, equity = [], cash
        for r in rows:
            px = (prices or {}).get(r["ticker"]) or last_price(r["ticker"])
            value = r["quantity"] * px
            equity += value
            positions.append(
                {"ticker": r["ticker"], "quantity": r["quantity"], "avg_cost": r["avg_cost"],
                 "last": px, "value": round(value, 2)}
            )
        return {"cash": round(cash, 2), "equity": round(equity, 2), "positions": positions}

    def place_order(self, ticker: str, side: str, quantity: int, rationale: str = "") -> Fill:
        ticker, side = ticker.upper(), side.lower()
        if side not in ("buy", "sell"):
            raise OrderRejected(f"side must be buy or sell, got {side!r}")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise OrderRejected("quantity must be a positive integer")
        if not rationale.strip():
            raise OrderRejected("a rationale is required for every order")
        price = last_price(ticker)
        notional = quantity * price
        if notional > MAX_ORDER_NOTIONAL:
            raise OrderRejected(f"order notional ${notional:,.0f} exceeds limit ${MAX_ORDER_NOTIONAL:,.0f}")

        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            cash = c.execute("SELECT cash FROM account").fetchone()["cash"]
            pos = c.execute("SELECT * FROM positions WHERE ticker = ?", (ticker,)).fetchone()
            held = pos["quantity"] if pos else 0
            if side == "buy":
                if notional > cash:
                    raise OrderRejected(f"insufficient cash: need ${notional:,.0f}, have ${cash:,.0f}")
                equity = self.snapshot({ticker: price})["equity"]
                if (held + quantity) * price > MAX_POSITION_PCT * equity:
                    raise OrderRejected(
                        f"position would exceed {MAX_POSITION_PCT:.0%} of equity (${equity:,.0f})"
                    )
                new_qty = held + quantity
                avg = ((pos["avg_cost"] * held if pos else 0) + notional) / new_qty
                c.execute(
                    "INSERT INTO positions (ticker, quantity, avg_cost) VALUES (?, ?, ?) "
                    "ON CONFLICT(ticker) DO UPDATE SET quantity = excluded.quantity, avg_cost = excluded.avg_cost",
                    (ticker, new_qty, avg),
                )
                c.execute("UPDATE account SET cash = cash - ?", (notional,))
            else:
                if quantity > held:
                    raise OrderRejected(f"cannot sell {quantity}: hold {held} (no shorting)")
                if held == quantity:
                    c.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
                else:
                    c.execute("UPDATE positions SET quantity = ? WHERE ticker = ?", (held - quantity, ticker))
                c.execute("UPDATE account SET cash = cash + ?", (notional,))
            cur = c.execute(
                "INSERT INTO orders (ticker, side, quantity, price, rationale) VALUES (?, ?, ?, ?, ?)",
                (ticker, side, quantity, price, rationale),
            )
            return Fill(cur.lastrowid, ticker, side, quantity, price)

    def orders(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM orders ORDER BY id")]
