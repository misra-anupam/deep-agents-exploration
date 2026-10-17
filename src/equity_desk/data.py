"""Filing/XBRL data providers: offline fixtures and live SEC EDGAR."""

from __future__ import annotations

import json
import re
import threading
import time
from html import unescape
from pathlib import Path
from typing import Protocol

import httpx

from . import config


class DataProvider(Protocol):
    def list_filings(self, ticker: str, form: str | None = None) -> list[dict]: ...
    def get_filing_text(self, ticker: str, accession: str) -> str: ...
    def get_company_facts(self, ticker: str) -> dict: ...


class FixtureProvider:
    """Reads synthetic filings from fixtures/<TICKER>/."""

    def __init__(self, root: Path | None = None):
        self.root = root or config.fixtures_dir()

    def _dir(self, ticker: str) -> Path:
        path = self.root / ticker.upper()
        if not path.is_dir():
            raise KeyError(f"No fixture data for ticker {ticker!r}")
        return path

    def list_filings(self, ticker: str, form: str | None = None) -> list[dict]:
        rows = json.loads((self._dir(ticker) / "index.json").read_text())
        return [r for r in rows if form is None or r["form"] == form]

    def get_filing_text(self, ticker: str, accession: str) -> str:
        for row in self.list_filings(ticker):
            if row["accession"] == accession:
                return (self._dir(ticker) / row["file"]).read_text()
        raise KeyError(f"Unknown accession {accession!r} for {ticker}")

    def get_company_facts(self, ticker: str) -> dict:
        return json.loads((self._dir(ticker) / "companyfacts.json").read_text())


class _RateLimiter:
    """SEC fair-access policy allows at most 10 requests/second."""

    def __init__(self, per_second: float = 8.0):
        self.min_gap = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_gap:
                time.sleep(self.min_gap - gap)
            self._last = time.monotonic()


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|h\d|li)>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{2,}", "\n\n", text).strip()


class EdgarProvider:
    """Live SEC EDGAR access. Requires EDGAR_USER_AGENT."""

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            headers={"User-Agent": config.edgar_user_agent()}, timeout=30.0
        )
        self._limiter = _RateLimiter()
        self._cik_cache: dict[str, int] = {}

    def _get(self, url: str) -> httpx.Response:
        self._limiter.wait()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def _cik(self, ticker: str) -> int:
        if not self._cik_cache:
            data = self._get("https://www.sec.gov/files/company_tickers.json").json()
            self._cik_cache = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
        try:
            return self._cik_cache[ticker.upper()]
        except KeyError:
            raise KeyError(f"Unknown ticker {ticker!r}") from None

    def list_filings(self, ticker: str, form: str | None = None) -> list[dict]:
        cik = self._cik(ticker)
        recent = self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()[
            "filings"
        ]["recent"]
        rows = []
        for i, f in enumerate(recent["form"]):
            if form is None or f == form:
                rows.append(
                    {
                        "accession": recent["accessionNumber"][i],
                        "form": f,
                        "filed": recent["filingDate"][i],
                        "period": recent["reportDate"][i],
                        "file": recent["primaryDocument"][i],
                    }
                )
        return rows

    def get_filing_text(self, ticker: str, accession: str) -> str:
        cik = self._cik(ticker)
        row = next(r for r in self.list_filings(ticker) if r["accession"] == accession)
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{row['file']}"
        )
        return html_to_text(self._get(url).text)

    def get_company_facts(self, ticker: str) -> dict:
        cik = self._cik(ticker)
        return self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()


def get_provider() -> DataProvider:
    return EdgarProvider() if config.data_mode() == "edgar" else FixtureProvider()
