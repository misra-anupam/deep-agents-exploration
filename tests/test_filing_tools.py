import json

import httpx
import pytest

from equity_desk import config
from equity_desk.data import EdgarProvider, html_to_text
from equity_desk.tools.filings import download_filing, get_xbrl_facts, list_filings

ACCN = "0009999999-25-000001"


def test_list_filings_filters_form():
    rows = json.loads(list_filings.invoke({"ticker": "ACME", "form": "10-K"}))
    assert [r["accession"] for r in rows][0] == ACCN
    assert json.loads(list_filings.invoke({"ticker": "ACME", "form": "8-K"})) == []


def test_download_filing_caches_and_returns_outline_not_text():
    out = json.loads(download_filing.invoke({"ticker": "acme", "accession": ACCN}))
    assert out["path"] == f"/filings/ACME/{ACCN}.txt"
    assert (config.cache_dir() / "ACME" / f"{ACCN}.txt").exists()
    headings = [h["heading"] for h in out["outline"]]
    assert any(h.startswith("Item 7") for h in headings)
    assert any(h.startswith("Note 4") for h in headings)
    assert "outline" in out and "text" not in out


def test_xbrl_facts_newest_first_and_cites_accession():
    out = json.loads(
        get_xbrl_facts.invoke(
            {"ticker": "ACME", "concepts": ["Revenues", "Bogus"], "periods": 2}
        )
    )
    assert [r["value"] for r in out["Revenues"]] == [1008000000, 900000000]
    assert out["Revenues"][0]["accn"] == ACCN
    assert out["Bogus"] == "not reported"


def test_unknown_ticker_raises():
    with pytest.raises(KeyError):
        list_filings.invoke({"ticker": "NOPE"})


def test_html_to_text_strips_markup():
    assert html_to_text("<p>Revenue &amp; costs</p><script>x()</script><p>Next</p>") == (
        "Revenue & costs\nNext"
    )
    assert html_to_text("<p>A</p><p></p><p>B</p>") == "A\n\nB"


def test_edgar_provider_parses_submissions_with_mock_transport(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"ticker": "ACME", "cik_str": 123}})
        if "submissions/CIK0000000123" in url:
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["10-K", "8-K"],
                            "accessionNumber": ["0000000123-25-000001", "0000000123-25-000002"],
                            "filingDate": ["2025-02-01", "2025-03-01"],
                            "reportDate": ["2024-12-31", ""],
                            "primaryDocument": ["a.htm", "b.htm"],
                        }
                    }
                },
            )
        return httpx.Response(404)

    provider = EdgarProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    provider._limiter.min_gap = 0
    rows = provider.list_filings("acme", "10-K")
    assert rows == [
        {
            "accession": "0000000123-25-000001",
            "form": "10-K",
            "filed": "2025-02-01",
            "period": "2024-12-31",
            "file": "a.htm",
        }
    ]
