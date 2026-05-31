import pytest

import financial_retrieval
from financial_retrieval import (
    select_latest_period,
    extract_profitability,
    extract_cash_flow,
    extract_debt,
    extract_valuation,
    fetch_financials,
)


def test_select_latest_period_picks_most_recent_eligible():
    periods = [
        {"calendarYear": "2025", "date": "2025-12-31", "revenue": 200},
        {"calendarYear": "2024", "date": "2024-12-31", "revenue": 100},
    ]

    selected = select_latest_period(periods, as_of_fiscal_year=2025)

    assert selected["revenue"] == 200


def test_select_latest_period_drops_future_year_leakage():
    periods = [
        {"calendarYear": "2026", "date": "2026-12-31", "revenue": 300},
        {"calendarYear": "2025", "date": "2025-12-31", "revenue": 200},
    ]

    selected = select_latest_period(periods, as_of_fiscal_year=2025)

    # The FY2026 statement postdates the FY2025 cutoff and must be excluded.
    assert selected["revenue"] == 200


def test_select_latest_period_falls_back_to_date_string():
    periods = [
        {"date": "2025-09-27", "revenue": 200},
        {"date": "2024-09-28", "revenue": 100},
    ]

    selected = select_latest_period(periods, as_of_fiscal_year=2025)

    assert selected["revenue"] == 200


def test_select_latest_period_returns_none_when_all_excluded():
    periods = [{"calendarYear": "2030", "date": "2030-12-31"}]

    assert select_latest_period(periods, as_of_fiscal_year=2025) is None


def test_extract_profitability_maps_fields():
    income = {"revenue": 1000, "netIncome": 250}
    ratios = {
        "grossProfitMargin": 0.4,
        "operatingProfitMargin": 0.3,
        "netProfitMargin": 0.25,
        "returnOnEquity": 0.5,
        "returnOnAssets": 0.2,
    }

    result = extract_profitability(income, ratios)

    assert result["revenue"] == 1000.0
    assert result["net_income"] == 250.0
    assert result["net_margin"] == 0.25
    assert result["return_on_equity"] == 0.5


def test_extract_cash_flow_maps_fields():
    cash_flow = {
        "operatingCashFlow": 500,
        "capitalExpenditure": -100,
        "freeCashFlow": 400,
    }

    result = extract_cash_flow(cash_flow)

    assert result["operating_cash_flow"] == 500.0
    assert result["capital_expenditure"] == -100.0
    assert result["free_cash_flow"] == 400.0


def test_extract_debt_maps_fields():
    balance = {"totalDebt": 900, "totalStockholdersEquity": 600}
    ratios = {"debtEquityRatio": 1.5, "currentRatio": 1.1, "interestCoverage": 8}

    result = extract_debt(balance, ratios)

    assert result["total_debt"] == 900.0
    assert result["total_equity"] == 600.0
    assert result["debt_to_equity"] == 1.5
    assert result["interest_coverage"] == 8.0


def test_extract_valuation_handles_missing_fields():
    result = extract_valuation({"priceEarningsRatio": 30})

    assert result["pe_ratio"] == 30.0
    assert result["pb_ratio"] is None
    assert result["ev_to_ebitda"] is None


def test_fetch_financials_integration(monkeypatch):
    payloads = {
        "income-statement": [
            {"calendarYear": "2025", "date": "2025-12-31",
             "revenue": 1000, "netIncome": 250},
            {"calendarYear": "2024", "date": "2024-12-31", "revenue": 800},
        ],
        "balance-sheet-statement": [
            {"calendarYear": "2025", "date": "2025-12-31",
             "totalDebt": 900, "totalStockholdersEquity": 600},
        ],
        "cash-flow-statement": [
            {"calendarYear": "2025", "date": "2025-12-31",
             "operatingCashFlow": 500, "freeCashFlow": 400},
        ],
        "ratios": [
            {"calendarYear": "2025", "date": "2025-12-31",
             "netProfitMargin": 0.25, "debtEquityRatio": 1.5,
             "priceEarningsRatio": 30},
        ],
    }

    def fake_fmp_get(endpoint, ticker, api_key, period="annual", limit=5):
        return payloads[endpoint]

    monkeypatch.setattr(financial_retrieval, "_fmp_get", fake_fmp_get)

    result = fetch_financials("aapl", api_key="dummy", as_of_fiscal_year=2025)

    assert result["ticker"] == "AAPL"
    assert result["fiscal_year"] == 2025
    assert result["report_date"] == "2025-12-31"
    assert result["profitability"]["revenue"] == 1000.0
    assert result["profitability"]["net_margin"] == 0.25
    assert result["cash_flow"]["free_cash_flow"] == 400.0
    assert result["debt"]["debt_to_equity"] == 1.5
    assert result["valuation"]["pe_ratio"] == 30.0


def test_fetch_financials_excludes_future_period(monkeypatch):
    payloads = {
        "income-statement": [
            {"calendarYear": "2026", "date": "2026-12-31", "revenue": 9999},
            {"calendarYear": "2025", "date": "2025-12-31", "revenue": 1000},
        ],
        "balance-sheet-statement": [],
        "cash-flow-statement": [],
        "ratios": [],
    }

    monkeypatch.setattr(
        financial_retrieval, "_fmp_get",
        lambda endpoint, *a, **k: payloads[endpoint],
    )

    result = fetch_financials("AAPL", api_key="dummy", as_of_fiscal_year=2025)

    assert result["fiscal_year"] == 2025
    assert result["profitability"]["revenue"] == 1000.0


def test_fetch_financials_stable_api_shape(monkeypatch):
    # The /stable API renamed fields (fiscalYear, debtToEquityRatio,
    # priceToEarningsRatio, interestCoverageRatio) and dropped ROE/ROA
    # from the ratios endpoint. This guards the compatibility shims.
    payloads = {
        "income-statement": [
            {"fiscalYear": "2025", "date": "2025-09-27",
             "revenue": 1000, "netIncome": 250},
        ],
        "balance-sheet-statement": [
            {"fiscalYear": "2025", "date": "2025-09-27",
             "totalDebt": 900, "totalStockholdersEquity": 500,
             "totalAssets": 2000},
        ],
        "cash-flow-statement": [
            {"fiscalYear": "2025", "date": "2025-09-27",
             "operatingCashFlow": 500, "freeCashFlow": 400},
        ],
        "ratios": [
            {"fiscalYear": "2025", "date": "2025-09-27",
             "netProfitMargin": 0.25, "debtToEquityRatio": 1.8,
             "interestCoverageRatio": 12.0, "priceToEarningsRatio": 30,
             "enterpriseValueMultiple": 20},
        ],
    }

    monkeypatch.setattr(
        financial_retrieval, "_fmp_get",
        lambda endpoint, *a, **k: payloads[endpoint],
    )

    result = fetch_financials("AAPL", api_key="dummy", as_of_fiscal_year=2025)

    assert result["fiscal_year"] == 2025
    assert result["debt"]["debt_to_equity"] == 1.8        # renamed field
    assert result["debt"]["interest_coverage"] == 12.0    # renamed field
    assert result["valuation"]["pe_ratio"] == 30.0        # renamed field
    assert result["valuation"]["ev_to_ebitda"] == 20.0
    # ROE/ROA absent from stable ratios -> computed from balance sheet.
    assert result["profitability"]["return_on_equity"] == 250 / 500
    assert result["profitability"]["return_on_assets"] == 250 / 2000


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

from financial_retrieval import _financials_cache_path  # noqa: E402


_CACHE_PAYLOADS = {
    "income-statement": [
        {"calendarYear": "2025", "date": "2025-12-31",
         "revenue": 1000, "netIncome": 250},
    ],
    "balance-sheet-statement": [
        {"calendarYear": "2025", "date": "2025-12-31",
         "totalDebt": 900, "totalStockholdersEquity": 600},
    ],
    "cash-flow-statement": [
        {"calendarYear": "2025", "date": "2025-12-31",
         "operatingCashFlow": 500, "freeCashFlow": 400},
    ],
    "ratios": [
        {"calendarYear": "2025", "date": "2025-12-31",
         "netProfitMargin": 0.25, "debtEquityRatio": 1.5,
         "priceEarningsRatio": 30},
    ],
}


def _install_fake_fmp(monkeypatch, call_log):
    def fake_fmp_get(endpoint, ticker, api_key, period="annual", limit=5):
        call_log.append(endpoint)
        return _CACHE_PAYLOADS[endpoint]

    monkeypatch.setattr(financial_retrieval, "_fmp_get", fake_fmp_get)


def test_fetch_financials_caches_and_reuses(monkeypatch, tmp_path):
    call_log = []
    _install_fake_fmp(monkeypatch, call_log)

    first = fetch_financials("AAPL", "dummy", cache_dir=tmp_path)
    second = fetch_financials("AAPL", "dummy", cache_dir=tmp_path)

    # First call hits all four endpoints; the second is served from disk.
    assert len(call_log) == 4
    assert first == second


def test_fetch_financials_writes_cache_file(monkeypatch, tmp_path):
    _install_fake_fmp(monkeypatch, [])

    fetch_financials("aapl", "dummy", period="annual", limit=5, cache_dir=tmp_path)

    expected = _financials_cache_path(tmp_path, "AAPL", "annual", 5)
    assert expected.exists()


def test_fetch_financials_no_cache_dir_always_calls_api(monkeypatch):
    call_log = []
    _install_fake_fmp(monkeypatch, call_log)

    fetch_financials("AAPL", "dummy")
    fetch_financials("AAPL", "dummy")

    # No cache_dir -> all four endpoints hit on every call.
    assert len(call_log) == 8
