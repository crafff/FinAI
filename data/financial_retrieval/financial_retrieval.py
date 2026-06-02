import json
from datetime import datetime, timezone
from pathlib import Path


FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Where the on-disk cache lives by default. Callers pass
# cache_dir=DEFAULT_FINANCIALS_CACHE to turn caching on; tests pass tmp_path.
DEFAULT_FINANCIALS_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "financials"

# The four statement endpoints fetched together, in this order.
_ENDPOINTS = (
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
    "ratios",
)


def _to_float(value):
    """
    Coerce an FMP numeric field to float, or None when missing/blank.
    """
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(period, *names):
    """
    Return the first present, non-empty value among several candidate
    field names. FMP renamed some fields between the legacy /api/v3 and
    the current /stable API (e.g. debtEquityRatio -> debtToEquityRatio),
    so callers pass both spellings.
    """
    for name in names:
        value = period.get(name)
        if value not in (None, ""):
            return value
    return None


def _period_year(period):
    """
    Return the integer fiscal year of one FMP statement period.

    Prefers the explicit calendarYear field; falls back to the first
    four characters of the date field. Returns None if neither parses.
    """
    # v3 used calendarYear; stable uses fiscalYear.
    year = period.get("calendarYear") or period.get("fiscalYear")

    if year is not None:
        try:
            return int(year)
        except (TypeError, ValueError):
            pass

    date_str = period.get("date", "")

    try:
        return int(str(date_str)[:4])
    except (TypeError, ValueError):
        return None


def select_latest_period(periods, as_of_fiscal_year):
    """
    Pick the most recent period at or before `as_of_fiscal_year`.

    FMP returns statements newest-first, but a run executed in 2026 may
    see a FY2026 statement that postdates the FY2025 10-K and its T0
    cutoff. Including it would be look-ahead leakage, so periods with a
    fiscal year greater than as_of_fiscal_year are dropped before
    selecting the latest remaining one.

    Returns the chosen period dict, or None if none qualify.
    """
    eligible = []

    for period in periods or []:
        year = _period_year(period)

        if year is None or year > as_of_fiscal_year:
            continue

        eligible.append((year, str(period.get("date", "")), period))

    if not eligible:
        return None

    eligible.sort(key=lambda item: (item[0], item[1]))

    return eligible[-1][2]


def extract_profitability(income, ratios, balance=None):
    """
    Profitability metrics. Margins come from the ratios endpoint; absolute
    figures from the income statement.

    ROE/ROA: the /stable `ratios` endpoint no longer returns them, so when
    absent they are computed from net income and the balance sheet
    (netIncome / equity, netIncome / assets).
    """
    income = income or {}
    ratios = ratios or {}
    balance = balance or {}

    net_income = _to_float(income.get("netIncome"))

    roe = _to_float(ratios.get("returnOnEquity"))
    if roe is None:
        equity = _to_float(balance.get("totalStockholdersEquity"))
        if net_income is not None and equity not in (None, 0):
            roe = net_income / equity

    roa = _to_float(ratios.get("returnOnAssets"))
    if roa is None:
        assets = _to_float(balance.get("totalAssets"))
        if net_income is not None and assets not in (None, 0):
            roa = net_income / assets

    return {
        "revenue": _to_float(income.get("revenue")),
        "net_income": net_income,
        "gross_margin": _to_float(ratios.get("grossProfitMargin")),
        "operating_margin": _to_float(ratios.get("operatingProfitMargin")),
        "net_margin": _to_float(ratios.get("netProfitMargin")),
        "return_on_equity": roe,
        "return_on_assets": roa,
    }


def extract_cash_flow(cash_flow):
    """
    Cash-flow metrics from the cash-flow statement.
    """
    cash_flow = cash_flow or {}

    return {
        "operating_cash_flow": _to_float(cash_flow.get("operatingCashFlow")),
        "capital_expenditure": _to_float(cash_flow.get("capitalExpenditure")),
        "free_cash_flow": _to_float(cash_flow.get("freeCashFlow")),
    }


def extract_debt(balance, ratios):
    """
    Leverage / solvency metrics from the balance sheet and ratios.
    """
    balance = balance or {}
    ratios = ratios or {}

    return {
        "total_debt": _to_float(balance.get("totalDebt")),
        "total_equity": _to_float(balance.get("totalStockholdersEquity")),
        "debt_to_equity": _to_float(
            _first(ratios, "debtToEquityRatio", "debtEquityRatio")
        ),
        "current_ratio": _to_float(ratios.get("currentRatio")),
        "interest_coverage": _to_float(
            _first(ratios, "interestCoverageRatio", "interestCoverage")
        ),
    }


def extract_valuation(ratios):
    """
    Valuation ratios from the ratios endpoint.

    NOTE on leakage: these ratios are computed by FMP at the fiscal
    period end, so they reflect the period-end price - not a live price.
    They are safe to feed the agent as point-in-time context. Any
    valuation ratio that must reflect the prediction-moment price should
    instead be recomputed downstream from the T0 baseline close (see
    price_retrieval), never pulled live, which would leak post-cutoff
    information.
    """
    ratios = ratios or {}

    return {
        "pe_ratio": _to_float(
            _first(ratios, "priceToEarningsRatio", "priceEarningsRatio")
        ),
        "pb_ratio": _to_float(ratios.get("priceToBookRatio")),
        "price_to_sales": _to_float(ratios.get("priceToSalesRatio")),
        "ev_to_ebitda": _to_float(
            _first(
                ratios,
                "enterpriseValueMultiple",
                "evToEBITDA",
                "enterpriseValueOverEBITDA",
            )
        ),
    }


def merge_financials(ticker, income, balance, cash_flow, ratios):
    """
    Combine the four selected period dicts into the project schema.
    """
    fiscal_year = _period_year(income or {})
    report_date = (income or {}).get("date")

    return {
        "ticker": ticker.upper(),
        "fiscal_year": fiscal_year,
        "report_date": report_date,
        "profitability": extract_profitability(income, ratios, balance),
        "cash_flow": extract_cash_flow(cash_flow),
        "debt": extract_debt(balance, ratios),
        "valuation": extract_valuation(ratios),
    }


def _fmp_get(endpoint, ticker, api_key, period="annual", limit=5):
    """
    Single network boundary for FMP requests. Tests monkeypatch this.

    Calls https://financialmodelingprep.com/stable/<endpoint>?symbol=<ticker>
    and returns the parsed JSON list (newest period first).
    """
    import requests

    # The /stable API takes the ticker as a `symbol` query param (the
    # legacy /api/v3 API put it in the path).
    url = f"{FMP_BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        params={
            "symbol": ticker.upper(),
            "period": period,
            "limit": limit,
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        # FMP returns {"Error Message": ...} on bad symbol / quota.
        raise ValueError(f"FMP error for {endpoint}/{ticker}: {data}")

    return data


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial writes
    from interrupted runs do not corrupt the cache.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def _financials_cache_path(cache_dir, ticker, period, limit):
    """
    Cache file for one (ticker, period, limit) request:
    <cache_dir>/<TICKER>/financials_<period>_<limit>.json
    """
    return (
        Path(cache_dir) / ticker.upper() / f"financials_{period}_{limit}.json"
    )


def _read_financials_cache(cache_dir, ticker, period, limit):
    """
    Return the cached raw endpoint responses for this request, or None.

    Returns a dict keyed by endpoint name (each value a list of period
    dicts, newest first), matching what `_fmp_get` returns per endpoint.
    """
    path = _financials_cache_path(cache_dir, ticker, period, limit)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("raw")


def _write_financials_cache(cache_dir, ticker, period, limit, raw):
    """
    Atomically persist the raw FMP responses for this request.
    """
    path = _financials_cache_path(cache_dir, ticker, period, limit)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": ticker.upper(),
        "period": period,
        "limit": limit,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
    }

    _atomic_write(path, json.dumps(payload, indent=2))


def fetch_financials(
    ticker,
    api_key,
    as_of_fiscal_year=2025,
    period="annual",
    limit=5,
    cache_dir=None,
):
    """
    Fetch structured fundamentals for one company from FMP.

    Pulls the income statement, balance sheet, cash-flow statement, and
    ratios; selects the latest period at or before `as_of_fiscal_year`
    (the leakage guard); and merges them into one schema.

    When `cache_dir` is given, the four raw endpoint responses are cached
    to disk keyed by (ticker, period, limit) and a repeated request is
    served from disk instead of spending FMP quota (4 calls per ticker).
    Pass DEFAULT_FINANCIALS_CACHE in production; None disables caching
    (used by tests). Mirrors the EDGAR / RAG / FinnHub cache convention.

    Schema returned:

        {
            ticker, fiscal_year, report_date,
            profitability: {revenue, net_income, gross_margin,
                            operating_margin, net_margin,
                            return_on_equity, return_on_assets},
            cash_flow:     {operating_cash_flow, capital_expenditure,
                            free_cash_flow},
            debt:          {total_debt, total_equity, debt_to_equity,
                            current_ratio, interest_coverage},
            valuation:     {pe_ratio, pb_ratio, price_to_sales,
                            ev_to_ebitda},
        }
    """
    # The cache stores the RAW endpoint responses (all periods). The
    # leakage guard (select_latest_period with as_of_fiscal_year) and the
    # merge are reapplied on every read, so they can change - and a
    # different as_of can be requested - without invalidating the cache.
    raw = None

    if cache_dir is not None:
        raw = _read_financials_cache(cache_dir, ticker, period, limit)

    if raw is None:
        raw = {
            endpoint: _fmp_get(endpoint, ticker, api_key, period, limit)
            for endpoint in _ENDPOINTS
        }

        if cache_dir is not None:
            _write_financials_cache(cache_dir, ticker, period, limit, raw)

    income = select_latest_period(raw["income-statement"], as_of_fiscal_year)
    balance = select_latest_period(
        raw["balance-sheet-statement"], as_of_fiscal_year
    )
    cash_flow = select_latest_period(
        raw["cash-flow-statement"], as_of_fiscal_year
    )
    ratios = select_latest_period(raw["ratios"], as_of_fiscal_year)

    return merge_financials(ticker, income, balance, cash_flow, ratios)
