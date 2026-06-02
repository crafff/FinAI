"""
Per-ticker shared data context.

`build_data_context` loads everything the agents need for one ticker - the
cached 10-K, its RAG index, financials, cutoff-safe news + social, and prices
- exactly once. The same DataContext is reused across every system in an
experiment, so the single-agent baseline and the full pipeline see identical
inputs, and the data caches are hit once per ticker rather than once per
system.

This consolidates the data-loading block that the three standalone runners
used to duplicate. All cache locations come from the data modules'
DEFAULT_*_CACHE constants (now centralized under .cache/), so there is one
source of truth for where things are stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag_10k import DEFAULT_INDEX_CACHE, build_or_load_index, make_retrieval_tool
from financial_retrieval import DEFAULT_FINANCIALS_CACHE, fetch_financials
from finnhub_retrieval import DEFAULT_NEWS_CACHE, fetch_company_news
from reddit_retrieval import DEFAULT_POSTS_CACHE, fetch_reddit_posts
from price_retrieval import DEFAULT_PRICES_CACHE, fetch_prices
from t0_logic import compute_t0
from edgar_retrieval import DEFAULT_FILING_CACHE, parse_acceptance_datetime


@dataclass
class DataContext:
    ticker: str
    accession: str
    t0: dict
    cutoff_timestamp: Any
    retrieval_tool: Callable
    financials: dict
    news: list
    social: list
    prices: dict
    baseline_price: float
    missing: list = None  # data sources that were unavailable (allow_missing)


def _find_cached_filing(ticker: str):
    """Return (meta, text) for the latest cached 10-K, or None."""
    ticker_dir = Path(DEFAULT_FILING_CACHE) / ticker.upper()
    metas = sorted(ticker_dir.glob("*.meta.json")) if ticker_dir.is_dir() else []

    if not metas:
        return None

    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    text_path = ticker_dir / f"{meta['accession_number']}.txt"

    if not text_path.exists():
        return None

    return meta, text_path.read_text(encoding="utf-8")


def build_data_context(ticker: str, settings, allow_missing: bool = False) -> DataContext:
    """
    Load all shared inputs for one ticker.

    The 10-K and prices are always required: the 10-K is the root of the RAG
    index every agent searches, and prices give the baseline anchor + the
    answer key. A missing 10-K raises FileNotFoundError (run
    data/EDGAR_retrieval/run_fetch.py first).

    When `allow_missing` is True, the *optional* sources - financials
    (FMP), news (FinnHub), and social (Reddit) - degrade to empty instead of
    aborting the ticker: a fetch that fails (e.g. an FMP 402, a quota error,
    a missing key) is caught, recorded in `DataContext.missing`, and the
    pipeline continues with `{}` / `[]`. When False (default), any such
    failure propagates as before.

    Reddit credentials are optional regardless: without them the social fetch
    uses the no-auth JSON backend. The answer-key `prices["target_price"]` is
    carried only in the returned context (and saved artifacts) - it is never
    passed to an agent.
    """
    ticker = ticker.upper()
    missing: list[str] = []

    def _optional(label, fetch, empty):
        """Run a fetch; under allow_missing, degrade to `empty` on failure."""
        if not allow_missing:
            return fetch()
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 - degrade, record, continue
            missing.append(label)
            print(
                f"[{ticker}] {label} unavailable "
                f"({type(exc).__name__}: {exc}); continuing with none.",
                flush=True,
            )
            return empty

    filing = _find_cached_filing(ticker)
    if filing is None:
        raise FileNotFoundError(
            f"No cached 10-K for {ticker} under {DEFAULT_FILING_CACHE}. "
            f"Run data/EDGAR_retrieval/run_fetch.py first."
        )

    meta, text = filing
    accession = meta["accession_number"]

    t0 = compute_t0(parse_acceptance_datetime(meta["filing_timestamp_et"]))
    cutoff_timestamp = t0["cutoff_timestamp_et"]

    index = build_or_load_index(ticker, accession, text, cache_dir=DEFAULT_INDEX_CACHE)
    retrieval_tool = make_retrieval_tool(index)

    financials = _optional(
        "financials",
        lambda: fetch_financials(
            ticker, settings.require_fmp_api_key(),
            cache_dir=DEFAULT_FINANCIALS_CACHE,
        ),
        {},
    )

    news = _optional(
        "news",
        lambda: fetch_company_news(
            ticker=ticker,
            cutoff_timestamp=cutoff_timestamp,
            api_key=settings.require_finnhub_api_key(),
            cache_dir=DEFAULT_NEWS_CACHE,
        ),
        [],
    )

    social = _optional(
        "social",
        lambda: fetch_reddit_posts(
            ticker=ticker,
            cutoff_timestamp=cutoff_timestamp,
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
            cache_dir=DEFAULT_POSTS_CACHE,
        ),
        [],
    )

    # Prices are required (baseline anchor + answer key), even under
    # allow_missing.
    prices = fetch_prices(
        ticker, t0["t0_date"], t0["target_date"], cache_dir=DEFAULT_PRICES_CACHE
    )

    return DataContext(
        ticker=ticker,
        accession=accession,
        t0=t0,
        cutoff_timestamp=cutoff_timestamp,
        retrieval_tool=retrieval_tool,
        financials=financials,
        news=news,
        social=social,
        prices=prices,
        baseline_price=prices["baseline_price"],
        missing=missing,
    )


def context_summary(ctx: DataContext) -> dict:
    """A JSON-able audit snapshot of the shared inputs for one ticker."""
    return {
        "ticker": ctx.ticker,
        "accession": ctx.accession,
        "t0_date": str(ctx.t0.get("t0_date")),
        "target_date": str(ctx.t0.get("target_date")),
        "cutoff_timestamp": str(ctx.cutoff_timestamp),
        "baseline_price": ctx.baseline_price,
        # answer key, persisted for evaluation only - never shown to an agent:
        "target_price": ctx.prices.get("target_price"),
        "news_count": len(ctx.news),
        "social_count": len(ctx.social),
        "has_financials": bool(ctx.financials),
        "missing": ctx.missing or [],
    }
