import types

import pytest

import context


def _stub(monkeypatch, *, fin_raises=False, news_raises=False, social_raises=False):
    """Replace every network/disk boundary in build_data_context with fakes."""
    monkeypatch.setattr(
        context, "_find_cached_filing",
        lambda t: ({"accession_number": "acc", "filing_timestamp_et": "x"}, "10-K text"),
    )
    monkeypatch.setattr(context, "parse_acceptance_datetime", lambda s: s)
    monkeypatch.setattr(context, "compute_t0", lambda dt: {
        "t0_date": "2025-11-03", "target_date": "2025-11-10",
        "cutoff_timestamp_et": "2025-11-03T16:00:00-05:00",
    })
    monkeypatch.setattr(context, "build_or_load_index", lambda *a, **k: {"idx": True})
    monkeypatch.setattr(
        context, "make_retrieval_tool",
        lambda idx: (lambda q, k=5, section=None: "[chunk]"),
    )
    monkeypatch.setattr(
        context, "fetch_prices",
        lambda *a, **k: {"baseline_price": 180.0, "target_price": 200.0},
    )

    def fin(*a, **k):
        if fin_raises:
            raise RuntimeError("402 Payment Required")
        return {"ticker": "AAPL", "fiscal_year": 2025}

    def news(*a, **k):
        if news_raises:
            raise RuntimeError("finnhub down")
        return [{"headline": "h"}]

    def social(*a, **k):
        if social_raises:
            raise RuntimeError("reddit down")
        return [{"title": "t"}]

    monkeypatch.setattr(context, "fetch_financials", fin)
    monkeypatch.setattr(context, "fetch_company_news", news)
    monkeypatch.setattr(context, "fetch_reddit_posts", social)


def _settings():
    return types.SimpleNamespace(
        require_fmp_api_key=lambda: "k",
        require_finnhub_api_key=lambda: "k",
        reddit_client_id=None,
        reddit_client_secret=None,
        reddit_user_agent=None,
    )


def test_allow_missing_degrades_failed_sources(monkeypatch):
    _stub(monkeypatch, fin_raises=True, news_raises=True, social_raises=True)

    ctx = context.build_data_context("AAPL", _settings(), allow_missing=True)

    assert ctx.financials == {}
    assert ctx.news == []
    assert ctx.social == []
    assert set(ctx.missing) == {"financials", "news", "social"}
    assert ctx.baseline_price == 180.0          # prices still required + present


def test_without_allow_missing_failure_propagates(monkeypatch):
    _stub(monkeypatch, fin_raises=True)

    with pytest.raises(RuntimeError):
        context.build_data_context("AAPL", _settings(), allow_missing=False)


def test_allow_missing_keeps_successful_sources(monkeypatch):
    _stub(monkeypatch)  # nothing raises

    ctx = context.build_data_context("AAPL", _settings(), allow_missing=True)

    assert ctx.financials["fiscal_year"] == 2025
    assert ctx.news and ctx.social
    assert ctx.missing == []


def test_missing_10k_always_raises_even_with_allow_missing(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setattr(context, "_find_cached_filing", lambda t: None)

    with pytest.raises(FileNotFoundError):
        context.build_data_context("AAPL", _settings(), allow_missing=True)


def test_context_summary_reports_missing(monkeypatch):
    _stub(monkeypatch, fin_raises=True)

    ctx = context.build_data_context("AAPL", _settings(), allow_missing=True)
    summary = context.context_summary(ctx)

    assert summary["has_financials"] is False
    assert "financials" in summary["missing"]
    assert summary["target_price"] == 200.0
