"""
Task 20 - shared interface contract: data schemas + agent I/O formats.

This module is the single source of truth for the *shapes* that flow
between every other part of the system. It is deliberately implemented
with typing.TypedDict so the contract is:

    - zero-runtime-cost and fully compatible with the plain dicts the
      data modules already return (no rewrite of Tasks 1-9 needed);
    - statically checkable in editors / type checkers;
    - introspectable at runtime via __required_keys__, which the
      `missing_keys` helper uses for lightweight validation in tests.

Three layers live here:

    1. Data records   - mirror the dicts returned by the Task 1-7 data
                        modules exactly (key names and types verified by
                        the tests against real module outputs).
    2. Agent I/O      - the reports / scores / predictions exchanged
                        between the Stage 1-4 agents (Tasks 10-17). This
                        is the layer that did not exist before and that
                        unblocks parallel agent development.
    3. Enums/helpers  - direction constants and a tiny validation util.

The LangGraph state that threads these together lives in state.py.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, NotRequired, TypedDict


# --------------------------------------------------------------------------
# Enums / constants
# --------------------------------------------------------------------------

# The Buy / Not-Buy labels. These MUST stay equal to evaluation/metrics.py's
# BUY / NOT_BUY constants - metrics.py predates this contract, so a test
# (test_contracts.py) asserts the two agree rather than refactoring the
# already-tested evaluation module.
BUY = "Buy"
NOT_BUY = "Not Buy"

Direction = Literal["Buy", "Not Buy"]

# Coarse subtask signal the Leader weighs. Sentiment may also be "mixed"
# because the spec requires it to report conflicting signals faithfully
# rather than resolving them.
Signal = Literal["bullish", "bearish", "neutral", "mixed"]

# Which ablation configuration produced a run (see evaluation README).
Variant = Literal["single", "paper", "full"]

RiskMethod = Literal["qualitative", "quantitative"]

Severity = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------
# Layer 1: data records (mirror Task 1-7 module outputs)
# --------------------------------------------------------------------------

class Filing(TypedDict):
    """Output of edgar_retrieval.fetch_10k (Task 1)."""

    ticker: str
    cik: str
    accession_number: str
    form: str
    filing_date: date
    filing_timestamp_et: datetime
    report_date: date
    primary_document: str
    primary_document_url: str
    html_path: str | None
    text_path: str | None
    text: str


class T0Window(TypedDict):
    """Output of t0_logic.compute_t0 (Task 2). The leakage anchor."""

    filing_timestamp_et: datetime
    t0_date: date
    cutoff_timestamp_et: datetime
    target_date: date


class TrendPoint(TypedDict):
    date: date
    close: float


class Prices(TypedDict):
    """Output of price_retrieval.fetch_prices (Task 3)."""

    ticker: str
    t0_date: date
    target_date: date
    baseline_price: float
    target_price: float          # answer key: never shown to an agent
    pre_release_trend: list[TrendPoint]


class Profitability(TypedDict):
    revenue: float | None
    net_income: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    return_on_equity: float | None
    return_on_assets: float | None


class CashFlow(TypedDict):
    operating_cash_flow: float | None
    capital_expenditure: float | None
    free_cash_flow: float | None


class Debt(TypedDict):
    total_debt: float | None
    total_equity: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    interest_coverage: float | None


class Valuation(TypedDict):
    pe_ratio: float | None
    pb_ratio: float | None
    price_to_sales: float | None
    ev_to_ebitda: float | None


class Financials(TypedDict):
    """Output of financial_retrieval.fetch_financials (Task 4)."""

    ticker: str
    fiscal_year: int | None
    report_date: str | None
    profitability: Profitability
    cash_flow: CashFlow
    debt: Debt
    valuation: Valuation


class NewsItem(TypedDict):
    """One item from finnhub_retrieval.fetch_company_news (Task 5)."""

    headline: str | None
    summary: str | None
    source: str | None
    url: str | None
    published_at_et: datetime
    published_unix: int


class RedditPost(TypedDict):
    """One item from reddit_retrieval.fetch_reddit_posts (Task 6)."""

    id: str | None
    title: str | None
    body: str | None
    subreddit: str | None
    score: int | None
    num_comments: int | None
    url: str | None
    permalink: str | None
    published_at_et: datetime
    published_unix: float | int


class RetrievedChunk(TypedDict):
    """One hit from rag_10k.retrieve (Task 7)."""

    chunk_id: int
    section_code: str
    section_title: str
    text: str
    char_start: int
    char_end: int
    similarity: float


# --------------------------------------------------------------------------
# Layer 2: agent I/O (Stage 1-4 agents, Tasks 10-17)
# --------------------------------------------------------------------------

class FundamentalReport(TypedDict):
    """
    Fundamental analyst output (Task 10, cooperative).

    `summary` is the free-text the Leader reads; `signal`/`confidence`
    give it a structured handle; `key_metrics` records the figures the
    agent actually cited; `citations` are RAG chunk ids / API field names.
    """

    ticker: str
    summary: str
    signal: Signal
    confidence: float            # 0..1
    key_metrics: dict[str, float | None]
    citations: list[str]


class SentimentReport(TypedDict):
    """
    Sentiment scout output (Task 11, cooperative).

    `disagreement` makes the spec's "report conflicting signals
    faithfully" requirement first-class instead of hidden in prose.
    """

    ticker: str
    summary: str
    signal: Signal
    confidence: float
    news_count: int
    social_count: int
    disagreement: bool
    citations: list[str]


class RiskScore(TypedDict):
    """
    One side of the competitive risk pair (Tasks 12/13).

    score is on a fixed 0-10 scale (10 = highest risk) so the two
    methods are comparable; `factors` are the phase-1 collected factors
    this score weighs; the pair is carried forward unaveraged.
    """

    method: RiskMethod
    score: float                 # 0..10, higher = riskier
    summary: str
    factors: list[str]
    justification: str


class RiskAssessment(TypedDict):
    """
    Output of the three-phase risk protocol (Task 14).

    `collected_factors` is the shared phase-1 cooperative enumeration;
    `scores` is the opposing phase-2 pair, deliberately NOT reduced to a
    single number.
    """

    collected_factors: list[str]
    scores: list[RiskScore]


class Prediction(TypedDict):
    """
    A buy/not-buy + target-price prediction with mandatory rationale.

    Produced by the Leader (Task 15) and emitted as the final output
    (Stage 4). `dominant_signal` and `risk_reconciliation` make the
    Leader's free-judgment weighting auditable, and are the surface the
    red team attacks.
    """

    direction: Direction
    target_price: float
    confidence: float
    rationale: str
    dominant_signal: str
    risk_reconciliation: str


class Rebuttal(TypedDict):
    """Red-team objection for one round (Task 16)."""

    round: int
    targeted_claim: str
    objections: list[str]
    severity: Severity


class LeaderResponse(TypedDict):
    """
    Leader's reply to a rebuttal (Task 17).

    `accepted=False` means the Leader held its position; `reason` then
    records the stated justification (the spec lets the Leader reject an
    unfounded rebuttal). `revised_prediction` is None when the position
    did not change.
    """

    round: int
    accepted: bool
    reason: str
    revised_prediction: NotRequired[Prediction | None]


# --------------------------------------------------------------------------
# Layer 3: validation helper
# --------------------------------------------------------------------------

def missing_keys(typed_dict_cls, value) -> set[str]:
    """
    Return the required keys of a TypedDict class that are absent from
    `value`. Empty set means the dict satisfies the contract's required
    fields. Used by tests to check real module outputs conform without
    pulling in a runtime validation dependency.
    """
    required = getattr(typed_dict_cls, "__required_keys__", frozenset())
    return set(required) - set(value.keys())
