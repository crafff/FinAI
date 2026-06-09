"""
Sub-task agent registry for the experiment harness.

Each registered SubtaskSpec knows two things: how to *run* the sub-task agent
against a DataContext (producing its report) and how to *render* that report
into the compact evidence the Leader / red-team consume. The Leader and
red-team are generic over a name->rendered-evidence map, so an experiment can
select any subset / number of sub-task agents by name; a new agent drops in by
registering one SubtaskSpec here - with no changes to the Leader, red-team,
loop, or harness.

Risk subtasks (all render to the same RiskAssessment evidence):
    - "risk"             - the full three-phase protocol (Task 14): cooperative
                           factor collection, then the qualitative (Task 12)
                           and quantitative (Task 13) analysts scoring in
                           competition. This is the real risk subtask.
    - "qualitative_risk" - the 10-K-narrative analyst alone (Task 12), kept for
                           ablations and as the interim single-score stand-in.
    - "quantitative_risk"- the financials analyst alone (Task 13), for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from leader_agent import (
    build_fundamental_evidence,
    build_risk_evidence,
    build_sentiment_evidence,
    risk_assessment_from_score,
)
from fundamental_agent import run_fundamental_agent
from sentiment_agent import run_sentiment_agent
from qualitative_risk_agent import run_qualitative_risk_agent
from quantitative_risk_agent import run_quantitative_risk_agent
from risk_protocol import run_risk_protocol


@dataclass(frozen=True)
class SubtaskSpec:
    """One pluggable sub-task agent."""

    name: str
    run: Callable          # (ctx: DataContext, client: LLMClient) -> report dict
    render: Callable       # (report dict) -> rendered evidence dict


def _run_fundamental(ctx, client):
    return run_fundamental_agent(
        ticker=ctx.ticker,
        financials=ctx.financials,
        retrieval_tool=ctx.retrieval_tool,
        client=client,
    )


def _run_sentiment(ctx, client):
    return run_sentiment_agent(
        ticker=ctx.ticker,
        news=ctx.news,
        social=ctx.social,
        client=client,
    )


def _price_trend(ctx):
    """The pre-release {date, close} trend, if prices were loaded (no
    answer-key leakage: only the trend is read, never target_price)."""
    return (getattr(ctx, "prices", None) or {}).get("pre_release_trend")


def _run_risk(ctx, client):
    # The full three-phase protocol (Task 14): returns a RiskAssessment with
    # both the qualitative and quantitative scores, unaveraged.
    return run_risk_protocol(
        ticker=ctx.ticker,
        retrieval_tool=ctx.retrieval_tool,
        financials=ctx.financials,
        client=client,
        price_trend=_price_trend(ctx),
    )


def _run_qualitative_risk(ctx, client):
    # The qualitative agent returns a single RiskScore; wrap it into a
    # RiskAssessment so the report is the same shape Task 14 produces (render
    # handles 1 or 2 scores identically).
    score = run_qualitative_risk_agent(
        ticker=ctx.ticker,
        retrieval_tool=ctx.retrieval_tool,
        client=client,
    )
    return risk_assessment_from_score(score)


def _run_quantitative_risk(ctx, client):
    # The quantitative agent returns a single RiskScore; wrap it the same way.
    score = run_quantitative_risk_agent(
        ticker=ctx.ticker,
        financials=ctx.financials,
        client=client,
        price_trend=_price_trend(ctx),
    )
    return risk_assessment_from_score(score)


REGISTRY: dict[str, SubtaskSpec] = {
    "fundamental": SubtaskSpec(
        "fundamental", _run_fundamental, build_fundamental_evidence
    ),
    "sentiment": SubtaskSpec(
        "sentiment", _run_sentiment, build_sentiment_evidence
    ),
    "risk": SubtaskSpec(
        "risk", _run_risk, build_risk_evidence
    ),
    "qualitative_risk": SubtaskSpec(
        "qualitative_risk", _run_qualitative_risk, build_risk_evidence
    ),
    "quantitative_risk": SubtaskSpec(
        "quantitative_risk", _run_quantitative_risk, build_risk_evidence
    ),
}


def available_subtasks() -> list[str]:
    """The registered sub-task agent names (for config validation / help)."""
    return list(REGISTRY)
