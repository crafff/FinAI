"""
Sub-task agent registry for the experiment harness.

Each registered SubtaskSpec knows two things: how to *run* the sub-task agent
against a DataContext (producing its report) and how to *render* that report
into the compact evidence the Leader / red-team consume. The Leader and
red-team are generic over a name->rendered-evidence map, so an experiment can
select any subset / number of sub-task agents by name, and a future agent
(e.g. the quantitative risk analyst, Task 13) drops in by registering one
SubtaskSpec here - with no changes to the Leader, red-team, loop, or harness.
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


def _run_qualitative_risk(ctx, client):
    # The qualitative agent returns a single RiskScore; wrap it into a
    # RiskAssessment so the report is the same shape Task 14 will eventually
    # produce (render handles 1 or 2 scores identically).
    score = run_qualitative_risk_agent(
        ticker=ctx.ticker,
        retrieval_tool=ctx.retrieval_tool,
        client=client,
    )
    return risk_assessment_from_score(score)


REGISTRY: dict[str, SubtaskSpec] = {
    "fundamental": SubtaskSpec(
        "fundamental", _run_fundamental, build_fundamental_evidence
    ),
    "sentiment": SubtaskSpec(
        "sentiment", _run_sentiment, build_sentiment_evidence
    ),
    "qualitative_risk": SubtaskSpec(
        "qualitative_risk", _run_qualitative_risk, build_risk_evidence
    ),
}


def available_subtasks() -> list[str]:
    """The registered sub-task agent names (for config validation / help)."""
    return list(REGISTRY)
