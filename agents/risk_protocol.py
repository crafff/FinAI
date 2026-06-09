"""
Three-phase risk protocol - Task 14.

Combines the two opposing risk analysts into the RiskAssessment contract
(contracts/schemas.py). The spec's risk subtask is deliberately a coopetition:
the analysts cooperate to agree on WHAT to worry about, then compete on HOW
risky it is.

  Phase 1 - cooperate:  a shared enumeration of the company's material risk
            factors (`collected_factors`), drawing on both the 10-K narrative
            and the financials, so both analysts score the same agenda.
  Phase 2 - compete:    the qualitative analyst (Task 12, 10-K narrative) and
            the quantitative analyst (Task 13, financials + price trend) each
            score risk 0-10 while weighing those shared factors.
  Phase 3 - carry both forward UNAVERAGED: the RiskAssessment holds the shared
            factors plus the method-tagged pair, NOT reduced to one number -
            the Leader (Task 15) reconciles the disagreement explicitly.

This is the real risk subtask that supersedes the interim qualitative-only
stand-in (`leader_agent.risk_assessment_from_score`): once wired in, the Leader
and red team receive both opposing scores with no change to their code (they
already render `scores` as a list).
"""

from __future__ import annotations

import json

from llm_client import LLMClient, run_tool_loop
from schemas import RiskAssessment

from qualitative_risk_agent import (
    build_search_tool,
    extract_json_object,
    run_qualitative_risk_agent,
)
from quantitative_risk_agent import run_quantitative_risk_agent


FACTORS_SYSTEM_PROMPT = """\
You are facilitating the cooperative first phase of a company risk review. Two \
analysts - one reading the 10-K narrative, one reading the financial metrics - \
need a single shared list of the company's material risk factors to assess. \
Your job is ONLY to enumerate that agenda, not to score it.

Use the `search_10k` tool to ground the qualitative side (start with Item 1A \
Risk Factors, then litigation, competition, regulation, cybersecurity, supply \
chain, operational risks), and incorporate the quantitative side from the \
financial metrics you are given (leverage, liquidity, coverage, margins, \
valuation). Merge both perspectives into one deduplicated list of concise, \
distinct factors. Do NOT assign scores and do NOT make a recommendation.

When complete, output ONLY a single JSON object:

{
  "factors": ["<concise risk factor>", "<another>", "..."]
}
"""


def build_factors_user_prompt(ticker: str, financials: dict) -> str:
    financials = financials or {}
    financials_json = json.dumps(financials, indent=2, default=str)

    fin_note = (
        f"Structured financial metrics (point-in-time, pre-cutoff):\n"
        f"{financials_json}\n\n"
        if financials
        else "No structured financials were available; rely on the 10-K for "
        "factor collection and note the missing financial coverage.\n\n"
    )

    return (
        f"Enumerate the shared material risk factors for {ticker.upper()}.\n\n"
        f"{fin_note}"
        f"Search the 10-K for narrative risks and combine them with the "
        f"financial picture. Return the JSON list of factors only."
    )


def parse_collected_factors(text: str) -> list[str]:
    """Pull the deduplicated factor list out of the Phase-1 output."""
    data = extract_json_object(text)

    factors = data.get("factors")
    if not isinstance(factors, list):
        return []

    seen: set[str] = set()
    out: list[str] = []
    for f in factors:
        s = str(f).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)

    return out


def collect_risk_factors(
    ticker: str,
    retrieval_tool,
    financials: dict,
    client: LLMClient,
    max_iterations: int = 8,
) -> list[str]:
    """
    Phase 1 (cooperative): produce the shared `collected_factors` agenda by
    merging 10-K narrative risks (via the search tool) with the financial
    picture into one deduplicated list.
    """
    tools = [build_search_tool(retrieval_tool)]
    messages = [{
        "role": "user",
        "text": build_factors_user_prompt(ticker, financials),
    }]

    response, _ = run_tool_loop(
        client,
        messages,
        tools,
        system=FACTORS_SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

    return parse_collected_factors(response.text)


def run_risk_protocol(
    ticker: str,
    retrieval_tool,
    financials: dict,
    client: LLMClient,
    price_trend: list | None = None,
    max_iterations: int = 8,
) -> RiskAssessment:
    """
    Run the full three-phase risk protocol and return a RiskAssessment.

    Phase 1 collects the shared factors; Phase 2 runs the qualitative and
    quantitative analysts over those factors; Phase 3 returns both scores
    unaveraged. The qualitative analyst's own collected factors backstop the
    shared list if Phase 1 came back empty, so the assessment is never
    factorless.
    """
    collected_factors = collect_risk_factors(
        ticker, retrieval_tool, financials, client,
        max_iterations=max_iterations,
    )

    qualitative = run_qualitative_risk_agent(
        ticker=ticker,
        retrieval_tool=retrieval_tool,
        client=client,
        max_iterations=max_iterations,
        shared_factors=collected_factors or None,
    )

    quantitative = run_quantitative_risk_agent(
        ticker=ticker,
        financials=financials,
        client=client,
        price_trend=price_trend,
        shared_factors=collected_factors or None,
    )

    if not collected_factors:
        # Backstop: never emit a factorless assessment. Merge whatever each
        # analyst surfaced on its own.
        merged: list[str] = []
        seen: set[str] = set()
        for f in (qualitative.get("factors") or []) + (quantitative.get("factors") or []):
            key = str(f).strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(str(f).strip())
        collected_factors = merged

    return RiskAssessment(
        collected_factors=collected_factors,
        scores=[qualitative, quantitative],
    )
