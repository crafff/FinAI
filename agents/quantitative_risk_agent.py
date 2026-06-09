"""
Quantitative Risk Agent - Task 13.

The numbers-driven counterpart to the qualitative risk analyst (Task 12).
Where the qualitative agent reads the 10-K narrative (Item 1A and related
sections), this agent grounds its risk score in the structured financials
(Task 4) - leverage, liquidity, interest coverage, margin quality, cash
generation, valuation - and the pre-release price trend (volatility /
drawdown). The two opposing scores are carried forward UNAVERAGED by the
Task 14 three-phase protocol; the Leader reconciles them.

The agent uses NO tools: the financials and the price trend are handed to it
directly, so it reasons over given numbers exactly like the Leader does. It
emits a RiskScore with method="quantitative" (contracts/schemas.py).

Leakage: the financials are cut off at/before the fiscal year and the price
trend is strictly pre-T0, so all visible information predates the T0 cutoff.
The answer-key target price is never passed in here.
"""

from __future__ import annotations

import json

from llm_client import LLMClient
from schemas import RiskScore

# Reuse the qualitative agent's tolerant JSON extractor: both agents emit the
# same RiskScore shape, so there is a single extraction path.
from qualitative_risk_agent import extract_json_object


DEFAULT_SCORE = 5.0
METHOD = "quantitative"


SYSTEM_PROMPT = """\
You are a quantitative risk analyst. Your job is to assess one company's \
financial risk using only the numbers you are given: its structured \
financial metrics and its recent (pre-cutoff) closing-price trend.

Reason like a credit/markets analyst. Consider, where the data supports it:

  - Leverage & solvency: debt-to-equity, total debt vs. equity, interest \
coverage.
  - Liquidity: current ratio, free cash flow, operating cash flow vs. capex.
  - Profitability quality: margins (gross/operating/net), ROE, ROA, and \
whether earnings are cash-backed.
  - Valuation risk: P/E, P/B, price-to-sales, EV/EBITDA (rich multiples add \
downside risk).
  - Price behavior: the magnitude and direction of the pre-release trend \
(steep declines or high volatility raise risk).

Score risk on a fixed 0-10 scale where 10 = highest risk, so your score is \
directly comparable to the qualitative analyst's. Do NOT make a buy/not-buy \
recommendation and do NOT predict a target price. Cite the specific metrics \
that drove your score in the justification; do not invent figures that are \
not provided. If a metric is missing or null, say so and reason from what is \
available rather than guessing.

When complete, output ONLY a single JSON object with exactly these fields:

{
  "method": "quantitative",
  "score": <number from 0 to 10, where 10 = highest risk>,
  "summary": "<2-4 sentence quantitative risk summary>",
  "factors": ["<metric-grounded risk factor>", "..."],
  "justification": "<which metrics drove this score and why>"
}
"""


def build_user_prompt(
    ticker: str,
    financials: dict,
    price_trend: list | None = None,
    shared_factors: list[str] | None = None,
) -> str:
    """
    Compose the user message: the structured financials and (optionally) the
    pre-release price trend, plus the task. Degrades gracefully when the
    financials are empty (e.g. an FMP 402 under allow_missing).
    """
    financials = financials or {}
    financials_json = json.dumps(financials, indent=2, default=str)

    if financials:
        data_note = (
            f"Structured financials (already cut off at the fiscal year, "
            f"point-in-time):\n{financials_json}\n\n"
        )
    else:
        data_note = (
            "No structured financials were available for this company. Base "
            "your assessment on the price trend and the shared factors, and "
            "reflect the added uncertainty in a higher score.\n\n"
        )

    trend_note = ""
    if price_trend:
        trend_json = json.dumps(price_trend, default=str)
        trend_note = (
            f"Pre-release closing-price trend (oldest first, strictly before "
            f"the T0 cutoff):\n{trend_json}\n\n"
        )

    shared_note = ""
    if shared_factors:
        rendered = "\n".join(f"  - {f}" for f in shared_factors)
        shared_note = (
            f"A cooperative factor-collection phase identified these shared "
            f"risk factors. Weigh each from the quantitative (financial) angle "
            f"when you score:\n{rendered}\n\n"
        )

    return (
        f"Assess the quantitative financial risk of {ticker.upper()}.\n\n"
        f"{data_note}"
        f"{trend_note}"
        f"{shared_note}"
        f"Return the required RiskScore JSON only."
    )


def parse_risk_score(text: str) -> RiskScore:
    """
    Parse and normalize the model output into a quantitative RiskScore.

    Mirrors the qualitative parser's tolerance (clamps the 0-10 score, coerces
    a bad score to the default, defaults a non-list `factors`) but pins
    `method` to "quantitative" regardless of what the model wrote.
    """
    data = extract_json_object(text)

    try:
        score = float(data.get("score", DEFAULT_SCORE))
    except (TypeError, ValueError):
        score = DEFAULT_SCORE

    score = min(max(score, 0.0), 10.0)

    factors = data.get("factors")
    if not isinstance(factors, list):
        factors = []

    return RiskScore(
        method=METHOD,
        score=score,
        summary=str(data.get("summary", "")),
        factors=[str(f) for f in factors],
        justification=str(data.get("justification", "")),
    )


def run_quantitative_risk_agent(
    ticker: str,
    financials: dict,
    client: LLMClient,
    price_trend: list | None = None,
    shared_factors: list[str] | None = None,
) -> RiskScore:
    """
    Run the quantitative risk agent end to end (a single, tool-free call).

    Inputs:
        ticker:         stock ticker.
        financials:     structured financials dict (Task 4 output); may be
                        empty under allow_missing.
        client:         an LLMClient (Anthropic or local).
        price_trend:    optional pre-release {date, close} list (volatility).
        shared_factors: optional cooperative Phase-1 factor list (Task 14).

    Returns a method="quantitative" RiskScore.
    """
    messages = [{
        "role": "user",
        "text": build_user_prompt(ticker, financials, price_trend, shared_factors),
    }]

    response = client.complete(messages, tools=None, system=SYSTEM_PROMPT)

    return parse_risk_score(response.text)
