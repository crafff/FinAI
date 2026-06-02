"""
Single-agent baseline (Task 19, ablation variant "single").

The simplest of the three ablation configurations: ONE LLM agent sees all
the evidence at once - the 10-K (via the RAG search tool), the structured
financials, the pre-cutoff news and social posts, and the T0 baseline
price - and directly emits the final buy/not-buy + one-week target-price
Prediction (contracts/schemas.py).

There is no subtask decomposition, no cooperative/competitive split, no
leader aggregation, and no red-team loop. That is the point: this baseline
isolates how much the full coopetition pipeline (variant "full") and the
paper-style aggregation (variant "paper") add over a single capable agent
given the same inputs.

Leakage: the agent fetches no live data. The 10-K, financials, news, and
social posts are all cut off at/before the T0 close by the data layer, and
the only price it is shown is the T0 baseline close (the prediction anchor,
which is allowed). The actual target-date close is the answer key and is
never passed in here.
"""

from __future__ import annotations

import json

from llm_client import LLMClient, run_tool_loop
from schemas import BUY, NOT_BUY, Prediction

# Reuse the building blocks the subtask agents already define so the single
# agent sees evidence in exactly the same shape the full pipeline does.
from fundamental_agent import build_search_tool, extract_json_object
from sentiment_agent import build_news_evidence, build_social_evidence


DEFAULT_CONFIDENCE = 0.5
DEFAULT_DIRECTION = NOT_BUY


SYSTEM_PROMPT = """\
You are a senior equity analyst making a self-contained investment call on \
one company, working only from the company's latest 10-K annual report and \
the supporting evidence provided to you.

You have everything you need in one place:
  - the structured financials (already point-in-time at the fiscal year),
  - pre-cutoff company news and Reddit/social posts,
  - the `search_10k` tool for semantic search over the full 10-K (business \
and competition in Item 1, risk factors in Item 1A, MD&A in Item 7),
  - the T0 baseline price (the most recent close), which anchors your \
target-price forecast.

All information is as of the T0 market close; do not assume any knowledge of \
events after it. Investigate the 10-K with `search_10k` as much as you need, \
weigh the fundamentals, sentiment, and risks yourself, and commit to a single \
judgment. Ground every figure in what you actually retrieve or are given; do \
not invent numbers.

Predict the stock's direction over the one trading week following T0 and the \
expected closing price on the 5th trading day after T0.

When your analysis is complete, output ONLY a single JSON object (no prose \
before or after it) with exactly these fields:

  {
    "direction": "Buy" | "Not Buy",
    "target_price": <number: expected close on the 5th trading day after T0>,
    "confidence": <number between 0 and 1>,
    "rationale": "<2-5 sentence justification for the call>",
    "dominant_signal": "<the single factor that drove the decision>",
    "risk_reconciliation": "<how you weighed the main risks against the thesis>"
  }
"""


def build_user_prompt(
    ticker: str,
    financials: dict,
    news: list[dict],
    social: list[dict],
    baseline_price: float | None,
) -> str:
    """
    Compose the initial user message: all evidence in one payload, plus the
    task. Financials are injected verbatim; news/social are compacted with
    the same evidence builders the sentiment agent uses (stable ids like
    "news_1" / "social_1" so the agent can cite them).
    """
    payload = {
        "ticker": ticker.upper(),
        "baseline_price": baseline_price,
        "financials": financials,
        "news_count": len(news),
        "social_count": len(social),
        "news_evidence": build_news_evidence(news),
        "social_evidence": build_social_evidence(social),
    }

    baseline_note = (
        f"The T0 baseline close is {baseline_price}. Anchor your target price "
        f"to it.\n\n"
        if baseline_price is not None
        else "No baseline price was provided; estimate the target price from "
        "the evidence.\n\n"
    )

    return (
        f"Make a one-week investment call on {ticker.upper()}.\n\n"
        f"{baseline_note}"
        f"Evidence JSON:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Investigate the 10-K with search_10k if useful, then produce the "
        f"JSON Prediction."
    )


def normalize_direction(value, target_price, baseline_price) -> str:
    """
    Coerce the model's direction to the BUY / NOT_BUY constants.

    Order matters: "Not Buy" contains "buy", so the not/sell check runs
    first. If the field is unusable, infer the direction from the predicted
    target vs. the baseline (the same rule the evaluation uses to derive the
    actual label), so a malformed direction can never contradict the price.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if "not" in v or "sell" in v or v == "no":
            return NOT_BUY
        if "buy" in v or v == "yes":
            return BUY

    if target_price is not None and baseline_price is not None:
        return BUY if target_price > baseline_price else NOT_BUY

    return DEFAULT_DIRECTION


def parse_prediction(
    text: str,
    ticker: str,
    baseline_price: float | None = None,
) -> Prediction:
    """
    Parse and normalize the model's final JSON into a Prediction.

    Lenient on formatting: coerces direction to a valid constant, clamps
    confidence to [0, 1], and falls back to the baseline price when the
    target price is missing or unparseable, so a slightly-off response still
    yields a contract-valid Prediction.
    """
    data = extract_json_object(text)

    try:
        target_price = float(data.get("target_price"))
    except (TypeError, ValueError):
        target_price = baseline_price

    direction = normalize_direction(
        data.get("direction"), target_price, baseline_price
    )

    try:
        confidence = float(data.get("confidence", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    confidence = min(max(confidence, 0.0), 1.0)

    return Prediction(
        direction=direction,
        target_price=target_price,
        confidence=confidence,
        rationale=str(data.get("rationale", "")),
        dominant_signal=str(data.get("dominant_signal", "")),
        risk_reconciliation=str(data.get("risk_reconciliation", "")),
    )


def run_single_agent(
    ticker: str,
    financials: dict,
    news: list[dict],
    social: list[dict],
    retrieval_tool,
    client: LLMClient,
    baseline_price: float | None = None,
    max_iterations: int = 8,
) -> Prediction:
    """
    Run the single-agent baseline end to end.

    Inputs:
        ticker:         stock ticker.
        financials:     structured financials dict (Task 4 output).
        news:           cutoff-safe FinnHub NewsItem records (Task 5).
        social:         cutoff-safe RedditPost records (Task 6).
        retrieval_tool: rag_10k retrieval callable bound to this ticker's
                        10-K index (Task 7, make_retrieval_tool).
        client:         an LLMClient (Anthropic or local).
        baseline_price: the T0 close, used to anchor the target price.
        max_iterations: tool-loop cap.

    Returns a Prediction (the run's final_prediction for variant "single").
    """
    tools = [build_search_tool(retrieval_tool)]
    messages = [{
        "role": "user",
        "text": build_user_prompt(
            ticker, financials, news, social, baseline_price
        ),
    }]

    response, _ = run_tool_loop(
        client, messages, tools,
        system=SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

    return parse_prediction(response.text, ticker, baseline_price)
