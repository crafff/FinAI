"""
Leader aggregation agent (Task 15, Stage 2).

The Leader reads the three Stage-1 subtask outputs - the FundamentalReport
(Task 10), the SentimentReport (Task 11), and the RiskAssessment (Task 14) -
and makes a FREE-JUDGMENT initial prediction: a buy/not-buy call plus a
one-week target price, with a mandatory rationale. There is no fixed
weighting formula; the Leader weighs the evidence itself and must justify
how it did so. Its output is a Prediction (contracts/schemas.py), written to
`leader_prediction` in the pipeline state and later attacked by the red team
(Tasks 16/17).

The Leader does NO data retrieval and uses NO tools: the subtask agents
already searched the 10-K, news, and social posts. It reasons only over the
reports they produced, so every figure it cites must come from them.

Risk input: the Leader is built against the *final* RiskAssessment contract
(`collected_factors` + a list of method-tagged RiskScores), not a bare
RiskScore. The risk evidence renders `scores` as a list, so it handles both
the eventual qualitative+quantitative pair (Task 14) and - via
`risk_assessment_from_score` below - the interim qualitative-only stand-in,
with no change to the prompt or parser.

Leakage: the only price the Leader sees is the T0 baseline close (the
prediction anchor, which is allowed). The actual target-date close is the
answer key and is never passed in here.
"""

from __future__ import annotations

import json

from llm_client import LLMClient
from schemas import LeaderResponse, Prediction, RiskAssessment, RiskScore

# Reuse the single agent's Prediction parser: the Leader emits the exact same
# six-field Prediction, so there is one normalization path for both.
from single_agent import extract_json_object, parse_prediction


SYSTEM_PROMPT = """\
You are the lead portfolio manager making the firm's initial investment call \
on one company. Three analysts have already done the legwork and handed you \
their reports:

  - a Fundamental analyst (financials + 10-K business/MD&A),
  - a Sentiment scout (pre-cutoff news and social media),
  - a Risk assessment (collected risk factors plus one or more method-tagged \
risk scores on a 0-10 scale, where 10 = highest risk).

Exercise free judgment. There is no fixed formula: weigh the fundamentals, \
the sentiment, and the risk yourself, decide which signal dominates, and \
commit to a single call. When the analysts disagree, or when the sentiment \
report flags disagreement, resolve it explicitly rather than averaging it \
away. Ground every figure in what the reports actually say; do not invent \
numbers and do not assume any knowledge of events after the T0 market close.

The T0 baseline price (the most recent close) anchors your target-price \
forecast. Predict the stock's direction over the one trading week following \
T0 and the expected closing price on the 5th trading day after T0.

Your rationale is mandatory and will be scrutinized by a red-team reviewer, \
so make your reasoning explicit and defensible.

Output ONLY a single JSON object (no prose before or after it) with exactly \
these fields:

  {
    "direction": "Buy" | "Not Buy",
    "target_price": <number: expected close on the 5th trading day after T0>,
    "confidence": <number between 0 and 1>,
    "rationale": "<2-5 sentence justification for the call>",
    "dominant_signal": "<the single factor that drove the decision>",
    "risk_reconciliation": "<how you weighed the risk score(s) against the thesis>"
  }
"""


def build_fundamental_evidence(report: dict) -> dict:
    """
    Compact a FundamentalReport to the fields the Leader weighs.
    """
    return {
        "summary": report.get("summary"),
        "signal": report.get("signal"),
        "confidence": report.get("confidence"),
        "key_metrics": report.get("key_metrics"),
        "citations": report.get("citations"),
    }


def build_sentiment_evidence(report: dict) -> dict:
    """
    Compact a SentimentReport to the fields the Leader weighs. `disagreement`
    is kept first-class so the Leader is forced to resolve conflicts.
    """
    return {
        "summary": report.get("summary"),
        "signal": report.get("signal"),
        "confidence": report.get("confidence"),
        "news_count": report.get("news_count"),
        "social_count": report.get("social_count"),
        "disagreement": report.get("disagreement"),
        "citations": report.get("citations"),
    }


def build_risk_evidence(risk_assessment: dict) -> dict:
    """
    Render a RiskAssessment for the Leader: the shared collected factors plus
    every method-tagged score as a list.

    Rendering `scores` as a list is what lets the Leader consume the interim
    qualitative-only assessment (one score) and the eventual
    qualitative+quantitative pair (two scores) with no behavioral change.
    """
    scores = []

    for score in risk_assessment.get("scores", []) or []:
        scores.append({
            "method": score.get("method"),
            "score": score.get("score"),
            "summary": score.get("summary"),
            "factors": score.get("factors"),
            "justification": score.get("justification"),
        })

    return {
        "collected_factors": risk_assessment.get("collected_factors"),
        "scores": scores,
    }


def build_user_prompt(
    ticker: str,
    reports: dict[str, dict],
    baseline_price: float | None,
) -> str:
    """
    Compose the initial user message: the subtask reports (an arbitrary
    name->rendered-evidence map) in one payload, plus the baseline anchor and
    the task.

    `reports` is already rendered by the caller (the experiment registry uses
    the `build_*_evidence` renderers below), so the Leader is agnostic to
    which or how many subtask agents ran.
    """
    payload = {
        "ticker": ticker.upper(),
        "baseline_price": baseline_price,
        "subtask_reports": reports,
    }

    baseline_note = (
        f"The T0 baseline close is {baseline_price}. Anchor your target price "
        f"to it.\n\n"
        if baseline_price is not None
        else "No baseline price was provided; estimate the target price from "
        "the reports.\n\n"
    )

    return (
        f"Make the firm's one-week investment call on {ticker.upper()} by "
        f"aggregating your analysts' reports.\n\n"
        f"{baseline_note}"
        f"Analyst reports JSON:\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Weigh the reports and produce the JSON Prediction."
    )


def risk_assessment_from_score(score: RiskScore) -> RiskAssessment:
    """
    Wrap a single RiskScore into a RiskAssessment - the interim
    qualitative-only stand-in used until Task 14's three-phase protocol
    exists.

    The Leader is agnostic to how many scores it receives, so when Task 14
    lands the runner simply passes the real (two-score) RiskAssessment
    instead of calling this adapter; nothing in this module changes.
    """
    factors = score.get("factors") or []

    return RiskAssessment(
        collected_factors=[str(f) for f in factors],
        scores=[score],
    )


def run_leader_agent(
    ticker: str,
    reports: dict[str, dict],
    client: LLMClient,
    baseline_price: float | None = None,
) -> Prediction:
    """
    Run the Leader aggregation agent end to end.

    Inputs:
        ticker:         stock ticker.
        reports:        a name->rendered-evidence map of the subtask outputs
                        (e.g. {"fundamental": ..., "sentiment": ...,
                        "qualitative_risk": ...}). Any subset/number works.
        client:         an LLMClient (Anthropic or local).
        baseline_price: the T0 close, used to anchor the target price.

    Returns the Leader's initial Prediction (leader_prediction).
    """
    messages = [{
        "role": "user",
        "text": build_user_prompt(ticker, reports, baseline_price),
    }]

    response = client.complete(messages, tools=None, system=SYSTEM_PROMPT)

    return parse_prediction(response.text, ticker, baseline_price)


# --------------------------------------------------------------------------
# Task 17: the Leader's reply to a red-team rebuttal
# --------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = """\
You are the lead portfolio manager defending your investment call against a \
red-team reviewer. You are shown your own current prediction, the reviewer's \
rebuttal (its targeted claim, objections, and severity), and the original \
analyst reports.

Judge the rebuttal honestly. You have two choices:

  - ACCEPT it: the objection is valid and material, so revise your \
prediction. Issue a complete, updated prediction that addresses the \
objection (you may change direction, target price, confidence, and the \
supporting fields).
  - HOLD your position: the objection is unfounded, immaterial, or already \
priced into your call. State clearly why you reject it; do not change the \
prediction.

Do not concede to a weak rebuttal just to appease the reviewer, and do not \
dig in against a strong one. Keep the same leakage discipline: no knowledge \
of events after the T0 market close, and anchor any revised target price to \
the T0 baseline.

Output ONLY a single JSON object (no prose before or after it):

  {
    "accepted": true | false,
    "reason": "<why you accepted or held, referencing the objection>",
    "revised_prediction": {
        "direction": "Buy" | "Not Buy",
        "target_price": <number>,
        "confidence": <number between 0 and 1>,
        "rationale": "<updated justification>",
        "dominant_signal": "<the factor that now drives the call>",
        "risk_reconciliation": "<how you weighed the risk(s)>"
    } | null
  }

Set "revised_prediction" to null when you hold your position; provide the \
full object when you accept and revise.
"""


def build_response_user_prompt(
    ticker: str,
    current_prediction: dict,
    rebuttal: dict,
    reports: dict[str, dict],
    baseline_price: float | None,
) -> str:
    """
    Compose the Leader's rebuttal-reply prompt: the standing prediction, the
    red-team rebuttal, and the same subtask reports, in one payload.
    """
    payload = {
        "ticker": ticker.upper(),
        "baseline_price": baseline_price,
        "current_prediction": current_prediction,
        "rebuttal": {
            "round": rebuttal.get("round"),
            "targeted_claim": rebuttal.get("targeted_claim"),
            "objections": rebuttal.get("objections"),
            "severity": rebuttal.get("severity"),
        },
        "subtask_reports": reports,
    }

    return (
        f"The red team has challenged your one-week call on {ticker.upper()}. "
        f"Decide whether to revise or hold.\n\n"
        f"Prediction, rebuttal, and evidence JSON:\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Return the JSON decision (accepted / reason / revised_prediction)."
    )


def parse_leader_response(
    text: str,
    round: int,
    ticker: str,
    baseline_price: float | None = None,
) -> LeaderResponse:
    """
    Parse the Leader's reply into a LeaderResponse.

    `round` is injected by the loop. A revised prediction is parsed through
    the shared `parse_prediction` (normalized like any other Prediction) only
    when the Leader accepted and actually supplied one; otherwise
    `revised_prediction` is None (the position was held / unchanged).
    """
    data = extract_json_object(text)

    accepted = data.get("accepted", False)
    if not isinstance(accepted, bool):
        accepted = str(accepted).strip().lower() in ("true", "yes", "1")

    revised = data.get("revised_prediction")
    revised_prediction = None

    if accepted and isinstance(revised, dict):
        revised_prediction = parse_prediction(
            json.dumps(revised), ticker, baseline_price
        )
    else:
        # A bare "accepted" with no usable revision is not a real change.
        accepted = False

    return LeaderResponse(
        round=int(round),
        accepted=accepted,
        reason=str(data.get("reason", "")),
        revised_prediction=revised_prediction,
    )


def run_leader_response(
    ticker: str,
    current_prediction: dict,
    rebuttal: dict,
    reports: dict[str, dict],
    client: LLMClient,
    round: int,
    baseline_price: float | None = None,
) -> LeaderResponse:
    """
    Run the Leader's reply to one red-team rebuttal (Task 17).

    `reports` is the same name->rendered-evidence map the Leader aggregated.
    Returns a LeaderResponse: `accepted=True` with a `revised_prediction`
    means the Leader took the objection and revised; `accepted=False` with
    `revised_prediction=None` means it held its position.
    """
    messages = [{
        "role": "user",
        "text": build_response_user_prompt(
            ticker,
            current_prediction,
            rebuttal,
            reports,
            baseline_price,
        ),
    }]

    response = client.complete(
        messages, tools=None, system=RESPONSE_SYSTEM_PROMPT
    )

    return parse_leader_response(response.text, round, ticker, baseline_price)
