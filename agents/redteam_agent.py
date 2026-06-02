"""
Red-team (Evaluation) agent (Task 16, Stage 3).

A dedicated adversarial reviewer that attacks the Leader's initial
prediction (Task 15). It is shown the Leader's Prediction and the same three
Stage-1 reports the Leader saw, and must find the single most vulnerable
claim, raise concrete objections, and rate their severity. Its output is a
Rebuttal (contracts/schemas.py), one per round of the rebuttal loop
(Task 17).

The red team does NO data retrieval and uses NO tools: it only critiques the
evidence already on the table, so it cannot smuggle in post-cutoff
information. Every objection must point at something in the prediction or the
reports.
"""

from __future__ import annotations

from llm_client import LLMClient
from schemas import Rebuttal

# Same lenient JSON extractor the other agents use.
from sentiment_agent import extract_json_object


VALID_SEVERITIES = ("low", "medium", "high")
DEFAULT_SEVERITY = "medium"


SYSTEM_PROMPT = """\
You are a skeptical evaluation analyst on a red team. The lead portfolio \
manager has made an investment call, and your job is to attack it - hard but \
fairly - before the firm commits.

You are given the Leader's prediction (direction, target price, confidence, \
rationale, dominant_signal, risk_reconciliation) and the same three analyst \
reports the Leader used (fundamental, sentiment, risk). You may not gather \
new information: critique only what is in front of you, and assume no \
knowledge of events after the T0 market close.

Find the SINGLE most vulnerable claim in the prediction - an overstated \
fundamental, a misread sentiment signal, an under-weighted risk, an \
unjustified target price, or an internal contradiction - and press on it. \
List concrete, specific objections (not vague doubts), each tied to the \
prediction or a report. Then rate how damaging your critique is: "high" if it \
should change the call, "medium" if it should lower confidence or the target, \
"low" if it is a minor caveat.

Output ONLY a single JSON object (no prose before or after it) with exactly \
these fields:

  {
    "targeted_claim": "<the single claim you are attacking, quoted or paraphrased>",
    "objections": ["<concrete objection>", "<concrete objection>", ...],
    "severity": "low" | "medium" | "high"
  }
"""


def build_user_prompt(
    ticker: str,
    prediction: dict,
    reports: dict[str, dict],
    baseline_price: float | None,
) -> str:
    """
    Compose the red-team prompt: the prediction under attack plus the subtask
    reports (the same name->rendered-evidence map the Leader saw).
    """
    import json

    payload = {
        "ticker": ticker.upper(),
        "baseline_price": baseline_price,
        "prediction_under_review": prediction,
        "subtask_reports": reports,
    }

    return (
        f"Attack the Leader's one-week call on {ticker.upper()}.\n\n"
        f"Prediction + evidence JSON:\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Identify the single most vulnerable claim and return the JSON "
        f"Rebuttal."
    )


def parse_rebuttal(text: str, round: int) -> Rebuttal:
    """
    Parse and normalize the model's JSON into a Rebuttal.

    `round` is injected by the loop (not trusted from the model). Severity is
    validated against the contract's Severity literal; objections are coerced
    to a list of strings.
    """
    data = extract_json_object(text)

    objections = data.get("objections")
    if not isinstance(objections, list):
        objections = [] if objections in (None, "") else [objections]

    severity = data.get("severity", DEFAULT_SEVERITY)
    if severity not in VALID_SEVERITIES:
        severity = DEFAULT_SEVERITY

    return Rebuttal(
        round=int(round),
        targeted_claim=str(data.get("targeted_claim", "")),
        objections=[str(o) for o in objections],
        severity=severity,
    )


def run_redteam_agent(
    ticker: str,
    prediction: dict,
    reports: dict[str, dict],
    client: LLMClient,
    round: int,
    baseline_price: float | None = None,
) -> Rebuttal:
    """
    Run the red-team agent for one round.

    Inputs:
        ticker:         stock ticker.
        prediction:     the Leader's current Prediction under attack.
        reports:        the same name->rendered-evidence map the Leader saw.
        client:         an LLMClient (Anthropic or local).
        round:          1-based round number, stamped on the Rebuttal.
        baseline_price: the T0 close (context only).

    Returns a Rebuttal for this round.
    """
    messages = [{
        "role": "user",
        "text": build_user_prompt(
            ticker,
            prediction,
            reports,
            baseline_price,
        ),
    }]

    response = client.complete(messages, tools=None, system=SYSTEM_PROMPT)

    return parse_rebuttal(response.text, round)
