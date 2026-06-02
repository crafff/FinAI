from llm_client import LLMResponse
from schemas import BUY, NOT_BUY, LeaderResponse, RiskAssessment, RiskScore, missing_keys
from leader_agent import (
    build_risk_evidence,
    build_user_prompt,
    parse_leader_response,
    risk_assessment_from_score,
    run_leader_agent,
    run_leader_response,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _fundamental_report():
    return {
        "ticker": "AAPL",
        "summary": "Strong margins and cash flow.",
        "signal": "bullish",
        "confidence": 0.7,
        "key_metrics": {"net_margin": 0.25},
        "citations": ["chunk_3"],
    }


def _sentiment_report():
    return {
        "ticker": "AAPL",
        "summary": "News positive, social mixed.",
        "signal": "mixed",
        "confidence": 0.5,
        "news_count": 4,
        "social_count": 2,
        "disagreement": True,
        "citations": ["news_1"],
    }


def _qual_score():
    return RiskScore(
        method="qualitative",
        score=6.0,
        summary="Litigation and supply-chain exposure.",
        factors=["litigation", "supply chain"],
        justification="Item 1A flags both.",
    )


def _quant_score():
    return RiskScore(
        method="quantitative",
        score=4.0,
        summary="Leverage is moderate.",
        factors=["debt_to_equity"],
        justification="D/E within peer range.",
    )


# --------------------------------------------------------------------------
# Risk adapter (the interim qualitative-only stand-in)
# --------------------------------------------------------------------------

def test_risk_assessment_from_score_is_contract_valid():
    assessment = risk_assessment_from_score(_qual_score())

    assert missing_keys(RiskAssessment, assessment) == set()
    assert assessment["scores"] == [_qual_score()]
    assert assessment["collected_factors"] == ["litigation", "supply chain"]


# --------------------------------------------------------------------------
# Risk evidence renders 1 or 2 scores identically (the swap is a no-op)
# --------------------------------------------------------------------------

def test_build_risk_evidence_renders_one_and_two_scores():
    one = build_risk_evidence(risk_assessment_from_score(_qual_score()))
    assert len(one["scores"]) == 1
    assert one["scores"][0]["method"] == "qualitative"

    two = build_risk_evidence(RiskAssessment(
        collected_factors=["litigation", "supply chain", "debt_to_equity"],
        scores=[_qual_score(), _quant_score()],
    ))
    assert len(two["scores"]) == 2
    assert {s["method"] for s in two["scores"]} == {"qualitative", "quantitative"}


# --------------------------------------------------------------------------
# User prompt
# --------------------------------------------------------------------------

def test_build_user_prompt_includes_all_three_reports_and_baseline():
    prompt = build_user_prompt(
        "AAPL",
        _fundamental_report(),
        _sentiment_report(),
        risk_assessment_from_score(_qual_score()),
        baseline_price=180.0,
    )

    assert "AAPL" in prompt
    assert "180.0" in prompt
    assert "Strong margins and cash flow." in prompt   # fundamental
    assert "News positive, social mixed." in prompt    # sentiment
    assert "Litigation and supply-chain exposure." in prompt  # risk


# --------------------------------------------------------------------------
# End-to-end with a scripted client (no tools / no tool loop)
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.recorder = None
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)


_PREDICTION_JSON = (
    '{"direction": "Buy", "target_price": 192.0, "confidence": 0.66, '
    '"rationale": "Fundamentals outweigh mixed sentiment and moderate risk.", '
    '"dominant_signal": "fundamentals", '
    '"risk_reconciliation": "Risk score 6/10 is priced in."}'
)


def test_run_leader_agent_aggregates_and_predicts():
    client = ScriptedClient([LLMResponse(text=_PREDICTION_JSON, tool_calls=[])])

    pred = run_leader_agent(
        ticker="AAPL",
        fundamental_report=_fundamental_report(),
        sentiment_report=_sentiment_report(),
        risk_assessment=risk_assessment_from_score(_qual_score()),
        client=client,
        baseline_price=180.0,
    )

    # Leader reads reports only: it must not be given any tools.
    assert client.calls[0]["tools"] is None
    assert pred["direction"] == BUY
    assert pred["target_price"] == 192.0
    assert pred["confidence"] == 0.66
    assert pred["rationale"]


def test_swapping_in_two_score_assessment_is_a_noop():
    """
    The future Task-14 swap (qual-only -> qual+quant) must not change the
    Leader's behavior: same scripted response, equivalent Prediction.
    """
    two_score = RiskAssessment(
        collected_factors=["litigation", "supply chain", "debt_to_equity"],
        scores=[_qual_score(), _quant_score()],
    )

    client = ScriptedClient([LLMResponse(text=_PREDICTION_JSON, tool_calls=[])])

    pred = run_leader_agent(
        ticker="AAPL",
        fundamental_report=_fundamental_report(),
        sentiment_report=_sentiment_report(),
        risk_assessment=two_score,
        client=client,
        baseline_price=180.0,
    )

    assert client.calls[0]["tools"] is None
    assert pred["direction"] == BUY
    assert pred["target_price"] == 192.0


# --------------------------------------------------------------------------
# Leader's reply to a rebuttal (Task 17)
# --------------------------------------------------------------------------

def _rebuttal():
    return {
        "round": 1,
        "targeted_claim": "Target 192 too high.",
        "objections": ["No catalyst."],
        "severity": "high",
    }


def _leader_response_args(client):
    return dict(
        ticker="AAPL",
        current_prediction={
            "direction": "Buy", "target_price": 192.0, "confidence": 0.66,
            "rationale": "x", "dominant_signal": "fundamentals",
            "risk_reconciliation": "y",
        },
        rebuttal=_rebuttal(),
        fundamental_report=_fundamental_report(),
        sentiment_report=_sentiment_report(),
        risk_assessment=risk_assessment_from_score(_qual_score()),
        client=client,
        round=1,
        baseline_price=180.0,
    )


def test_run_leader_response_accept_yields_revised_prediction():
    revised_json = (
        '{"accepted": true, "reason": "Fair point on the catalyst.", '
        '"revised_prediction": {"direction": "Buy", "target_price": 186.0, '
        '"confidence": 0.55, "rationale": "Lower the target.", '
        '"dominant_signal": "fundamentals", "risk_reconciliation": "z"}}'
    )
    client = ScriptedClient([LLMResponse(text=revised_json, tool_calls=[])])

    resp = run_leader_response(**_leader_response_args(client))

    assert client.calls[0]["tools"] is None
    assert resp["accepted"] is True
    assert resp["revised_prediction"] is not None
    assert resp["revised_prediction"]["target_price"] == 186.0
    assert missing_keys(LeaderResponse, resp) == set()


def test_run_leader_response_hold_has_no_revision():
    held_json = (
        '{"accepted": false, "reason": "Catalyst is the filing itself.", '
        '"revised_prediction": null}'
    )
    client = ScriptedClient([LLMResponse(text=held_json, tool_calls=[])])

    resp = run_leader_response(**_leader_response_args(client))

    assert resp["accepted"] is False
    assert resp["revised_prediction"] is None
    assert resp["reason"]


def test_parse_leader_response_accept_without_revision_is_treated_as_hold():
    # accepted=true but no usable revision -> not a real change.
    resp = parse_leader_response(
        '{"accepted": true, "reason": "ok", "revised_prediction": null}',
        round=2, ticker="AAPL", baseline_price=180.0,
    )

    assert resp["accepted"] is False
    assert resp["revised_prediction"] is None
    assert resp["round"] == 2
