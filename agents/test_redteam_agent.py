from llm_client import LLMResponse
from schemas import Rebuttal, RiskAssessment, RiskScore, missing_keys
from redteam_agent import parse_rebuttal, run_redteam_agent


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _prediction():
    return {
        "direction": "Buy",
        "target_price": 192.0,
        "confidence": 0.66,
        "rationale": "Fundamentals outweigh mixed sentiment.",
        "dominant_signal": "fundamentals",
        "risk_reconciliation": "Risk 6/10 priced in.",
    }


def _fundamental_report():
    return {"summary": "Strong margins.", "signal": "bullish", "confidence": 0.7,
            "key_metrics": {"net_margin": 0.25}, "citations": ["chunk_3"]}


def _sentiment_report():
    return {"summary": "News positive, social mixed.", "signal": "mixed",
            "confidence": 0.5, "news_count": 4, "social_count": 2,
            "disagreement": True, "citations": ["news_1"]}


def _risk_assessment():
    return RiskAssessment(
        collected_factors=["litigation"],
        scores=[RiskScore(method="qualitative", score=6.0, summary="Litigation.",
                          factors=["litigation"], justification="Item 1A.")],
    )


# --------------------------------------------------------------------------
# parse_rebuttal
# --------------------------------------------------------------------------

def test_parse_rebuttal_full_and_round_injection():
    text = """{
        "targeted_claim": "Target price 192 is too high.",
        "objections": ["No catalyst for a 7% move.", "Sentiment is mixed."],
        "severity": "high"
    }"""

    reb = parse_rebuttal(text, round=2)

    assert reb["round"] == 2                       # injected, not from model
    assert reb["targeted_claim"].startswith("Target price")
    assert reb["objections"] == ["No catalyst for a 7% move.", "Sentiment is mixed."]
    assert reb["severity"] == "high"
    assert missing_keys(Rebuttal, reb) == set()


def test_parse_rebuttal_validates_severity_and_coerces_objections():
    # Bad severity -> default; a scalar objection -> wrapped in a list.
    reb = parse_rebuttal(
        '{"targeted_claim": "x", "objections": "only one", "severity": "catastrophic"}',
        round=1,
    )

    assert reb["severity"] == "medium"
    assert reb["objections"] == ["only one"]


# --------------------------------------------------------------------------
# End-to-end with a scripted client (no tools)
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.recorder = None
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)


def test_run_redteam_agent_returns_contract_valid_rebuttal_without_tools():
    client = ScriptedClient([LLMResponse(
        text='{"targeted_claim": "Confidence too high.", '
             '"objections": ["Mixed sentiment undercuts 0.66."], '
             '"severity": "medium"}',
        tool_calls=[],
    )])

    reb = run_redteam_agent(
        ticker="AAPL",
        prediction=_prediction(),
        fundamental_report=_fundamental_report(),
        sentiment_report=_sentiment_report(),
        risk_assessment=_risk_assessment(),
        client=client,
        round=1,
    )

    assert client.calls[0]["tools"] is None        # red team uses no tools
    assert reb["round"] == 1
    assert reb["severity"] == "medium"
    assert reb["objections"]
    assert missing_keys(Rebuttal, reb) == set()
