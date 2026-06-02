"""
Shared offline test helpers: a scripted LLM client, a fake DataContext, and
canned agent-output JSON strings valid for each agent's lenient parser.
"""

import types

from llm_client import LLMResponse


class _Cfg:
    model = "test-model"
    backend = "test"


class ScriptedClient:
    """Returns canned response texts in order; records nothing."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.recorder = None
        self.config = _Cfg()

    def complete(self, messages, tools=None, system=None):
        return LLMResponse(text=self.responses.pop(0), tool_calls=[])


def fake_context(ticker="AAPL", target_price=200.0, baseline_price=180.0):
    return types.SimpleNamespace(
        ticker=ticker,
        accession="0000-acc",
        t0={"t0_date": "2025-11-03", "target_date": "2025-11-10"},
        cutoff_timestamp="2025-11-03T16:00:00-05:00",
        retrieval_tool=lambda query, k=5, section=None: "[chunk]",
        financials={"fiscal_year": 2025},
        news=[],
        social=[],
        prices={"target_price": target_price},
        baseline_price=baseline_price,
        missing=[],
    )


PREDICTION_JSON = (
    '{"direction": "Buy", "target_price": 192.0, "confidence": 0.6, '
    '"rationale": "r", "dominant_signal": "fundamentals", '
    '"risk_reconciliation": "x"}'
)
FUNDAMENTAL_JSON = (
    '{"summary": "s", "signal": "bullish", "confidence": 0.6, '
    '"key_metrics": {}, "citations": []}'
)
SENTIMENT_JSON = (
    '{"summary": "s", "signal": "mixed", "confidence": 0.5, "news_count": 0, '
    '"social_count": 0, "disagreement": false, "citations": []}'
)
RISK_JSON = (
    '{"method": "qualitative", "score": 6, "summary": "s", '
    '"factors": ["x"], "justification": "j"}'
)
REBUTTAL_JSON = (
    '{"targeted_claim": "c", "objections": ["o"], "severity": "high"}'
)
HOLD_JSON = (
    '{"accepted": false, "reason": "held", "revised_prediction": null}'
)
