from llm_client import LLMResponse, ToolCall
from schemas import BUY, NOT_BUY
from single_agent import (
    build_user_prompt,
    normalize_direction,
    parse_prediction,
    run_single_agent,
)


# --------------------------------------------------------------------------
# Direction normalization
# --------------------------------------------------------------------------

def test_normalize_direction_buy_and_not_buy():
    assert normalize_direction("Buy", 110, 100) == BUY
    # "Not Buy" contains "buy"; the not/sell check must win.
    assert normalize_direction("Not Buy", 90, 100) == NOT_BUY
    assert normalize_direction("sell", 90, 100) == NOT_BUY


def test_normalize_direction_infers_from_price_when_unparseable():
    assert normalize_direction("???", 110, 100) == BUY
    assert normalize_direction(None, 90, 100) == NOT_BUY


def test_normalize_direction_defaults_when_nothing_usable():
    assert normalize_direction(None, None, None) == NOT_BUY


# --------------------------------------------------------------------------
# Prediction parsing / normalization
# --------------------------------------------------------------------------

def test_parse_prediction_full():
    text = """{
        "direction": "Buy",
        "target_price": 192.5,
        "confidence": 0.7,
        "rationale": "Strong cash flow.",
        "dominant_signal": "fundamentals",
        "risk_reconciliation": "Risks priced in."
    }"""

    pred = parse_prediction(text, "aapl", baseline_price=180.0)

    assert pred["direction"] == BUY
    assert pred["target_price"] == 192.5
    assert pred["confidence"] == 0.7
    assert pred["rationale"] == "Strong cash flow."
    assert pred["dominant_signal"] == "fundamentals"
    assert pred["risk_reconciliation"] == "Risks priced in."


def test_parse_prediction_clamps_confidence_and_falls_back_target():
    text = '{"direction": "Buy", "confidence": 5}'

    pred = parse_prediction(text, "AAPL", baseline_price=150.0)

    assert pred["confidence"] == 1.0          # clamped to [0, 1]
    assert pred["target_price"] == 150.0      # missing -> baseline anchor


def test_parse_prediction_infers_direction_from_target():
    # No/garbled direction: infer from target vs. baseline.
    text = '{"target_price": 200, "confidence": 0.4}'

    pred = parse_prediction(text, "AAPL", baseline_price=180.0)

    assert pred["direction"] == BUY


# --------------------------------------------------------------------------
# User prompt
# --------------------------------------------------------------------------

def test_build_user_prompt_includes_evidence_and_baseline():
    prompt = build_user_prompt(
        "AAPL",
        financials={"fiscal_year": 2025},
        news=[{"headline": "Earnings beat", "summary": "Up"}],
        social=[{"title": "to the moon", "body": "buy"}],
        baseline_price=180.0,
    )

    assert "AAPL" in prompt
    assert "180.0" in prompt
    assert "news_1" in prompt
    assert "social_1" in prompt


# --------------------------------------------------------------------------
# End-to-end with a scripted client + fake retrieval tool
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.recorder = None

    def complete(self, messages, tools=None, system=None):
        return self.responses.pop(0)


def test_run_single_agent_uses_rag_then_predicts():
    retrieval_calls = []

    def fake_retrieval_tool(query, k=5, section=None):
        retrieval_calls.append((query, k, section))
        return f"[chunk] {query}"

    client = ScriptedClient([
        LLMResponse(
            text="checking risk factors",
            tool_calls=[ToolCall("t1", "search_10k",
                                 {"query": "risk factors", "section": "1A"})],
        ),
        LLMResponse(
            text='{"direction": "Buy", "target_price": 195.0, '
                 '"confidence": 0.65, "rationale": "Solid.", '
                 '"dominant_signal": "fundamentals", '
                 '"risk_reconciliation": "Manageable."}',
            tool_calls=[],
        ),
    ])

    pred = run_single_agent(
        ticker="AAPL",
        financials={"fiscal_year": 2025, "profitability": {"net_margin": 0.25}},
        news=[{"headline": "Earnings beat"}],
        social=[{"title": "bullish"}],
        retrieval_tool=fake_retrieval_tool,
        client=client,
        baseline_price=180.0,
    )

    assert retrieval_calls == [("risk factors", 5, "1A")]
    assert pred["direction"] == BUY
    assert pred["target_price"] == 195.0
    assert pred["confidence"] == 0.65
