import pytest

from llm_client import LLMResponse
from quantitative_risk_agent import (
    build_user_prompt,
    parse_risk_score,
    run_quantitative_risk_agent,
)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)


FINANCIALS = {
    "ticker": "AAPL",
    "debt": {"debt_to_equity": 1.5, "current_ratio": 0.9, "interest_coverage": 30},
    "profitability": {"net_margin": 0.25, "return_on_equity": 1.4},
}
TREND = [{"date": "2025-10-01", "close": 180.0}, {"date": "2025-10-02", "close": 175.0}]


def test_build_user_prompt_includes_ticker_and_financials():
    prompt = build_user_prompt("aapl", FINANCIALS, price_trend=TREND)

    assert "AAPL" in prompt
    assert "debt_to_equity" in prompt
    assert "price" in prompt.lower()


def test_build_user_prompt_degrades_on_empty_financials():
    prompt = build_user_prompt("AAPL", {}, price_trend=None)

    assert "No structured financials" in prompt
    assert "AAPL" in prompt


def test_build_user_prompt_renders_shared_factors():
    prompt = build_user_prompt("AAPL", FINANCIALS, shared_factors=["leverage", "fx"])

    assert "leverage" in prompt
    assert "fx" in prompt
    assert "shared" in prompt.lower()


def test_parse_risk_score_forces_method_quantitative():
    text = (
        '{"method": "qualitative", "score": 4.2, "summary": "s", '
        '"factors": ["leverage"], "justification": "j"}'
    )
    result = parse_risk_score(text)

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(4.2)
    assert result["factors"] == ["leverage"]


def test_parse_risk_score_clamps_and_defaults():
    high = parse_risk_score('{"score": 99, "factors": []}')
    bad = parse_risk_score('{"score": "n/a", "factors": "leverage"}')

    assert high["score"] == 10.0
    assert bad["score"] == 5.0
    assert bad["factors"] == []


def test_run_quantitative_risk_agent_single_call_returns_score():
    client = ScriptedClient([
        LLMResponse(
            text='{"method": "quantitative", "score": 3.5, "summary": "Low '
                 'leverage.", "factors": ["valuation"], "justification": "j"}',
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    result = run_quantitative_risk_agent(
        ticker="aapl",
        financials=FINANCIALS,
        client=client,
        price_trend=TREND,
    )

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(3.5)
    # Tool-free: exactly one completion, no tools passed.
    assert len(client.calls) == 1
    assert client.calls[0]["tools"] is None
    assert client.calls[0]["system"] is not None
