import json

import pytest

from llm_client import LLMResponse, ToolCall
from quantitative_risk_agent import (
    build_financials_tool,
    build_search_tool,
    build_user_prompt,
    extract_json_object,
    parse_risk_score,
    run_quantitative_risk_agent,
)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
        })
        return self.responses.pop(0)


SAMPLE_FINANCIALS = {
    "ticker": "AAPL",
    "fiscal_year": 2025,
    "report_date": "2025-09-27",
    "profitability": {
        "revenue": 400000.0,
        "net_income": 100000.0,
        "gross_margin": 0.45,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "return_on_equity": 1.5,
        "return_on_assets": 0.28,
    },
    "cash_flow": {
        "operating_cash_flow": 110000.0,
        "capital_expenditure": -11000.0,
        "free_cash_flow": 99000.0,
    },
    "debt": {
        "total_debt": 120000.0,
        "total_equity": 60000.0,
        "debt_to_equity": 2.0,
        "current_ratio": 0.98,
        "interest_coverage": 30.0,
    },
    "valuation": {
        "pe_ratio": 33.0,
        "pb_ratio": 50.0,
        "price_to_sales": 8.5,
        "ev_to_ebitda": 25.0,
    },
}


def fake_retrieval_tool(query, k=5, section=None):
    return (
        f"[chunk 1 | Item {section or '7'}. MD&A | sim=0.88]\n"
        f"Financial context for query: {query}"
    )


# --------------------------------------------------------------------------
# build_user_prompt
# --------------------------------------------------------------------------

def test_build_user_prompt_mentions_ticker_and_quantitative_focus():
    prompt = build_user_prompt("aapl", SAMPLE_FINANCIALS)

    assert "AAPL" in prompt
    assert "quantitative" in prompt.lower()
    assert "get_financials" in prompt
    assert "RiskScore" in prompt


def test_build_user_prompt_lists_available_categories():
    prompt = build_user_prompt("MSFT", SAMPLE_FINANCIALS)

    assert "profitability" in prompt
    assert "cash_flow" in prompt
    assert "debt" in prompt
    assert "valuation" in prompt


def test_build_user_prompt_handles_missing_categories():
    sparse = {
        "ticker": "TSLA",
        "fiscal_year": 2025,
        "report_date": None,
        "profitability": {"revenue": None, "net_income": None, "gross_margin": None,
                          "operating_margin": None, "net_margin": None,
                          "return_on_equity": None, "return_on_assets": None},
        "cash_flow": {"operating_cash_flow": None, "capital_expenditure": None,
                      "free_cash_flow": None},
        "debt": {"total_debt": None, "total_equity": None, "debt_to_equity": None,
                 "current_ratio": None, "interest_coverage": None},
        "valuation": {"pe_ratio": None, "pb_ratio": None, "price_to_sales": None,
                      "ev_to_ebitda": None},
    }
    prompt = build_user_prompt("TSLA", sparse)

    assert "TSLA" in prompt
    # All None means no categories listed.
    assert "none available" in prompt


# --------------------------------------------------------------------------
# build_financials_tool
# --------------------------------------------------------------------------

def test_build_financials_tool_returns_json():
    tool = build_financials_tool(SAMPLE_FINANCIALS)

    result = tool.impl()

    parsed = json.loads(result)
    assert parsed["ticker"] == "AAPL"
    assert parsed["profitability"]["revenue"] == 400000.0
    assert parsed["debt"]["debt_to_equity"] == 2.0


def test_build_financials_tool_metadata():
    tool = build_financials_tool(SAMPLE_FINANCIALS)

    assert tool.name == "get_financials"
    assert "profitability" in tool.description
    assert "leverage" in tool.description


# --------------------------------------------------------------------------
# build_search_tool
# --------------------------------------------------------------------------

def test_build_search_tool_calls_retrieval_tool():
    calls = []

    def retrieval_tool(query, k=5, section=None):
        calls.append((query, k, section))
        return "retrieved text"

    tool = build_search_tool(retrieval_tool)

    result = tool.impl(query="debt covenants", k=3, section="7")

    assert result == "retrieved text"
    assert calls == [("debt covenants", 3, "7")]


# --------------------------------------------------------------------------
# extract_json_object
# --------------------------------------------------------------------------

def test_extract_json_object_plain_json():
    result = extract_json_object('{"score": 6.5}')

    assert result["score"] == 6.5


def test_extract_json_object_with_fence():
    result = extract_json_object(
        '```json\n{"score": 7.0}\n```'
    )

    assert result["score"] == 7.0


def test_extract_json_object_with_surrounding_prose():
    result = extract_json_object(
        'Here is the result: {"score": 4.0, "method": "quantitative"}'
    )

    assert result["score"] == 4.0


def test_extract_json_object_raises_on_no_json():
    with pytest.raises(ValueError, match="No JSON object"):
        extract_json_object("There is no JSON here at all.")


# --------------------------------------------------------------------------
# parse_risk_score
# --------------------------------------------------------------------------

def test_parse_risk_score_valid_output():
    text = """
    {
      "method": "quantitative",
      "score": 6.0,
      "summary": "Elevated leverage with strong profitability offset.",
      "factors": ["high debt-to-equity", "stretched valuation"],
      "justification": "Debt/equity of 2.0 is above average; P/B of 50 is extreme."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(6.0)
    assert result["summary"].startswith("Elevated leverage")
    assert result["factors"] == ["high debt-to-equity", "stretched valuation"]
    assert "Debt/equity" in result["justification"]


def test_parse_risk_score_forces_method_to_quantitative():
    text = """
    {
      "method": "qualitative",
      "score": 6,
      "summary": "Moderate risk.",
      "factors": ["competition"],
      "justification": "Competitive pressure is material."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "quantitative"


def test_parse_risk_score_clamps_score():
    high = parse_risk_score("""
    {
      "score": 15,
      "summary": "Very high risk.",
      "factors": [],
      "justification": "Too high."
    }
    """)

    low = parse_risk_score("""
    {
      "score": -3,
      "summary": "Low risk.",
      "factors": [],
      "justification": "Too low."
    }
    """)

    assert high["score"] == 10.0
    assert low["score"] == 0.0


def test_parse_risk_score_defaults_bad_score_and_factors():
    text = """
    {
      "score": "not-a-number",
      "summary": "Unclear risk.",
      "factors": "high debt",
      "justification": "Model returned malformed fields."
    }
    """

    result = parse_risk_score(text)

    assert result["score"] == 5.0
    assert result["factors"] == []


def test_parse_risk_score_defaults_missing_score():
    text = """
    {
      "summary": "Moderate risk profile.",
      "factors": ["high valuation"],
      "justification": "Valuation multiples are stretched."
    }
    """

    result = parse_risk_score(text)

    assert result["score"] == pytest.approx(5.0)


def test_parse_risk_score_defaults_missing_factors():
    text = """
    {
      "method": "quantitative",
      "score": 4.5,
      "summary": "The company has moderate quantitative risk.",
      "justification": "Metrics are largely healthy."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(4.5)
    assert result["factors"] == []
    assert result["summary"] == "The company has moderate quantitative risk."


# --------------------------------------------------------------------------
# run_quantitative_risk_agent (end-to-end with scripted client)
# --------------------------------------------------------------------------

def test_run_quantitative_risk_agent_executes_tools_then_returns_score():
    client = ScriptedClient([
        # Turn 1: model calls get_financials
        LLMResponse(
            text="I will examine the financials.",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="get_financials",
                    arguments={},
                )
            ],
            stop_reason="tool_use",
        ),
        # Turn 2: model calls search_10k for MD&A context
        LLMResponse(
            text="Now I will check the 10-K for leverage context.",
            tool_calls=[
                ToolCall(
                    id="t2",
                    name="search_10k",
                    arguments={
                        "query": "debt covenants leverage",
                        "k": 3,
                        "section": "7",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        # Turn 3: model returns final RiskScore JSON
        LLMResponse(
            text="""
            {
              "method": "quantitative",
              "score": 5.5,
              "summary": "Moderate risk from elevated leverage offset by strong cash flow.",
              "factors": ["high debt-to-equity", "strong free cash flow"],
              "justification": "D/E of 2.0 is elevated but interest coverage of 30x and FCF of $99B provide a strong safety margin."
            }
            """,
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    result = run_quantitative_risk_agent(
        ticker="aapl",
        financials=SAMPLE_FINANCIALS,
        retrieval_tool=fake_retrieval_tool,
        client=client,
    )

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(5.5)
    assert result["factors"] == ["high debt-to-equity", "strong free cash flow"]

    # 3 LLM calls: get_financials, search_10k, final answer
    assert len(client.calls) == 3
    assert client.calls[0]["tools"] is not None
    assert client.calls[0]["system"] is not None


def test_run_quantitative_risk_agent_single_turn():
    """Model returns the score in a single turn (no tool calls)."""
    client = ScriptedClient([
        LLMResponse(
            text=json.dumps({
                "method": "quantitative",
                "score": 3.0,
                "summary": "Low risk; healthy balance sheet.",
                "factors": ["low leverage"],
                "justification": "D/E below 1, strong margins.",
            }),
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    result = run_quantitative_risk_agent(
        ticker="MSFT",
        financials=SAMPLE_FINANCIALS,
        retrieval_tool=fake_retrieval_tool,
        client=client,
    )

    assert result["method"] == "quantitative"
    assert result["score"] == pytest.approx(3.0)
    assert len(client.calls) == 1
