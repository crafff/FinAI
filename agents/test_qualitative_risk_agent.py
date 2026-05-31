import pytest

from llm_client import LLMResponse, ToolCall
from qualitative_risk_agent import (
    build_search_tool,
    build_user_prompt,
    extract_json_object,
    parse_risk_score,
    run_qualitative_risk_agent,
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


def fake_retrieval_tool(query, k=5, section=None):
    return (
        f"[chunk 1 | Item {section or '1A'}. Risk Factors | sim=0.90]\n"
        f"Risk text for query: {query}"
    )


def test_build_user_prompt_mentions_ticker_and_risk_focus():
    prompt = build_user_prompt("aapl")

    assert "AAPL" in prompt
    assert "Risk Factors" in prompt
    assert "RiskScore" in prompt


def test_build_search_tool_calls_retrieval_tool():
    calls = []

    def retrieval_tool(query, k=5, section=None):
        calls.append((query, k, section))
        return "retrieved text"

    tool = build_search_tool(retrieval_tool)

    result = tool.impl(query="litigation risk", k=3, section="1A")

    assert result == "retrieved text"
    assert calls == [("litigation risk", 3, "1A")]


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
        'Here is the result: {"score": 4.0, "method": "qualitative"}'
    )

    assert result["score"] == 4.0


def test_parse_risk_score_valid_output():
    text = """
    {
      "method": "qualitative",
      "score": 7.2,
      "summary": "The company faces elevated regulatory and supply chain risk.",
      "factors": ["regulatory risk", "supply chain risk"],
      "justification": "The 10-K emphasizes material uncertainty in these areas."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "qualitative"
    assert result["score"] == pytest.approx(7.2)
    assert result["summary"].startswith("The company faces")
    assert result["factors"] == ["regulatory risk", "supply chain risk"]
    assert "10-K" in result["justification"]


def test_parse_risk_score_forces_method_to_qualitative():
    text = """
    {
      "method": "quantitative",
      "score": 6,
      "summary": "Moderate risk.",
      "factors": ["competition"],
      "justification": "Competitive pressure is material."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "qualitative"


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
      "factors": "competition",
      "justification": "Model returned malformed fields."
    }
    """

    result = parse_risk_score(text)

    assert result["score"] == 5.0
    assert result["factors"] == []


def test_run_qualitative_risk_agent_executes_tool_then_returns_score():
    client = ScriptedClient([
        LLMResponse(
            text="I will inspect risk factors.",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="search_10k",
                    arguments={
                        "query": "material risk factors",
                        "k": 3,
                        "section": "1A",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        LLMResponse(
            text="""
            {
              "method": "qualitative",
              "score": 6.8,
              "summary": "Risk is moderately elevated.",
              "factors": ["competition", "regulatory risk"],
              "justification": "The retrieved 10-K chunks emphasize these risks."
            }
            """,
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    result = run_qualitative_risk_agent(
        ticker="aapl",
        retrieval_tool=fake_retrieval_tool,
        client=client,
    )

    assert result["method"] == "qualitative"
    assert result["score"] == pytest.approx(6.8)
    assert result["factors"] == ["competition", "regulatory risk"]

    assert len(client.calls) == 2
    assert client.calls[0]["tools"] is not None
    assert client.calls[0]["system"] is not None

def test_parse_risk_score_defaults_missing_score():
    text = """
    {
      "summary": "Moderate risk profile.",
      "factors": ["competition"],
      "justification": "Competition remains intense."
    }
    """

    result = parse_risk_score(text)

    assert result["score"] == pytest.approx(5.0)

def test_parse_risk_score_defaults_missing_factors():
    text = """
    {
      "method": "qualitative",
      "score": 6.0,
      "summary": "The company faces moderate qualitative risk.",
      "justification": "Several material risks are discussed in the filing."
    }
    """

    result = parse_risk_score(text)

    assert result["method"] == "qualitative"
    assert result["score"] == pytest.approx(6.0)
    assert result["factors"] == []
    assert result["summary"] == "The company faces moderate qualitative risk."
