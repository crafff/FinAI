import pytest

from llm_client import LLMResponse, ToolCall
from fundamental_agent import (
    build_user_prompt,
    extract_json_object,
    parse_fundamental_report,
    run_fundamental_agent,
)


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

def test_extract_json_object_plain():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_fenced():
    text = 'Here is the report:\n```json\n{"a": 1, "b": 2}\n```\nThanks.'

    assert extract_json_object(text) == {"a": 1, "b": 2}


def test_extract_json_object_with_surrounding_prose():
    text = 'Final answer: {"a": 1, "nested": {"x": 2}} -- done.'

    assert extract_json_object(text) == {"a": 1, "nested": {"x": 2}}


def test_extract_json_object_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json_object("no json here")


# --------------------------------------------------------------------------
# Report parsing / normalization
# --------------------------------------------------------------------------

def test_parse_fundamental_report_full():
    text = """{
        "summary": "Solid margins and cash flow.",
        "signal": "bullish",
        "confidence": 0.8,
        "key_metrics": {"net_margin": 0.25},
        "citations": ["Item 7 MD&A", "net_margin"]
    }"""

    report = parse_fundamental_report(text, "aapl")

    assert report["ticker"] == "AAPL"
    assert report["signal"] == "bullish"
    assert report["confidence"] == 0.8
    assert report["key_metrics"] == {"net_margin": 0.25}
    assert report["citations"] == ["Item 7 MD&A", "net_margin"]


def test_parse_fundamental_report_coerces_bad_signal_and_confidence():
    text = '{"summary": "x", "signal": "very-bullish", "confidence": 5}'

    report = parse_fundamental_report(text, "AAPL")

    assert report["signal"] == "neutral"     # invalid -> default
    assert report["confidence"] == 1.0       # clamped to [0, 1]


def test_parse_fundamental_report_defaults_missing_fields():
    report = parse_fundamental_report('{"summary": "x"}', "AAPL")

    assert report["signal"] == "neutral"
    assert report["confidence"] == 0.5
    assert report["key_metrics"] == {}
    assert report["citations"] == []


def test_build_user_prompt_includes_financials():
    prompt = build_user_prompt("AAPL", {"fiscal_year": 2025, "report_date": "2025-09-27"})

    assert "AAPL" in prompt
    assert "2025" in prompt


# --------------------------------------------------------------------------
# End-to-end with a scripted client + fake retrieval tool
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, tools=None, system=None):
        return self.responses.pop(0)


def test_run_fundamental_agent_uses_rag_then_reports():
    retrieval_calls = []

    def fake_retrieval_tool(query, k=5, section=None):
        retrieval_calls.append((query, k, section))
        return f"[chunk] {query}"

    client = ScriptedClient([
        LLMResponse(
            text="let me check the MD&A",
            tool_calls=[ToolCall("t1", "search_10k",
                                 {"query": "liquidity", "section": "7"})],
        ),
        LLMResponse(
            text='{"summary": "Healthy.", "signal": "bullish", '
                 '"confidence": 0.7, "key_metrics": {}, "citations": ["Item 7"]}',
            tool_calls=[],
        ),
    ])

    report = run_fundamental_agent(
        ticker="AAPL",
        financials={"fiscal_year": 2025, "profitability": {"net_margin": 0.25}},
        retrieval_tool=fake_retrieval_tool,
        client=client,
    )

    # The agent actually invoked RAG with the model's query + section.
    assert retrieval_calls == [("liquidity", 5, "7")]

    assert report["ticker"] == "AAPL"
    assert report["signal"] == "bullish"
    assert report["confidence"] == 0.7
    assert report["citations"] == ["Item 7"]
