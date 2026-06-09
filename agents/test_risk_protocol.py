import pytest

from llm_client import LLMResponse, ToolCall
from risk_protocol import (
    collect_risk_factors,
    parse_collected_factors,
    run_risk_protocol,
)


class ScriptedClient:
    """Returns canned responses in order; records each call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.recorder = None

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)


def fake_retrieval_tool(query, k=5, section=None):
    return f"[chunk | Item {section or '1A'}] risk text for {query}"


FACTORS = LLMResponse(
    text='{"factors": ["leverage", "Leverage", "competition"]}',
    tool_calls=[],
    stop_reason="end_turn",
)
QUAL = LLMResponse(
    text='{"method": "qualitative", "score": 6.0, "summary": "s", '
         '"factors": ["competition"], "justification": "j"}',
    tool_calls=[],
    stop_reason="end_turn",
)
QUANT = LLMResponse(
    text='{"method": "quantitative", "score": 3.0, "summary": "s", '
         '"factors": ["leverage"], "justification": "j"}',
    tool_calls=[],
    stop_reason="end_turn",
)


def test_parse_collected_factors_dedupes_case_insensitively():
    factors = parse_collected_factors('{"factors": ["A", "a", "B", " a "]}')

    assert factors == ["A", "B"]


def test_parse_collected_factors_handles_bad_shape():
    assert parse_collected_factors('{"factors": "nope"}') == []
    assert parse_collected_factors("{}") == []


def test_collect_risk_factors_uses_tool_loop_and_dedupes():
    client = ScriptedClient([FACTORS])

    factors = collect_risk_factors(
        ticker="AAPL",
        retrieval_tool=fake_retrieval_tool,
        financials={"debt": {"debt_to_equity": 1.2}},
        client=client,
    )

    assert factors == ["leverage", "competition"]
    # Phase 1 is a tool-loop call: tools are offered.
    assert client.calls[0]["tools"] is not None


def test_run_risk_protocol_three_phases_both_scores_unaveraged():
    client = ScriptedClient([FACTORS, QUAL, QUANT])

    assessment = run_risk_protocol(
        ticker="AAPL",
        retrieval_tool=fake_retrieval_tool,
        financials={"debt": {"debt_to_equity": 1.2}},
        client=client,
        price_trend=[{"date": "2025-10-01", "close": 180.0}],
    )

    # Phase 1 shared factors carried through (deduped).
    assert assessment["collected_factors"] == ["leverage", "competition"]
    # Phase 3: both opposing scores, not reduced to one.
    assert len(assessment["scores"]) == 2
    assert {s["method"] for s in assessment["scores"]} == {
        "qualitative", "quantitative"
    }
    assert assessment["scores"][0]["score"] == pytest.approx(6.0)
    assert assessment["scores"][1]["score"] == pytest.approx(3.0)
    # Three LLM interactions: factors, qualitative, quantitative.
    assert len(client.calls) == 3


def test_run_risk_protocol_backstops_empty_factors_from_scores():
    empty_factors = LLMResponse(
        text='{"factors": []}', tool_calls=[], stop_reason="end_turn"
    )
    client = ScriptedClient([empty_factors, QUAL, QUANT])

    assessment = run_risk_protocol(
        ticker="AAPL",
        retrieval_tool=fake_retrieval_tool,
        financials={},
        client=client,
    )

    # With no shared factors, the assessment is backstopped from each
    # analyst's own factors (merged, deduped) rather than left empty.
    assert set(assessment["collected_factors"]) == {"competition", "leverage"}
    assert len(assessment["scores"]) == 2


def test_collect_risk_factors_executes_tool_then_parses():
    client = ScriptedClient([
        LLMResponse(
            text="searching",
            tool_calls=[ToolCall(id="t1", name="search_10k",
                                 arguments={"query": "risk", "section": "1A"})],
            stop_reason="tool_use",
        ),
        FACTORS,
    ])

    factors = collect_risk_factors(
        ticker="AAPL",
        retrieval_tool=fake_retrieval_tool,
        financials={},
        client=client,
    )

    assert factors == ["leverage", "competition"]
    assert len(client.calls) == 2
