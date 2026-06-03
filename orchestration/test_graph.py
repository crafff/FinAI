import types

import pytest

from llm_client import LLMResponse
from experiment_config import SystemConfig
from graph import build_langgraph, run_langgraph_system

from graph import (
    _as_risk_assessment,
    build_langgraph,
    run_langgraph_system,
)

class FakeClient:
    def __init__(self):
        self.config = types.SimpleNamespace(model="test-model")
        self.recorder = None

    def complete(self, messages, tools=None, system=None):
        return LLMResponse(
            text='{"direction": "Buy", "target_price": 192.0, '
                 '"confidence": 0.6, "rationale": "r", '
                 '"dominant_signal": "fundamentals", '
                 '"risk_reconciliation": "x"}',
            tool_calls=[],
        )


def test_build_langgraph_compiles(monkeypatch):
    system = SystemConfig(
        name="full",
        mode="leader",
        subtasks=[],
        red_team=False,
        max_rounds=0,
    )

    graph = build_langgraph(system, settings=object(), client=FakeClient())

    assert graph is not None


def test_run_langgraph_system_with_no_subtasks(monkeypatch):
    import graph as graph_module

    fake_ctx = types.SimpleNamespace(
        t0={"cutoff_timestamp_et": "2025-11-03T16:00:00-05:00"},
        cutoff_timestamp="2025-11-03T16:00:00-05:00",
        financials={},
        news=[],
        social=[],
        prices={"target_price": 200.0},
        baseline_price=180.0,
    )

    monkeypatch.setattr(
        graph_module,
        "build_data_context",
        lambda ticker, settings, allow_missing=False: fake_ctx,
    )

    system = SystemConfig(
        name="leader_only",
        mode="leader",
        subtasks=[],
        red_team=False,
        max_rounds=0,
    )

    state = run_langgraph_system(
        system=system,
        ticker="AAPL",
        settings=object(),
        client=FakeClient(),
    )

    assert state["ticker"] == "AAPL"
    assert state["baseline_price"] == 180.0
    assert state["final_prediction"]["direction"] == "Buy"
    assert state["converged"] is True


def test_qualitative_risk_wrapped_into_risk_assessment():
    report = {
        "method": "qualitative",
        "score": 6.0,
        "summary": "Moderate risk",
        "factors": ["competition"],
        "justification": "Reason",
    }

    result = _as_risk_assessment(report)

    assert result["collected_factors"] == ["competition"]
    assert len(result["scores"]) == 1
    assert result["scores"][0]["score"] == pytest.approx(6.0)

def test_as_risk_assessment_preserves_existing_assessment():
    assessment = {
        "collected_factors": ["competition"],
        "scores": [
            {
                "method": "qualitative",
                "score": 6.0,
                "summary": "Moderate risk",
                "factors": ["competition"],
                "justification": "Reason",
            }
        ],
    }

    result = _as_risk_assessment(assessment)

    assert result is assessment

def test_as_risk_assessment_wraps_raw_risk_score():
    report = {
        "method": "qualitative",
        "score": 6.0,
        "summary": "Moderate risk",
        "factors": ["competition"],
        "justification": "Reason",
    }

    result = _as_risk_assessment(report)

    assert result["collected_factors"] == ["competition"]
    assert len(result["scores"]) == 1
    assert result["scores"][0]["method"] == "qualitative"
    assert result["scores"][0]["score"] == pytest.approx(6.0)


def test_as_risk_assessment_preserves_existing_assessment():
    assessment = {
        "collected_factors": ["competition"],
        "scores": [
            {
                "method": "qualitative",
                "score": 6.0,
                "summary": "Moderate risk",
                "factors": ["competition"],
                "justification": "Reason",
            }
        ],
    }

    result = _as_risk_assessment(assessment)

    assert result is assessment
    assert result["collected_factors"] == ["competition"]
    assert len(result["scores"]) == 1


def test_route_after_leader_skips_redteam_when_disabled():
    from graph import route_after_leader
    from experiment_config import SystemConfig
    from state import new_state

    system = SystemConfig(
        name="leader_no_redteam",
        mode="leader",
        subtasks=["fundamental"],
        red_team=False,
        max_rounds=3,
    )

    state = new_state("AAPL", variant="full", max_rounds=3)

    router = route_after_leader(system)

    assert router(state) == "finalize"


def test_route_after_leader_skips_redteam_when_max_rounds_zero():
    from graph import route_after_leader
    from experiment_config import SystemConfig
    from state import new_state

    system = SystemConfig(
        name="leader_no_rounds",
        mode="leader",
        subtasks=["fundamental"],
        red_team=True,
        max_rounds=0,
    )

    state = new_state("AAPL", variant="full", max_rounds=0)

    router = route_after_leader(system)

    assert router(state) == "finalize"


def test_route_after_leader_enters_redteam_when_enabled():
    from graph import route_after_leader
    from experiment_config import SystemConfig
    from state import new_state

    system = SystemConfig(
        name="full",
        mode="leader",
        subtasks=["fundamental", "sentiment", "qualitative_risk"],
        red_team=True,
        max_rounds=3,
    )

    state = new_state("AAPL", variant="full", max_rounds=3)

    router = route_after_leader(system)

    assert router(state) == "redteam"


def test_route_after_leader_response_continues_until_round_cap():
    from graph import route_after_leader_response
    from state import new_state

    state = new_state("AAPL", variant="full", max_rounds=3)
    state["round_count"] = 1
    state["converged"] = False

    assert route_after_leader_response(state) == "redteam"


def test_route_after_leader_response_finalizes_when_converged():
    from graph import route_after_leader_response
    from state import new_state

    state = new_state("AAPL", variant="full", max_rounds=3)
    state["round_count"] = 1
    state["converged"] = True

    assert route_after_leader_response(state) == "finalize"


def test_route_after_leader_response_finalizes_at_round_cap():
    from graph import route_after_leader_response
    from state import new_state

    state = new_state("AAPL", variant="full", max_rounds=3)
    state["round_count"] = 3
    state["converged"] = False

    assert route_after_leader_response(state) == "finalize"
