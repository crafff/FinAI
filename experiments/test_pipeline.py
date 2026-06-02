from _kit import (
    FUNDAMENTAL_JSON,
    HOLD_JSON,
    PREDICTION_JSON,
    REBUTTAL_JSON,
    RISK_JSON,
    SENTIMENT_JSON,
    ScriptedClient,
    fake_context,
)
from experiment_config import SystemConfig
from pipeline import run_system


def test_single_mode_emits_only_final_prediction():
    state = run_system(
        SystemConfig("single", mode="single"),
        fake_context(),
        ScriptedClient([PREDICTION_JSON]),
    )

    assert state["final_prediction"]["target_price"] == 192.0
    assert state["round_count"] == 0
    assert state["converged"] is True
    assert "subtask_reports" not in state          # no sub-tasks in single mode


def test_leader_without_redteam_final_is_leader_prediction():
    state = run_system(
        SystemConfig("nrt", mode="leader", subtasks=["fundamental"], red_team=False),
        fake_context(),
        ScriptedClient([FUNDAMENTAL_JSON, PREDICTION_JSON]),
    )

    assert state["round_count"] == 0
    assert state["final_prediction"] == state["leader_prediction"]
    assert set(state["subtask_reports"]) == {"fundamental"}


def test_leader_with_redteam_runs_loop_and_converges():
    # subtasks(3) -> leader -> round1 rebuttal -> leader holds -> converged.
    client = ScriptedClient([
        FUNDAMENTAL_JSON, SENTIMENT_JSON, RISK_JSON,   # sub-task agents
        PREDICTION_JSON,                                # leader initial
        REBUTTAL_JSON, HOLD_JSON,                       # round 1
    ])
    state = run_system(
        SystemConfig(
            "full", mode="leader",
            subtasks=["fundamental", "sentiment", "qualitative_risk"],
            red_team=True, max_rounds=3,
        ),
        fake_context(),
        client,
    )

    assert state["round_count"] == 1
    assert state["converged"] is True
    assert len(state["rebuttals"]) == 1
    assert set(state["subtask_reports"]) == {
        "fundamental", "sentiment", "qualitative_risk"
    }
    assert "final_prediction" in state


def test_leader_arbitrary_single_subtask_count():
    # length-1 subtask set works just like length-3.
    state = run_system(
        SystemConfig("one", mode="leader", subtasks=["sentiment"], red_team=False),
        fake_context(),
        ScriptedClient([SENTIMENT_JSON, PREDICTION_JSON]),
    )
    assert set(state["subtask_reports"]) == {"sentiment"}
    assert state["final_prediction"]["direction"] == "Buy"
