from llm_client import LLMResponse
from state import new_state
from redteam_loop import run_rebuttal_loop


# --------------------------------------------------------------------------
# A scripted client that returns red-team then leader-response payloads in
# the exact order the loop calls .complete (rebuttal, response, rebuttal, ...).
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.recorder = None

    def complete(self, messages, tools=None, system=None):
        return LLMResponse(text=self.responses.pop(0), tool_calls=[])


def _rebuttal_json(severity="high", objections=None):
    objs = objections if objections is not None else ["No catalyst."]
    import json
    return json.dumps({
        "targeted_claim": "Target too high.",
        "objections": objs,
        "severity": severity,
    })


def _revise_json(target):
    import json
    return json.dumps({
        "accepted": True,
        "reason": "Valid objection.",
        "revised_prediction": {
            "direction": "Buy", "target_price": target, "confidence": 0.5,
            "rationale": "Revised.", "dominant_signal": "fundamentals",
            "risk_reconciliation": "ok",
        },
    })


_HOLD_JSON = (
    '{"accepted": false, "reason": "Already priced in.", '
    '"revised_prediction": null}'
)


def _seed_state(max_rounds=3):
    state = new_state("AAPL", variant="full", max_rounds=max_rounds)
    state["baseline_price"] = 180.0
    state["subtask_reports_rendered"] = {
        "fundamental": {"summary": "Strong.", "signal": "bullish",
                        "confidence": 0.7, "key_metrics": {}, "citations": []},
        "sentiment": {"summary": "Mixed.", "signal": "mixed", "confidence": 0.5,
                      "news_count": 1, "social_count": 0, "disagreement": True,
                      "citations": []},
        "qualitative_risk": {"collected_factors": ["litigation"],
                             "scores": [{"method": "qualitative", "score": 6.0,
                                         "summary": "s", "factors": ["litigation"],
                                         "justification": "j"}]},
    }
    state["leader_prediction"] = {
        "direction": "Buy", "target_price": 192.0, "confidence": 0.66,
        "rationale": "Initial.", "dominant_signal": "fundamentals",
        "risk_reconciliation": "ok",
    }
    return state


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_loop_revise_then_hold_converges_with_last_revision():
    # Round 1: red team attacks, leader revises to 186.
    # Round 2: red team attacks, leader holds -> converged.
    client = ScriptedClient([
        _rebuttal_json(), _revise_json(186.0),
        _rebuttal_json(), _HOLD_JSON,
    ])

    state = run_rebuttal_loop(_seed_state(max_rounds=3), client)

    assert state["round_count"] == 2
    assert state["converged"] is True
    assert len(state["rebuttals"]) == 2
    assert len(state["leader_responses"]) == 2
    assert state["final_prediction"]["target_price"] == 186.0


def test_loop_hold_round_one_keeps_initial_prediction():
    client = ScriptedClient([_rebuttal_json(), _HOLD_JSON])

    state = run_rebuttal_loop(_seed_state(max_rounds=3), client)

    assert state["round_count"] == 1
    assert state["converged"] is True
    assert state["final_prediction"]["target_price"] == 192.0   # unchanged


def test_loop_stops_at_round_cap_when_leader_keeps_revising():
    # Leader revises every round: only the hard cap can stop the loop.
    client = ScriptedClient([
        _rebuttal_json(), _revise_json(190.0),
        _rebuttal_json(), _revise_json(188.0),
        _rebuttal_json(), _revise_json(185.0),
        # a 4th round would pop here and raise IndexError if the cap failed
    ])

    state = run_rebuttal_loop(_seed_state(max_rounds=3), client)

    assert state["round_count"] == 3
    assert state["converged"] is False        # cap stopped it, not convergence
    assert state["final_prediction"]["target_price"] == 185.0


def test_loop_converges_when_rebuttal_has_no_objections():
    client = ScriptedClient([_rebuttal_json(objections=[]), _HOLD_JSON])

    state = run_rebuttal_loop(_seed_state(max_rounds=3), client)

    assert state["round_count"] == 1
    assert state["converged"] is True
