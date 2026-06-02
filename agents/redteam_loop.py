"""
Rebuttal-revision loop + convergence cap (Task 17, Stage 3 orchestration).

Drives the Leader <-> red-team exchange: each round the red team (Task 16)
attacks the standing prediction and the Leader (Task 15/17) either revises or
holds. The loop runs until it converges or the hard round cap is hit, then
writes the surviving prediction as `final_prediction`.

The loop operates on the shared PipelineState (contracts/state.py) and uses
`should_continue_rebuttal` as the single place the convergence cap is
enforced. This is deliberately the exact control flow Item 18 (LangGraph)
will wrap: each `run_*` call is a future node, and the `while` guard is the
future conditional edge.

Convergence: if the Leader revises, the new prediction becomes the standing
one and the loop continues; if the Leader holds (or the rebuttal raises no
objections), the positions are stable and `converged` is set. `max_rounds`
(default 3, from RunConfig) is the backstop.
"""

from __future__ import annotations

from llm_client import LLMClient
from state import current_prediction, should_continue_rebuttal

from leader_agent import run_leader_response
from redteam_agent import run_redteam_agent


def run_rebuttal_loop(state: dict, client: LLMClient) -> dict:
    """
    Run the red-team rebuttal loop in place on `state`.

    Reads from state: `ticker`, `subtask_reports_rendered` (the name->rendered
    evidence map), `baseline_price`, and the standing prediction
    (`leader_prediction`). Mutates: `rebuttals`, `leader_responses`,
    `round_count`, `converged`, and finally `final_prediction`.

    Returns the same state object for convenience.
    """
    ticker = state["ticker"]
    reports = state["subtask_reports_rendered"]
    baseline_price = state.get("baseline_price")

    prediction = current_prediction(state)

    while should_continue_rebuttal(state):
        round = state["round_count"] + 1

        rebuttal = run_redteam_agent(
            ticker=ticker,
            prediction=prediction,
            reports=reports,
            client=client,
            round=round,
            baseline_price=baseline_price,
        )
        state["rebuttals"].append(rebuttal)

        response = run_leader_response(
            ticker=ticker,
            current_prediction=prediction,
            rebuttal=rebuttal,
            reports=reports,
            client=client,
            round=round,
            baseline_price=baseline_price,
        )
        state["leader_responses"].append(response)
        state["round_count"] = round

        revised = response.get("revised_prediction")

        if response["accepted"] and revised is not None:
            # Leader took the objection: adopt the revision and keep probing.
            prediction = revised
        else:
            # Leader held its ground: positions are stable.
            state["converged"] = True

        if not rebuttal["objections"]:
            # Red team had nothing material to add: also stable.
            state["converged"] = True

    state["final_prediction"] = prediction

    return state
