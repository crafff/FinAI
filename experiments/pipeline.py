"""
The configurable pipeline: build + run one system on one ticker.

`run_system` is the single source of truth for what each ablation
configuration does, replacing the bodies of the three standalone runners:

  - mode "single": one agent sees all evidence and emits the final prediction.
  - mode "leader": the selected sub-task agents run, the Leader aggregates
    their reports, and (optionally) the red-team rebuttal loop runs under a
    configurable round cap.

Everything threads through a PipelineState (contracts/state.py) so a finished
run projects straight into the evaluation row.
"""

from __future__ import annotations

from registry import REGISTRY
from state import new_state
from single_agent import run_single_agent
from leader_agent import run_leader_agent
from redteam_loop import run_rebuttal_loop


def run_system(system, ctx, client) -> dict:
    """
    Run one SystemConfig against one DataContext with the given LLMClient.
    Returns the finished PipelineState (with `final_prediction` set).
    """
    state = new_state(
        ctx.ticker,
        variant=system.name,
        model=client.config.model,
        max_rounds=system.max_rounds,
    )
    state["prices"] = ctx.prices
    state["baseline_price"] = ctx.baseline_price

    if system.mode == "single":
        final = run_single_agent(
            ticker=ctx.ticker,
            financials=ctx.financials,
            news=ctx.news,
            social=ctx.social,
            retrieval_tool=ctx.retrieval_tool,
            client=client,
            baseline_price=ctx.baseline_price,
        )
        state["final_prediction"] = final
        state["round_count"] = 0
        state["converged"] = True
        return state

    # mode == "leader": run the selected sub-task agents, then aggregate.
    raw = {n: REGISTRY[n].run(ctx, client) for n in system.subtasks}
    rendered = {n: REGISTRY[n].render(raw[n]) for n in system.subtasks}
    state["subtask_reports"] = raw
    state["subtask_reports_rendered"] = rendered

    leader_prediction = run_leader_agent(
        ticker=ctx.ticker,
        reports=rendered,
        client=client,
        baseline_price=ctx.baseline_price,
    )
    state["leader_prediction"] = leader_prediction

    if system.red_team and system.max_rounds > 0:
        run_rebuttal_loop(state, client)
    else:
        # Red team disabled: the Leader's initial call is final.
        state["final_prediction"] = leader_prediction
        state["round_count"] = 0
        state["converged"] = True

    return state
