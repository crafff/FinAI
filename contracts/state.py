"""
Task 20 - shared interface contract: the LangGraph pipeline state.

PipelineState is the single mutable object that threads through every
node of the LangGraph graph (Task 18). Each agent is a node that reads
some keys and writes others; the keys and their owners are documented
below so the agents (Tasks 10-17) can be built in parallel against one
agreed state shape.

The state is `total=False`: it is populated incrementally as the
pipeline advances, so early nodes see only the keys written so far.
`new_state` seeds the inputs and the loop bookkeeping; `to_ablation_record`
projects a finished run into the wide-row schema the evaluation code
(Tasks 8/9) consumes, closing the loop from orchestration to metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from schemas import (
    Direction,
    Filing,
    Financials,
    FundamentalReport,
    LeaderResponse,
    NewsItem,
    Prediction,
    Prices,
    RedditPost,
    RiskAssessment,
    SentimentReport,
    T0Window,
    Variant,
)


DEFAULT_MAX_ROUNDS = 3


class RunConfig(TypedDict, total=False):
    """
    Per-run configuration. `variant` selects the ablation configuration
    (see evaluation README); `max_rounds` caps the red-team loop (the
    spec's hard convergence cap, default 3); `model` is the shared
    backbone model id.
    """

    variant: Variant
    model: str
    max_rounds: int
    trend_days: int


class PipelineState(TypedDict, total=False):
    """
    The LangGraph state object. Grouped by who writes each key.
    """

    # --- run config ---
    config: RunConfig

    # --- inputs / shared context (written by the data layer, Tasks 1-7;
    #     read-only for the agents) ---
    ticker: str
    filing: Filing
    t0_window: T0Window
    cutoff_timestamp_et: datetime     # leakage anchor every tool obeys
    baseline_price: float             # = prices.baseline_price (T0 close)
    prices: Prices
    financials: Financials
    news: list[NewsItem]
    social: list[RedditPost]

    # --- Stage 1 subtask outputs (Tasks 10/11/14) ---
    fundamental_report: FundamentalReport
    sentiment_report: SentimentReport
    risk_assessment: RiskAssessment

    # Generalized subtask outputs for the configurable experiment harness:
    # an arbitrary {agent_name: report} map plus the {agent_name: rendered
    # evidence} the Leader / red-team actually consume. These supersede the
    # three fixed keys above when running with a configurable subtask set.
    subtask_reports: dict          # {name: raw report}, for artifacts/audit
    subtask_reports_rendered: dict  # {name: rendered evidence}, fed to agents

    # --- Stage 2 aggregation (Task 15) ---
    leader_prediction: Prediction

    # --- Stage 3 red-team loop (Tasks 16/17) ---
    rebuttals: list  # list[Rebuttal]
    leader_responses: list  # list[LeaderResponse]
    round_count: int
    converged: bool

    # --- Stage 4 final output ---
    final_prediction: Prediction


def new_state(
    ticker: str,
    variant: Variant = "full",
    model: str = "claude-opus-4-8",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    trend_days: int = 30,
) -> PipelineState:
    """
    Construct a fresh PipelineState with config seeded and the red-team
    loop bookkeeping initialised (round_count=0, converged=False, empty
    rebuttal/response logs). Data and agent nodes fill in the rest.
    """
    return PipelineState(
        config=RunConfig(
            variant=variant,
            model=model,
            max_rounds=max_rounds,
            trend_days=trend_days,
        ),
        ticker=ticker.upper(),
        rebuttals=[],
        leader_responses=[],
        round_count=0,
        converged=False,
    )


def should_continue_rebuttal(state: PipelineState) -> bool:
    """
    Loop guard for the red-team rebuttal stage (Task 17).

    Returns True only if the position has not converged AND the hard
    round cap has not been reached. This is the single place the
    convergence cap is enforced, so the graph cannot loop unbounded.
    """
    if state.get("converged", False):
        return False

    max_rounds = state.get("config", {}).get("max_rounds", DEFAULT_MAX_ROUNDS)

    return state.get("round_count", 0) < max_rounds


def current_prediction(state: PipelineState) -> Prediction | None:
    """
    The prediction that currently stands: the final one if emitted, else
    the Leader's latest, else None. Convenience for nodes that need "the
    answer so far" without caring which stage produced it.
    """
    if "final_prediction" in state:
        return state["final_prediction"]

    return state.get("leader_prediction")


def to_ablation_record(state: PipelineState) -> dict:
    """
    Project a finished run into one wide row for the evaluation code.

    Produces the shared columns plus this run's variant-prefixed
    prediction columns, exactly matching the schema evaluation/metrics.py
    and evaluation/significance.py expect:

        ticker, baseline_price, actual_target_price,
        <variant>_direction, <variant>_target_price

    Rows for the three variants are merged on `ticker` downstream to
    build the full ablation DataFrame (Task 19).
    """
    variant = state["config"]["variant"]
    prediction = state["final_prediction"]

    direction: Direction = prediction["direction"]

    return {
        "ticker": state["ticker"],
        "baseline_price": state["baseline_price"],
        # actual_target_price is the real close on the target date - the
        # answer key fetched by Task 3, never exposed to an agent.
        "actual_target_price": state["prices"]["target_price"],
        f"{variant}_direction": direction,
        f"{variant}_target_price": prediction["target_price"],
    }
