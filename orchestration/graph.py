"""
LangGraph orchestration for Task 18.

This module wires Tasks 1-17 into a complete state graph.

The graph preserves the Task 20 PipelineState contract. Runtime-only
dependencies such as settings, LLMClient, SystemConfig, DataContext,
and rendered Leader/red-team report inputs are kept inside GraphContext
rather than being written into PipelineState.

The existing non-LangGraph pipeline remains untouched. This module provides
a parallel LangGraph implementation for Task 18.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.graph import END, StateGraph

from context import build_data_context
from registry import REGISTRY
from state import PipelineState, current_prediction, new_state, should_continue_rebuttal

from leader_agent import run_leader_agent, run_leader_response
from redteam_agent import run_redteam_agent
from single_agent import run_single_agent


@dataclass
class GraphContext:
    """
    Runtime dependencies that should not be stored directly in PipelineState.

    raw_reports and rendered_reports are orchestration-only structures used
    to connect the Stage 1 agents to the Leader and red-team agents without
    adding non-contract keys to PipelineState.
    """

    settings: object
    client: object
    system: object
    data_context: object | None = None
    raw_reports: dict = field(default_factory=dict)
    rendered_reports: dict = field(default_factory=dict)


def _as_risk_assessment(report: dict) -> dict:
    """
    Ensure the qualitative risk output has the Task 20 RiskAssessment shape.

    The final Task 14 contract expects:

        {
            "collected_factors": list[str],
            "scores": list[RiskScore],
        }

    The current qualitative risk agent may produce either:
        - a raw RiskScore, or
        - a RiskAssessment already wrapped by the registry.

    This helper accepts either shape and returns a RiskAssessment.
    """
    if isinstance(report, dict) and "scores" in report and "collected_factors" in report:
        return report

    factors = report.get("factors", []) if isinstance(report, dict) else []

    return {
        "collected_factors": [str(f) for f in factors],
        "scores": [report],
    }


def load_data_node(graph_ctx: GraphContext):
    """
    Build the data-loading node.

    This node represents Tasks 1-7:
        - EDGAR 10-K retrieval
        - T0 computation
        - price retrieval
        - financial retrieval
        - FinnHub news retrieval
        - Reddit retrieval
        - 10-K RAG retrieval tool
    """

    def node(state: PipelineState) -> PipelineState:
        # Reuse a context injected by the caller (the experiment runner builds
        # one per ticker and shares it across systems) instead of loading the
        # data again; only fetch when none was provided.
        ctx = graph_ctx.data_context
        if ctx is None:
            ctx = build_data_context(
                state["ticker"],
                graph_ctx.settings,
                allow_missing=getattr(graph_ctx.system, "allow_missing", False),
            )
            graph_ctx.data_context = ctx

        state["t0_window"] = ctx.t0
        state["cutoff_timestamp_et"] = ctx.cutoff_timestamp
        state["financials"] = ctx.financials
        state["news"] = ctx.news
        state["social"] = ctx.social
        state["prices"] = ctx.prices
        state["baseline_price"] = ctx.baseline_price

        return state

    return node


def single_agent_node(graph_ctx: GraphContext):
    """
    Build the single-agent baseline node.

    This supports SystemConfig(mode="single") without changing the existing
    experiment harness or single-agent implementation.
    """

    def node(state: PipelineState) -> PipelineState:
        ctx = graph_ctx.data_context
        if ctx is None:
            raise RuntimeError("DataContext has not been loaded yet.")

        final_prediction = run_single_agent(
            ticker=ctx.ticker,
            financials=ctx.financials,
            news=ctx.news,
            social=ctx.social,
            retrieval_tool=ctx.retrieval_tool,
            client=graph_ctx.client,
            baseline_price=ctx.baseline_price,
        )

        state["final_prediction"] = final_prediction
        state["round_count"] = 0
        state["converged"] = True

        return state

    return node


def subtask_node(name: str, graph_ctx: GraphContext):
    """
    Build one Stage 1 subtask node from the registry.

    Each subtask writes its contract field into PipelineState and stores
    temporary raw/rendered report versions inside GraphContext.
    """

    def node(state: PipelineState) -> PipelineState:
        ctx = graph_ctx.data_context
        if ctx is None:
            raise RuntimeError("DataContext has not been loaded yet.")

        spec = REGISTRY[name]
        report = spec.run(ctx, graph_ctx.client)

        if name == "fundamental":
            state["fundamental_report"] = report
            report_for_rendering = report

        elif name == "sentiment":
            state["sentiment_report"] = report
            report_for_rendering = report

        elif name in ("risk", "qualitative_risk", "quantitative_risk"):
            # All risk subtasks land in the same Task 20 contract field. The
            # full protocol ("risk", Task 14) already returns a RiskAssessment;
            # the single-method variants return one wrapped by the registry -
            # _as_risk_assessment is a no-op on an assessment and wraps a bare
            # RiskScore, so every shape normalizes here.
            risk_assessment = _as_risk_assessment(report)
            state["risk_assessment"] = risk_assessment
            report_for_rendering = risk_assessment

        else:
            # Future subtasks can still be run and rendered through the registry
            # without changing the Task 20 state contract.
            report_for_rendering = report

        rendered = spec.render(report_for_rendering)
        graph_ctx.raw_reports[name] = report_for_rendering
        graph_ctx.rendered_reports[name] = rendered

        # Mirror the plain-Python pipeline's state shape so downstream saving
        # (subtask_reports/<name>.json) is identical across both engines.
        state.setdefault("subtask_reports", {})[name] = report_for_rendering
        state.setdefault("subtask_reports_rendered", {})[name] = rendered

        return state

    return node


def leader_node(graph_ctx: GraphContext):
    """
    Build the Leader aggregation node.
    """

    def node(state: PipelineState) -> PipelineState:
        prediction = run_leader_agent(
            ticker=state["ticker"],
            reports=graph_ctx.rendered_reports,
            client=graph_ctx.client,
            baseline_price=state.get("baseline_price"),
        )

        state["leader_prediction"] = prediction

        return state

    return node


def redteam_node(graph_ctx: GraphContext):
    """
    Build one red-team rebuttal node.

    This is one explicit LangGraph round. The conditional edge decides
    whether another round should run.
    """

    def node(state: PipelineState) -> PipelineState:
        prediction = current_prediction(state)
        if prediction is None:
            raise RuntimeError("No prediction available for red-team review.")

        round_num = state.get("round_count", 0) + 1

        rebuttal = run_redteam_agent(
            ticker=state["ticker"],
            prediction=prediction,
            reports=graph_ctx.rendered_reports,
            client=graph_ctx.client,
            round=round_num,
            baseline_price=state.get("baseline_price"),
        )

        state["rebuttals"].append(rebuttal)

        return state

    return node


def leader_response_node(graph_ctx: GraphContext):
    """
    Build one Leader-response node.

    If the Leader accepts the rebuttal and revises, the revised prediction
    becomes the standing leader_prediction. If the Leader holds, the graph
    is marked converged.
    """

    def node(state: PipelineState) -> PipelineState:
        if not state.get("rebuttals"):
            raise RuntimeError("No rebuttal available for Leader response.")

        rebuttal = state["rebuttals"][-1]
        prediction = current_prediction(state)

        if prediction is None:
            raise RuntimeError("No prediction available for Leader response.")

        round_num = state.get("round_count", 0) + 1

        response = run_leader_response(
            ticker=state["ticker"],
            current_prediction=prediction,
            rebuttal=rebuttal,
            reports=graph_ctx.rendered_reports,
            client=graph_ctx.client,
            round=round_num,
            baseline_price=state.get("baseline_price"),
        )

        state["leader_responses"].append(response)
        state["round_count"] = round_num

        revised = response.get("revised_prediction")

        if response["accepted"] and revised is not None:
            state["leader_prediction"] = revised
        else:
            state["converged"] = True

        if not rebuttal["objections"]:
            state["converged"] = True

        return state

    return node


def finalize_node():
    def node(state: PipelineState) -> PipelineState:
        prediction = current_prediction(state)

        if prediction is None:
            raise RuntimeError("Cannot finalize without a prediction.")

        state["final_prediction"] = prediction
        state["converged"] = True

        return state

    return node

def route_after_leader(system):
    """
    Decide whether to enter the red-team loop after the Leader prediction.
    """

    def router(state: PipelineState) -> str:
        if getattr(system, "red_team", False) and getattr(system, "max_rounds", 0) > 0:
            return "redteam"

        return "finalize"

    return router


def route_after_leader_response(state: PipelineState) -> str:
    """
    Continue the rebuttal loop or finalize the prediction.
    """
    if should_continue_rebuttal(state):
        return "redteam"

    return "finalize"


def build_langgraph(system, settings, client, data_context=None):
    """
    Build and compile the Task 18 LangGraph graph.

    `data_context`, when given, is reused by the load_data node instead of
    fetching the ticker's data again (the experiment runner shares one
    DataContext across all systems for a ticker).

    For leader-mode systems, the graph wires:

        load_data
            -> selected Stage 1 subtasks
            -> leader
            -> redteam
            -> leader_response
            -> conditional edge back to redteam or forward to finalize
            -> END

    For single-mode systems, the graph wires:

        load_data
            -> single_agent
            -> finalize
            -> END
    """

    graph_ctx = GraphContext(
        settings=settings,
        client=client,
        system=system,
        data_context=data_context,
    )

    graph = StateGraph(PipelineState)

    graph.add_node("load_data", load_data_node(graph_ctx))
    graph.set_entry_point("load_data")

    if getattr(system, "mode", "leader") == "single":
        graph.add_node("single_agent", single_agent_node(graph_ctx))
        graph.add_node("finalize", finalize_node())

        graph.add_edge("load_data", "single_agent")
        graph.add_edge("single_agent", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    previous = "load_data"

    for name in getattr(system, "subtasks", []):
        node_name = f"subtask_{name}"
        graph.add_node(node_name, subtask_node(name, graph_ctx))
        graph.add_edge(previous, node_name)
        previous = node_name

    graph.add_node("leader", leader_node(graph_ctx))
    graph.add_edge(previous, "leader")

    graph.add_node("redteam", redteam_node(graph_ctx))
    graph.add_node("leader_response", leader_response_node(graph_ctx))
    graph.add_node("finalize", finalize_node())

    graph.add_conditional_edges(
        "leader",
        route_after_leader(system),
        {
            "redteam": "redteam",
            "finalize": "finalize",
        },
    )

    graph.add_edge("redteam", "leader_response")

    graph.add_conditional_edges(
        "leader_response",
        route_after_leader_response,
        {
            "redteam": "redteam",
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


def run_langgraph_system(
    system, ticker, settings, client, data_context=None
) -> PipelineState:
    """
    Run one system configuration on one ticker through the LangGraph graph.

    `data_context`, when provided, is reused instead of reloading the ticker's
    data inside the graph.
    """

    state = new_state(
        ticker,
        variant=system.name,
        model=client.config.model,
        max_rounds=system.max_rounds,
    )

    graph = build_langgraph(system, settings, client, data_context=data_context)

    return graph.invoke(state)


def run_system_graph(system, ctx, client, settings=None) -> PipelineState:
    """
    Runner-facing adapter mirroring `pipeline.run_system`'s signature, so the
    experiment harness can switch engines by swapping this in. Drives the
    Task 18 LangGraph state machine over an already-built DataContext.
    """
    return run_langgraph_system(
        system=system,
        ticker=ctx.ticker,
        settings=settings,
        client=client,
        data_context=ctx,
    )
