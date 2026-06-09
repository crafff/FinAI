# LangGraph Orchestration (Task 18)

This module implements Task 18: LangGraph orchestration.

The purpose of this module is to wire Tasks 1–17 into a complete state graph using LangGraph while preserving the shared Task 20 `PipelineState` contract.

The orchestration layer does not perform new financial analysis itself. Instead, it coordinates the execution of the data retrieval layer, Stage 1 agents, Leader Agent, red-team review process, and final prediction generation.

All communication between nodes occurs through the shared `PipelineState` structure defined in Task 20.

---

# Purpose

The project contains many independently developed components:

* Data retrieval modules (Tasks 1–7)
* Evaluation metrics (Tasks 8–9)
* Stage 1 analysis agents (Tasks 10–14)
* Leader Agent (Task 15)
* Red-Team Agent (Task 16)
* Rebuttal–Revision Loop (Task 17)

Task 18 connects these components into a single executable workflow.

The orchestration layer is responsible for:

1. Loading all required company data.
2. Running the selected Stage 1 agents.
3. Collecting and rendering agent reports.
4. Executing Leader aggregation.
5. Managing the rebuttal–revision cycle.
6. Producing the final prediction.
7. Maintaining a valid `PipelineState` throughout execution.

---

# Files

```text
orchestration/
├── graph.py
├── test_graph.py
└── README.md
```

---

# High-Level Graph

The graph supports both the single-agent baseline and the full multi-agent system.

## Single-Agent Baseline

```text
START
  ↓
load_data
  ↓
single_agent
  ↓
finalize
  ↓
END
```

The single-agent baseline loads the complete data context and produces a prediction directly without Leader aggregation or red-team review.

---

## Multi-Agent System

```text
START
  ↓
load_data
  ↓
subtask_fundamental
  ↓
subtask_sentiment
  ↓
subtask_risk
  ↓
leader
  ↓
redteam
  ↓
leader_response
  ↓
continue?
 ↙       ↘
yes      no
 ↓        ↓
redteam  finalize
          ↓
          END
```

The graph explicitly represents the rebuttal–revision cycle as part of the LangGraph state machine.

The exact Stage 1 nodes are determined by the selected `SystemConfig` and are generated dynamically from the shared registry.

---

# Data Loading Node (Tasks 1–7)

The `load_data` node represents the complete retrieval layer.

It loads:

* EDGAR 10-K filings
* T₀ timing information
* historical prices
* financial statements
* FinnHub news
* Reddit posts
* RAG retrieval tools

Rather than reimplementing these retrieval systems, the orchestration layer reuses the existing `build_data_context()` helper.

The resulting information is written into `PipelineState`.

Example fields include:

```python
state["t0_window"]
state["cutoff_timestamp_et"]
state["financials"]
state["news"]
state["social"]
state["prices"]
state["baseline_price"]
```

This guarantees that all downstream agents operate on the same shared company context.

---

# Stage 1 Agent Nodes (Tasks 10–14)

Stage 1 agents generate specialized analyses of the company.

The graph creates these nodes dynamically from the shared registry.

Each registered subtask provides:

```python
run(...)
render(...)
```

The orchestration layer executes each subtask and stores the contract-defined outputs in `PipelineState`.

The graph stores:

```python
state["fundamental_report"]
state["sentiment_report"]
state["risk_assessment"]
```

Rendered versions of the reports are maintained internally inside `GraphContext` for Leader aggregation and red-team review. They are intentionally not stored in `PipelineState` so that the Task 20 contract remains unchanged.

---

## Risk Subtask

The risk subtask now runs the full Task 14 three-phase protocol (`risk` in the
registry):

1. **Phase 1 — cooperate:** the qualitative (Task 12, 10-K narrative) and
   quantitative (Task 13, financials + price trend) perspectives agree on one
   shared list of material risk factors (`collected_factors`).
2. **Phase 2 — compete:** each analyst scores risk 0–10 while weighing those
   shared factors, producing a method-tagged `RiskScore`.
3. **Phase 3 — carry both forward unaveraged:** the resulting `RiskAssessment`
   holds the shared factors plus both opposing scores, so the Leader reconciles
   the disagreement rather than seeing a single blended number.

The graph stores the protocol output in `state["risk_assessment"]`. The
single-method variants (`qualitative_risk`, `quantitative_risk`) remain
available for ablations and land in the same field; `_as_risk_assessment`
normalizes a bare `RiskScore` into a one-element `RiskAssessment` and is a
no-op on an assessment that already has both scores, so every risk subtask
shape is handled uniformly.

---

# Leader Node (Task 15)

The Leader Agent receives all rendered Stage 1 reports.

Inputs:

```python
state["baseline_price"]
rendered_reports
```

The Leader performs free-form reasoning across the available evidence and produces:

```python
state["leader_prediction"]
```

The prediction includes:

* direction (Buy / Not Buy)
* target price
* confidence
* rationale

The Leader does not retrieve new information. It reasons exclusively over the outputs generated by the Stage 1 agents.

---

# Red-Team Review (Tasks 16–17)

The red-team process is represented explicitly in the graph.

Each cycle consists of:

```text
Current Prediction
        ↓
   Red Team
        ↓
Leader Response
        ↓
Convergence Check
```

The red-team agent critiques the current prediction and identifies weaknesses in the reasoning.

The Leader then decides whether to:

* revise the prediction, or
* defend the current prediction

The process repeats until convergence criteria are met.

The graph stores:

```python
state["rebuttals"]
state["leader_responses"]
state["round_count"]
state["converged"]
```

throughout the review process.

The rebuttal loop is controlled by the shared Task 20 helper:

```python
should_continue_rebuttal(...)
```

which centralizes the convergence and maximum-round logic.

---

# Finalization

The `finalize` node determines the final prediction for the run.

The final prediction may originate from:

* the single-agent baseline
* the initial Leader prediction
* a revised Leader prediction produced during rebuttal

The result is written to:

```python
state["final_prediction"]
```

This is the prediction consumed by the evaluation pipeline.

---

# Runtime Context

Certain objects are required during execution but should not be stored inside `PipelineState`.

Examples include:

* settings
* LLM clients
* system configurations
* rendered report caches
* DataContext objects

These are stored inside a separate `GraphContext` object.

This keeps `PipelineState` aligned with the Task 20 contract while still allowing nodes to access runtime dependencies.

---

# State Flow

The primary state fields used throughout execution are:

```python
ticker
variant
model

t0_window
cutoff_timestamp_et
financials
news
social
prices
baseline_price

fundamental_report
sentiment_report
risk_assessment

leader_prediction

rebuttals
leader_responses
round_count
converged

final_prediction
```

Temporary rendered reports and other runtime-only structures are maintained in `GraphContext` rather than `PipelineState`.

The graph ensures that each node receives a consistent and valid state representation.

---

# Relationship to the Experiment Harness

The project already includes an experiment framework responsible for:

* loading datasets
* selecting system configurations
* running ablation studies
* saving results
* computing evaluation metrics

The LangGraph orchestration layer integrates with that framework rather than replacing it.

Existing components such as:

```text
context.py
registry.py
pipeline.py
runner.py
```

continue to provide experiment infrastructure.

Task 18 adds an explicit LangGraph state machine on top of those components.

The experiment harness can drive either implementation: set `engine: langgraph`
in the config (or pass `--engine langgraph`) and the runner executes each
`(ticker, system)` cell through `run_system_graph`, which reuses the runner's
per-ticker `DataContext` (injected into the graph's `load_data` node) and
returns a `PipelineState` shaped exactly like the plain pipeline's — including
`subtask_reports` — so saving, records, and metrics are engine-agnostic. The
default `engine: pipeline` keeps the plain-Python path. Both share the same
registry, agents, and convergence logic, so they coexist for validation and
regression testing and should produce equivalent predictions.

---

# Testing

The orchestration layer is tested independently of external APIs.

Unit tests verify:

* graph construction
* state transitions
* risk-assessment wrapping
* routing logic
* single-agent execution
* Leader execution
* rebuttal-loop control flow

The tests rely on mocked data contexts and scripted model outputs, allowing them to run fully offline.

Run tests from the repository root:

```bash
pytest orchestration -v
```

or:

```bash
uv run pytest orchestration -v
```

All orchestration tests should execute without requiring access to EDGAR, FinnHub, Reddit, or external LLM services.

---

# Output

The graph returns a fully populated `PipelineState`.

This state can be passed directly to:

```python
to_ablation_record(...)
```

and subsequently into the evaluation pipeline for:

* directional accuracy
* confidence intervals
* target-price error
* correlated-error analysis

This closes the loop from retrieval, through agent reasoning, to final evaluation.

