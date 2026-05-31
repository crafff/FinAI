# Shared Interface Contract (Task 20)

This module is the foundation the rest of the system is built against. It
fixes the **shapes** that flow between every part of the pipeline, so the
data modules (Tasks 1-9), the agents (Tasks 10-17), the orchestration
(Task 18), and the evaluation (Tasks 8/9/19) can be developed in parallel
without clashing.

It produces no behaviour of its own. It is pure type/shape definitions
plus a few tiny helpers.

## Why it exists

The agents are built in parallel. Without one agreed contract, the
fundamental agent might emit `{"summary": ...}` while the Leader expects
`{"fundamental_report": ...}`, and integration breaks. Fixing the
contract first lets each agent be written and tested against a stable
interface.

## Files

- `schemas.py` - three layers of shapes:
  1. **Data records** that mirror the dicts the Task 1-7 data modules
     already return (`Filing`, `T0Window`, `Prices`, `Financials`,
     `NewsItem`, `RedditPost`, `RetrievedChunk`). These document existing
     outputs; the tests assert real module shapes conform.
  2. **Agent I/O** - the previously-missing layer: `FundamentalReport`,
     `SentimentReport`, `RiskScore` / `RiskAssessment`, `Prediction`,
     `Rebuttal`, `LeaderResponse`.
  3. **Enums/helpers** - `Direction` (Buy / Not Buy), `Signal`,
     `Variant`, and `missing_keys` for lightweight validation.
- `state.py` - the LangGraph `PipelineState` (the single object threaded
  through every node), `RunConfig`, and helpers: `new_state`,
  `should_continue_rebuttal` (the single place the red-team round cap is
  enforced), `current_prediction`, and `to_ablation_record`.

## Implementation choice: TypedDict

The contract is built on `typing.TypedDict`, not pydantic or dataclasses,
for three reasons:

- it is byte-for-byte compatible with the plain dicts the data modules
  already return, so Tasks 1-9 need no rewrite;
- it is statically checkable in editors / type checkers;
- it is introspectable at runtime via `__required_keys__`, which
  `missing_keys` uses so tests can verify conformance without a runtime
  validation dependency.

LangGraph consumes a `TypedDict` state directly, so `PipelineState` plugs
into the graph (Task 18) without adaptation.

## How the pieces connect

```
data layer (Tasks 1-7)  ->  PipelineState inputs
   Filing, T0Window, Prices, Financials, news[], social[]

Stage 1 agents (10/11/14)  ->  fundamental_report, sentiment_report,
                               risk_assessment
Stage 2 Leader (15)        ->  leader_prediction (Prediction + rationale)
Stage 3 red team (16/17)   ->  rebuttals[], leader_responses[],
                               round_count, converged
Stage 4                    ->  final_prediction

to_ablation_record(state)  ->  one wide row for evaluation/metrics.py
                               and evaluation/significance.py (Tasks 8/9),
                               merged across variants for Task 19.
```

### Leakage anchor

`cutoff_timestamp_et` lives in the state and is the single timestamp every
tool-calling agent obeys, so fundamental / sentiment / risk agents all cut
their information off at the same T0 close (no per-agent drift).

### Red-team convergence cap

`should_continue_rebuttal` is the only gate on the rebuttal loop: it stops
when `converged` is true or `round_count` reaches `config.max_rounds`
(default 3). Centralising it guarantees the graph cannot loop unbounded.

## Usage

This module is part of the unified uv project at the repo root.

    uv run pytest contracts

All tests run offline. They assert the direction constants agree with
`evaluation/metrics.py`, that representative data-record shapes conform,
that the state helpers behave, and that `to_ablation_record` output feeds
straight into `evaluate_predictions` - proving the contract closes the
loop from orchestration to evaluation.

Agents import the contract by bare name (the repo's convention):

    from schemas import FundamentalReport, Prediction, RiskScore
    from state import PipelineState, new_state, should_continue_rebuttal
