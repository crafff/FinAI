# Agents

The multi-agent layer (Stages 1-4). Each agent is built against the
shared contract (`contracts/schemas.py`) and the unified LLM client here.

## `llm_client.py` — unified LLM client

One tool-calling interface over two backends, selected by
`config.LLMConfig` (`LLM_BACKEND`):

- **`anthropic`** — hosted Anthropic Messages API.
- **`local`** — any OpenAI-compatible server (this project uses **vLLM**)
  via the `openai` SDK.

The two providers have different request/response shapes for tool use, so
the client keeps a single backend-agnostic message format and converts at
the edges:

```
internal messages  --(pure)-->  provider request  --(boundary)-->  SDK call
provider response dict  --(pure)-->  LLMResponse
```

- **Pure functions** (`to_anthropic_*` / `to_openai_*` / `build_*_request`
  / `parse_*_response`) do all the conversion and are unit-tested with
  plain dicts.
- **Network boundaries** (`LLMClient._raw_anthropic` / `_raw_local`)
  import the SDK and return `.model_dump()`, so parsers only ever see
  dicts. These are the only un-mocked parts.
- **`run_tool_loop`** drives a multi-turn tool-use conversation until the
  model answers without calling a tool, or an iteration cap is hit (then
  one final tool-less call forces an answer). A failing/unknown tool
  returns an error string to the model instead of raising.

### Key types

- `Tool(name, description, parameters, impl)` — a JSON-schema tool plus
  its Python implementation.
- `ToolCall`, `ToolResult`, `LLMResponse` — normalized request/response
  pieces.

### Usage

```python
from settings import load_settings
from llm_client import LLMClient

client = LLMClient(load_settings().llm)
response = client.complete(
    [{"role": "user", "text": "Hello"}],
    system="You are helpful.",
)
```

## `fundamental_agent.py` — Fundamental agent (Task 10)

A single tool-equipped agent (Stage 1, cooperative). It investigates the
10-K through the RAG tool and combines it with the structured financials
(Task 4) to emit a `FundamentalReport`.

```python
from rag_10k import make_retrieval_tool
from financial_retrieval import fetch_financials
from llm_client import LLMClient
from fundamental_agent import run_fundamental_agent
from settings import load_settings

cfg = load_settings()
retrieval_tool = make_retrieval_tool(index)          # bound to one 10-K
financials = fetch_financials("AAPL", cfg.require_fmp_api_key())

report = run_fundamental_agent(
    ticker="AAPL",
    financials=financials,
    retrieval_tool=retrieval_tool,
    client=LLMClient(cfg.llm),
)
```

The agent runs the full tool-use loop (it decides when/what to retrieve),
then its final message is parsed into a `FundamentalReport`. Parsing is
lenient: it tolerates ```json fences / surrounding prose, coerces an
out-of-range signal to `neutral`, and clamps confidence to `[0, 1]`.

**Leakage:** the agent fetches no live data. The 10-K is the filing and
the financials are already cut off at the fiscal year by the data layer,
so everything visible predates the T0 cutoff.

## `leader_agent.py` — Leader aggregation agent (Task 15)

Stage 2. The Leader reads the three Stage-1 reports — `FundamentalReport`
(Task 10), `SentimentReport` (Task 11), and a `RiskAssessment` (Task 14) —
and makes a **free-judgment** initial prediction with a **mandatory
rationale**: a `Prediction` (buy/not-buy + one-week target price). It uses
**no tools and does no retrieval** (the subtask agents already did that), so
it is a single `client.complete` call whose output is parsed by the single
agent's `parse_prediction`.

```python
from leader_agent import risk_assessment_from_score, run_leader_agent

# Interim: Tasks 13/14 (quantitative risk + three-phase protocol) are not
# built yet, so wrap the Task-12 qualitative RiskScore into a one-score
# RiskAssessment. The Leader is written against the final RiskAssessment
# contract, so when Task 14 lands you pass its real (two-score) assessment
# here instead — the Leader, its prompt, and its parser do not change.
risk_assessment = risk_assessment_from_score(qual_score)

prediction = run_leader_agent(
    ticker="AAPL",
    fundamental_report=fundamental_report,
    sentiment_report=sentiment_report,
    risk_assessment=risk_assessment,
    client=LLMClient(cfg.llm),
    baseline_price=baseline_price,
)
```

`build_risk_evidence` renders `risk_assessment["scores"]` as a list, so one
score (qualitative-only) and two scores (qualitative + quantitative) are
handled identically — the future swap is a one-line runner change. The
end-to-end runner is `run_leader_agent.py` (Stage-1 agents → adapter →
Leader, saving one combined transcript plus all reports).

**Leakage:** the only price the Leader sees is the T0 baseline close (the
allowed forecast anchor); the actual target-date close is never passed in.

## `redteam_agent.py` — Red-team / Evaluation agent (Task 16)

Stage 3, adversarial. Given the Leader's `Prediction` and the same three
Stage-1 reports, it attacks the single most vulnerable claim and emits a
`Rebuttal` (`targeted_claim`, `objections`, `severity`). It uses **no tools
and gathers no new data** — it only critiques what is on the table, so it
cannot smuggle in post-cutoff information. The report compactors are reused
from `leader_agent` so the red team sees exactly what the Leader saw.

## `redteam_loop.py` — rebuttal–revision loop + convergence cap (Task 17)

`run_rebuttal_loop(state, client)` drives the Leader↔red-team exchange on the
shared `PipelineState`. Each round: the red team raises a `Rebuttal`, then the
Leader replies with a `LeaderResponse` (`leader_agent.run_leader_response`) —
either **accepting** (issuing a revised `Prediction`) or **holding** its
position with a stated reason.

The loop is bounded by `contracts/state.py:should_continue_rebuttal`, the
single place the hard round cap (`max_rounds`, default 3) is enforced.
Convergence is reached when the Leader holds (positions stable) or the red
team raises no objections; otherwise the standing prediction is the latest
revision. The surviving prediction is written to `final_prediction`. This is
the exact control flow Item 18 (LangGraph) will wrap — each `run_*` a node,
the loop guard a conditional edge.

## `run_full_agent.py` — full coopetition pipeline (ablation variant "full")

Wires Stage 1 → Leader → red-team loop end to end on a `PipelineState`,
seeding the data/Stage-1 keys and projecting the finished run with
`to_ablation_record` (→ `full_direction` / `full_target_price`). Saves the
full transcript plus the three reports, `rebuttals.json`,
`leader_responses.json`, `final_prediction.json`, and `ablation_record.json`.

```bash
uv run --extra rag python agents/run_full_agent.py AAPL
```

Risk input is still the qualitative-only stand-in
(`risk_assessment_from_score`); swapping in Task 14's `RiskAssessment` is the
same one-line change as in `run_leader_agent.py`.

## Tests

```bash
uv run pytest agents
```

All offline. The pure converters/parsers are tested with hand-built
dicts; the tool loop and the agent are tested with a scripted client
(returning canned `LLMResponse`s) and a fake retrieval tool, so no SDK,
no API key, and no network are needed.
