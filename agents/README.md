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

## Tests

```bash
uv run pytest agents
```

All offline. The pure converters/parsers are tested with hand-built
dicts; the tool loop and the agent are tested with a scripted client
(returning canned `LLMResponse`s) and a fake retrieval tool, so no SDK,
no API key, and no network are needed.
