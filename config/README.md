# Configuration (`config/settings.py`)

One place to manage every secret and environment-specific setting, so
anyone who checks out the code can supply their own keys and run it.

## Quick start

```bash
cp .env.example .env     # then edit .env and fill in your keys
uv sync
uv run pytest            # passes with an empty .env (no keys needed)
```

`.env` is gitignored. `.env.example` is the committed template listing
every variable.

## How it works

- **Source of truth: environment variables.** `load_settings()` reads
  them, optionally seeding from `.env` at the repo root first (via
  `python-dotenv`). Real shell / CI environment variables override `.env`.

- **Nothing is required at load time.** `load_settings()` always
  succeeds; unset values are `None`. A value is only *required* when a
  code path needs it, via a `require_*` accessor that raises a clear
  `MissingConfigError` naming the variable and how to set it. This is why
  the offline test suite is green with no keys configured, while a real
  network/LLM call without its credential fails with a precise message.

## Usage

```python
from settings import load_settings

cfg = load_settings()

# Data tools (Tasks 1, 4, 5, 6)
fetch_10k(ticker, user_agent=cfg.require_sec_user_agent())
fetch_financials(ticker, api_key=cfg.require_fmp_api_key())
fetch_company_news(ticker, cutoff, api_key=cfg.require_finnhub_api_key())
client_id, client_secret, user_agent = cfg.require_reddit()

# LLM backbone (agents, Tasks 10-17)
llm = cfg.llm                       # LLMConfig
model = llm.model                   # model id to call
api_key = llm.require_api_key()     # validates per backend
base_url = llm.require_base_url()   # endpoint URL, or Anthropic default
```

### Pre-flight check in a runner

To fail fast before a long loop instead of midway through:

```python
cfg = load_settings()
missing = cfg.missing("sec", "fmp")
if missing:
    print("Set these in .env:", ", ".join(missing))
    sys.exit(1)
```

## Variables

| Variable | Used by | Notes |
|---|---|---|
| `SEC_USER_AGENT` | Task 1 (EDGAR) | SEC requires `"Name email"`. |
| `FMP_API_KEY` | Task 4 (FMP) | financialmodelingprep.com. |
| `FINNHUB_API_KEY` | Task 5 (news) | finnhub.io. |
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | Task 6 (Reddit) | reddit.com/prefs/apps, app type "script". |
| `LLM_BACKEND` | agents | `anthropic`, or OpenAI-compatible (`local`/`openai`/`deepseek`/`vllm`). Default `anthropic`. |
| `LLM_MODEL` | agents | Default `claude-opus-4-8` / `llama3.1`. |
| `ANTHROPIC_API_KEY` | agents (anthropic) | Hosted API key. |
| `LOCAL_MODEL_BASE_URL` / `LOCAL_MODEL_API_KEY` | agents (OpenAI-compat) | local vLLM or a remote API (OpenAI / DeepSeek). |

## LLM backends

The agents support two kinds of backbone, selected by `LLM_BACKEND`:

- **`anthropic`** — the hosted Claude API. Needs `ANTHROPIC_API_KEY`.
  `LLM_MODEL` defaults to `claude-opus-4-8`.
- **OpenAI-compatible** — set `LLM_BACKEND` to `local`, `openai`,
  `deepseek`, or `vllm` (all equivalent). Routes through the OpenAI SDK at
  `LOCAL_MODEL_BASE_URL`, so the endpoint can be a local **vLLM** or a
  remote API (**OpenAI**, **DeepSeek**). Needs `LOCAL_MODEL_BASE_URL`
  (defaults to vLLM `http://localhost:8000/v1`) and `LOCAL_MODEL_API_KEY`.
  See `.env.example` for OpenAI / DeepSeek / Qwen examples. The model must
  support OpenAI-style tool calling (the agents use tools).

`config` only *describes* the backend; the actual client
(`agents/llm_client.py`) constructs the Anthropic or OpenAI-compatible
client from it (`backend`, `model`, key, base URL).

## Tests

```bash
uv run pytest config
```

All offline. They set the environment via `monkeypatch` and load with
`use_dotenv=False`, so a developer's real `.env` never affects the tests.
