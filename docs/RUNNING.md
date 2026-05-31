# Running the pipeline — from setup to the Fundamental agent

This is an end-to-end runbook: install the environment, configure
credentials, prepare the data, and run the **Fundamental agent**
(Task 10) standalone for one ticker.

All commands are run from the repo root.

---

## 0. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (the project's package manager).
- Python 3.12 — `uv` will fetch it automatically if missing.
- Git, to clone the repo.

```bash
git clone <repo-url> FinAI
cd FinAI
```

---

## 1. Environment setup

The whole project is one unified `uv` project (see `pyproject.toml`).

```bash
uv sync                # base deps: data retrieval, evaluation, agents
uv sync --extra rag    # ALSO installs the embedding stack (torch, ~GB)
```

- `uv sync` installs everything except the RAG embedding model stack.
- `uv sync --extra rag` adds `sentence-transformers` (pulls in PyTorch,
  several hundred MB). **You need this extra** to build the 10-K RAG
  index, which the Fundamental agent depends on.

Sanity-check the install (all offline, no keys needed):

```bash
uv run pytest -q       # expect: passed, 1 skipped (RAG smoke test)
```

The one skipped test is the real-embedder smoke test; it runs only with
`--extra rag` installed.

---

## 2. Configure credentials

All secrets live in one `.env` file at the repo root.

```bash
cp .env.example .env    # then edit .env
```

`.env` is gitignored. Real shell environment variables override it.

### What the Fundamental agent path needs

| Variable | Needed for | Required here? |
|---|---|---|
| `SEC_USER_AGENT` | EDGAR 10-K download (step 3a) | ✅ (format: `"Name email"`) |
| `FMP_API_KEY` | structured financials (step 4) | ✅ (free: financialmodelingprep.com) |
| `ANTHROPIC_API_KEY` *or* `LOCAL_MODEL_*` | the LLM backbone | ✅ (one of them) |
| `FINNHUB_API_KEY`, `REDDIT_*` | sentiment agent (Task 11) | ❌ not for fundamental |

### Choosing the LLM backend

You can run the agent against a hosted API (Anthropic, OpenAI, DeepSeek, etc.)
or a local OpenAI-compatible model server (for example, a vLLM or Ollama
instance). The instructions and examples below show both approaches; if you
intend to run a model locally, see the appendices for detailed vLLM and
Ollama setup and notes.

**Hosted Anthropic (example):**

```dotenv
LLM_BACKEND=anthropic
LLM_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...
```

**Hosted OpenAI-compatible (example):**

```dotenv
# OpenAI
LLM_BACKEND=openai
LLM_MODEL=gpt-4o
LOCAL_MODEL_BASE_URL=https://api.openai.com/v1
LOCAL_MODEL_API_KEY=sk-...
```


**Local OpenAI-compatible servers (examples):**

You may also run an OpenAI-compatible inference server locally and point the
agent at it. This is useful when you want to self-host an open model or use a
local quantized build. Two supported local server options are vLLM and
Ollama.

For full, step-by-step setup (server flags, tool-call parser choices,
context-length recommendations, and troubleshooting), see the appendices
below: [Appendix: vLLM (Qwen2.5-7B-Instruct)](#appendix-vllm-qwen2.5-7b-instruct)
and [Appendix: Ollama](#appendix-ollama). The short examples below only
show the minimal `.env` variables to point FinAI at an already-running
server.

vLLM example (server running at http://localhost:8000/v1):

```dotenv
LLM_BACKEND=vllm
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_MODEL_BASE_URL=http://localhost:8000/v1
LOCAL_MODEL_API_KEY=vllm
```

Ollama example (server running at http://localhost:11434/v1):

```dotenv
LLM_BACKEND=local
LLM_MODEL=qwen2.5:7b
LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL_API_KEY=ollama
```

Note: the agent relies on OpenAI-style tool/function calling. Pick a model
and server configuration that reliably supports tool calling (see the vLLM
and Ollama appendices for parser flags and context-length recommendations).

See `config/README.md` for the full variable reference.

---

## 3. Data preparation

The Fundamental agent reads the **10-K** (via RAG) and **financials**.
Steps 3a–3b prepare the 10-K; financials are fetched at run time (step 4).

### 3a. Download the 10-K (Task 1, needs `SEC_USER_AGENT`)

```bash
uv run python data/EDGAR_retrieval/run_fetch.py
```

Downloads every Dow-30 FY2025 10-K into
`data/EDGAR_retrieval/cache/<TICKER>/` as `<accession>.{html,txt,meta.json}`.
Re-runs are fast (cached filings skip the network). Per-ticker failures
don't stop the run.

> Only need one company? The runner fetches all 30, but that's the
> simplest path and reruns are cheap. To fetch a single ticker instead,
> call `fetch_10k("AAPL", user_agent=...)` from a REPL (see
> `data/EDGAR_retrieval/README.md`).

### 3b. Build the RAG index (Task 7, needs `--extra rag`)

```bash
uv run --extra rag python data/rag_10k/run_build_indexes.py
```

Chunks and embeds every cached 10-K into
`data/rag_10k/cache/<TICKER>/`. The first run downloads the embedding
model (`BAAI/bge-small-en-v1.5`, ~130 MB) once. Already-built indexes are
loaded instantly.

---

## 4. Run the Fundamental agent (standalone)

One command ties it together — load the cached 10-K, build/load its RAG
index, fetch financials from FMP, and run the agent:

```bash
uv run --extra rag python agents/run_fundamental_agent.py AAPL
```

What it prints:

```
[1/3] RAG index for AAPL (0000320193-25-000079) ...
[2/3] Financials for AAPL from FMP ...
[3/3] Fundamental agent (anthropic:claude-opus-4-8) ...

{
  "ticker": "AAPL",
  "summary": "Strong margins and cash generation; ...",
  "signal": "bullish",
  "confidence": 0.72,
  "key_metrics": {"net_margin": 0.25, "free_cash_flow": 1.0e11},
  "citations": ["Item 7 MD&A", "net_margin"]
}
```

The output is a `FundamentalReport` (see `contracts/schemas.py`). The
agent runs the full tool-use loop: it decides when to call `search_10k`,
retrieves from the 10-K, and combines that with the financials.

### Calling it from Python instead

```python
import sys; sys.path[:0] = [
    "config", "contracts", "agents",
    "data/rag_10k", "data/financial_retrieval",
]
from pathlib import Path
from settings import load_settings
from rag_10k import build_or_load_index, make_retrieval_tool
from financial_retrieval import fetch_financials
from llm_client import LLMClient
from fundamental_agent import run_fundamental_agent

cfg = load_settings()
text = Path("data/EDGAR_retrieval/cache/AAPL/0000320193-25-000079.txt").read_text()
index = build_or_load_index("AAPL", "0000320193-25-000079", text,
                            cache_dir=Path("data/rag_10k/cache"))

report = run_fundamental_agent(
    ticker="AAPL",
    financials=fetch_financials("AAPL", cfg.require_fmp_api_key()),
    retrieval_tool=make_retrieval_tool(index),
    client=LLMClient(cfg.llm),
)
print(report)
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing config in .env: FMP_API_KEY, ANTHROPIC_API_KEY` | Fill those in `.env`. The message names exactly what's missing. |
| `No cached 10-K for AAPL ...` | Run step 3a (`run_fetch.py`) first. |
| `ModuleNotFoundError: sentence_transformers` | You skipped `--extra rag`. Use `uv run --extra rag ...`. |
| `MissingConfigError: ... SEC_USER_AGENT` (during 3a) | Set `SEC_USER_AGENT="Name email"` in `.env`. |
| Model ignores tools / returns no JSON | Use a tool-calling model (see the note in step 2). |
| FMP error / quota | Check `FMP_API_KEY`; the free tier has rate limits. |

---

## Summary (minimal happy path)

```bash
uv sync --extra rag
cp .env.example .env                      # set SEC_USER_AGENT, FMP_API_KEY, ANTHROPIC_API_KEY
uv run python data/EDGAR_retrieval/run_fetch.py
uv run --extra rag python data/rag_10k/run_build_indexes.py
uv run --extra rag python agents/run_fundamental_agent.py AAPL
```

---

<a id="appendix-vllm-qwen2.5-7b-instruct"></a>
## Appendix: running a local model with vLLM (Qwen2.5-7B-Instruct)

How to self-host an open model as the OpenAI-compatible backend. vLLM is
a **separate inference server** — install it in its own environment (keep
it out of the FinAI project deps); it exposes an OpenAI-compatible API
that FinAI connects to via `.env`.

### Picking the model

The agent runs a tool-use loop, so the **first** requirement is reliable
OpenAI-style tool calling, then financial reasoning + clean JSON. Qwen
(Instruct) is the best-supported open choice in vLLM. By GPU:

| VRAM | Model |
|---|---|
| 24 GB (4090 / A5000) | `Qwen2.5-14B-Instruct` or `Qwen3-14B` |
| 48 GB (A6000 / L40S) | `Qwen2.5-32B-Instruct` (best balance) |
| 2×80 GB (A100/H100) | `Qwen2.5-72B-Instruct` / `Llama-3.3-70B-Instruct` |
| save VRAM | `Qwen3-30B-A3B` (MoE, ~3B active) |
| smallest viable | `Qwen2.5-7B-Instruct` (used below) |

Use the **Instruct** variant (not base); avoid pure reasoning models
(e.g. `deepseek-reasoner`) for tool calling.

### 0. Check the GPU

```bash
nvidia-smi      # need an NVIDIA GPU + CUDA driver
```

Qwen2.5-7B in bf16 needs ~15 GB. ≥24 GB is comfortable; ~16 GB needs a
smaller context or a quantized model (see below).

### 1. Install vLLM (its own environment)

```bash
uv venv ~/.venvs/vllm --python 3.12
source ~/.venvs/vllm/bin/activate
uv pip install vllm
```

### 2. Start the server — tool parsing MUST be enabled

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 16384
```

- First launch downloads ~15 GB of weights to `~/.cache/huggingface`
  (Qwen is open — **no HF token needed**).
- Keep this terminal running; it *is* the server.
- `--enable-auto-tool-choice --tool-call-parser hermes` is what makes
  Qwen emit tool calls. **Without them the agent's tool loop breaks.**
  (Qwen3 uses `hermes` too; Llama uses `llama3_json`; Mistral uses
  `mistral`.)
- **≤16 GB VRAM?** Use the quantized model and a smaller context:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.92
  ```

### 3. Verify the server (new terminal)

```bash
curl http://localhost:8000/v1/models
```

### 4. Point FinAI at it — `.env`

```dotenv
LLM_BACKEND=vllm
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_MODEL_BASE_URL=http://localhost:8000/v1
LOCAL_MODEL_API_KEY=vllm
```

`LLM_MODEL` must exactly match the name vLLM serves (the HF repo id by
default, as shown by `/v1/models`; for the AWQ build use
`Qwen/Qwen2.5-7B-Instruct-AWQ`).

### 5. Run the agent

From the FinAI project dir (its own uv env — separate from the vLLM venv,
which is intended):

```bash
uv run --extra rag python agents/run_fundamental_agent.py AAPL
```

### Other backends

If vLLM won't start (old driver, too little VRAM) or you have no GPU, see
the two appendices below: **Ollama** (the easiest path for older drivers /
small or multiple GPUs) and **troubleshooting vLLM startup**.

---

<a id="appendix-ollama"></a>
## Appendix: Ollama (older drivers, small / multiple GPUs)

Ollama bundles its own CUDA runtime (tolerant of older NVIDIA drivers),
serves quantized GGUF models (a 7B fits in ~5 GB), exposes an
OpenAI-compatible API, and **auto-shards a model across multiple GPUs**
when it doesn't fit on one. This makes it the most robust local option on
modest hardware (e.g. RTX 2080 Ti, 11 GB).

### Install + serve

```bash
curl -fsSL https://ollama.com/install.sh | sh

# key env vars, then start the server (keep this terminal running)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OLLAMA_CONTEXT_LENGTH=16384 \
OLLAMA_KEEP_ALIVE=30m \
ollama serve
```

- `OLLAMA_CONTEXT_LENGTH=16384` — **important.** The agent feeds the
  financials JSON + retrieved 10-K chunks + tool results into the context;
  Ollama's small default truncates them, degrading or breaking tool use.
- `OLLAMA_KEEP_ALIVE=30m` — keep the model resident between agent runs.
- `CUDA_VISIBLE_DEVICES` — pick which GPUs (default: all). A model that
  doesn't fit on one card is auto-sharded across cards; add
  `OLLAMA_SCHED_SPREAD=1` only to force spreading (unnecessary, and slower,
  for a model that already fits on fewer cards).

### Pick a model by VRAM (Ollama defaults to Q4)

Multiple 11 GB cards combine their VRAM, so you can run a bigger,
better-at-tool-calling model than 7B (tool calling is this agent's
bottleneck):

| VRAM available | Model | Q4 size |
|---|---|---|
| 1×11 GB | `qwen2.5:7b` | ~4.7 GB |
| 2×11 GB | `qwen2.5:14b` | ~9 GB |
| 3-4×11 GB | `qwen2.5:32b` | ~20 GB |

```bash
ollama pull qwen2.5:32b      # or 14b / 7b per the table
```

### Configure + run

```dotenv
LLM_BACKEND=local
LLM_MODEL=qwen2.5:32b
LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL_API_KEY=ollama
```

`LLM_MODEL` must match `ollama list`.

```bash
curl http://localhost:11434/v1/models     # lists the model -> OK
nvidia-smi                                # model spread across the GPUs
uv run --extra rag python agents/run_fundamental_agent.py AAPL
```

Prefer one bigger model over many parallel copies: tool-calling quality
matters more than throughput here, and running Dow-30 sequentially is
fine. If the agent fails to call `search_10k`, first confirm the context
length, then move up a size (7B → 14B → 32B).

---

## Appendix: troubleshooting vLLM startup

### "The NVIDIA driver on your system is too old"

vLLM ships a torch built for a recent CUDA (e.g. 12.8+). If your driver is
older (e.g. CUDA 12.6, reported as `found version 12060`) the engine won't
start. Options, easiest first:

1. **Use Ollama instead** (appendix above) — it bundles its own CUDA, so
   the system driver version stops mattering.
2. **Use a hosted API** — no GPU at all; one `.env` change (see step 2,
   "OpenAI-compatible"), e.g. DeepSeek / OpenAI / Anthropic.
3. **Downgrade vLLM** to a release whose bundled torch targets your CUDA
   (≤ your driver's version), then check the torch build it pulls:
   ```bash
   uv pip install "vllm==0.9.2"
   ```
   This is fiddly version/CUDA matchmaking — prefer options 1-2.

### Not enough VRAM (model won't fit)

A 7B model in fp16 needs ~15 GB, so it won't fit an 11 GB card even with a
fine driver. Either:

- use **Ollama** (Q4 GGUF ~5 GB; auto-shards across multiple cards), or
- with vLLM, use a quantized model + smaller context, and split across
  cards with tensor parallelism:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 2
  ```
  Set `LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ` to match.

