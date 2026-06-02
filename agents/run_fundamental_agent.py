"""
Run the Fundamental agent (Task 10) standalone for one ticker.

Ties the data layer to the agent end to end:

  1. read the cached 10-K text  (Task 1; run data/EDGAR_retrieval/run_fetch.py first)
  2. build / load the RAG index over it  (Task 7; needs the `rag` extra)
  3. fetch structured financials from FMP  (Task 4)
  4. run the agent with an LLMClient, recording the full transcript
  5. save transcript.json / transcript.md / report.json under RUNS_DIR
     and print the FundamentalReport as JSON

Usage from the repo root:
    uv run --extra rag python agents/run_fundamental_agent.py AAPL

Requires in .env: FMP_API_KEY and the LLM backend credential
(ANTHROPIC_API_KEY, or LOCAL_MODEL_* for a local/OpenAI-compatible backend).
SEC_USER_AGENT is only needed earlier, to populate the EDGAR cache.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


for _sub in (
    "config",
    "contracts",
    "data/EDGAR_retrieval",
    "data/rag_10k",
    "data/financial_retrieval",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from rag_10k import build_or_load_index, make_retrieval_tool  # noqa: E402
from financial_retrieval import (  # noqa: E402
    DEFAULT_FINANCIALS_CACHE,
    fetch_financials,
)
from llm_client import LLMClient  # noqa: E402
from fundamental_agent import run_fundamental_agent  # noqa: E402
from transcript import TranscriptRecorder, new_run_dir  # noqa: E402

EDGAR_CACHE = ROOT / ".cache" / "edgar"
RAG_CACHE = ROOT / ".cache" / "rag"


def _find_cached_filing(ticker):
    """Return (accession_number, text) for the cached 10-K, or None."""
    ticker_dir = EDGAR_CACHE / ticker.upper()

    metas = sorted(ticker_dir.glob("*.meta.json")) if ticker_dir.is_dir() else []

    if not metas:
        return None

    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    text_path = ticker_dir / f"{meta['accession_number']}.txt"

    if not text_path.exists():
        return None

    return meta["accession_number"], text_path.read_text(encoding="utf-8")


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    cfg = load_settings()

    # Pre-flight: FMP (financials) + the chosen LLM backend's credential.
    missing = cfg.missing("fmp", "llm")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    filing = _find_cached_filing(ticker)
    if filing is None:
        print(f"No cached 10-K for {ticker} under {EDGAR_CACHE}.")
        print("Run first: uv run python data/EDGAR_retrieval/run_fetch.py")
        sys.exit(1)

    accession, text = filing

    print(f"[1/3] RAG index for {ticker} ({accession}) ...", flush=True)
    index = build_or_load_index(ticker, accession, text, cache_dir=RAG_CACHE)
    retrieval_tool = make_retrieval_tool(index)

    print(f"[2/3] Financials for {ticker} from FMP ...", flush=True)
    financials = fetch_financials(
        ticker,
        cfg.require_fmp_api_key(),
        cache_dir=DEFAULT_FINANCIALS_CACHE,
    )

    print(
        f"[3/3] Fundamental agent ({cfg.llm.backend}:{cfg.llm.model}) ...",
        flush=True,
    )
    from datetime import datetime, timezone

    recorder = TranscriptRecorder(metadata={
        "agent": "fundamental",
        "ticker": ticker,
        "accession": accession,
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    report = run_fundamental_agent(
        ticker=ticker,
        financials=financials,
        retrieval_tool=retrieval_tool,
        client=LLMClient(cfg.llm, recorder=recorder),
    )

    # Persist the run: transcript (conversation + tool calls) + the report.
    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, ticker)
    recorder.save(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
