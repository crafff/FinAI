"""
Run the Qualitative Risk Agent (Task 12) standalone for one ticker.

Usage from the repo root:

    uv run --extra rag python agents/run_qualitative_risk_agent.py AAPL
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _sub in (
    "config",
    "contracts",
    "data/EDGAR_retrieval",
    "data/rag_10k",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from rag_10k import build_or_load_index, make_retrieval_tool  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from transcript import TranscriptRecorder, new_run_dir  # noqa: E402
from qualitative_risk_agent import run_qualitative_risk_agent  # noqa: E402


EDGAR_CACHE = ROOT / ".cache" / "edgar"
RAG_CACHE = ROOT / ".cache" / "rag"


def _find_cached_filing(ticker):
    """
    Return (accession_number, text) for the cached 10-K, or None.
    """
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

    missing = cfg.missing("llm")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    filing = _find_cached_filing(ticker)
    if filing is None:
        print(f"No cached 10-K for {ticker} under {EDGAR_CACHE}.")
        print("Run first: uv run python data/EDGAR_retrieval/run_fetch.py")
        sys.exit(1)

    accession, text = filing

    print(f"[1/2] RAG index for {ticker} ({accession}) ...", flush=True)
    index = build_or_load_index(
        ticker,
        accession,
        text,
        cache_dir=RAG_CACHE,
    )

    retrieval_tool = make_retrieval_tool(index)

    print(
        f"[2/2] Qualitative risk agent ({cfg.llm.backend}:{cfg.llm.model}) ...",
        flush=True,
    )

    recorder = TranscriptRecorder(metadata={
        "agent": "qualitative_risk",
        "ticker": ticker,
        "accession": accession,
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    score = run_qualitative_risk_agent(
        ticker=ticker,
        retrieval_tool=retrieval_tool,
        client=LLMClient(cfg.llm, recorder=recorder),
    )

    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, ticker)
    recorder.save(run_dir)

    (run_dir / "qualitative_risk_score.json").write_text(
        json.dumps(score, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(json.dumps(score, indent=2, default=str))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
