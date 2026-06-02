"""
Run the single-agent baseline (Task 19, variant "single") for one ticker.

This is the simplest ablation configuration: one agent sees all the
evidence and emits the final Prediction directly. The runner wires the
whole data layer to it end to end:

  1. read the cached 10-K text  (Task 1; run data/EDGAR_retrieval/run_fetch.py first)
  2. build / load the RAG index over it  (Task 7; needs the `rag` extra)
  3. fetch structured financials from FMP  (Task 4)
  4. fetch cutoff-safe news (FinnHub, Task 5) and social posts (Reddit, Task 6)
  5. compute T0 / target date (Task 2) and fetch prices (Task 3) for the
     baseline anchor and the answer-key target price
  6. run the single agent with an LLMClient, recording the full transcript
  7. save transcript.json / transcript.md / prediction.json / ablation_record.json
     under RUNS_DIR and print the Prediction as JSON

The baseline (T0 close) is shown to the agent as the prediction anchor.
The actual target-date close is the answer key: it is written only to
ablation_record.json for evaluation and is never passed to the agent.

Usage from the repo root:
    uv run --extra rag python agents/run_single_agent.py AAPL

Requires in .env: FMP_API_KEY, FINNHUB_API_KEY, REDDIT_* and the LLM
backend credential. SEC_USER_AGENT is only needed earlier, to populate
the EDGAR cache.
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
    "data/financial_retrieval",
    "data/FinnHub_retrieval",
    "data/reddit_retrieval",
    "data/t0_logic",
    "data/price_retrieval",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from rag_10k import build_or_load_index, make_retrieval_tool  # noqa: E402
from financial_retrieval import (  # noqa: E402
    DEFAULT_FINANCIALS_CACHE,
    fetch_financials,
)
from finnhub_retrieval import DEFAULT_NEWS_CACHE, fetch_company_news  # noqa: E402
from reddit_retrieval import DEFAULT_POSTS_CACHE, fetch_reddit_posts  # noqa: E402
from t0_logic import compute_t0  # noqa: E402
from price_retrieval import fetch_prices  # noqa: E402
from edgar_retrieval import parse_acceptance_datetime  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from single_agent import run_single_agent  # noqa: E402
from transcript import TranscriptRecorder, new_run_dir  # noqa: E402

EDGAR_CACHE = ROOT / "data" / "EDGAR_retrieval" / "cache"
RAG_CACHE = ROOT / "data" / "rag_10k" / "cache"


def _find_cached_filing(ticker):
    """Return (meta, text) for the cached 10-K, or None."""
    ticker_dir = EDGAR_CACHE / ticker.upper()

    metas = sorted(ticker_dir.glob("*.meta.json")) if ticker_dir.is_dir() else []

    if not metas:
        return None

    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    text_path = ticker_dir / f"{meta['accession_number']}.txt"

    if not text_path.exists():
        return None

    return meta, text_path.read_text(encoding="utf-8")


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    cfg = load_settings()

    # Pre-flight: every data source the single agent consumes + the LLM.
    # Reddit credentials are intentionally NOT required: without them the
    # social fetch falls back to Reddit's no-auth JSON endpoint.
    missing = cfg.missing("fmp", "finnhub", "llm")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    filing = _find_cached_filing(ticker)
    if filing is None:
        print(f"No cached 10-K for {ticker} under {EDGAR_CACHE}.")
        print("Run first: uv run python data/EDGAR_retrieval/run_fetch.py")
        sys.exit(1)

    meta, text = filing
    accession = meta["accession_number"]

    # T0 / target date and prices: baseline anchors the forecast, the target
    # close is the answer key (kept out of everything the agent sees).
    t0 = compute_t0(parse_acceptance_datetime(meta["filing_timestamp_et"]))
    cutoff_timestamp = t0["cutoff_timestamp_et"]

    print(f"[1/6] RAG index for {ticker} ({accession}) ...", flush=True)
    index = build_or_load_index(ticker, accession, text, cache_dir=RAG_CACHE)
    retrieval_tool = make_retrieval_tool(index)

    print(f"[2/6] Financials for {ticker} from FMP ...", flush=True)
    financials = fetch_financials(
        ticker,
        cfg.require_fmp_api_key(),
        cache_dir=DEFAULT_FINANCIALS_CACHE,
    )

    print(f"[3/6] FinnHub news for {ticker} ...", flush=True)
    news = fetch_company_news(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        api_key=cfg.require_finnhub_api_key(),
        cache_dir=DEFAULT_NEWS_CACHE,
    )

    # With credentials -> authenticated PRAW; without -> no-auth JSON
    # endpoint (backend="auto" picks based on what's set in .env).
    print(f"[4/6] Reddit posts for {ticker} ...", flush=True)
    social = fetch_reddit_posts(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        client_id=cfg.reddit_client_id,
        client_secret=cfg.reddit_client_secret,
        user_agent=cfg.reddit_user_agent,
        cache_dir=DEFAULT_POSTS_CACHE,
    )

    print(f"[5/6] Prices for {ticker} ...", flush=True)
    prices = fetch_prices(ticker, t0["t0_date"], t0["target_date"])
    baseline_price = prices["baseline_price"]

    print(
        f"[6/6] Single agent ({cfg.llm.backend}:{cfg.llm.model}) ...",
        flush=True,
    )

    recorder = TranscriptRecorder(metadata={
        "agent": "single",
        "variant": "single",
        "ticker": ticker,
        "accession": accession,
        "cutoff_timestamp": str(cutoff_timestamp),
        "baseline_price": baseline_price,
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "news_count": len(news),
        "social_count": len(social),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    prediction = run_single_agent(
        ticker=ticker,
        financials=financials,
        news=news,
        social=social,
        retrieval_tool=retrieval_tool,
        client=LLMClient(cfg.llm, recorder=recorder),
        baseline_price=baseline_price,
    )

    # Persist the run: transcript + the prediction + an evaluation row.
    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, f"{ticker}_single")
    recorder.save(run_dir)
    (run_dir / "prediction.json").write_text(
        json.dumps(prediction, indent=2, default=str), encoding="utf-8"
    )

    # One wide row for evaluation/metrics.py (variant-prefixed columns,
    # matching contracts.state.to_ablation_record's "single" projection).
    ablation_record = {
        "ticker": ticker,
        "baseline_price": baseline_price,
        "actual_target_price": prices["target_price"],
        "single_direction": prediction["direction"],
        "single_target_price": prediction["target_price"],
    }
    (run_dir / "ablation_record.json").write_text(
        json.dumps(ablation_record, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(json.dumps(prediction, indent=2, default=str))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
