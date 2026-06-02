"""
Run the FULL coopetition pipeline (ablation variant "full") for one ticker.

This is the complete system end to end - the configuration the three-way
ablation (Task 19) compares against the single-agent and paper-style
baselines:

  1. read the cached 10-K text  (Task 1; run data/EDGAR_retrieval/run_fetch.py first)
  2. build / load the RAG index over it  (Task 7; needs the `rag` extra)
  3. Stage 1 - Fundamental (Task 10), Sentiment (Task 11), Qualitative risk
     (Task 12) analysts. The qualitative RiskScore is wrapped into a
     RiskAssessment via risk_assessment_from_score (INTERIM stand-in until
     Task 14's three-phase protocol exists - swap that one call then).
  4. Stage 2 - Leader aggregation (Task 15): initial Prediction.
  5. Stage 3 - red-team rebuttal loop (Tasks 16/17): the Evaluation agent
     attacks, the Leader revises or holds, capped by max_rounds.
  6. Stage 4 - final_prediction (the surviving prediction).
  7. compute T0 / target date (Task 2) and prices (Task 3): the baseline
     anchors the forecast, the target close is the answer key.

Everything threads through a PipelineState (contracts/state.py), so the run
projects directly into the evaluation row via to_ablation_record (variant
"full" -> full_direction / full_target_price).

Usage from the repo root:
    uv run --extra rag python agents/run_full_agent.py AAPL

Requires in .env: FMP_API_KEY, FINNHUB_API_KEY and the LLM backend
credential. REDDIT_* is optional (no-auth JSON fallback). SEC_USER_AGENT is
only needed earlier, to populate the EDGAR cache.
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
from state import new_state, to_ablation_record  # noqa: E402
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
from fundamental_agent import run_fundamental_agent  # noqa: E402
from sentiment_agent import run_sentiment_agent  # noqa: E402
from qualitative_risk_agent import run_qualitative_risk_agent  # noqa: E402
from leader_agent import risk_assessment_from_score, run_leader_agent  # noqa: E402
from redteam_loop import run_rebuttal_loop  # noqa: E402
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

    # Pre-flight: the data sources the agents consume + the LLM. Reddit
    # credentials are intentionally NOT required (no-auth JSON fallback).
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

    t0 = compute_t0(parse_acceptance_datetime(meta["filing_timestamp_et"]))
    cutoff_timestamp = t0["cutoff_timestamp_et"]

    print(f"[1/7] RAG index for {ticker} ({accession}) ...", flush=True)
    index = build_or_load_index(ticker, accession, text, cache_dir=RAG_CACHE)
    retrieval_tool = make_retrieval_tool(index)

    print(f"[2/7] Financials for {ticker} from FMP ...", flush=True)
    financials = fetch_financials(
        ticker, cfg.require_fmp_api_key(), cache_dir=DEFAULT_FINANCIALS_CACHE,
    )

    print(f"[3/7] FinnHub news for {ticker} ...", flush=True)
    news = fetch_company_news(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        api_key=cfg.require_finnhub_api_key(),
        cache_dir=DEFAULT_NEWS_CACHE,
    )

    print(f"[4/7] Reddit posts for {ticker} ...", flush=True)
    social = fetch_reddit_posts(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        client_id=cfg.reddit_client_id,
        client_secret=cfg.reddit_client_secret,
        user_agent=cfg.reddit_user_agent,
        cache_dir=DEFAULT_POSTS_CACHE,
    )

    print(f"[5/7] Prices for {ticker} ...", flush=True)
    prices = fetch_prices(ticker, t0["t0_date"], t0["target_date"])
    baseline_price = prices["baseline_price"]

    # One recorder + one client across every agent, so the saved transcript
    # captures the whole Stage-1 -> Leader -> red-team conversation.
    recorder = TranscriptRecorder(metadata={
        "agent": "full",
        "variant": "full",
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
    client = LLMClient(cfg.llm, recorder=recorder)

    print(f"[6/7] Stage-1 + Leader ({cfg.llm.backend}:{cfg.llm.model}) ...", flush=True)
    fundamental_report = run_fundamental_agent(
        ticker=ticker, financials=financials,
        retrieval_tool=retrieval_tool, client=client,
    )
    sentiment_report = run_sentiment_agent(
        ticker=ticker, news=news, social=social, client=client,
    )
    qual_score = run_qualitative_risk_agent(
        ticker=ticker, retrieval_tool=retrieval_tool, client=client,
    )
    # INTERIM: qualitative-only stand-in. Replace with Task 14 when ready.
    risk_assessment = risk_assessment_from_score(qual_score)

    leader_prediction = run_leader_agent(
        ticker=ticker,
        fundamental_report=fundamental_report,
        sentiment_report=sentiment_report,
        risk_assessment=risk_assessment,
        client=client,
        baseline_price=baseline_price,
    )

    # Seed the pipeline state and run the red-team loop on it.
    state = new_state(ticker, variant="full", model=cfg.llm.model)
    state.update({
        "prices": prices,
        "baseline_price": baseline_price,
        "fundamental_report": fundamental_report,
        "sentiment_report": sentiment_report,
        "risk_assessment": risk_assessment,
        "leader_prediction": leader_prediction,
    })

    print("[7/7] Red-team rebuttal loop ...", flush=True)
    run_rebuttal_loop(state, client)

    # Persist the run.
    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, f"{ticker}_full")
    recorder.save(run_dir)

    def _dump(name, obj):
        (run_dir / name).write_text(
            json.dumps(obj, indent=2, default=str), encoding="utf-8"
        )

    _dump("fundamental_report.json", fundamental_report)
    _dump("sentiment_report.json", sentiment_report)
    _dump("risk_assessment.json", risk_assessment)
    _dump("leader_prediction.json", leader_prediction)
    _dump("rebuttals.json", state["rebuttals"])
    _dump("leader_responses.json", state["leader_responses"])
    _dump("final_prediction.json", state["final_prediction"])
    _dump("ablation_record.json", to_ablation_record(state))

    print()
    print(json.dumps(state["final_prediction"], indent=2, default=str))
    print(
        f"\nrounds={state['round_count']} converged={state['converged']}"
    )
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
