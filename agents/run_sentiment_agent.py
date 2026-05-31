"""
Run the Sentiment agent (Task 11) standalone for one ticker.

This runner:
  1. takes a ticker and cutoff timestamp
  2. fetches cutoff-safe FinnHub news using Task 5
  3. fetches cutoff-safe Reddit posts using Task 6
  4. runs the Sentiment agent with an LLMClient
  5. saves transcript.json / transcript.md / sentiment_report.json

Usage from the repo root:

    uv run python agents/run_sentiment_agent.py AAPL 2025-11-03T16:00:00-05:00

In the full LangGraph pipeline, cutoff_timestamp_et should come from
the Task 2 T0Window stored in PipelineState.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _sub in (
    "config",
    "contracts",
    "data/FinnHub_retrieval",
    "data/reddit_retrieval",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from transcript import TranscriptRecorder, new_run_dir  # noqa: E402
from sentiment_agent import run_sentiment_agent  # noqa: E402
from finnhub_retrieval import (  # noqa: E402
    DEFAULT_NEWS_CACHE,
    fetch_company_news,
)
from reddit_retrieval import (  # noqa: E402
    DEFAULT_POSTS_CACHE,
    fetch_reddit_posts,
)


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    if len(sys.argv) > 2:
        cutoff_timestamp = datetime.fromisoformat(sys.argv[2])
    else:
        cutoff_timestamp = datetime.fromisoformat("2025-11-03T16:00:00-05:00")

    cfg = load_settings()

    missing = cfg.missing("finnhub", "reddit", "llm")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    client_id, client_secret, reddit_user_agent = cfg.require_reddit()

    print(f"[1/3] FinnHub news for {ticker} ...", flush=True)
    news = fetch_company_news(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        api_key=cfg.require_finnhub_api_key(),
        cache_dir=DEFAULT_NEWS_CACHE,
    )

    print(f"[2/3] Reddit posts for {ticker} ...", flush=True)
    social = fetch_reddit_posts(
        ticker=ticker,
        cutoff_timestamp=cutoff_timestamp,
        client_id=client_id,
        client_secret=client_secret,
        user_agent=reddit_user_agent,
        cache_dir=DEFAULT_POSTS_CACHE,
    )

    print(
        f"[3/3] Sentiment agent ({cfg.llm.backend}:{cfg.llm.model}) ...",
        flush=True,
    )

    recorder = TranscriptRecorder(metadata={
        "agent": "sentiment",
        "ticker": ticker,
        "cutoff_timestamp": cutoff_timestamp.isoformat(),
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "news_count": len(news),
        "social_count": len(social),
    })

    report = run_sentiment_agent(
        ticker=ticker,
        news=news,
        social=social,
        client=LLMClient(cfg.llm, recorder=recorder),
    )

    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, ticker)
    recorder.save(run_dir)

    (run_dir / "sentiment_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
