"""
Thin wrapper: run the single-agent baseline (variant "single") for one ticker.

This is equivalent to a one-ticker experiment with a single `mode: single`
system; all the data loading, agent wiring, and result saving live in the
experiment harness (experiments/). For multi-ticker or multi-system runs use
experiments/run_experiment.py with a YAML config.

Usage from the repo root:
    uv run --extra rag python agents/run_single_agent.py AAPL

Requires in .env: FMP_API_KEY, FINNHUB_API_KEY and the LLM backend credential
(REDDIT_* optional). The 10-K must already be cached
(data/EDGAR_retrieval/run_fetch.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _sub in (
    "config", "contracts",
    "data/EDGAR_retrieval", "data/rag_10k", "data/financial_retrieval",
    "data/FinnHub_retrieval", "data/reddit_retrieval", "data/t0_logic",
    "data/price_retrieval", "agents", "evaluation", "experiments",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from experiment_config import ExperimentConfig, SystemConfig  # noqa: E402
from runner import run_experiment  # noqa: E402


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    settings = load_settings()
    missing = settings.missing("fmp", "finnhub", "llm")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    config = ExperimentConfig(
        name="single_agent",
        tickers=[ticker],
        systems=[SystemConfig("single", mode="single")],
        resume=False,
    )
    run_experiment(config, settings)


if __name__ == "__main__":
    main()
