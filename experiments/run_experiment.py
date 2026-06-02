"""
CLI for the experiment harness.

Runs every (ticker, system) cell described by a YAML config and writes the
results + metrics under runs/<experiment_name>/.

Usage from the repo root:
    uv run --extra rag python experiments/run_experiment.py experiments/configs/ablation.yaml
    uv run --extra rag python experiments/run_experiment.py <config.yaml> --tickers AAPL,MSFT
    uv run --extra rag python experiments/run_experiment.py <config.yaml> --no-resume

Requires in .env: FMP_API_KEY, FINNHUB_API_KEY and the LLM backend
credential. REDDIT_* is optional. 10-Ks must already be cached
(data/EDGAR_retrieval/run_fetch.py).
"""

import argparse
import sys
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
    "agents",
    "evaluation",
    "experiments",
):
    sys.path.insert(0, str(ROOT / _sub))

from settings import load_settings  # noqa: E402
from experiment_config import (  # noqa: E402
    TICKER_SETS,
    expand_ticker_set,
    load_experiment_config,
)
from runner import run_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a FinAI ablation experiment.")
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "--tickers",
        help="Override the config's tickers: a named set "
        f"({', '.join(TICKER_SETS)}) or a comma-separated list "
        "(handy for a quick single-stock run).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run every cell even if a cached record.json exists.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    if args.tickers:
        if args.tickers.strip().lower() in TICKER_SETS:
            config.tickers = expand_ticker_set(args.tickers)
        else:
            config.tickers = [
                t.strip().upper() for t in args.tickers.split(",") if t.strip()
            ]
    if args.no_resume:
        config.resume = False

    settings = load_settings()
    # With allow_missing, FMP/FinnHub are optional (their data degrades to
    # empty); only the LLM credential is always required.
    required = ["llm"] if config.allow_missing else ["fmp", "finnhub", "llm"]
    missing = settings.missing(*required)
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    run_experiment(config, settings)


if __name__ == "__main__":
    main()
