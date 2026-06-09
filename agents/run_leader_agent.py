"""
Thin wrapper: run Stage 1 -> Leader (no red-team) for one ticker.

Equivalent to a one-ticker experiment with a single `mode: leader` system
(all three sub-task agents, red-team loop disabled), so the saved
final_prediction is the Leader's initial call. All wiring lives in the
experiment harness (experiments/); for multi-ticker / multi-system runs use
experiments/run_experiment.py.

Usage from the repo root:
    uv run --extra rag python agents/run_leader_agent.py AAPL

Requires in .env: FMP_API_KEY, FINNHUB_API_KEY and the LLM backend credential
(REDDIT_* optional). The 10-K must already be cached.
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
        name="leader_agent",
        tickers=[ticker],
        systems=[SystemConfig(
            "leader",
            mode="leader",
            subtasks=["fundamental", "sentiment", "risk"],
            red_team=False,
        )],
        resume=False,
    )
    run_experiment(config, settings)


if __name__ == "__main__":
    main()
