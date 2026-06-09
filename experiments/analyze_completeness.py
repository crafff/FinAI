"""
CLI: break down an experiment's accuracy / price-error by data completeness.

Reads a finished experiment directory (runs/<name>/), groups its tickers by
which data sources were missing (from each data_context.json), and reports
directional accuracy + target-price error per group per system. Writes
`completeness_metrics.json` and `completeness_metrics.md` into the experiment
directory and prints the markdown table.

Usage from the repo root:
    uv run python experiments/analyze_completeness.py runs/<experiment_name>
    # or just the name (resolved under the configured runs dir):
    uv run python experiments/analyze_completeness.py <experiment_name>
"""

import argparse
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
from completeness import analyze_experiment, render_markdown  # noqa: E402


def _resolve_exp_dir(arg: str) -> Path:
    """Accept a path to the experiment dir, or a bare name under runs/."""
    path = Path(arg)
    if (path / "config.json").exists():
        return path

    runs_base = Path(load_settings().runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base
    candidate = runs_base / arg
    if (candidate / "config.json").exists():
        return candidate

    raise SystemExit(
        f"No experiment found at {path} or {candidate} "
        f"(expected a config.json inside)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Per-data-completeness accuracy/error breakdown."
    )
    parser.add_argument(
        "experiment",
        help="Path to runs/<name>/ (or just <name>, resolved under runs/).",
    )
    args = parser.parse_args()

    exp_dir = _resolve_exp_dir(args.experiment)
    report = analyze_experiment(exp_dir)

    (exp_dir / "completeness_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (exp_dir / "completeness_metrics.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Saved to {exp_dir}/completeness_metrics.{{json,md}}")


if __name__ == "__main__":
    main()
