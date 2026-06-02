"""
Experiment orchestration: run every (ticker, system) cell and aggregate.

`run_experiment` loads each ticker's DataContext once, runs every system on
it, saves per-cell artifacts, and then merges all records into the wide
ablation table + metrics. It is robust for multi-stock sweeps: one failing
cell is logged and skipped, never aborting the run, and completed cells are
skipped on re-run (resume).

Output tree (under runs/<experiment_name>/):

    config.json            resolved config snapshot
    results.csv            wide ablation table (one row per ticker)
    metrics.json / .md     per-system accuracy + Wilson CI + MAPE
    compare.json           pairwise significance + correlated-error (>= 2 systems)
    summary.json           headline numbers + run metadata + errors
    errors.log
    per_ticker/<TICKER>/
        data_context.json
        <system_name>/  transcript.{json,md}, subtask_reports/<name>.json,
                        leader_prediction.json, rebuttals.json,
                        leader_responses.json, final_prediction.json,
                        record.json, meta.json
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from settings import load_settings
from llm_client import LLMClient
from transcript import TranscriptRecorder

from experiment_config import ExperimentConfig, to_jsonable
from context import build_data_context, context_summary
from pipeline import run_system
from results import build_wide_df, compute_metrics, record_for_system


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _llm_config(settings, config: ExperimentConfig):
    overrides = {}
    if config.model:
        overrides["model"] = config.model
    if config.backend:
        overrides["backend"] = config.backend
    return replace(settings.llm, **overrides) if overrides else settings.llm


def _save_cell(cell_dir: Path, system, state, recorder) -> None:
    """Persist one finished (ticker, system) run."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    recorder.save(cell_dir)

    if state.get("subtask_reports"):
        sub_dir = cell_dir / "subtask_reports"
        sub_dir.mkdir(exist_ok=True)
        for name, report in state["subtask_reports"].items():
            _dump(sub_dir / f"{name}.json", report)

    if "leader_prediction" in state:
        _dump(cell_dir / "leader_prediction.json", state["leader_prediction"])
    if state.get("rebuttals"):
        _dump(cell_dir / "rebuttals.json", state["rebuttals"])
    if state.get("leader_responses"):
        _dump(cell_dir / "leader_responses.json", state["leader_responses"])

    _dump(cell_dir / "final_prediction.json", state["final_prediction"])

    meta = {
        "system": system.name,
        "mode": system.mode,
        "subtasks": system.subtasks,
        "red_team": system.red_team,
        "max_rounds": system.max_rounds,
        "round_count": state.get("round_count", 0),
        "converged": state.get("converged", False),
    }
    _dump(cell_dir / "meta.json", meta)


def run_experiment(config: ExperimentConfig, settings=None) -> Path:
    """Run the whole experiment; return the experiment output directory."""
    settings = settings or load_settings()
    llm_config = _llm_config(settings, config)

    runs_base = Path(settings.runs_dir)
    if not runs_base.is_absolute():
        runs_base = Path(__file__).resolve().parents[1] / runs_base

    # Stable per-experiment dir (no timestamp) so re-runs can resume.
    exp_dir = runs_base / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    _dump(exp_dir / "config.json", to_jsonable(config))

    errors_path = exp_dir / "errors.log"
    rows: list[dict] = []
    errors: list[dict] = []
    started = time.monotonic()

    for ticker in config.tickers:
        ticker_dir = exp_dir / "per_ticker" / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        try:
            ctx = build_data_context(ticker, settings, allow_missing=config.allow_missing)
            _dump(ticker_dir / "data_context.json", context_summary(ctx))
        except Exception as exc:  # noqa: BLE001 - one ticker must not abort the sweep
            msg = f"[{ticker}] data context: {type(exc).__name__}: {exc}"
            errors.append({"ticker": ticker, "stage": "data_context", "error": str(exc)})
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n" + traceback.format_exc() + "\n")
            print(msg, flush=True)
            continue

        for system in config.systems:
            cell_dir = ticker_dir / system.name
            record_path = cell_dir / "record.json"

            if config.resume and record_path.exists():
                rows.append(json.loads(record_path.read_text(encoding="utf-8")))
                print(f"[{ticker}/{system.name}] resume: cached", flush=True)
                continue

            try:
                recorder = TranscriptRecorder(metadata={
                    "experiment": config.name,
                    "system": system.name,
                    "ticker": ticker,
                    "mode": system.mode,
                    "subtasks": system.subtasks,
                    "red_team": system.red_team,
                    "max_rounds": system.max_rounds,
                    "backend": llm_config.backend,
                    "model": llm_config.model,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                })
                client = LLMClient(llm_config, recorder=recorder)

                state = run_system(system, ctx, client)
                _save_cell(cell_dir, system, state, recorder)

                record = record_for_system(state, system.name)
                _dump(record_path, record)
                rows.append(record)

                print(
                    f"[{ticker}/{system.name}] {state['final_prediction']['direction']} "
                    f"-> {state['final_prediction']['target_price']} "
                    f"(rounds={state.get('round_count', 0)})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - log + continue
                msg = f"[{ticker}/{system.name}] {type(exc).__name__}: {exc}"
                errors.append({"ticker": ticker, "system": system.name, "error": str(exc)})
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n" + traceback.format_exc() + "\n")
                print(msg, flush=True)

    _aggregate(exp_dir, config, rows, errors, time.monotonic() - started)
    return exp_dir


def _aggregate(exp_dir, config, rows, errors, elapsed) -> None:
    """Build the wide table, metrics, and summary from all collected rows."""
    if not rows:
        _dump(exp_dir / "summary.json", {
            "experiment": config.name,
            "num_tickers": 0,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1),
        })
        print("No successful runs; wrote empty summary.")
        return

    df = build_wide_df(rows)
    df.to_csv(exp_dir / "results.csv", index=False)

    metrics = compute_metrics(
        df, config.systems, seed=config.seed, alpha=config.alpha, n_boot=config.n_boot
    )
    _dump(exp_dir / "metrics.json", metrics)
    (exp_dir / "metrics.md").write_text(_metrics_markdown(config, metrics), encoding="utf-8")

    if "comparison" in metrics or "correlated_error_rate" in metrics:
        _dump(exp_dir / "compare.json", {
            "comparison": metrics.get("comparison"),
            "comparison_error": metrics.get("comparison_error"),
            "correlated_error_rate": metrics.get("correlated_error_rate"),
        })

    _dump(exp_dir / "summary.json", {
        "experiment": config.name,
        "num_tickers": metrics["num_tickers"],
        "systems": [s.name for s in config.systems],
        "per_system": metrics["per_system"],
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    })

    print(f"\nSaved experiment to {exp_dir}")


def _metrics_markdown(config, metrics) -> str:
    lines = [
        f"# Experiment: {config.name}",
        "",
        f"Tickers evaluated: {metrics['num_tickers']}",
        "",
        "| System | Accuracy | 95% CI | n | Mean APE | Median APE |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in metrics["per_system"].items():
        if m.get("num_total", 0) == 0:
            lines.append(f"| {name} | - | - | 0 | - | - |")
            continue
        ci = m.get("confidence_interval", {})
        lo, hi = ci.get("lower"), ci.get("upper")
        ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "-"
        lines.append(
            f"| {name} | {m['directional_accuracy']:.2%} | {ci_str} | "
            f"{m['num_total']} | {m['mean_target_price_percentage_error']:.2%} | "
            f"{m['median_target_price_percentage_error']:.2%} |"
        )
    return "\n".join(lines) + "\n"
