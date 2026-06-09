"""
Data-completeness breakdown of an experiment's results.

`allow_missing` lets a ticker run with some sources degraded to empty (e.g. an
FMP 402 leaves it without financials); each ticker's `data_context.json`
records which sources were `missing`. This module reads a finished experiment
directory and reports directional accuracy + target-price error **grouped by
data completeness**, so you can see how much the missing-data tickers drag the
numbers down (and per system, when the experiment ran several).

It reuses the same metric definitions as the main harness
(`evaluation/metrics.py` via `results._per_system_df` + `evaluate_predictions`),
so a group's numbers match what `metrics.json` would report on that subset.

Reads (under runs/<name>/):
    config.json                       -> the system names
    per_ticker/<T>/data_context.json  -> that ticker's `missing` list
    per_ticker/<T>/<system>/record.json -> that cell's prediction + answer key
"""

from __future__ import annotations

import json
from pathlib import Path

from metrics import evaluate_predictions
from results import build_wide_df, _per_system_df


def experiment_systems(exp_dir: Path) -> list[str]:
    """System names from the resolved config snapshot."""
    config = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    return [s["name"] for s in config.get("systems", [])]


def load_ticker_missing(exp_dir: Path) -> dict[str, list[str]]:
    """Map each ticker to its list of missing data sources (empty = complete)."""
    out: dict[str, list[str]] = {}
    for dc in sorted((exp_dir / "per_ticker").glob("*/data_context.json")):
        data = json.loads(dc.read_text(encoding="utf-8"))
        out[data["ticker"]] = list(data.get("missing") or [])
    return out


def load_records(exp_dir: Path, systems: list[str]) -> list[dict]:
    """Collect every cell's record.json (one per (ticker, system) that ran)."""
    rows: list[dict] = []
    per_ticker = exp_dir / "per_ticker"
    if not per_ticker.is_dir():
        return rows
    for ticker_dir in sorted(p for p in per_ticker.iterdir() if p.is_dir()):
        for system in systems:
            record_path = ticker_dir / system / "record.json"
            if record_path.exists():
                rows.append(json.loads(record_path.read_text(encoding="utf-8")))
    return rows


def completeness_label(missing: list[str]) -> str:
    """A stable group label: 'complete' or 'missing: a,b' (sorted)."""
    return "complete" if not missing else "missing: " + ", ".join(sorted(missing))


def _metrics_per_system(sub_df, systems: list[str]) -> dict:
    """evaluate_predictions for each system over one ticker subset."""
    per_system: dict[str, dict] = {}
    for name in systems:
        psub = _per_system_df(sub_df, name)
        if len(psub) == 0:
            per_system[name] = {"num_total": 0}
            continue
        per_system[name] = evaluate_predictions(psub)
    return per_system


def analyze_experiment(exp_dir) -> dict:
    """
    Build the completeness report for one experiment directory.

    Returns a dict with three views, each carrying per-system metrics:
      - `overall`               : all tickers.
      - `by_completeness`       : one bucket per exact missing-set
                                  (e.g. 'complete', 'missing: financials').
      - `complete_vs_degraded`  : the coarse two-bucket split.
    """
    exp_dir = Path(exp_dir)
    systems = experiment_systems(exp_dir)
    ticker_missing = load_ticker_missing(exp_dir)
    rows = load_records(exp_dir, systems)

    df = build_wide_df(rows)
    if len(df) == 0:
        return {
            "experiment": exp_dir.name,
            "systems": systems,
            "num_tickers": 0,
            "overall": {"num_tickers": 0, "per_system": {}},
            "by_completeness": {},
            "complete_vs_degraded": {},
        }

    labels = df["ticker"].map(lambda t: completeness_label(ticker_missing.get(t, [])))
    buckets = labels.map(lambda lbl: "complete" if lbl == "complete" else "degraded")

    report: dict = {
        "experiment": exp_dir.name,
        "systems": systems,
        "num_tickers": int(len(df)),
        "overall": {
            "num_tickers": int(len(df)),
            "per_system": _metrics_per_system(df, systems),
        },
        "by_completeness": {},
        "complete_vs_degraded": {},
    }

    for label in sorted(labels.unique()):
        sub = df[labels == label]
        report["by_completeness"][label] = {
            "num_tickers": int(len(sub)),
            "tickers": sorted(sub["ticker"].tolist()),
            "per_system": _metrics_per_system(sub, systems),
        }

    for bucket in sorted(buckets.unique()):
        sub = df[buckets == bucket]
        report["complete_vs_degraded"][bucket] = {
            "num_tickers": int(len(sub)),
            "per_system": _metrics_per_system(sub, systems),
        }

    return report


def _rows_for(view: dict, systems: list[str]) -> list[str]:
    lines = []
    for group, body in view.items():
        per_system = body["per_system"]
        for name in systems:
            m = per_system.get(name, {"num_total": 0})
            n = m.get("num_total", 0)
            if not n:
                lines.append(f"| {group} | {name} | 0 | - | - | - | - |")
                continue
            ci = m.get("confidence_interval", {})
            lo, hi = ci.get("lower"), ci.get("upper")
            ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "-"
            lines.append(
                f"| {group} | {name} | {n} | {m['directional_accuracy']:.2%} | "
                f"{ci_str} | {m['mean_target_price_percentage_error']:.2%} | "
                f"{m['median_target_price_percentage_error']:.2%} |"
            )
    return lines


def render_markdown(report: dict) -> str:
    systems = report["systems"]
    header = [
        f"# Completeness breakdown: {report['experiment']}",
        "",
        f"Tickers: {report['num_tickers']} | systems: {', '.join(systems) or '-'}",
        "",
    ]

    def section(title, view):
        out = [f"## {title}", "",
               "| Group | System | n | Accuracy | 95% CI | Mean APE | Median APE |",
               "|---|---|---|---|---|---|---|"]
        out += _rows_for(view, systems)
        out.append("")
        return out

    lines = header
    lines += section("By data completeness", report["by_completeness"])
    lines += section("Complete vs. degraded", report["complete_vs_degraded"])
    lines += section("Overall", {"all": report["overall"]})
    return "\n".join(lines) + "\n"
