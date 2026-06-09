import json
from pathlib import Path

from completeness import (
    analyze_experiment,
    completeness_label,
    render_markdown,
)


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=str), encoding="utf-8")


def _make_cell(exp_dir: Path, ticker, system, missing, baseline, actual,
               direction, target):
    """Write one ticker's data_context + one system's record."""
    _write(exp_dir / "per_ticker" / ticker / "data_context.json", {
        "ticker": ticker,
        "baseline_price": baseline,
        "target_price": actual,
        "missing": missing,
    })
    _write(exp_dir / "per_ticker" / ticker / system / "record.json", {
        "ticker": ticker,
        "baseline_price": baseline,
        "actual_target_price": actual,
        f"{system}_direction": direction,
        f"{system}_target_price": target,
    })


def _toy_experiment(tmp_path) -> Path:
    exp = tmp_path / "exp"
    _write(exp / "config.json", {"name": "exp", "systems": [{"name": "full"}]})

    # complete group: A correct (Buy, up), B wrong (Buy, but down) -> 0.5
    _make_cell(exp, "A", "full", [], 100.0, 110.0, "Buy", 108.0)
    _make_cell(exp, "B", "full", [], 100.0, 90.0, "Buy", 95.0)
    # degraded group (missing financials): C correct (Buy, up) -> 1.0
    _make_cell(exp, "C", "full", ["financials"], 100.0, 120.0, "Buy", 118.0)
    return exp


def test_completeness_label():
    assert completeness_label([]) == "complete"
    assert completeness_label(["financials"]) == "missing: financials"
    assert completeness_label(["news", "financials"]) == "missing: financials, news"


def test_analyze_groups_by_completeness(tmp_path):
    report = analyze_experiment(_toy_experiment(tmp_path))

    assert report["num_tickers"] == 3
    assert report["systems"] == ["full"]

    by = report["by_completeness"]
    assert set(by) == {"complete", "missing: financials"}

    complete = by["complete"]["per_system"]["full"]
    assert complete["num_total"] == 2
    assert complete["directional_accuracy"] == 0.5
    assert by["complete"]["tickers"] == ["A", "B"]

    degraded = by["missing: financials"]["per_system"]["full"]
    assert degraded["num_total"] == 1
    assert degraded["directional_accuracy"] == 1.0


def test_complete_vs_degraded_and_overall(tmp_path):
    report = analyze_experiment(_toy_experiment(tmp_path))

    cvd = report["complete_vs_degraded"]
    assert cvd["complete"]["per_system"]["full"]["num_total"] == 2
    assert cvd["degraded"]["per_system"]["full"]["num_total"] == 1

    overall = report["overall"]["per_system"]["full"]
    assert overall["num_total"] == 3
    assert overall["num_correct"] == 2          # A and C correct, B wrong


def test_render_markdown_has_rows(tmp_path):
    report = analyze_experiment(_toy_experiment(tmp_path))
    md = render_markdown(report)

    assert "Completeness breakdown" in md
    assert "missing: financials" in md
    assert "| complete | full |" in md


def test_empty_experiment_is_handled(tmp_path):
    exp = tmp_path / "empty"
    _write(exp / "config.json", {"name": "empty", "systems": [{"name": "full"}]})
    (exp / "per_ticker").mkdir(parents=True)

    report = analyze_experiment(exp)
    assert report["num_tickers"] == 0
    assert report["by_completeness"] == {}
