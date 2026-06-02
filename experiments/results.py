"""
Turn finished runs into evaluation records, a wide ablation table, and metrics.

- `record_for_system` projects one PipelineState into a single wide row
  contribution (generalizing contracts.state.to_ablation_record to an
  arbitrary system-name column prefix).
- `build_wide_df` merges per-(ticker, system) rows into the one-row-per-ticker
  DataFrame the evaluation code expects.
- `compute_metrics` reuses evaluation/metrics.py (per-system accuracy + Wilson
  CI + target-price MAPE) and evaluation/significance.py (pairwise
  McNemar/Wilcoxon/bootstrap + correlated-error) across systems.
"""

from __future__ import annotations

import pandas as pd

from metrics import correlated_error_rate, evaluate_predictions
from significance import compare_systems


def record_for_system(state: dict, system_name: str) -> dict:
    """
    One wide-row contribution for a finished run. Mirrors
    `to_ablation_record` but uses the (arbitrary) system name as the column
    prefix, so systems beyond the single|paper|full literals work.
    """
    prediction = state["final_prediction"]
    return {
        "ticker": state["ticker"],
        "baseline_price": state["baseline_price"],
        # answer key (real close on the target date); never shown to an agent.
        "actual_target_price": state["prices"]["target_price"],
        f"{system_name}_direction": prediction["direction"],
        f"{system_name}_target_price": prediction["target_price"],
    }


def build_wide_df(rows: list[dict]) -> pd.DataFrame:
    """
    Merge per-(ticker, system) records into one row per ticker. Shared
    columns (baseline_price, actual_target_price) coincide across systems.
    """
    by_ticker: dict[str, dict] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], {}).update(row)

    return pd.DataFrame(list(by_ticker.values()))


def _per_system_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    A frame with the column names evaluate_predictions expects, dropping
    rows where this system has no prediction (e.g. an errored cell). Returns
    an empty frame if the system produced no columns at all.
    """
    dir_col, tp_col = f"{name}_direction", f"{name}_target_price"

    if dir_col not in df.columns or tp_col not in df.columns:
        return pd.DataFrame(columns=[
            "baseline_price", "actual_target_price",
            "predicted_direction", "predicted_target_price",
        ])

    sub = df[["baseline_price", "actual_target_price", dir_col, tp_col]].dropna(
        subset=[dir_col, tp_col]
    )
    return sub.rename(columns={
        dir_col: "predicted_direction",
        tp_col: "predicted_target_price",
    })


def compute_metrics(df, systems, seed=None, alpha=0.05, n_boot=10_000) -> dict:
    """
    Per-system metrics plus (when >= 2 systems) pairwise significance and the
    correlated-error rate. `systems` is a list of SystemConfig.
    """
    names = [s.name for s in systems]

    per_system = {}
    for name in names:
        sub = _per_system_df(df, name)
        if len(sub) == 0:
            per_system[name] = {"num_total": 0}
            continue
        per_system[name] = evaluate_predictions(sub)

    result: dict = {"num_tickers": int(len(df)), "per_system": per_system}

    # Only compare systems that actually produced columns (errored / skipped
    # systems contribute nothing).
    present = [n for n in names if f"{n}_direction" in df.columns]

    if len(present) >= 2:
        dir_cols = [f"{n}_direction" for n in present]
        tp_cols = [f"{n}_target_price" for n in present]
        complete = df.dropna(subset=dir_cols + tp_cols)

        if len(complete) >= 2:
            try:
                result["comparison"] = compare_systems(
                    complete,
                    [
                        {"name": n, "dir_col": f"{n}_direction",
                         "tp_col": f"{n}_target_price"}
                        for n in present
                    ],
                    alpha=alpha,
                    n_boot=n_boot,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                result["comparison_error"] = f"{type(exc).__name__}: {exc}"

            result["correlated_error_rate"] = correlated_error_rate(
                complete, dir_cols
            )

    return result
