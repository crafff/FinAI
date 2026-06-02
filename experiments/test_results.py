from experiment_config import SystemConfig
from results import build_wide_df, compute_metrics, record_for_system


def _state(direction, tp, baseline=180.0, actual=200.0, ticker="AAPL"):
    return {
        "ticker": ticker,
        "baseline_price": baseline,
        "prices": {"target_price": actual},
        "final_prediction": {"direction": direction, "target_price": tp},
    }


def test_record_uses_system_name_prefix():
    r = record_for_system(_state("Buy", 190.0), "full")

    assert r["full_direction"] == "Buy"
    assert r["full_target_price"] == 190.0
    assert r["actual_target_price"] == 200.0
    assert r["baseline_price"] == 180.0


def test_build_wide_df_merges_systems_on_ticker():
    rows = [
        record_for_system(_state("Buy", 190.0), "single"),
        record_for_system(_state("Not Buy", 170.0), "full"),
    ]
    df = build_wide_df(rows)

    assert len(df) == 1
    assert {"single_direction", "full_direction"} <= set(df.columns)


def test_compute_metrics_per_system_and_comparison():
    # 3 tickers, two systems (a, b).
    data = [
        ("AAPL", 180, 200, "Buy", 198, "Buy", 205),       # actual Buy
        ("MSFT", 100, 90, "Not Buy", 92, "Buy", 110),     # actual Not Buy
        ("NVDA", 50, 60, "Buy", 59, "Not Buy", 44),       # actual Buy
    ]
    rows = []
    for tk, base, act, ad, atp, bd, btp in data:
        rows.append({"ticker": tk, "baseline_price": base, "actual_target_price": act,
                     "a_direction": ad, "a_target_price": atp})
        rows.append({"ticker": tk, "baseline_price": base, "actual_target_price": act,
                     "b_direction": bd, "b_target_price": btp})

    df = build_wide_df(rows)
    metrics = compute_metrics(
        df,
        [SystemConfig("a", mode="single"), SystemConfig("b", mode="single")],
        seed=1, n_boot=200,
    )

    assert metrics["num_tickers"] == 3
    assert metrics["per_system"]["a"]["num_total"] == 3
    assert metrics["per_system"]["b"]["num_total"] == 3
    assert "comparison" in metrics
    assert "correlated_error_rate" in metrics


def test_compute_metrics_handles_missing_system_columns():
    # Only system 'a' produced columns; 'b' errored everywhere.
    rows = [{"ticker": "AAPL", "baseline_price": 180, "actual_target_price": 200,
             "a_direction": "Buy", "a_target_price": 190}]
    df = build_wide_df(rows)

    metrics = compute_metrics(
        df, [SystemConfig("a", mode="single"), SystemConfig("b", mode="single")]
    )

    assert metrics["per_system"]["a"]["num_total"] == 1
    assert metrics["per_system"]["b"]["num_total"] == 0
    assert "comparison" not in metrics            # only one system present
