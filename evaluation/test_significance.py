import numpy as np
import pandas as pd
import pytest

from significance import (
    compare_systems,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
    wilcoxon_apes_test,
)


def _three_system_df():
    """
    A small mock DataFrame used by several tests. Truths:
        row 0  Buy (110 > 100)
        row 1  Not Buy (90 < 100)
        row 2  Buy (105 > 100)
        row 3  Not Buy (95 < 100)
    """
    return pd.DataFrame({
        "baseline_price": [100.0, 100.0, 100.0, 100.0],
        "actual_target_price": [110.0, 90.0, 105.0, 95.0],
        "a_dir": ["Buy", "Not Buy", "Buy", "Not Buy"],
        "b_dir": ["Buy", "Buy", "Not Buy", "Not Buy"],
        "c_dir": ["Not Buy", "Buy", "Not Buy", "Buy"],
        "a_tp": [108.0, 92.0, 104.0, 96.0],
        "b_tp": [105.0, 95.0, 108.0, 93.0],
        "c_tp": [120.0, 80.0, 115.0, 85.0],
    })


def test_mcnemar_counts_b_and_c():
    df = _three_system_df()
    result = mcnemar_test(df, "a_dir", "b_dir")

    assert result["b"] == 2
    assert result["c"] == 0
    assert result["n"] == 2


def test_mcnemar_p_value_against_binom_table():
    """
    Construct 4 discordant pairs all of one type (b=4, c=0).
    Exact two-sided binomial p for min(b,c)=0 out of 4 with p=0.5
    is 2 * P(X=0) = 2 * 0.5^4 = 0.125.
    """
    df = pd.DataFrame({
        "baseline_price": [100.0] * 4,
        "actual_target_price": [110.0] * 4,
        "a_dir": ["Buy"] * 4,
        "b_dir": ["Not Buy"] * 4,
    })

    result = mcnemar_test(df, "a_dir", "b_dir")

    assert result["b"] == 4
    assert result["c"] == 0
    assert result["p_value"] == pytest.approx(0.125)


def test_mcnemar_returns_one_when_no_discordant_pairs():
    df = pd.DataFrame({
        "baseline_price": [100.0, 100.0],
        "actual_target_price": [110.0, 90.0],
        "a_dir": ["Buy", "Not Buy"],
        "b_dir": ["Buy", "Not Buy"],
    })

    result = mcnemar_test(df, "a_dir", "b_dir")

    assert result["n"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_rejects_invalid_direction_values():
    df = pd.DataFrame({
        "baseline_price": [100.0],
        "actual_target_price": [110.0],
        "a_dir": ["BUY"],
        "b_dir": ["Buy"],
    })

    with pytest.raises(ValueError):
        mcnemar_test(df, "a_dir", "b_dir")


def test_wilcoxon_detects_consistent_improvement():
    """
    Build a df where system A's APE is uniformly smaller than B's.
    The signed-rank test should reject with all positive ranks for
    (b - a) and median_diff < 0.
    """
    actual = np.linspace(100, 200, 30)
    pred_a = actual * 1.01
    pred_b = actual * 1.10

    df = pd.DataFrame({
        "actual_target_price": actual,
        "a_tp": pred_a,
        "b_tp": pred_b,
    })

    result = wilcoxon_apes_test(df, "a_tp", "b_tp")

    assert result["n"] == 30
    assert result["p_value"] < 0.05
    assert result["median_diff"] < 0


def test_wilcoxon_no_difference_returns_high_p():
    df = pd.DataFrame({
        "actual_target_price": [100.0, 110.0, 120.0],
        "a_tp": [101.0, 109.0, 121.0],
        "b_tp": [101.0, 109.0, 121.0],
    })

    result = wilcoxon_apes_test(df, "a_tp", "b_tp")

    assert result["p_value"] == 1.0
    assert result["median_diff"] == 0.0


def test_wilcoxon_raises_on_zero_actual():
    df = pd.DataFrame({
        "actual_target_price": [100.0, 0.0],
        "a_tp": [101.0, 1.0],
        "b_tp": [100.0, 1.0],
    })

    with pytest.raises(ValueError):
        wilcoxon_apes_test(df, "a_tp", "b_tp")


def test_paired_bootstrap_diff_matches_raw_accuracy_diff():
    df = _three_system_df()

    result = paired_bootstrap_accuracy_diff(
        df, "a_dir", "b_dir", n_boot=500, seed=0,
    )

    # a_dir is correct on rows 0, 1, 2, 3 -> accuracy 1.0
    # b_dir is correct on rows 0, 3        -> accuracy 0.5
    assert result["diff"] == pytest.approx(0.5)
    assert result["lower"] <= result["diff"] <= result["upper"]


def test_paired_bootstrap_seed_is_deterministic():
    df = _three_system_df()

    a = paired_bootstrap_accuracy_diff(
        df, "a_dir", "b_dir", n_boot=200, seed=42,
    )
    b = paired_bootstrap_accuracy_diff(
        df, "a_dir", "b_dir", n_boot=200, seed=42,
    )

    assert a["lower"] == b["lower"]
    assert a["upper"] == b["upper"]


def test_paired_bootstrap_rejects_zero_n_boot():
    df = _three_system_df()

    with pytest.raises(ValueError):
        paired_bootstrap_accuracy_diff(
            df, "a_dir", "b_dir", n_boot=0, seed=0,
        )


def test_holm_bonferroni_classic_example():
    """
    pvalues = [0.01, 0.02, 0.04], alpha = 0.05, m = 3
    sorted: 0.01 -> 3*0.01 = 0.03 (rank 1)
            0.02 -> 2*0.02 = 0.04 (rank 2)
            0.04 -> 1*0.04 = 0.04 (rank 3)
    cumulative max: 0.03, 0.04, 0.04
    all <= 0.05 -> all reject
    """
    result = holm_bonferroni([0.01, 0.02, 0.04], alpha=0.05)

    assert [r["adjusted_p_value"] for r in result] == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    assert [r["reject"] for r in result] == [True, True, True]


def test_holm_bonferroni_stops_at_first_non_rejection():
    """
    pvalues = [0.01, 0.04, 0.045], alpha = 0.05
    sorted adjusted: 0.03 (reject), 0.08 (no), max-propagate -> 0.08 (no)
    """
    result = holm_bonferroni([0.01, 0.04, 0.045], alpha=0.05)

    assert [r["reject"] for r in result] == [True, False, False]


def test_holm_bonferroni_preserves_input_order():
    result = holm_bonferroni([0.04, 0.01, 0.02], alpha=0.05)

    # Adjusted values in input order:
    # input[0]=0.04 sorted-rank 3 -> 1 * 0.04 = 0.04 (after monotone max)
    # input[1]=0.01 sorted-rank 1 -> 3 * 0.01 = 0.03
    # input[2]=0.02 sorted-rank 2 -> 2 * 0.02 = 0.04
    assert [r["adjusted_p_value"] for r in result] == pytest.approx(
        [0.04, 0.03, 0.04]
    )


def test_holm_bonferroni_caps_at_one():
    result = holm_bonferroni([0.5, 0.6], alpha=0.05)

    for r in result:
        assert r["adjusted_p_value"] <= 1.0
        assert r["reject"] is False


def test_holm_bonferroni_empty_input():
    assert holm_bonferroni([], alpha=0.05) == []


def test_compare_systems_three_way_shape():
    df = _three_system_df()

    systems = [
        {"name": "alpha", "dir_col": "a_dir", "tp_col": "a_tp"},
        {"name": "beta",  "dir_col": "b_dir", "tp_col": "b_tp"},
        {"name": "gamma", "dir_col": "c_dir", "tp_col": "c_tp"},
    ]

    report = compare_systems(df, systems, n_boot=100, seed=0)

    assert len(report["pairs"]) == 3
    assert {(p["system_a"], p["system_b"]) for p in report["pairs"]} == {
        ("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma"),
    }
    for pair in report["pairs"]:
        assert pair["mcnemar"]["test"] == "mcnemar_exact"
        assert pair["wilcoxon"]["test"] == "wilcoxon_signed_rank"
        assert "diff" in pair["bootstrap"]

    assert len(report["mcnemar_holm_bonferroni"]) == 3
    assert len(report["wilcoxon_holm_bonferroni"]) == 3


def test_compare_systems_seed_determinism():
    df = _three_system_df()
    systems = [
        {"name": "alpha", "dir_col": "a_dir", "tp_col": "a_tp"},
        {"name": "beta",  "dir_col": "b_dir", "tp_col": "b_tp"},
        {"name": "gamma", "dir_col": "c_dir", "tp_col": "c_tp"},
    ]

    a = compare_systems(df, systems, n_boot=100, seed=123)
    b = compare_systems(df, systems, n_boot=100, seed=123)

    for pa, pb in zip(a["pairs"], b["pairs"]):
        assert pa["bootstrap"]["lower"] == pb["bootstrap"]["lower"]
        assert pa["bootstrap"]["upper"] == pb["bootstrap"]["upper"]


def test_compare_systems_rejects_single_system():
    df = _three_system_df()

    with pytest.raises(ValueError):
        compare_systems(
            df,
            [{"name": "alpha", "dir_col": "a_dir", "tp_col": "a_tp"}],
            n_boot=10,
            seed=0,
        )


def test_compare_systems_full_beats_single_on_realistic_mock():
    """
    Sanity: a full system that gets 70% of directional calls correct
    should beat a single agent at 55% strongly enough to register a
    positive accuracy gap on a 30-row sample with seed pinned.
    """
    rng = np.random.default_rng(0)
    n = 30
    baseline = np.full(n, 100.0)
    actual = baseline * (1 + rng.normal(0, 0.05, n))
    truth = actual > baseline

    def labels_from(prob_correct):
        flip = rng.random(n) > prob_correct
        pred = np.where(flip, ~truth, truth)
        return np.where(pred, "Buy", "Not Buy")

    df = pd.DataFrame({
        "baseline_price": baseline,
        "actual_target_price": actual,
        "single_dir": labels_from(0.55),
        "full_dir": labels_from(0.85),
        "single_tp": baseline * (1 + rng.normal(0, 0.10, n)),
        "full_tp": baseline * (1 + rng.normal(0, 0.03, n)),
    })

    systems = [
        {"name": "single", "dir_col": "single_dir", "tp_col": "single_tp"},
        {"name": "full",   "dir_col": "full_dir",   "tp_col": "full_tp"},
    ]

    report = compare_systems(df, systems, n_boot=500, seed=42)

    pair = report["pairs"][0]
    # bootstrap diff = single - full, so full beating single -> diff < 0
    assert pair["bootstrap"]["diff"] < 0
