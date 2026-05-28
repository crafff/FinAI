import itertools

import numpy as np
import pandas as pd

from metrics import BUY, NOT_BUY, actual_direction


VALID_DIRECTIONS = {BUY, NOT_BUY}


def _validate_direction_column(df, column):
    """
    Confirm that a column contains only the BUY / NOT_BUY string
    constants. Other values mean the prediction format is wrong and
    the paired tests would silently mis-count.
    """
    invalid_mask = ~df[column].isin(VALID_DIRECTIONS)

    if invalid_mask.any():
        offending = list(df.index[invalid_mask])
        raise ValueError(
            f"Column {column!r} contains values outside "
            f"{{{BUY!r}, {NOT_BUY!r}}} at rows {offending}."
        )


def _actual_directions(df, baseline_price_col, actual_target_price_col):
    """
    Compute the true Buy / Not Buy label for each row using the same
    rule as evaluation.metrics.actual_direction.
    """
    return np.array([
        actual_direction(b, a)
        for b, a in zip(
            df[baseline_price_col].values,
            df[actual_target_price_col].values,
        )
    ])


def mcnemar_test(
    df,
    system_a_dir_col,
    system_b_dir_col,
    baseline_price_col="baseline_price",
    actual_target_price_col="actual_target_price",
):
    """
    Exact McNemar test on paired directional predictions.

    Counts discordant pairs:
        b = system A correct AND system B wrong
        c = system A wrong   AND system B correct

    Under the null hypothesis that the two systems have equal error
    rates, each discordant pair is equally likely to fall into b or c,
    so the count min(b, c) is Binomial(b + c, 0.5). The two-sided
    exact p-value is taken from scipy.stats.binomtest, which avoids
    the small-sample issues of the chi-square approximation.
    """
    from scipy.stats import binomtest

    _validate_direction_column(df, system_a_dir_col)
    _validate_direction_column(df, system_b_dir_col)

    true_dir = _actual_directions(df, baseline_price_col, actual_target_price_col)

    correct_a = df[system_a_dir_col].values == true_dir
    correct_b = df[system_b_dir_col].values == true_dir

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c

    if n == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue
        )

    return {
        "b": b,
        "c": c,
        "n": n,
        "p_value": p_value,
        "test": "mcnemar_exact",
    }


def _absolute_percentage_errors(df, predicted_col, actual_col):
    if (df[actual_col] == 0).any():
        raise ValueError(f"Column {actual_col!r} contains zero values.")

    return (
        (df[predicted_col] - df[actual_col]).abs()
        / df[actual_col].abs()
    ).values


def wilcoxon_apes_test(
    df,
    system_a_tp_col,
    system_b_tp_col,
    actual_target_price_col="actual_target_price",
):
    """
    Paired Wilcoxon signed-rank test on per-row target-price APE.

    Compares APE_A and APE_B element-wise. Non-parametric, so robust
    to MAPE's heavy right tail. median_diff = median(APE_A - APE_B):
    negative means system A is more accurate.

    NaN rows are dropped before the test. When every difference is
    zero (identical predictions) the test is undefined and we report
    p_value = 1.0 / statistic = 0.0 rather than letting scipy raise.
    """
    from scipy.stats import wilcoxon

    ape_a = _absolute_percentage_errors(df, system_a_tp_col, actual_target_price_col)
    ape_b = _absolute_percentage_errors(df, system_b_tp_col, actual_target_price_col)

    finite_mask = np.isfinite(ape_a) & np.isfinite(ape_b)
    ape_a = ape_a[finite_mask]
    ape_b = ape_b[finite_mask]

    n = len(ape_a)

    diffs = ape_a - ape_b

    if n == 0 or np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "median_diff": 0.0,
            "n": n,
            "test": "wilcoxon_signed_rank",
        }

    result = wilcoxon(ape_a, ape_b, zero_method="wilcox", alternative="two-sided")

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_diff": float(np.median(diffs)),
        "n": n,
        "test": "wilcoxon_signed_rank",
    }


def paired_bootstrap_accuracy_diff(
    df,
    system_a_dir_col,
    system_b_dir_col,
    baseline_price_col="baseline_price",
    actual_target_price_col="actual_target_price",
    n_boot=10_000,
    alpha=0.05,
    seed=None,
):
    """
    Percentile bootstrap CI for accuracy_A - accuracy_B.

    Resamples rows with replacement so the pairing between system A
    and system B is preserved on every replicate. Reports the point
    estimate on the original sample alongside the alpha/2 and
    1 - alpha/2 quantiles of the bootstrap distribution.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    _validate_direction_column(df, system_a_dir_col)
    _validate_direction_column(df, system_b_dir_col)

    true_dir = _actual_directions(df, baseline_price_col, actual_target_price_col)
    correct_a = (df[system_a_dir_col].values == true_dir).astype(np.int8)
    correct_b = (df[system_b_dir_col].values == true_dir).astype(np.int8)

    n = len(df)
    diff = float(correct_a.mean() - correct_b.mean()) if n > 0 else 0.0

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    lower = float(np.quantile(boot_diffs, alpha / 2))
    upper = float(np.quantile(boot_diffs, 1 - alpha / 2))

    return {
        "diff": diff,
        "lower": lower,
        "upper": upper,
        "alpha": alpha,
        "n_boot": n_boot,
        "n": n,
    }


def holm_bonferroni(pvalues, alpha=0.05):
    """
    Holm-Bonferroni step-down correction.

    For p-values sorted ascending (p_(1), ..., p_(m)) the adjusted
    value is

        adj_(i) = max over j<=i of (m - j + 1) * p_(j),  capped at 1.

    Monotone non-decreasing in i, so once adj_(i) > alpha, every
    later test is also non-rejected. Returns results in the order the
    caller supplied so they can be paired back with the originating
    comparisons.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    m = len(pvalues)

    if m == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])

    adjusted_sorted = [0.0] * m
    running_max = 0.0

    for rank, (_, p) in enumerate(indexed, start=1):
        candidate = (m - rank + 1) * p
        running_max = max(running_max, candidate)
        adjusted_sorted[rank - 1] = min(running_max, 1.0)

    results = [None] * m

    for sorted_pos, (orig_idx, p) in enumerate(indexed):
        adj = adjusted_sorted[sorted_pos]
        results[orig_idx] = {
            "p_value": float(p),
            "adjusted_p_value": float(adj),
            "reject": bool(adj <= alpha),
        }

    return results


def compare_systems(
    df,
    systems,
    alpha=0.05,
    n_boot=10_000,
    seed=None,
    baseline_price_col="baseline_price",
    actual_target_price_col="actual_target_price",
):
    """
    Run every pairwise paired test among the supplied systems.

    `systems` is a list of dicts:
        [{"name": ..., "dir_col": ..., "tp_col": ...}, ...]

    For each unordered pair (A, B) with A appearing before B in the
    list this runs mcnemar_test, wilcoxon_apes_test, and
    paired_bootstrap_accuracy_diff. The McNemar p-values are then
    corrected with Holm-Bonferroni as one family, the Wilcoxon
    p-values as a separate family.

    Bootstrap seeds for each pair are drawn from a parent RNG seeded
    with `seed`, so the whole report is reproducible from a single
    seed without correlating bootstrap draws across pairs.
    """
    if len(systems) < 2:
        raise ValueError("Need at least two systems to compare.")

    parent_rng = np.random.default_rng(seed)

    pair_results = []
    mcnemar_pvalues = []
    wilcoxon_pvalues = []

    for sys_a, sys_b in itertools.combinations(systems, 2):
        pair_seed = int(parent_rng.integers(0, 2**32))

        mcnemar_result = mcnemar_test(
            df,
            sys_a["dir_col"],
            sys_b["dir_col"],
            baseline_price_col=baseline_price_col,
            actual_target_price_col=actual_target_price_col,
        )

        wilcoxon_result = wilcoxon_apes_test(
            df,
            sys_a["tp_col"],
            sys_b["tp_col"],
            actual_target_price_col=actual_target_price_col,
        )

        bootstrap_result = paired_bootstrap_accuracy_diff(
            df,
            sys_a["dir_col"],
            sys_b["dir_col"],
            baseline_price_col=baseline_price_col,
            actual_target_price_col=actual_target_price_col,
            n_boot=n_boot,
            alpha=alpha,
            seed=pair_seed,
        )

        pair_results.append({
            "system_a": sys_a["name"],
            "system_b": sys_b["name"],
            "mcnemar": mcnemar_result,
            "wilcoxon": wilcoxon_result,
            "bootstrap": bootstrap_result,
        })

        mcnemar_pvalues.append(mcnemar_result["p_value"])
        wilcoxon_pvalues.append(wilcoxon_result["p_value"])

    return {
        "pairs": pair_results,
        "mcnemar_holm_bonferroni": holm_bonferroni(mcnemar_pvalues, alpha),
        "wilcoxon_holm_bonferroni": holm_bonferroni(wilcoxon_pvalues, alpha),
        "alpha": alpha,
    }
