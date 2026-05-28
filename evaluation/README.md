# Evaluation Metrics

This module implements the evaluation code for Task 8.

The evaluation framework is fully independent of the
agent pipeline and can be tested using mock prediction
outputs before the multi-agent system is completed.

The module computes four metrics:

1. Directional Accuracy

The system predicts either:

    - Buy
    - Not Buy

The actual label is determined by comparing the baseline
price at the T₀ close with the actual closing price on the
5th trading day after T₀.

If:

    actual_target_price > baseline_price

then:

    actual label = Buy

Otherwise:

    actual label = Not Buy

Directional accuracy is the fraction of companies where
the predicted direction matches the actual direction.

2. Confidence Interval

The module computes a Wilson confidence interval for
directional accuracy.

This provides a more statistically informative estimate
than raw accuracy alone, especially when the evaluation
sample size is small.

The Wilson interval is preferred over the standard
normal approximation because it remains more stable
near 0 or 1 accuracy values and for limited datasets.

3. Target-Price Percentage Error

The target-price error measures how close the predicted
5th-trading-day price is to the true market price.

The error formula is:

    abs(predicted_target_price - actual_target_price)
    /
    actual_target_price

The module reports:

    - mean absolute percentage error (MAPE)
    - median absolute percentage error

The median error is included because stock-price
prediction errors may contain outliers, and the median
provides a more robust summary of typical performance.

4. Correlated-Error Rate

This metric measures how often multiple systems make
the same directional mistake on the same company.

A correlated error occurs only when:

    1. all systems are incorrect
    2. all systems predicted the same wrong direction

For example, if the:

    - single-agent baseline
    - paper-style aggregation baseline
    - full coopetition system

all predict Buy while the true label is Not Buy,
that counts as a correlated error.

The implementation explicitly checks for the same wrong
prediction even though this project currently uses a
binary Buy / Not Buy label. In the binary case, all
systems being wrong implies they selected the opposite
label, but the explicit check keeps the metric easier
to understand and extend if additional labels are added
later.

Together, these metrics evaluate:

    - directional correctness
    - numerical target-price quality
    - statistical uncertainty
    - shared failure behavior across systems

This directly supports the project’s planned ablation
experiments comparing:

    - the single-agent baseline
    - the paper-style aggregation baseline
    - the full coopetition pipeline

# Significance Tests

The companion module significance.py implements Task 9:
paired statistical tests between configurations.

Because every system predicts every company in the
ablation, observations are paired. Unpaired tests would
discard that structure and overstate variance, so the
module supplies the paired counterparts:

1. McNemar exact test

   For each pair of systems, count the discordant cells:

       b = system A correct AND system B wrong
       c = system A wrong   AND system B correct

   Under the null hypothesis of equal error rates,
   min(b, c) is Binomial(b + c, 0.5). The two-sided p
   value is computed exactly via scipy.stats.binomtest,
   avoiding the small-sample issues of the chi-square
   approximation.

2. Wilcoxon signed-rank test

   For each pair of systems, compute the per-row absolute
   percentage error and apply scipy.stats.wilcoxon to the
   paired error vectors. Non-parametric, so robust to the
   right-skewed MAPE distribution. A negative median_diff
   means system A is more accurate.

3. Paired bootstrap accuracy CI

   Resamples rows (with replacement) so the pairing is
   preserved on every replicate, then takes percentile
   bounds on accuracy_A minus accuracy_B. Reports the
   point estimate plus the lower and upper bounds.

4. Holm-Bonferroni correction

   Applied separately to the McNemar p-value family and
   the Wilcoxon p-value family. Step-down adjustment
   preserves the original input order so each adjusted
   value can be matched back to its system pair.

The compare_systems convenience function runs every
pairwise McNemar, Wilcoxon, and bootstrap, applies the
two corrections, and returns one nested dict ready to
print as an ablation comparison table. All bootstrap
draws are reproducible from a single seed.

The module imports BUY, NOT_BUY, and actual_direction
from metrics.py rather than reimplementing the buy /
sell rule. Direction columns are validated to contain
only the BUY / NOT_BUY string constants.

## Usage

### 1. Install dependencies

    cd evaluation
    pip install -r requirements.txt

requirements.txt pulls in pandas, numpy, scipy, and pytest.
metrics.py only uses pandas; scipy is needed for significance.py.

### 2. Run the tests

    pytest -v

All tests run offline. test_metrics.py exercises the four metrics on
small hand-built DataFrames. test_significance.py uses a 4-row mock
DataFrame plus a 30-row synthetic ablation built with a seeded RNG so
the McNemar / Wilcoxon / bootstrap paths are tested without any
external data.

### 3. Run against real ablation data

Both modules operate on a wide DataFrame: one row per company, one
column per system. They require:

    - baseline_price                   T0 close (shared)
    - actual_target_price              5th-trading-day close (shared)
    - <system>_direction               "Buy" or "Not Buy" per system
    - <system>_target_price            per-system price prediction

Once Task 19 produces this DataFrame at the end of the ablation,
run the full evaluation as follows. Until then, use any DataFrame
matching the same schema (e.g. a mock built from yfinance baselines
and randomly generated predictions, as shown in test_significance.py).

    import pandas as pd
    from metrics import evaluate_predictions, correlated_error_rate
    from significance import compare_systems

    df = pd.read_csv("ablation_predictions.csv")

    # Per-system summary metrics.
    for sys_prefix in ("single", "paper", "full"):
        per_system_df = df.assign(
            predicted_direction=df[f"{sys_prefix}_direction"],
            predicted_target_price=df[f"{sys_prefix}_target_price"],
        )
        print(sys_prefix, evaluate_predictions(per_system_df))

    print("correlated errors:", correlated_error_rate(df, [
        "single_direction", "paper_direction", "full_direction",
    ]))

    # Pairwise significance + multiple-comparison correction.
    report = compare_systems(
        df,
        systems=[
            {"name": "single", "dir_col": "single_direction",
             "tp_col": "single_target_price"},
            {"name": "paper",  "dir_col": "paper_direction",
             "tp_col": "paper_target_price"},
            {"name": "full",   "dir_col": "full_direction",
             "tp_col": "full_target_price"},
        ],
        alpha=0.05,
        n_boot=10_000,
        seed=42,
    )

    for pair, mc, wx in zip(
        report["pairs"],
        report["mcnemar_holm_bonferroni"],
        report["wilcoxon_holm_bonferroni"],
    ):
        print(
            f"{pair['system_a']} vs {pair['system_b']}: "
            f"mcnemar p={pair['mcnemar']['p_value']:.3f} "
            f"(adj {mc['adjusted_p_value']:.3f}, "
            f"reject={mc['reject']}); "
            f"wilcoxon p={pair['wilcoxon']['p_value']:.3f} "
            f"(adj {wx['adjusted_p_value']:.3f}, "
            f"reject={wx['reject']}); "
            f"acc diff = {pair['bootstrap']['diff']:+.3f} "
            f"[{pair['bootstrap']['lower']:+.3f}, "
            f"{pair['bootstrap']['upper']:+.3f}]"
        )

The seed argument makes the bootstrap reproducible across runs.
