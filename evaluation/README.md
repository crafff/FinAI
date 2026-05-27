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
