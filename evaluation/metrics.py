import math
import pandas as pd


BUY = "Buy"
NOT_BUY = "Not Buy"


def actual_direction(baseline_price, target_price):
    """
    Actual label:
        Buy if target price is greater than baseline price.
        Not Buy otherwise.
    """
    return BUY if target_price > baseline_price else NOT_BUY


def directional_accuracy(df):
    """
    Compute directional accuracy for one system.

    Required columns:
        predicted_direction
        baseline_price
        actual_target_price
    """
    actual = df.apply(
        lambda row: actual_direction(
            row["baseline_price"],
            row["actual_target_price"]
        ),
        axis=1
    )

    correct = df["predicted_direction"] == actual

    return {
        "accuracy": correct.mean(),
        "num_correct": int(correct.sum()),
        "num_total": int(len(df)),
    }


def wilson_confidence_interval(num_correct, num_total, z=1.96):
    """
    Wilson confidence interval for directional accuracy.

    Default z=1.96 gives an approximate 95% confidence interval.
    """
    if num_total <= 0:
        raise ValueError("num_total must be greater than 0.")

    p_hat = num_correct / num_total

    denominator = 1 + (z ** 2 / num_total)

    center = (
        p_hat + (z ** 2 / (2 * num_total))
    ) / denominator

    margin = (
        z * math.sqrt(
            (p_hat * (1 - p_hat) / num_total)
            + (z ** 2 / (4 * num_total ** 2))
        )
    ) / denominator

    return {
        "lower": center - margin,
        "upper": center + margin,
    }


def target_price_percentage_error(df):
    """
    Compute absolute percentage error for target-price prediction.

    Formula:
        abs(predicted - actual) / actual

    Required columns:
        predicted_target_price
        actual_target_price
    """
    if (df["actual_target_price"] == 0).any():
        raise ValueError("actual_target_price cannot be zero.")

    errors = (
        (df["predicted_target_price"] - df["actual_target_price"]).abs()
        / df["actual_target_price"]
    )

    return {
        "mean_absolute_percentage_error": errors.mean(),
        "median_absolute_percentage_error": errors.median(),
        "errors": errors.tolist(),
    }


def correlated_error_rate(df, system_direction_columns):
    """
    Compute how often multiple systems make the same directional mistake.

    A correlated error occurs when:
        1. all systems are incorrect, and
        2. all systems predicted the same wrong direction

    Required columns:
        baseline_price
        actual_target_price
        one predicted-direction column per system

    Example system_direction_columns:
        [
            "single_agent_direction",
            "paper_style_direction",
            "full_system_direction"
        ]
    """
    if len(system_direction_columns) < 2:
        raise ValueError(
            "Need at least two systems to compute correlated error rate."
        )

    actual = df.apply(
        lambda row: actual_direction(
            row["baseline_price"],
            row["actual_target_price"]
        ),
        axis=1
    )

    predictions = df[system_direction_columns]

    all_wrong = predictions.ne(actual, axis=0).all(axis=1)
    same_wrong_prediction = predictions.nunique(axis=1) == 1

    correlated_errors = all_wrong & same_wrong_prediction

    return {
        "correlated_error_rate": correlated_errors.mean(),
        "num_correlated_errors": int(correlated_errors.sum()),
        "num_total": int(len(df)),
    }


def evaluate_predictions(df):
    """
    Combined evaluation for one system.
    """
    direction = directional_accuracy(df)

    ci = wilson_confidence_interval(
        direction["num_correct"],
        direction["num_total"]
    )

    price_error = target_price_percentage_error(df)

    return {
        "directional_accuracy": direction["accuracy"],
        "num_correct": direction["num_correct"],
        "num_total": direction["num_total"],
        "confidence_interval": ci,
        "mean_target_price_percentage_error":
            price_error["mean_absolute_percentage_error"],
        "median_target_price_percentage_error":
            price_error["median_absolute_percentage_error"],
    }
