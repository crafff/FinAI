import pytest
import pandas as pd

from metrics import (
    actual_direction,
    directional_accuracy,
    wilson_confidence_interval,
    target_price_percentage_error,
    correlated_error_rate,
    evaluate_predictions,
)


def test_actual_direction_buy():
    assert actual_direction(100, 105) == "Buy"


def test_actual_direction_not_buy_when_lower():
    assert actual_direction(100, 95) == "Not Buy"


def test_actual_direction_not_buy_when_flat():
    assert actual_direction(100, 100) == "Not Buy"


def test_directional_accuracy():
    df = pd.DataFrame({
        "predicted_direction": ["Buy", "Not Buy", "Buy"],
        "baseline_price": [100, 100, 100],
        "actual_target_price": [110, 90, 95],
    })

    result = directional_accuracy(df)

    assert result["num_correct"] == 2
    assert result["num_total"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_wilson_confidence_interval():
    result = wilson_confidence_interval(8, 10)

    assert 0 <= result["lower"] <= result["upper"] <= 1


def test_target_price_percentage_error():
    df = pd.DataFrame({
        "predicted_target_price": [110, 90],
        "actual_target_price": [100, 100],
    })

    result = target_price_percentage_error(df)

    assert result["mean_absolute_percentage_error"] == pytest.approx(0.10)
    assert result["median_absolute_percentage_error"] == pytest.approx(0.10)


def test_target_price_percentage_error_rejects_zero_actual():
    df = pd.DataFrame({
        "predicted_target_price": [100],
        "actual_target_price": [0],
    })

    with pytest.raises(ValueError):
        target_price_percentage_error(df)


def test_correlated_error_rate_same_wrong_direction():
    df = pd.DataFrame({
        "baseline_price": [100, 100, 100, 100],
        "actual_target_price": [110, 90, 110, 90],
        "single_agent_direction": ["Not Buy", "Buy", "Buy", "Not Buy"],
        "paper_style_direction": ["Not Buy", "Buy", "Not Buy", "Not Buy"],
        "full_system_direction": ["Not Buy", "Buy", "Buy", "Buy"],
    })

    result = correlated_error_rate(
        df,
        [
            "single_agent_direction",
            "paper_style_direction",
            "full_system_direction",
        ]
    )

    assert result["num_correlated_errors"] == 2
    assert result["num_total"] == 4
    assert result["correlated_error_rate"] == pytest.approx(0.50)


def test_correlated_error_rate_does_not_count_when_one_system_is_correct():
    df = pd.DataFrame({
        "baseline_price": [100],
        "actual_target_price": [100],
        # Actual direction is Not Buy because target is flat.
        # The paper-style system predicts Not Buy, so one system is correct.
        # Therefore, this should not count as a correlated error.
        "single_agent_direction": ["Buy"],
        "paper_style_direction": ["Not Buy"],
        "full_system_direction": ["Buy"],
    })

    result = correlated_error_rate(
        df,
        [
            "single_agent_direction",
            "paper_style_direction",
            "full_system_direction",
        ]
    )

    assert result["num_correlated_errors"] == 0
    assert result["correlated_error_rate"] == pytest.approx(0.0)


def test_evaluate_predictions():
    df = pd.DataFrame({
        "predicted_direction": ["Buy", "Not Buy"],
        "predicted_target_price": [110, 95],
        "baseline_price": [100, 100],
        "actual_target_price": [105, 90],
    })

    result = evaluate_predictions(df)

    assert result["num_total"] == 2
    assert "directional_accuracy" in result
    assert "confidence_interval" in result
    assert "mean_target_price_percentage_error" in result

def test_correlated_error_rate_requires_at_least_two_systems():
    df = pd.DataFrame({
        "baseline_price": [100],
        "actual_target_price": [110],
        "single_agent_direction": ["Buy"],
    })

    with pytest.raises(ValueError):
        correlated_error_rate(
            df,
            ["single_agent_direction"]
        )
