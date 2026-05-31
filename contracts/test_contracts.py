from datetime import date, datetime
from zoneinfo import ZoneInfo

import schemas
import state
from schemas import (
    Filing,
    Financials,
    Prediction,
    Prices,
    RiskScore,
    missing_keys,
)
from state import (
    DEFAULT_MAX_ROUNDS,
    current_prediction,
    new_state,
    should_continue_rebuttal,
    to_ablation_record,
)


NY = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------
# Cross-module consistency: the contract's direction constants must equal
# the ones evaluation/metrics.py already uses.
# --------------------------------------------------------------------------

def test_direction_constants_match_metrics():
    import metrics

    assert schemas.BUY == metrics.BUY
    assert schemas.NOT_BUY == metrics.NOT_BUY


# --------------------------------------------------------------------------
# missing_keys validation helper
# --------------------------------------------------------------------------

def test_missing_keys_empty_when_conforming():
    prediction: Prediction = {
        "direction": "Buy",
        "target_price": 150.0,
        "confidence": 0.7,
        "rationale": "fundamentals strong, sentiment positive",
        "dominant_signal": "fundamental",
        "risk_reconciliation": "qual high but quant moderate; net acceptable",
    }

    assert missing_keys(Prediction, prediction) == set()


def test_missing_keys_detects_absent_field():
    incomplete = {"direction": "Buy", "target_price": 150.0}

    assert "rationale" in missing_keys(Prediction, incomplete)


def test_risk_score_conforms():
    score: RiskScore = {
        "method": "quantitative",
        "score": 6.5,
        "summary": "elevated leverage",
        "factors": ["high debt", "litigation"],
        "justification": "weighted model over balance-sheet ratios",
    }

    assert missing_keys(RiskScore, score) == set()


# --------------------------------------------------------------------------
# Conformance of representative data-record shapes (mirroring the real
# Task 1-4 module outputs).
# --------------------------------------------------------------------------

def test_filing_record_shape_conforms():
    filing: Filing = {
        "ticker": "AAPL",
        "cik": "0000320193",
        "accession_number": "0000320193-25-000079",
        "form": "10-K",
        "filing_date": date(2025, 10, 31),
        "filing_timestamp_et": datetime(2025, 10, 31, 18, 1, tzinfo=NY),
        "report_date": date(2025, 9, 27),
        "primary_document": "aapl-20250927.htm",
        "primary_document_url": "https://example.com/doc.htm",
        "html_path": None,
        "text_path": None,
        "text": "...",
    }

    assert missing_keys(Filing, filing) == set()


def test_prices_and_financials_shapes_conform():
    prices: Prices = {
        "ticker": "AAPL",
        "t0_date": date(2025, 11, 3),
        "target_date": date(2025, 11, 10),
        "baseline_price": 150.0,
        "target_price": 155.0,
        "pre_release_trend": [{"date": date(2025, 11, 3), "close": 150.0}],
    }

    financials: Financials = {
        "ticker": "AAPL",
        "fiscal_year": 2025,
        "report_date": "2025-09-27",
        "profitability": {
            "revenue": 1000.0, "net_income": 250.0, "gross_margin": 0.4,
            "operating_margin": 0.3, "net_margin": 0.25,
            "return_on_equity": 0.5, "return_on_assets": 0.2,
        },
        "cash_flow": {
            "operating_cash_flow": 500.0, "capital_expenditure": -100.0,
            "free_cash_flow": 400.0,
        },
        "debt": {
            "total_debt": 900.0, "total_equity": 600.0,
            "debt_to_equity": 1.5, "current_ratio": 1.1,
            "interest_coverage": 8.0,
        },
        "valuation": {
            "pe_ratio": 30.0, "pb_ratio": 5.0,
            "price_to_sales": 7.0, "ev_to_ebitda": 20.0,
        },
    }

    assert missing_keys(Prices, prices) == set()
    assert missing_keys(Financials, financials) == set()


# --------------------------------------------------------------------------
# State construction and loop control
# --------------------------------------------------------------------------

def test_new_state_seeds_config_and_loop():
    st = new_state("aapl")

    assert st["ticker"] == "AAPL"
    assert st["config"]["variant"] == "full"
    assert st["config"]["max_rounds"] == DEFAULT_MAX_ROUNDS
    assert st["round_count"] == 0
    assert st["converged"] is False
    assert st["rebuttals"] == []
    assert st["leader_responses"] == []


def test_should_continue_rebuttal_respects_convergence():
    st = new_state("AAPL")
    st["converged"] = True

    assert should_continue_rebuttal(st) is False


def test_should_continue_rebuttal_respects_cap():
    st = new_state("AAPL", max_rounds=3)
    st["round_count"] = 3

    assert should_continue_rebuttal(st) is False


def test_should_continue_rebuttal_true_mid_loop():
    st = new_state("AAPL", max_rounds=3)
    st["round_count"] = 1

    assert should_continue_rebuttal(st) is True


def test_current_prediction_prefers_final():
    st = new_state("AAPL")
    leader: Prediction = {
        "direction": "Buy", "target_price": 150.0, "confidence": 0.6,
        "rationale": "x", "dominant_signal": "fundamental",
        "risk_reconciliation": "y",
    }
    final: Prediction = {**leader, "target_price": 152.0}

    st["leader_prediction"] = leader
    assert current_prediction(st)["target_price"] == 150.0

    st["final_prediction"] = final
    assert current_prediction(st)["target_price"] == 152.0


def test_current_prediction_none_when_empty():
    assert current_prediction(new_state("AAPL")) is None


# --------------------------------------------------------------------------
# to_ablation_record bridges orchestration -> evaluation
# --------------------------------------------------------------------------

def test_to_ablation_record_maps_to_eval_schema():
    st = new_state("AAPL", variant="full")
    st["baseline_price"] = 150.0
    st["prices"] = {
        "ticker": "AAPL",
        "t0_date": date(2025, 11, 3),
        "target_date": date(2025, 11, 10),
        "baseline_price": 150.0,
        "target_price": 155.0,
        "pre_release_trend": [],
    }
    st["final_prediction"] = {
        "direction": "Buy", "target_price": 154.0, "confidence": 0.7,
        "rationale": "r", "dominant_signal": "fundamental",
        "risk_reconciliation": "rr",
    }

    record = to_ablation_record(st)

    assert record["ticker"] == "AAPL"
    assert record["baseline_price"] == 150.0
    assert record["actual_target_price"] == 155.0
    assert record["full_direction"] == "Buy"
    assert record["full_target_price"] == 154.0


def test_to_ablation_record_feeds_evaluate_predictions():
    """
    The projected record must be directly consumable by
    evaluation.metrics, proving the contract closes the loop.
    """
    import pandas as pd

    from metrics import evaluate_predictions

    rows = []

    for variant, pred_price, pred_dir in [
        ("full", 154.0, "Buy"),
    ]:
        st = new_state("AAPL", variant=variant)
        st["baseline_price"] = 150.0
        st["prices"] = {
            "ticker": "AAPL", "t0_date": date(2025, 11, 3),
            "target_date": date(2025, 11, 10), "baseline_price": 150.0,
            "target_price": 155.0, "pre_release_trend": [],
        }
        st["final_prediction"] = {
            "direction": pred_dir, "target_price": pred_price,
            "confidence": 0.7, "rationale": "r",
            "dominant_signal": "f", "risk_reconciliation": "rr",
        }
        rows.append(to_ablation_record(st))

    df = pd.DataFrame(rows).assign(
        predicted_direction=lambda d: d["full_direction"],
        predicted_target_price=lambda d: d["full_target_price"],
    )

    result = evaluate_predictions(df)

    # Actual went 150 -> 155 (Buy); prediction was Buy -> correct.
    assert result["directional_accuracy"] == 1.0
