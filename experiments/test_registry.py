from _kit import (
    FUNDAMENTAL_JSON,
    RISK_JSON,
    SENTIMENT_JSON,
    ScriptedClient,
    fake_context,
)
from registry import REGISTRY, available_subtasks


def test_registry_has_expected_agents():
    assert set(available_subtasks()) == {
        "fundamental", "sentiment", "qualitative_risk"
    }


def test_fundamental_spec_runs_and_renders():
    spec = REGISTRY["fundamental"]
    report = spec.run(fake_context(), ScriptedClient([FUNDAMENTAL_JSON]))

    assert report["signal"] == "bullish"
    evidence = spec.render(report)
    assert evidence["summary"] == "s"
    assert "key_metrics" in evidence


def test_sentiment_spec_runs_and_renders():
    spec = REGISTRY["sentiment"]
    report = spec.run(fake_context(), ScriptedClient([SENTIMENT_JSON]))

    assert report["signal"] == "mixed"
    evidence = spec.render(report)
    assert "disagreement" in evidence


def test_qualitative_risk_spec_wraps_score_into_assessment():
    spec = REGISTRY["qualitative_risk"]
    report = spec.run(fake_context(), ScriptedClient([RISK_JSON]))

    # The single RiskScore is wrapped into a RiskAssessment.
    assert "scores" in report
    assert report["scores"][0]["method"] == "qualitative"

    evidence = spec.render(report)
    assert len(evidence["scores"]) == 1
