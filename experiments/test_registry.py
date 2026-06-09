from _kit import (
    FACTORS_JSON,
    FUNDAMENTAL_JSON,
    QUANT_RISK_JSON,
    RISK_JSON,
    SENTIMENT_JSON,
    ScriptedClient,
    fake_context,
)
from registry import REGISTRY, available_subtasks


def test_registry_has_expected_agents():
    assert set(available_subtasks()) == {
        "fundamental", "sentiment", "risk",
        "qualitative_risk", "quantitative_risk",
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


def test_quantitative_risk_spec_wraps_score_into_assessment():
    spec = REGISTRY["quantitative_risk"]
    report = spec.run(fake_context(), ScriptedClient([QUANT_RISK_JSON]))

    assert "scores" in report
    assert report["scores"][0]["method"] == "quantitative"

    evidence = spec.render(report)
    assert len(evidence["scores"]) == 1


def test_risk_spec_runs_three_phase_protocol_with_both_scores():
    spec = REGISTRY["risk"]
    # Phase 1 (factors) -> qualitative score -> quantitative score.
    report = spec.run(
        fake_context(),
        ScriptedClient([FACTORS_JSON, RISK_JSON, QUANT_RISK_JSON]),
    )

    # The full protocol carries the shared factors plus BOTH scores, unaveraged.
    assert report["collected_factors"] == ["leverage", "competition"]
    methods = {s["method"] for s in report["scores"]}
    assert methods == {"qualitative", "quantitative"}

    evidence = spec.render(report)
    assert len(evidence["scores"]) == 2
