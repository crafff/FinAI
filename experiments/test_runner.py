import json
import types

import runner
from experiment_config import ExperimentConfig, SystemConfig


def _settings(tmp_path):
    return types.SimpleNamespace(
        llm=types.SimpleNamespace(model="test-model", backend="test"),
        runs_dir=str(tmp_path),
    )


def _fake_ctx(ticker, settings, allow_missing=False):
    from _kit import fake_context
    return fake_context(ticker)


def _ok_state(ctx):
    return {
        "ticker": ctx.ticker,
        "baseline_price": ctx.baseline_price,
        "prices": ctx.prices,
        "final_prediction": {"direction": "Buy", "target_price": 190.0},
        "round_count": 0,
        "converged": True,
    }


def test_run_experiment_saves_aggregates_and_isolates_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_data_context", _fake_ctx)

    def fake_run_system(system, ctx, client):
        if system.name == "boom":
            raise RuntimeError("kaboom")
        return _ok_state(ctx)

    monkeypatch.setattr(runner, "run_system", fake_run_system)

    config = ExperimentConfig(
        name="exp", tickers=["AAPL", "MSFT"],
        systems=[SystemConfig("single", mode="single"),
                 SystemConfig("boom", mode="single")],
    )

    exp_dir = runner.run_experiment(config, _settings(tmp_path))

    # Aggregates written from the successful cells.
    assert (exp_dir / "results.csv").exists()
    assert (exp_dir / "metrics.json").exists()
    assert (exp_dir / "config.json").exists()

    # Good cells saved; the failing system has no record and is logged.
    assert (exp_dir / "per_ticker" / "AAPL" / "single" / "record.json").exists()
    assert not (exp_dir / "per_ticker" / "AAPL" / "boom" / "record.json").exists()
    assert (exp_dir / "errors.log").exists()

    summary = json.loads((exp_dir / "summary.json").read_text())
    assert summary["num_tickers"] == 2          # both tickers have a 'single' row
    assert len(summary["errors"]) == 2          # boom failed on both tickers


def test_engine_setting_is_resolved_recorded_and_used(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_data_context", _fake_ctx)

    used = {}

    def fake_resolve(engine):
        used["engine"] = engine

        def run_cell(system, ctx, client, settings=None):
            used["called"] = True
            return _ok_state(ctx)

        return run_cell

    monkeypatch.setattr(runner, "_resolve_engine", fake_resolve)

    config = ExperimentConfig(
        name="exp_engine", tickers=["AAPL"],
        systems=[SystemConfig("single", mode="single")],
        engine="langgraph",
    )

    exp_dir = runner.run_experiment(config, _settings(tmp_path))

    # The configured engine is resolved once and used to run the cell.
    assert used == {"engine": "langgraph", "called": True}

    # ...and recorded in both the resolved config snapshot and the cell meta.
    cfg = json.loads((exp_dir / "config.json").read_text())
    assert cfg["engine"] == "langgraph"
    meta = json.loads(
        (exp_dir / "per_ticker" / "AAPL" / "single" / "meta.json").read_text()
    )
    assert meta["engine"] == "langgraph"


def test_resume_skips_completed_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_data_context", _fake_ctx)

    calls = {"n": 0}

    def counting_run_system(system, ctx, client):
        calls["n"] += 1
        return _ok_state(ctx)

    monkeypatch.setattr(runner, "run_system", counting_run_system)

    config = ExperimentConfig(
        name="exp_resume", tickers=["AAPL"],
        systems=[SystemConfig("single", mode="single")],
        resume=True,
    )

    runner.run_experiment(config, _settings(tmp_path))
    assert calls["n"] == 1

    # Second run: the completed cell is served from its cached record.json.
    runner.run_experiment(config, _settings(tmp_path))
    assert calls["n"] == 1
