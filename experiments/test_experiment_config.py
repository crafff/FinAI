import pytest

from experiment_config import (
    TICKER_SETS,
    ConfigError,
    ExperimentConfig,
    SystemConfig,
    expand_ticker_set,
    load_experiment_config,
    to_jsonable,
    validate,
)


def _write(tmp_path, text):
    p = tmp_path / "exp.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_inline_tickers_and_systems(tmp_path):
    cfg = load_experiment_config(_write(tmp_path, """
name: t1
tickers: [aapl, MSFT]
systems:
  - {name: single, mode: single}
  - {name: full, mode: leader, subtasks: [fundamental, sentiment], max_rounds: 2}
"""))

    assert cfg.name == "t1"
    assert cfg.tickers == ["AAPL", "MSFT"]            # upper-cased
    assert len(cfg.systems) == 2
    assert cfg.systems[1].subtasks == ["fundamental", "sentiment"]
    assert cfg.systems[1].max_rounds == 2


def test_tickers_file_resolution(tmp_path):
    (tmp_path / "tickers.txt").write_text("AAPL\n# comment\nMSFT\n\n", encoding="utf-8")
    cfg = load_experiment_config(_write(tmp_path, """
name: t2
tickers_file: tickers.txt
systems:
  - {name: single, mode: single}
"""))

    assert cfg.tickers == ["AAPL", "MSFT"]


def test_ticker_set_dow30_expands_via_ticker_set_key(tmp_path):
    cfg = load_experiment_config(_write(tmp_path, """
name: t_dow
ticker_set: dow30
systems:
  - {name: single, mode: single}
"""))

    assert cfg.tickers == TICKER_SETS["dow30"]
    assert len(cfg.tickers) == 30
    assert "AAPL" in cfg.tickers


def test_ticker_set_dow30_shorthand_in_tickers_field(tmp_path):
    cfg = load_experiment_config(_write(tmp_path, """
name: t_dow2
tickers: dow30
systems:
  - {name: single, mode: single}
"""))

    assert cfg.tickers == TICKER_SETS["dow30"]


def test_expand_ticker_set_rejects_unknown():
    with pytest.raises(ConfigError):
        expand_ticker_set("sp500")


def test_to_jsonable_roundtrips_systems(tmp_path):
    cfg = load_experiment_config(_write(tmp_path, """
name: t3
tickers: [AAPL]
systems:
  - {name: full, mode: leader, subtasks: [fundamental], red_team: false}
"""))
    d = to_jsonable(cfg)
    assert d["systems"][0]["name"] == "full"
    assert d["systems"][0]["red_team"] is False


def test_allow_missing_parses_and_defaults_false(tmp_path):
    on = load_experiment_config(_write(tmp_path, """
name: a
tickers: [AAPL]
allow_missing: true
systems:
  - {name: single, mode: single}
"""))
    assert on.allow_missing is True
    assert to_jsonable(on)["allow_missing"] is True

    off = load_experiment_config(_write(tmp_path, """
name: b
tickers: [AAPL]
systems:
  - {name: single, mode: single}
"""))
    assert off.allow_missing is False


def test_engine_parses_and_defaults_pipeline(tmp_path):
    default = load_experiment_config(_write(tmp_path, """
name: a
tickers: [AAPL]
systems:
  - {name: single, mode: single}
"""))
    assert default.engine == "pipeline"
    assert to_jsonable(default)["engine"] == "pipeline"

    lg = load_experiment_config(_write(tmp_path, """
name: b
tickers: [AAPL]
engine: langgraph
systems:
  - {name: single, mode: single}
"""))
    assert lg.engine == "langgraph"
    assert to_jsonable(lg)["engine"] == "langgraph"


def test_validate_rejects_bad_engine():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"],
        systems=[SystemConfig("a", mode="single")],
        engine="quantum",
    )
    with pytest.raises(ConfigError):
        validate(cfg)


def test_validate_rejects_duplicate_names():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"],
        systems=[SystemConfig("a", mode="single"), SystemConfig("a", mode="single")],
    )
    with pytest.raises(ConfigError):
        validate(cfg)


def test_validate_rejects_bad_mode():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"], systems=[SystemConfig("a", mode="wizard")]
    )
    with pytest.raises(ConfigError):
        validate(cfg)


def test_validate_rejects_unknown_subtask():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"],
        systems=[SystemConfig("a", mode="leader", subtasks=["nope"])],
    )
    with pytest.raises(ConfigError):
        validate(cfg)


def test_validate_rejects_negative_rounds():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"],
        systems=[SystemConfig("a", mode="leader", subtasks=["fundamental"], max_rounds=-1)],
    )
    with pytest.raises(ConfigError):
        validate(cfg)


def test_validate_rejects_leader_without_subtasks():
    cfg = ExperimentConfig(
        name="x", tickers=["AAPL"], systems=[SystemConfig("a", mode="leader")]
    )
    with pytest.raises(ConfigError):
        validate(cfg)
