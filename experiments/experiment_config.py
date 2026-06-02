"""
Experiment configuration: dataclasses + YAML loader + validation.

An ExperimentConfig describes one experiment - which tickers to run and which
*systems* (system configurations) to run on each. A SystemConfig is one point
in the ablation space: the single-agent baseline, or the leader pipeline with
an arbitrary subset of sub-task agents, the red-team loop on or off, and a
configurable round cap.

YAML keeps configs human-friendly (comments, terse lists). See
experiments/configs/*.yaml for ready-made examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from edgar_retrieval import DOW_30
from registry import REGISTRY


VALID_MODES = ("single", "leader")

# Named ticker universes usable directly from a config (`ticker_set: dow30`
# or the shorthand `tickers: dow30`) and from the CLI (`--tickers dow30`).
TICKER_SETS = {
    "dow30": list(DOW_30),
}


@dataclass
class SystemConfig:
    """
    One system configuration (one column in the ablation table).

    name:       unique label; used as the output dir and the ablation column
                prefix (`<name>_direction` / `<name>_target_price`).
    mode:       "single" (one agent sees everything) or "leader" (sub-task
                agents -> leader -> optional red-team loop).
    subtasks:   leader mode only - any subset/order of registered agent names.
    red_team:   leader mode only - run the Stage-3 rebuttal loop.
    max_rounds: red-team loop cap (0, or red_team=False, => no loop).
    """

    name: str
    mode: str = "leader"
    subtasks: list[str] = field(default_factory=list)
    red_team: bool = True
    max_rounds: int = 3


@dataclass
class ExperimentConfig:
    name: str
    tickers: list[str]
    systems: list[SystemConfig]
    model: str | None = None        # overrides settings.llm.model
    backend: str | None = None      # overrides settings.llm.backend
    seed: int | None = None         # bootstrap reproducibility for compare_systems
    alpha: float = 0.05
    n_boot: int = 10_000
    resume: bool = True             # skip (ticker, system) cells already done
    allow_missing: bool = False     # degrade missing financials/news/social to empty


class ConfigError(ValueError):
    """Raised when an experiment config is malformed."""


def expand_ticker_set(name: str) -> list[str]:
    """Resolve a named ticker universe (e.g. 'dow30') to its ticker list."""
    key = name.strip().lower()
    if key not in TICKER_SETS:
        raise ConfigError(
            f"unknown ticker_set {name!r}; available: {list(TICKER_SETS)}."
        )
    return list(TICKER_SETS[key])


def _resolve_tickers(data: dict, base_dir: Path) -> list[str]:
    """
    Read tickers from, in priority order:
      - `ticker_set: dow30`   (a named universe), or
      - `tickers: dow30`      (string shorthand for a named universe), or
      - `tickers_file: path`  (one ticker per line), or
      - `tickers: [...]`      (an inline list).
    """
    if data.get("ticker_set"):
        tickers = expand_ticker_set(data["ticker_set"])
    elif isinstance(data.get("tickers"), str):
        tickers = expand_ticker_set(data["tickers"])
    elif data.get("tickers_file"):
        path = Path(data["tickers_file"])
        if not path.is_absolute():
            path = base_dir / path
        lines = path.read_text(encoding="utf-8").splitlines()
        tickers = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    else:
        tickers = list(data.get("tickers", []))

    return [t.upper() for t in tickers]


def _build_systems(raw_systems) -> list[SystemConfig]:
    systems = []
    for raw in raw_systems or []:
        systems.append(SystemConfig(
            name=raw["name"],
            mode=raw.get("mode", "leader"),
            subtasks=list(raw.get("subtasks", [])),
            red_team=bool(raw.get("red_team", True)),
            max_rounds=int(raw.get("max_rounds", 3)),
        ))
    return systems


def validate(config: ExperimentConfig) -> None:
    """Raise ConfigError on any malformed field."""
    if not config.name:
        raise ConfigError("experiment 'name' is required.")
    if not config.tickers:
        raise ConfigError("at least one ticker is required.")
    if not config.systems:
        raise ConfigError("at least one system is required.")

    seen = set()
    for sys in config.systems:
        if sys.name in seen:
            raise ConfigError(f"duplicate system name: {sys.name!r}")
        seen.add(sys.name)

        if sys.mode not in VALID_MODES:
            raise ConfigError(
                f"system {sys.name!r}: mode must be one of {VALID_MODES}, "
                f"got {sys.mode!r}."
            )

        if sys.max_rounds < 0:
            raise ConfigError(f"system {sys.name!r}: max_rounds must be >= 0.")

        if sys.mode == "leader":
            if not sys.subtasks:
                raise ConfigError(
                    f"system {sys.name!r}: leader mode needs at least one subtask."
                )
            unknown = [s for s in sys.subtasks if s not in REGISTRY]
            if unknown:
                raise ConfigError(
                    f"system {sys.name!r}: unknown subtasks {unknown}; "
                    f"registered: {list(REGISTRY)}."
                )


def load_experiment_config(path) -> ExperimentConfig:
    """Parse + validate a YAML experiment config file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    config = ExperimentConfig(
        name=data.get("name", ""),
        tickers=_resolve_tickers(data, path.parent),
        systems=_build_systems(data.get("systems")),
        model=data.get("model"),
        backend=data.get("backend"),
        seed=data.get("seed"),
        alpha=float(data.get("alpha", 0.05)),
        n_boot=int(data.get("n_boot", 10_000)),
        resume=bool(data.get("resume", True)),
        allow_missing=bool(data.get("allow_missing", False)),
    )

    validate(config)
    return config


def to_jsonable(config: ExperimentConfig) -> dict:
    """A plain-dict snapshot of the resolved config (for config.json)."""
    return {
        "name": config.name,
        "tickers": config.tickers,
        "model": config.model,
        "backend": config.backend,
        "seed": config.seed,
        "alpha": config.alpha,
        "n_boot": config.n_boot,
        "resume": config.resume,
        "allow_missing": config.allow_missing,
        "systems": [
            {
                "name": s.name,
                "mode": s.mode,
                "subtasks": s.subtasks,
                "red_team": s.red_team,
                "max_rounds": s.max_rounds,
            }
            for s in config.systems
        ],
    }
