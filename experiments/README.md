# Experiments

A config-driven harness (Task 19) for running and comparing system
configurations over one or many tickers. It supersedes the three standalone
agent runners by putting the data loading and pipeline wiring in one place.

## Run

```bash
uv run --extra rag python experiments/run_experiment.py experiments/configs/ablation.yaml
# quick single stock (override the config's ticker list):
uv run --extra rag python experiments/run_experiment.py experiments/configs/full.yaml --tickers AAPL
# force a clean re-run (ignore cached cells):
uv run --extra rag python experiments/run_experiment.py experiments/configs/ablation.yaml --no-resume
```

Needs `FMP_API_KEY`, `FINNHUB_API_KEY`, and the LLM backend credential in
`.env`; `REDDIT_*` is optional. 10-Ks must already be cached
(`data/EDGAR_retrieval/run_fetch.py`).

## Config (YAML)

```yaml
name: ablation_v1
tickers: [AAPL, MSFT, NVDA]      # inline list, OR a named set / file (see below)
seed: 42                          # reproducible bootstrap in compare_systems
systems:
  - {name: single, mode: single}                       # single-agent baseline
  - name: full                                          # full coopetition system
    mode: leader
    subtasks: [fundamental, sentiment, qualitative_risk]
    red_team: true
    max_rounds: 3
  - {name: full_no_redteam, mode: leader,
     subtasks: [fundamental, sentiment, qualitative_risk], red_team: false}
  - {name: fundamentals_only, mode: leader,
     subtasks: [fundamental], red_team: true, max_rounds: 1}
```

Tickers can be given four ways (priority order): `ticker_set: dow30` (a named
universe), the shorthand `tickers: dow30`, `tickers_file: path/to/list.txt`
(one per line), or an inline `tickers: [...]` list. The only named set today is
`dow30` (the 30 Dow constituents, from `edgar_retrieval.DOW_30`); the CLI
accepts it too: `--tickers dow30`. See `configs/ablation_dow30.yaml`.

- **mode** — `single` (one agent sees all evidence) or `leader` (sub-task
  agents → leader → optional red-team loop).
- **subtasks** — any subset/order of the registered agents
  (`fundamental`, `sentiment`, `qualitative_risk`). Add a new agent by
  registering one `SubtaskSpec` in `registry.py`; it becomes selectable by
  name with no other changes.
- **red_team** / **max_rounds** — toggle the Stage-3 rebuttal loop and cap its
  rounds (`red_team: false` or `max_rounds: 0` ⇒ the Leader's initial call is
  final).

Experiment-level option **`allow_missing: true`** (default false): if a ticker's
financials (FMP), news (FinnHub), or social (Reddit) can't be fetched — e.g. an
FMP `402 Payment Required`, a quota error, or a missing key — that source
degrades to empty (`{}` / `[]`) and the run continues, instead of skipping the
whole ticker. The 10-K and prices remain required. Each ticker's
`data_context.json` records which sources were `missing`, and with
`allow_missing` the CLI no longer hard-requires `FMP_API_KEY` / `FINNHUB_API_KEY`
(only the LLM credential).

## Output (`runs/<name>/`)

```
config.json            resolved config snapshot (reproducible)
results.csv            wide ablation table, one row per ticker
metrics.json / .md     per-system accuracy + Wilson CI + target-price MAPE
compare.json           pairwise McNemar/Wilcoxon/bootstrap + correlated-error
summary.json           headline numbers + run metadata + errors
errors.log
per_ticker/<TICKER>/
  data_context.json    t0, baseline, target (answer key), counts
  <system_name>/       transcript.{json,md}, subtask_reports/<name>.json,
                       leader_prediction.json, rebuttals.json,
                       leader_responses.json, final_prediction.json,
                       record.json, meta.json
```

Robustness: one failing `(ticker, system)` cell is logged to `errors.log` and
skipped, never aborting the sweep; completed cells are skipped on re-run
(`resume`, on by default), so a long multi-stock run is restartable.

## Re-running: resume vs. fresh results

The experiment directory is **stable** (`runs/<name>/`, no timestamp), so
re-running the same config updates that directory in place rather than creating
a new one. What happens on a re-run depends on `resume`:

- **Resume (default, `resume: true`)** — each `(ticker, system)` cell whose
  `record.json` already exists is **skipped** (its cached result is reused);
  cells that are missing or previously **errored are retried**; the aggregate
  files (`results.csv`, `metrics.json`, `summary.json`, …) are always
  recomputed from all records. This is the restartable "continue / fill in the
  gaps" mode — e.g. after an FMP quota resets or you add an API key, just re-run
  and only the failed tickers run again.

- **Force a full re-run (`--no-resume`, or `resume: false`)** — every cell is
  re-executed and **overwritten in the same directory** (cached records are
  ignored). Note this does *not* keep the previous results; it overwrites them.

- **Keep an independent snapshot** — change the experiment `name` (e.g.
  `single_baseline_v2`) so it writes to a new `runs/<new_name>/`. Use this to
  compare different models / dates / prompt versions side by side. (Different
  names are separate directories and do not resume from one another.)

Finer control: to re-run just one stock or one cell, delete its
`runs/<name>/per_ticker/<TICKER>/` directory (whole ticker) or a single
`.../<TICKER>/<system>/record.json` (one cell); the next run rebuilds only what
you removed and leaves everything else untouched.

## Modules

- `experiment_config.py` — `ExperimentConfig` / `SystemConfig` + YAML loader + validation.
- `registry.py` — the `SubtaskSpec` registry (run + render per sub-task agent).
- `context.py` — `build_data_context`: load each ticker's shared inputs once.
- `pipeline.py` — `run_system`: the single/leader pipeline on a `PipelineState`.
- `results.py` — records → wide DataFrame → metrics (reuses `evaluation/`).
- `runner.py` / `run_experiment.py` — orchestration + saving + CLI.

## Tests

```bash
uv run pytest experiments
```

All offline: a scripted client + fake `DataContext` (`_kit.py`) drive the
pipeline, registry, results, and runner (incl. error isolation and resume)
without keys or network.
