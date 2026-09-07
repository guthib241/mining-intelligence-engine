# MIE Architecture

Repository = source of truth. `python -m mie state` rebuilds `research_state/project_state.json` from ledger, queue, manifests, champion, governor and git.

## Control loop (implemented)
DATA (`mie.data`, immutable raw + manifests) → VALIDATE (`validate_all`) → FEATURES (`mie.features`, leakage-safe by construction) → HYPOTHESIS/QUEUE (`mie.research.queue`) → EXPERIMENT DESIGN (`mie.experiments.library` registered callables) → MODEL (`mie.models`, common interface) → BACKTEST (`mie.forecasting.walk_forward`, strict chronological) → RESULT COMPARISON (`mie.evaluation.compare`: baseline/champion/candidate/null, deltas, paired CI, percentiles, coverage, regimes) → RED TEAM (null column, gates) → ECONOMIC TEST (`mie.economics`, `mie.optimization`) → PROMOTION/REJECTION (`mie.evaluation.gates`; never automatic) → RESEARCH MEMORY (`mie.research.ledger`, `reports/runs/*.json`) → NEXT EXPERIMENT (`ResearchQueue.next()`, `Governor`).

## Modules
| module | responsibility |
|---|---|
| infrastructure | paths, git provenance, resource tracking, failure taxonomy |
| data | dataset registry, loaders, manifests, validation |
| features | epoch/pace features (`build_frame`) |
| models | `BaseModel` (fit/predict/predict_distribution/evaluate/serialize/load/complexity); NaiveZero, Drift, OLSBlend (=BLEND-v1), Kalman |
| forecasting | walk-forward runner producing identical rows for every model |
| latent_state / anomaly / discovery / statistics | Kalman, CUSUM, lead-lag screen, paired tests, bootstrap, MI-null |
| evaluation | protocols (P1), metrics, strict comparison, promotion gates |
| economics / optimization | parametric simulator from forecast quantiles; VOI and decision value |
| experiments | runner (provenance, resources, failure class, ledger), synthetic lab, experiment library |
| research | ledger, executable queue, 3-experiment governor, project state |

## Protocols
P1 (official): BTCHDR-2025-12-14, rolling 60-epoch OLS, min 20, test epochs 350–458 (n=109), horizons k∈{0,100,300,500,1000,1500}, tuning only on epochs <330. Champion BLEND-v1 scores are a permanent regression test (`tests/test_champion_regression.py`).

## Failure protocol
Runs end in COMPLETED, FAILED(CODE|DATA|ENVIRONMENT|EXPERIMENT|MODEL), or BLOCKED. A FAILED run's verdict is always "NOT A SCIENTIFIC RESULT".

## Adding things
- Dataset: add loader + manifest entry in `mie/data/registry.py` (manifest must carry source, schema, coverage, checksum, validation, limitations, authority).
- Model: subclass `BaseModel`, register in `mie/models/difficulty.py::REGISTRY`.
- Experiment: decorate a function with `@register(id, hypothesis, datasets, primary=True|False)` in `mie/experiments/library.py`; run with `python -m mie run ID '{json cfg}'`.
- Queue item: `ResearchQueue().add(id=..., hypothesis=..., info_gain=, econ_value=, p_success=, compute_cost=, data_cost=, fdr_risk=, experiment="CHALLENGE", ...)`.
