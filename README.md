# Mining Intelligence Engine (MIE)

A reproducible research machine that asks one question honestly: **where, if anywhere, is there a real predictive advantage in cryptocurrency mining?**

It is built so that every claim is a rerunnable experiment, every negative result is preserved, and no model is promoted without passing explicit gates.

## Headline results

**1. Predicting the hash itself is impossible — proven, not assumed.**
Two independent proofs: 927,905 real blocks show zero exploitable structure, and a round-reduced SHA-256 ladder shows learnability dies between round 4 and 6 of 64 (Bitcoin uses 192). Full-round accuracy: **51.0% vs 50% chance**. See [FINDINGS.md](FINDINGS.md).

**2. Predicting the environment works.**
Champion **BLEND-v1** forecasts difficulty adjustments **33% better than the standard estimator** at k=300 and **43% better than naive** at k=1000, with calibrated 90% intervals (88–93% coverage), on 109 untouched out-of-sample epochs.

**3. Economic value realized so far: $0.** Forecast value ≈ $4.9k/EH/s/epoch vs naive. Stated plainly because it is the truth.

## Quick start
```bash
pip install numpy pandas scipy scikit-learn pytest ruff mypy
make ci                       # lint + types + tests + synthetic/null regression lab
python -m mie state           # rebuild machine-readable project state from the repo
python -m mie validate        # dataset validation
python -m mie next            # next executable experiments from the research queue
python -m mie run REPRO-BLEND-v1                                  # reproduce the champion
python -m mie run EXP-SHA-LADDER '{"N":40000,"rounds":[1,2,4,8,64]}'   # the SHA-256 proof
python -m mie new-iteration "note"                                # reset the 3-experiment governor
```

## Data setup (not stored in git — regenerable)
```bash
git clone https://github.com/nip-333/btc-archive && git clone https://github.com/nip-333/btc-current
python data/build_compact.py             # -> data/processed/headers_compact.npz  (927,905 verified headers)
git clone https://github.com/jptrustlearning/btc btc            # daily BTC/USD
git clone https://github.com/bitcoin-data/block-arrival-times   # CC0 multi-node arrivals
python data/ingest_arrivals.py
```

## Architecture
```
mie/
  data/          dataset registry, manifests, validation (raw data immutable)
  features/      leakage-safe epoch/pace features
  models/        BaseModel: fit/predict/predict_distribution/evaluate/serialize/complexity
  forecasting/   strict chronological walk-forward
  evaluation/    protocols, metrics, strict comparison, promotion gates
  economics/     parametric profit simulator from forecast quantiles
  optimization/  value-of-information across decisions
  statistics/    paired tests, block bootstrap, mutual information + null
  latent_state/  Kalman hashrate filter
  anomaly/ discovery/    CUSUM, lead-lag screen with multiple-testing control
  experiments/   experiment runner, synthetic/null lab, experiment library, SHA-256 ladder
  research/      append-only ledger, executable queue, 3-experiment governor, project state
```
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What makes this different from a typical crypto-ML repo
- **Negative results are preserved, not deleted.** Ten hypotheses were destroyed, including one that reached OR 15.3 / AUC 0.775 before an independent-panel replication killed it.
- **A synthetic regression lab** must pass on every commit: the pipeline has to recover known signals, reject pure noise, survive a 50-feature multiple-testing trap, and flag look-ahead leakage.
- **Promotion gates in code**, not prose: a candidate cannot become champion without beating baseline, champion and a null on identical rows, with paired CIs, calibration, regime checks and an economic test.
- **A 3-experiment governor** enforces compute discipline.
- **The repository is the source of truth.** `python -m mie state` reconstructs the whole project without any chat history.

## Honest limitations
- The champion's realized economic value is zero; it has never been traded or operated on.
- 73% of its remaining error is real hashrate movement that our data does not observe.
- The electricity side of mining — the largest cost — is entirely unobserved here.
- Fee data is subsampled and transcribed; price data is third-party.

## License
MIT (code). Data sources retain their own licenses; see `research_state/dataset_manifest.json`.
