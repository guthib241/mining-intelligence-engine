# FINDINGS — Can AI predict Bitcoin mining?

Every number below is produced by code in this repository, on data described in `research_state/dataset_manifest.json`, under a registered evaluation protocol. Nothing is quoted from memory.

---

## Part 1 — Predicting the mining OUTPUT (the hash itself)

**Answer: no. Proven twice, independently.**

### Proof A — real blockchain data (`experiments/exp1_hash.py`)
927,905 real Bitcoin block headers (genesis → 2025-12-14), verified against consensus rules (correct genesis hash, 0 broken prev-hash links, 0 PoW failures).

| Test | Result | Expected if unpredictable |
|---|---|---|
| hash/target ratio distribution | KS p = 0.80 | uniform |
| autocorrelation, lags 1/2/3/10/100/2016 | all abs(r) < 0.002 (2se = 0.002) | zero |
| previous nonce parity → next hash bit | chi2 p = 0.71 | no relation |
| previous outcome → next hash bit | chi2 p = 0.65 | no relation |
| hour of day → outcome | ANOVA p = 0.48 | no relation |
| mutual information, 6 header features | all p > 0.14 (permutation null) | zero |

### Proof B — the algorithm itself (`mie/experiments/sha_ladder.py`)
SHA-256 implemented from FIPS 180-4 with a configurable round count (verified bit-for-bit against Python `hashlib`), unlimited generated data, three model families (logistic regression, MLP, gradient boosting), 40,000 samples per rung, 5 output bits each.

| Rounds (of 64) | 1 input-bit flip changes | Best model accuracy | Verdict |
|---|---|---|---|
| 1 | 0.4% of output bits | **100%** | fully learnable |
| 2 | 0% | **100%** | learnable |
| 4 | 6.8% | **100%** | learnable |
| 6 | — | 50.8% (z=1.6) | chance |
| 8 | 26% | 51.4% (z=2.7) | chance |
| 16 | 49.8% | 50.8% | chance |
| 24 | 50.0% | 50.6% | chance |
| **64 (full)** | **50.0%** | **51.0%** — logreg 50.5, MLP 49.4, GBT 50.3 | **chance** |

Bonferroni threshold z = 3.4. **The learnability wall sits between round 4 and round 6.** Bitcoin hashes an 80-byte header with SHA-256 *twice* — three compression passes, **192 rounds**. Learnable structure dies at round ~5.

**Conclusion: direct proof-of-work output prediction is closed. Not "hard" — measured at chance, from two independent directions.**

---

## Part 2 — Predicting the mining ENVIRONMENT (what does work)

Champion **BLEND-v1**: walk-forward linear model on (from-start pace extrapolation, previous epoch's adjustment), predicting the next difficulty adjustment. Protocol P1: test epochs 350–458 (n=109, 2020–2025), tuning permitted only on epochs <330.

| Horizon | Naive (no change) | Standard estimator | **BLEND-v1** | vs naive | vs standard |
|---|---|---|---|---|---|
| k=0 | 3.61% | — | **3.27%** | 9% better | — |
| k=300 | 3.61% | 4.79% | **3.23%** | 11% better | **33% better** |
| k=500 | 3.61% | 3.74% | **2.96%** | 18% better | 21% better |
| k=1000 | 3.61% | 2.15% | **2.06%** | 43% better | 4% better |
| k=1500 | 3.61% | 1.15% | **1.10%** | 70% better | 4% better |

Key practical finding: **the pace extrapolation every block explorer displays is worse than assuming no change for the first ~600 blocks of an epoch.** Proper shrinkage fixes this.

Prediction intervals are conformal and calibrated: 90% intervals cover 88–93%.

### How much of the possible value is captured
At block 500, the remaining 1,515 blocks are random; their timing alone imposes a floor of 1.54pp MAE that **no model can cross**.

| Horizon | Champion | Achievable floor | **Captured** |
|---|---|---|---|
| k=0 | 3.27 | 1.78 | **18.3%** |
| k=500 | 2.96 | 1.54 | **31.6%** |
| k=1000 | 2.06 | 1.26 | **66.1%** |
| k=1500 | 1.10 | 0.90 | **92.5%** |

### Where the remaining error lives (`reports/blend_residual_decomposition.json`)
- 27% of residual variance = irreducible randomness.
- **73% = real hashrate moving during the unobserved remainder of the epoch** — a variable absent from our data (fleet deployment, curtailment).
- Of six covariates screened, exactly one survives Bonferroni correction: prior-adjustment magnitude predicts error size (rho = −0.348, p = 0.0002).

---

## Part 3 — What was tested and DESTROYED

Kept as evidence: negative results are results.

| Hypothesis | Why it looked good | How it died |
|---|---|---|
| Epoch-to-epoch momentum | r = +0.57 over full history | Entirely 2010–2015 growth regime; r = 0.03 since 2016 |
| 2023+ mean reversion | r = −0.27, p = 0.04 | Mechanical MA(1) artifact of Poisson block timing |
| Kalman filter on hashrate | principled state-space model | No gain vs the simple blend (p = 0.66) |
| Price → hashrate | plausible economics | Null at epoch (n=109) and daily (n=2,190) resolution |
| Nonce/ASIC fingerprint | strong, real, undocumented pattern | Coincident with hashrate, not leading (16 tests, none survive) |
| Recent-window / EWMA pace | should react faster | Never better; worse at k≥1000 |
| CUSUM early warning | classic change detection | Uncorrelated with error; rule never fires out-of-sample |
| **Propagation → orphan risk** | **OR 15.3, AUC 0.775 on a temporal holdout** | **Killed by disjoint-observer replication (AUC 0.54). Feature and label shared observer nodes.** |
| Arrival-time pace | true timestamps beat miner timestamps | Zero difference (all CIs straddle 0) |
| Residual covariates (12) | many candidates | 0 of 24 survive Bonferroni |
| **Within-epoch pace trend (`slope`) + ridge** | **corr with champion residual on tuning epochs: rho = −0.189, p = 0.004 at k=1500; in-sample residual sd 1.657 → 1.578pp** | **Did not replicate out-of-sample. Worse than champion at k=500 and k=1000; at k=1500 the paired test is null (p = 0.70) and the candidate wins only 48% of epochs — the −3.3% headline is carried by a few large residuals, not consistent gain.** |

**Lesson recorded:** a temporal holdout is *not* independent validation when the feature and the label come from the same measurement source.

**Lesson recorded (EXP26):** a covariate that correlates with the champion's residual *on the tuning epochs* is not a predictor. Selecting `alpha` and the feature set on the test epochs would have reported k=1500 as a 3.3% improvement; the tuning/test split is what prevented that. Ridge bought nothing — with 2–3 features the 60-row window is not over-fitting enough for a penalty to help, and the selected alpha was 0.0 at two of five horizons.

---

## Part 4 — Economics, honestly

Per EH/s per 14-day epoch, from `reports/voi_target_map.json`:

| Component | Share of revenue variance | Perfect-information value | Captured today |
|---|---|---|---|
| Price | 39–82% | $61k | $0 (unforecastable; it is a hedging problem) |
| Fees | 1–46% (episodic) | $46k | ~$4–5k |
| **Difficulty** | **8–26%** | **$32k** | **$4.9k vs naive / $8.8k vs standard** |

- Binary run/stop decision value of the difficulty forecast: **≈ $0** — the decision only flips at near-zero margin.
  **EXP28 generalized this across 32 scenarios** ($0.03–$0.25/kWh × 17–30 J/TH): best forecast gain **$2.10 per epoch**
  on $1,620 of revenue (0.13%), *perfect foresight caps at the same $2.10*, and in 5 of 32 scenarios the forecast makes
  the decision **worse**. The decision is driven by electricity price (109/109 epochs run at $0.03/kWh, 33/109 at $0.25),
  not by difficulty. **An AI deciding when to mine should forecast electricity price, not difficulty.**
- **Realized economic value to date: $0.** Nothing has been traded or operated on.

**The strategic finding:** forecast value is bounded by the target's variance share. Six iterations optimized the *smallest* component.

---

## Current status
- Champion: BLEND-v1, level L4 (out-of-sample, calibrated), reproduced by the engine and locked by a regression test.
- Direct PoW prediction: **closed with proof**.
- Next bottleneck: fleet-deployment data (73% of the residual), then the electricity side (the biggest cost in mining and entirely unobserved).
