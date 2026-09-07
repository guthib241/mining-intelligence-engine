# ITERATION 003 — 2026-09-07 — Infrastructure + Latent-State + Data-Acquisition Sprint
Code: git d34a4e5 → HEAD. Protocol P1 = walk-forward, test epochs 350–458 (n=109), tuning only on epochs <330.

## SCIENTIFIC STATUS
**Established (survived attack):**
- Hash outcomes, single-block timing, and all header fields carry no information about next interval or next adjustment (EXP1/2/11; permutation-null MI all p>0.14).
- Difficulty-adjustment log series is MA(1) in Poisson noise (ac1 = −0.5); daily hashrate proxy is 98% Poisson noise (EXP13, ac1 = −0.494 measured).
- Champion BLEND-v1 beats naive and the standard pace estimator under P1, with calibrated conformal intervals (cov90 0.88–0.93). Its edge over the *standard estimator* is concentrated in the first half of the epoch (k=300: 4.79→3.23; k=500: 3.74→2.95; k=1000: 2.15→2.06).
- Price momentum does not predict hashrate change at epoch (n=109) **or** daily (n=2,190) resolution.

**Provisional:** none carried. The 2.64/2.95 ambiguity is RESOLVED (AUDIT-003-1): 2.95 is the P1 score; 2.64 was the same model on subsample e≥370.

**Rejected this iteration:** Kalman (EXP4), recent-window/EWMA pace (EXP8), CUSUM early-warning switch (EXP9), fingerprint-leads-hashrate (EXP6), orphan/spread association (EXP12, artifact), price→hashrate (EXP7, EXP13).

## SYSTEM STATUS (added)
`lib/forecast.py` (models + conformal quantiles + pinball/coverage eval), `lib/kalman.py`, `lib/econ.py` (parametric simulator), `lib/infotheory.py` (MI + permutation null), `experiments/bench_synthetic.py` (A–J regression suite + power curve), `tools/data_discovery.py`, `research_state/{eval_protocols,dataset_manifest,manifest_arrivals,tool_evaluations,data_sources_examined}.json`, compact dataset (load 2.63s→0.045s).

## LATENT STATE
Within-epoch hashrate transitions are **not detectable before the standard estimator** in normal conditions: recent-window pace is never better and is worse at k≥1000 (2.40 vs 2.06). Poisson noise on any sub-epoch window exceeds typical intra-epoch drift. Only in the single largest structural break (China ban, e=341, outside the test set) did a 300-block window beat from-start pace (−25.8 vs −18.8, actual −32.8). Regime-conditional, n≈1, not promoted.

## EARLY WARNING
No evidence. CUSUM score uncorrelated with forecast error (r=−0.07); a train-selected switching rule never fires on test; model disagreement (contradiction engine) uncorrelated with error at 3 of 4 horizons.

## SYNTHETIC TESTING
False-discovery control PASS: pure noise, 50-feature multiple-testing trap not detected; look-ahead feature flagged by leak check (r_future=1.0). Power: R²≥0.20 → 1.0, R²=0.14 → 0.65, R²=0.08 → 0.20 at N=400. **The pipeline is honest but cannot validate weak signals at epoch n.** Lesson applied: EXP13 moved to a daily target (n≈2,200).

## ECONOMIC SIMULATION
Binary run/curtail decision value of the forecast ≈ **$0** per PH/s per epoch (perfect foresight ≈ $1): decisions flip only at ~zero margin. Linear-exposure forecast value (forward-selling hashrate, revenue hedging): ≈ $3.7k per EH/s per epoch vs naive at k=500, P=$90k (~$96k/yr/EH); vs the standard estimator ≈ 0.8pp of 14-day revenue. Realized economic value: **none demonstrated**. Example MC at D=1.48e14, P=90k, 25 J/TH, $0.06/kWh: 14-day profit/PH mean $11, P(loss)=0.27, break-even $0.064/kWh.

## DATA DISCOVERY (all sources recorded in research_state/data_sources_examined.json)
- **Adopted:** bitcoin-data/block-arrival-times (CC0; 545,791 blocks 2015-08→2026-03; 20 sources incl. stratum job timings). Validated against headers (246 mismatches = competing blocks).
- Evaluated, no offline data: bboerst/stratum-work (collector code only).
- Not found on GitHub via 13 paced queries: fee/mempool history, hashprice, pool share stats, ASIC fleet data, energy/curtailment. Erojo/mempool (9 KB, 2021) too small; CBECI-derived repo is energy-model output, not miner data.

## GITHUB RESEARCH (research_state/tool_evaluations.json)
MLflow / Optuna / PyG / TransformerLens / OpenAI evals / MCP servers: **not adopted**, each with a stated reason (Optuna deliberately avoided: n=109 makes hyperparameter search a multiple-testing engine).

## MODEL BEHAVIOR (measured, not narrated)
Champion responds to two inputs. Ablation under P1: intercept-only 3.34; +prev-epoch adj 3.27 (k=0); +from-start pace: 2.95 (k=500), 2.06 (k=1000), 1.10 (k=1500). Pace weight grows with k; momentum term ≈ drift correction only. Intervals: residual-quantile conformal, width90 13.2% (k=500) vs 17.4% drift-only.

## PERFORMANCE
Data load 2.63s → 0.045s (compact npz, 161 MB → 8 MB). Full P1 evaluation across 6 horizons: 5.4s. Synthetic bench: 21s. Peak memory 280 MB (pickle) → <10 MB (npz).

## FAILURES (destroyed)
Momentum (regime artifact) · behavioural mean-reversion (mechanical MA(1)) · Kalman (no gain) · fingerprint-as-predictor · early-warning/CUSUM · orphan-rate trend (source-composition confound) · propagation-spread metric (observer heterogeneity: one node lagged 54s median in 2024Q4) · price→hashrate at two resolutions.

## NEW OBSERVATIONS (L0/L1, no predictive use found)
Header timestamps lag arrival by median 18.5s. Fastest pool issues empty work 0.19s *before* observer nodes see the block; full template +0.36s later (empty-block window ≈0.06% of blocks). Version-bit rolling adoption 0.7% (2017) → 96.7% (2025). Nonce low-byte fingerprint (iteration 002).

## HEADER-ONLY EXHAUSTION TEST
Feature families: hash, nonce (bits/bytes/eras), version, timestamp (raw, mod, error vs arrival), bits/difficulty, intervals, epoch position — tested. Statistical families: uniformity/KS, autocorrelation, chi²/ANOVA, OLS walk-forward, Kalman, conformal, MI/permutation, CUSUM, block-bootstrap — tested. Temporal structures: block, daily, epoch, lead/lag ±2 — tested. Null controls, multiple-testing correction, synthetic recovery — completed. Independent validation (external reproduction) — not done. Classification: **TEMPORARILY EXHAUSTED for forecasting**. Not impossible; not blocked.

## NEXT RESEARCH QUEUE (value = gain×impact×p / (cost×risk))
1. H016 arrival-based pace for small k (1.2) — cheap, low expected gain.
2. H017 header-ts-error as pool fingerprint (1.4) — descriptive.
3. Per-node calibrated propagation metric from peer-observer set (new) — prerequisite for any stale-risk work.
4. H012/H009 fee-conditional hypotheses — BLOCKED on fee data (pre-escalation search recorded; Kaggle/BigQuery/blockchair are outside the sandbox allowlist).
5. H006 tradeability — BLOCKED on derivative prices.

## NEXT EXECUTED EXPERIMENT
H016 (queued, executable now). Pre-escalation status for fee/hashprice data: GitHub exhausted for this pass; remaining avenues are PyPI-bundled samples and public research archives, to be searched next iteration before any request is made.
