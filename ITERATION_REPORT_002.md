# ITERATION 002 — 2026-09-07

## HYPOTHESES TESTED
N001 pipeline null tests · H010 Kalman hashrate filter · H004 nonce fingerprint leads hashrate · H005 price → hashrate · (H008 mean-reversion resolved by mechanism)

## CHANGE
Refactored EXP3 into `lib/difficulty.py` (pure functions, identical path for real and synthetic data). Added `lib/kalman.py`. Research memory (`research_state/`, `research_ledger/`) and git initialised. Commits: 0b492aa → HEAD.

## DATA
Headers genesis..927,904 (2025-12-14), verified. BTC/USDT daily (Binance via jptrustlearning/btc), validated on 4 reference dates, 0 gaps. Third-party; not authoritative.

## EXPERIMENTS & RESULTS (MAE, % of next difficulty, walk-forward, epochs 2020+)
| horizon | naive | champion blend | Kalman | blend+price |
|---|---|---|---|---|
| k=0 | 3.61 | 3.27 | 3.32 | 3.31 |
| k=500 | 3.61 | 2.95 | 3.07 | 2.79 vs 2.64 same-sample |
| k=1000 | 3.61 | 2.06 | 2.01 | — |

Champion k=0 signed error percentiles (%): P1 −8.3, P5 −7.0, P10 −5.2, P25 −3.2, P50 −0.1, P75 +2.9, P90 +5.2, P95 +6.5, P99 +9.4; mean −0.14, sd 4.17, n=109.

## NULL / SYNTHETIC
Shuffled targets: extrapolation gives zero gain (pipeline clean). Constant-hashrate synthetic: naive 2.83% (theory corrected: MA(1) Poisson, not 1.8%); Kalman with correct q reaches 1.91% vs 1.78% floor (implementation validated).

## RED-TEAM
- Momentum (r=+.57) destroyed as regime artifact (iteration 1).
- 2023+ mean-reversion (r=−.27) explained by MA(1) mechanism — not behavioural, not exploitable beyond what blend already captures.
- Fingerprint lead/lag: 16 tests, none pass Bonferroni.
- Price: fails Bonferroni, Spearman collapses, hurts walk-forward.

## ROBUSTNESS
Champion beats naive in 2023, 2024, 2025 separately (~25%); 2021–2022 no gain (China-ban + bear-market hashrate shocks).

## ECONOMICS
Unquantified. Requires hashprice / difficulty-derivative data. Interpretation: ~0.9 pp less error on a 2-week revenue forecast.

## VERDICTS
- Difficulty blend: **L4 — VALIDATE** (needs economic test + live paper trial).
- Kalman: REJECT (no gain; complexity tax).
- Nonce fingerprint as predictor: REJECT. As index: L1, known method, new instance.
- Price momentum: REJECT.

## LESSONS
1. Log-adjustments under constant hashrate are MA(1) with autocorr −0.5; naive-forecast Poisson floor is 3.15% sd, not 2.2%.
2. Hashrate process noise (~5%/epoch) dominates observation noise, so filtering history buys little; the residual is genuine supply-side hashrate change.
3. Epoch-horizon hashrate change is not price-driven (ASIC lead times).
4. Coincident indicators (fleet fingerprint) are not forecasts.

## NEXT
Header-only track is near exhaustion for forecasting. Remaining unblocked queue items are low value (H007, H011). High-value items are BLOCKED on data: mempool/fee history (H009, H012), hashprice/derivative prices (H006), pool/share logs and machine telemetry (stale-share, thermal, failure tracks).
