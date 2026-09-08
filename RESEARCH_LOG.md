# Mining Intelligence Research Log

## Data
- Bitcoin block headers, genesis → height 927,904 (2025-12-14). Source: github nip-333/btc-archive + btc-current (raw 80-byte headers).
- Verified: genesis hash matches, 0 broken prev-hash links, 0 PoW failures, difficulty-adjustment formula reproduced to 1e-5 (2016+).
- Gap: no mempool/fee/tx data, no price, no pool data yet. Sandbox cannot reach chain APIs; GitHub/PyPI only.

## Sprint 1 — Experiments

### EXP1 Hash-outcome predictability — PROVEN UNPREDICTABLE
- hash/target ratio u: mean .5001, KS-uniform p=.80, autocorr all lags |r|<.002 (2se .002), n=928k.
- Prior-block nonce parity, prior u, hour-of-day: no effect on next hash bits (p=.71,.65,.48).
- Lesson: stop spending budget on hash-level prediction. Door closed with evidence, not assumption.
- LEAD (parked): nonce top-4-bit histogram strongly non-uniform in recent 100k blocks (4972–7066 per bin vs 6250 expected), flat in 2012. ASIC nonce-space fingerprint. Possible use: on-chain hardware-mix estimation → input to hashrate/difficulty models. Untested.

### EXP2 Block-interval memorylessness — PROVEN UNPREDICTABLE (single block)
- Last 150k blocks: E[remaining | elapsed>t] ≈ 590s for t=0..1800s. Interval autocorr |r|<.005. Prior 6-block pace → next interval r=.003.
- KS rejects exponential (p=1e-9) but only via timestamp noise (670 negative intervals). Not exploitable.
- SIGNAL: mean interval 597.6s early-epoch vs 585.1s late-epoch → within-epoch hashrate growth. Feeds EXP3.

### EXP3 Difficulty-adjustment forecasting — PREDICTIVE SIGNAL FOUND (operational, modest)
- Walk-forward, epochs 2020+ (n≈109), MAE in % of next difficulty:
  k(blocks in) naive extrap blend
  0     3.61  —     3.27*  (*drift+momentum; drift alone 3.34 → momentum adds ~0)
  300   3.61  4.79  3.23
  500   3.61  3.74  2.95
  1000  3.61  2.15  2.06
  1500  3.61  1.15  1.10
- KEY: raw pace extrapolation (what public estimators show) is WORSE than "no change" until ~600 blocks into the epoch. Shrinkage blend beats extrap at k=300 (Wilcoxon p=3e-7, win 68%), beats naive by ~25% in 2023/2024/2025 each (regime-stable).
- DESTROYED: epoch-to-epoch momentum. r=+.57 full history is a 2010–2015 growth artifact; 2016+ r=.03. 2023+ r=−.27 (p=.04, n=59, single test) — possible mean-reversion, NOT CURRENTLY PREDICTABLE.
- Poisson floor: adjustment sd from block randomness alone ≈ 2.2%; naive implies total sd ≈ 4.5%. So ~half the variance is hashrate change — the part external data could predict.

## Open hypotheses (ranked by expected info gain)
1. Hashrate-change prediction from external drivers (BTC price, hashprice, curtailment/weather, ASIC shipments) — would attack the ~2.3% residual in EXP3.
2. Do difficulty-derivative markets price off raw extrapolation? If so EXP3 blend is directly tradeable early-epoch. Needs market data.
3. Nonce-fingerprint → hardware-mix time series → leading indicator of hashrate.
4. Fee/mempool dynamics conditional on interval length (long gap → fat next block). Needs tx-level data.
5. Stale-share / pool latency — needs pool logs (operator data).

## Iteration 007 — EXP26 within-epoch pace trend + ridge (NEGATIVE)

**Question.** BLEND-v1 prices the unobserved remainder of the epoch at the *from-start mean pace*. That discards
any trend in the pace across the observed window. If within-epoch hashrate drift persists into the remainder, a
slope term should carry information `extrap` averages away.

**Why this was not already dead.** EXP8 tested win300/win150/ewma, but those *replace* the pace level with a
recent-window pace, losing the level; `slope` is a trend term used *alongside* the level, so the regression can
use both. EXP22 screened `disp` (var/mean²), a spread measure, not a directional trend.

**Design.** `RidgeBlend(features=[extrap, mom, slope])` vs BLEND-v1 under P1. Ridge motivated by the 60-row
training window, not capacity. `alpha` grid [0, 0.3, 1, 3, 10, 30] selected on epochs <330 only, then the spec
was locked and evaluated on the untouched test epochs 350–458 (n=109).

**Tuning-stage evidence (epochs <330, n=230)** — this is why the candidate was built:
- slope vs champion residual: k=1000 rho=−0.134 (p=.042), k=1500 rho=−0.189 (p=.0041); k=300/k=500 null.
- in-sample residual sd: k=1500 1.657 → 1.578pp, k=1000 3.272 → 3.172pp.

**Out-of-sample result (P1, n=109) — did not replicate:**

| k | champion | candidate | rel | paired p | win rate | verdict |
|---|---|---|---|---|---|---|
| 0 | 3.273 | 3.273 | +0.00% | 1.000 | 0.34 | REJECT |
| 300 | 3.228 | 3.209 | +0.61% | 0.530 | 0.56 | NO MATERIAL DIFFERENCE |
| 500 | 2.955 | 3.026 | **−2.41%** | 0.766 | 0.49 | REJECT |
| 1000 | 2.057 | 2.079 | **−1.09%** | 0.869 | 0.50 | REJECT |
| 1500 | 1.101 | 1.064 | +3.31% | 0.700 | 0.48 | INTERESTING |

**Reading.** Worse at k=500 and k=1000. At k=1500, where the tuning evidence was strongest, the paired test is
null and the win rate is *below half* (0.48) despite the lower mean MAE — the headline is a few large residuals,
not consistent improvement. Ridge added nothing: the alpha curve is nearly flat and alpha=0.0 was selected at two
horizons, so with 2–3 features the 60-row window is not over-fitting enough for shrinkage to help.

**Verdict.** H023 DESTROYED. Champion unchanged. Had alpha and the feature set been chosen on the test epochs,
k=1500 would have been reported as a 3.3% win — the tuning/test split is the only reason it wasn't.

**Consequence.** This was the last untested representation idea on the existing header-only feature set. It
corroborates `research_state/remaining_information_gap.json`: the residual is a NEW-INFORMATION problem, not a
representation or model-complexity problem. Kept in the codebase (`pace_slope`, `RidgeBlend`, tests) as
infrastructure for when external hashrate-driver data arrives.

**Data note.** `data/processed/headers_compact.npz` rebuilt this iteration from nip-333/btc-archive +
btc-current and re-verified: 927,905 headers, correct genesis, 0 broken prev-hash links, 0 PoW failures,
460 epochs, last block 2025-12-14. `data/parse_headers.py` emits `headers.pkl`, not the compact npz the
registry loads; `data/build_compact.py` now covers that gap.
