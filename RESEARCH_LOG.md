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

## Iteration 007b — EXP27 directional accuracy (NOT PROVEN)

**Why.** Every iteration so far optimized MAE. Stated as a plain accuracy — how often does the engine call the
*sign* of the next adjustment right — the champion had never been measured. It is also the number a
non-specialist actually reads as "how good is it".

**Champion, measured (P1, n=109):** k=0 66.1%, k=300 72.5%, **k=500 69.7%**, k=1000 79.8%, k=1500 90.8%.
Base rate ("always guess up", since difficulty rises far more often than it falls) = **67.9%**. So the
champion's edge at k=500 is +1.8pp, and at k=0 it is *negative*.

**Candidate.** Walk-forward L2 logistic regression on [extrap, mom, slope] trained on the up/down label
directly, rather than taking the sign of an MAE-optimized regression. Tuned on epochs <330, locked, tested.

**Tuning said:** logistic wins only at k=500 (85.7 vs 84.8); sign-of-regression wins at k=1000 and k=1500.
**Test confirmed that ordering at all four horizons** — k=500 73.4 vs 71.6, k=1000 79.8 vs 81.7, k=1500 85.3
vs 89.0. The ordering replicated; the *magnitude* did not survive testing.

**k=500 head-to-head vs champion:** 69.7% → 73.4% (+3.7pp), but McNemar p=0.48 — the models disagree on only
18 of 109 epochs, split 11–7. Candidate 95% CI [64.1%, 81.4%] **contains the 67.9% base rate**.

**Verdict: NOT PROVEN.** This is a power limit, not an idea limit: at the observed discordance rate,
demonstrating a 3.7pp gain at p<0.05 needs ~600 test epochs (~23 years of Bitcoin). 109 exist. No directional
forecast from headers alone can be shown to beat "assume difficulty rises" at k=500.

**Kept.** `dir_acc` / `dir_base_rate` / `dir_edge` are now emitted by `mie.evaluation.metrics.score`, so every
future experiment reports the accuracy percentage beside MAE — always with its baseline attached, because the
number is misleading without it.

## Iteration 008 — EXP28 curtailment decision map (STRATEGIC NEGATIVE)

**Framing.** Operator question: is this project even doing the right thing — an AI that mines instead of a
traditional rig? Re-ran EXP-SHA-LADDER live to settle it: full-round accuracy **50.4%** vs 50% chance,
avalanche 0.500, learnability wall at round 4 of 64 (Bitcoin uses 192). An AI cannot do the hashing. Its only
route to value is deciding **when** to hash. So this iteration asks what that decision is actually worth.

**Why a new experiment.** FINDINGS already recorded run/stop decision value ≈ $0, but at a single scenario
(E=$0.06/kWh) where the miner is profitable regardless of difficulty, so the decision never flips. One scenario
cannot support a general claim. EXP28 maps 32 scenarios: $0.03–$0.25/kWh × 17–30 J/TH, 1 PH/s, BTC $90k,
champion vs naive on the P1 test epochs, economics parametric.

**Result.** Best forecast gain: **$2.10 per epoch** on ~$1,620 of epoch revenue — **0.13%**. Perfect foresight
caps at the *same* $2.10. In 5 of 32 scenarios the forecast makes the decision **worse** than ignoring it
(−$0.80 worst). Summed across the grid the gain is indistinguishable from zero.

**Why.** The decision only flips when profit is within one difficulty move (~3.3% of revenue) of zero. Inside
that band profit is by construction ≈ $0, so calling it right earns ≈ $0. The forecast is accurate; the
decision it feeds is worthless.

**What does drive it.** Electricity price. A 25 J/TH miner runs 109/109 epochs at $0.03/kWh and 33/109 at
$0.25/kWh. Breakeven electricity: 17 J/TH $0.228, 21 J/TH $0.185, 25 J/TH $0.155, 30 J/TH $0.129.

**Consequence.** An AI meant to decide *when to mine* should forecast **electricity price**, not difficulty.
Difficulty forecasting stays defensible for forward-selling / hashprice exposure — a linear exposure where 3.3%
does translate into dollars — but not for the on/off decision. Third independent line of evidence that the
difficulty track is the smallest target: EXP25 (8–26% of revenue variance), the VOI map ($32k of $139k), and
now the curtail decision (~$0).
