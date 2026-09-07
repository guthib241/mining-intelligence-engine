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
