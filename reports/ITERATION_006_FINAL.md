# ITERATION 006 — FINAL — 2026-09-07
Governor: 3-experiment cap. Used: **1** (EXP25). EXP2 cancelled pre-run; EXP3 not run (information value collapsed). Commit: c5f62f7 → HEAD.

## EXPERIMENTS EXECUTED
| ID | Question | Data | Runtime | Result | Verdict |
|---|---|---|---|---|---|
| EXP25 | Which component of 14-day-ahead revenue-per-hash uncertainty carries the variance, and what is the VOI of each? | MEMPOOL-FEES-3Y-S4 (402 rows, 2023-09→2025-11) + BTCUSD-BINANCE + BTCHDR difficulty; 388 windows | 1.5 s | price 0.39 · fees 0.46 [CI 0.21–0.62] · difficulty 0.08 (2025 only: 0.82 / 0.01 / 0.26). VOI $/EH/epoch: price 61k, fees 46k, difficulty 32k. Fee 14d forecast OOS (n=246): persistence MAE 0.417, 28d-mean 0.377, 90d-mean 0.485 | L1 attribution, era-replicated; **strategic: difficulty is the smallest slice** |

## CODE CHANGES
experiments/exp25_revenue_attribution.py; data/external/mempool_fees_3y_stride4.csv; research_state/{manifest_fees.json, research_frontier.json (F5 added, F1 blocked), data_sources_examined.json (+3)}; reports/voi_target_map.json.

## DATA USED / ACQUIRED
New channel: `web_fetch` reaches mempool.space and api.blockchain.info where the sandbox cannot. Acquired 2-day-stride per-block fee averages (transcribed; integrity check vs independent Binance closes: median ratio 0.9995, 1/402 rows >5% on a 7% intraday-move day). Dead source recorded: blockchain.info mining charts 404. No BCH header archive on GitHub (F1 blocked at data step).

## RESULT COMPARISONS
No champion challenge this iteration. Fee-forecast baselines compared under identical OOS windows (2024-07+): 28d-mean beats persistence by 0.040 log-MAE (9.6% relative); 90d and expanding means are worse (bias −0.26 / −1.40): fee level is non-stationary and spike-driven.

## PERCENTILES
Fee share of block reward 2023-09→2025-11: P10 0.008, P50 0.022, mean 0.047, P90 0.114, max 0.643 (halving-day). 14d log-revenue sd: 13.7% overall, 7.5% in 2025.

## UNCERTAINTY
Fees variance-share CI95 [0.21, 0.62] (28-day block bootstrap, 500 draws). Era split reproduces ordering pre/post-halving; 2025 differs (fees quiescent).

## RED-TEAM
Halving-spanning windows excluded (deterministic subsidy step). Transcription validated on an independent column. Era stability checked (pre-halving / post-halving / 2025). Price is USDT-quoted (Binance) — immaterial at this precision.

## FAILURES
None new. Two planned experiments cancelled on evidence (documented in decisions.jsonl).

## DISCOVERIES
None promotable. Strategic finding (L1): forecasting value is bounded by variance share; the program's champion target (difficulty) carries 8–26% of revenue variance; price (unforecastable) carries 39–82%; fees carry 46% only in spike regimes.

## CURRENT CHAMPION / DELTA
BLEND-v1, L4, unchanged. Delta 0. Its VOI ceiling is now quantified: ≤ $32k/EH/epoch (perfect information), realised value still none.

## ECONOMIC VALUE
Per EH/s per epoch, perfect-information VOI: price $61k, fees $46k, difficulty $32k. Capturable today: difficulty ≈ $4.9k (champion vs naive), fees ≈ $4–5k (28d-mean vs persistence). Realised: none.

## BLOCKED ITEMS
F1 (BCH headers), H009/H012 (tx-level), H006 (market quotes), H018c/d (pool identity), stale-share (private). F5 needs mempool statistics via web_fetch (executable next iteration, data-first).

## NEW HYPOTHESES
H023/F5: fee-regime onset forecastable from mempool state (events ~4 in current data; needs mempool statistics history).

## RESEARCH QUEUE (rescored)
F5 (0.4, data-first) · F4 VOI map (0.5) · F3 (0.4) · F2 dormant · F1/H009/H006 blocked.

## NEXT RECOMMENDED EXPERIMENT
F5 step 1: acquire mempool.space `/api/v1/statistics/3y` (vsize-by-fee-level) via web_fetch, subsample, integrity-check, manifest; then pre-register EXP26: does mempool backlog/fee-floor state lead per-block fee level by ≥2 days, tested against a persistence baseline with block-bootstrap CIs. Stop rule: if fewer than 8 independent spike episodes exist, do not run.
