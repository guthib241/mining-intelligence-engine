# ITERATION 005 — FINAL — 2026-09-07
Governor: 3-experiment cap. Experiments used: 3 (EXP22, EXP23, EXP24). Commit: c26e9c3 → HEAD.

## EXPERIMENTS EXECUTED (pre-registered in research_state/iteration_005_plan.json)
| ID | Question | Runtime | Result | Verdict |
|---|---|---|---|---|
| EXP22 | Where does the champion residual live? 12 covariates × 2 horizons | 3.3 s | 0/24 survive Bonferroni (0.0021); no sign-consistent covariate across 2021–23 / 2024–25 | REJECT (residual unstructured) |
| EXP23 | Post-halving residual across 2016/2020/2024; 2024 held out | 4 s | epochs 0–4 mean residual −1.0 / −1.2 / −4.5 pp (k=500); shift fit on 2016+2020 → 2024 MAE 4.60→3.96 (n=5), null p=0.048 | INTERESTING (L2), dormant |
| EXP24 | Stratum finder-lead as observer-independent propagation; disjoint-source orphan replication; H020 | 7.1 s | L bimodal (ΔBIC 45,269; 7% finder component at −1.24 s); orphan AUC 0.539 in null [0.40, 0.62]; delay~interval −0.02 | M3 = L0 measurement; orphan branch CLOSED; H020 REJECT |

## CODE CHANGES
experiments/exp22–24; research_state/{iteration_005_plan.json, math_candidates.md, research_frontier.json, github_candidates.json}; reports/{champion_residual_map.json, information_loss_audit.json, hidden_signal_screen_005.json, exp23_halving.json}.

## DATA USED
BTCHDR-2025-12-14 (927,905 headers) · BLOCK-ARRIVALS-CC0 (545,791 blocks; 80,011 with stratum + ≥3 observers) · BTCUSD-BINANCE-DAILY. No new data acquired this iteration.

## RESULT COMPARISONS (P1, same epochs)
Champion k=500 MAE 2.955 pp unchanged. EXP23 halving rule applied only to 5 epochs: P1 overall 2.932→2.902 pp (paired Δ on touched epochs +0.65 pp); not promoted (single held-out event, heterogeneous effect: 2020's hashrate grew +3.4%/epoch after its halving).

## PERCENTILES / UNCERTAINTY
Champion residual k=500: sd 3.74 pp, mean +0.27 pp (from EXP22). k=0: sd 4.14, mean +0.29. Post-halving 2024 residuals k=500 offsets 0–5: +0.3, −8.0, −2.6, −8.1, −4.0, −4.2 pp.
Finder-lead L (s): P1 −2.3, P5 −1.05, P25 −0.44, P50 −0.19, P75 +0.03, P95 +0.37, P99 +0.75.

## RED-TEAM
EXP22: Bonferroni over 24 tests, era-split sign check. EXP23: null = same shift applied to 5 random non-halving epochs (2,000 draws). EXP24: bootstrap AUC CI, permutation-null AUC band, three Fisher thresholds.

## FAILURES (destroyed this iteration)
Residual covariates (all 12) · ensembling (M5: error corr 0.99) · M1 surprise, M2 dispersion (deleted) · orphan-risk from stratum-derived delay (second disjoint source) · H020 · H019 cancelled pre-run (no new information).

## DISCOVERIES
None promotable. L2: post-halving negative residual. L0: finder-lead fraction rose 0.10 → 0.35–0.53 during 2025 (pool-behaviour transition visible in the stratum stream).

## CURRENT CHAMPION / DELTA
BLEND-v1, L4, unchanged. Champion delta this iteration: 0.

## ECONOMIC VALUE
No change: forecast value median $4.9k/EH/s/epoch vs naive (k=500); decision value ≈ $0; realized value none. Halving prior's value is deferred to 2028 and unquantifiable at n=1.

## INFORMATION-LOSS AUDIT
No loss from aggregation (Poisson sufficiency confirmed empirically, rho −0.11); the only representational gap is event-conditioning (reports/information_loss_audit.json).

## BLOCKED ITEMS
Unchanged (research_state/data_need_map.json): tx-level fees/mempool; market quotes; per-pool stratum/coinbase identity; private share logs.

## NEW HYPOTHESES / RESEARCH FRONTIER (research_state/research_frontier.json)
F1 cross-chain SHA-256 hashrate allocation (BTC vs BCH/BSV; DAA-driven oscillation; a real chain-choice decision) — TOP. F4 VOI map across mining decisions. F3 pool-software event timeline vs finder-lead fraction. F2 halving prior (dormant).

## RESEARCH QUEUE
F1 (1.0) · H009 (0.8, blocked) · H018c (0.6, blocked) · F4 (0.5) · F3 (0.4) · F2 (0.3) · H006 (0.25, blocked).

## NEXT RECOMMENDED EXPERIMENT
F1, step 1 (data acquisition deliverable): `research_state/bch_header_source.json` — locate, verify (PoW, links, DAA reproduction) and manifest a BCH header archive; then EXP25: predict next-window BCH hashrate share from BTC/BCH profitability ratio and ASERT state, baseline persistence, protocol P2 to be registered before running.
