# ITERATION 004 — 2026-09-07
Commit: e9fe164 → HEAD. Protocol P1 unchanged. Datasets: BTCHDR-2025-12-14 (927,905 headers), BLOCK-ARRIVALS-CC0 (2,219,426 observer rows, 17 node observers + 2 stratum files), BTCUSD-BINANCE-DAILY.

## EXPERIMENTS RUN
EXP14a H016 · EXP14b H017 · EXP15 propagation calibration · EXP16 orphan pre-outcome + network regimes · EXP17 red-team · EXP18 leakage-safe · EXP19 econ ranges · EXP20 temporal holdout · EXP21 independent implementation. Repo inspection of 7 listed GitHub projects; PyPI pass (30 names).

## CODE CHANGES
`lib/propagation.py` (vectorized per-observer calibration: monthly offset, MAD, reliability, offset-corrected spread, LOO), `experiments/exp14–21`, `research_state/{repo_inspection,pypi_pass,data_need_map}.json`. All runs <15 s; peak memory <1 GB.

## RESULTS (strict comparison protocol)
**H016 — arrival-based pace vs BLEND-v1 (P1, n=109 per horizon)**
| k | drift MAE | champion MAE | candidate MAE | Δ abs (pp) | Δ rel | paired CI95 (pp) | cov90 champ/cand |
|---|---|---|---|---|---|---|---|
| 50 | 3.335 | 3.259 | 3.259 | 0.000 | 0.01% | [−0.001, +0.002] | .945/.945 |
| 100 | 3.335 | 3.338 | 3.338 | 0.000 | 0.01% | [−0.001, +0.002] | .936/.936 |
| 300 | 3.335 | 3.228 | 3.230 | −0.001 | −0.03% | [−0.003, +0.000] | .908/.908 |
| 500 | 3.335 | 2.955 | 2.955 | 0.000 | 0.00% | [−0.001, +0.001] | .917/.917 |
Signed-error percentiles identical to 0.01 pp (e.g. k=500 P5 −6.28/−6.28, P50 0.33/0.33, P95 5.50/5.50, P99 9.45/9.44). Year-by-year identical. **Verdict: NO MATERIAL DIFFERENCE → REJECT.** Why: arrival endpoints alter measurement timing by ~20 s over ~60,000 s; no information added. Would change verdict: nothing plausible.

**H017 — timestamp-error fingerprint.** Unimodal; ac1 = 0.016; adjacent-quarter KS median 0.031 (max 0.154); level shift −20 s → −13 s at 2024Q4. **Verdict: DESCRIPTIVE ONLY** (no pool labels available).

**Propagation calibration engine.** Automatically flags `peer-observer-frank` (MAD 95–143 s, Nov–Dec 2024) and `n-thumann` (22 s median lag); 49/527 observer-months excluded. Leave-one-out: dropping frank moves quarterly medians up to 1.18 s; the 2023Q2 jump coincides with observer-set change. **Cross-era propagation trend: SUSPECT (observer-dominated). Within KIT era 2019–2023Q1: P50 0.1–0.4 s, P90 0.3–2.2 s, credible.**

**Network-state regimes.** One change point (2023-04), within 2 months of an observer transition; bootstrap recovery 0/30. **Verdict: REJECT (artifact).**

**Orphan-risk pre-outcome signal — the iteration's main event.** Feature: leakage-safe calibrated spread of block h−1 (only arrivals before first arrival of h). Base rate 0.33/1000; flagged (>2 s) 2.18/1000, permutation p<1e-4. Survived: adjacency exclusion (1 adjacent pair), dropping flagged observers (4.63 vs 0.24), both eras, interval >30 s, slow-observer attribution across 5 nodes. Temporal holdout (threshold 3 s fixed on 2021–23; test 2024–25): 3.08 vs 0.20/1000, Fisher OR 15.3, p=4.9e-9, AUC 0.775 [0.678, 0.856].
**Then EXP21:** disjoint observer panel (KIT, 2020–23): 0 competing blocks after any high-spread event, AUC 0.538. Alternative feature (time-to-2nd-observer) on original panel: AUC 0.484, negative slope, skill −0.001, calibration flat.
**Verdict: REJECT (UNSTABLE — observer-visibility artifact).** Why: feature (slowest node) and label (node that logged a competitor) share the node set; a transiently partitioned node is both. Evidence supporting: OR 15 on holdout. Evidence weakening: two independent implementations null. Untested: a race label from nodes disjoint from the feature panel (H018d, blocked). Would change verdict: replication on a disjoint label source.

## PERCENTILES / UNCERTAINTY
Champion k=500 signed error (%): P1 −8.3, P5 −6.3, P10 −5.2, P25 −3.2, P50 +0.3, P75 +2.9, P90 +5.2, P95 +5.5, P99 +9.4; conformal cov50 .47, cov90 .92, width90 13.2 pp (drift 17.4).

## ECONOMIC VALUE (EXP19, per EH/s per epoch, k=500, 144 parameter cells)
| quantity | P10 | P50 | P90 |
|---|---|---|---|
| forward mispricing, naive | $25.1k | $46.5k | $75.4k |
| forward mispricing, standard estimator | $27.2k | $50.4k | $81.7k |
| forward mispricing, champion | $22.4k | $41.6k | $67.4k |
| value champion vs naive | $2.6k | $4.9k | $8.0k |
| value champion vs standard | $4.8k | $8.8k | $14.3k |
| value of perfect information | $25.1k | $46.5k | $75.4k |
Value scales with price and uptime (corr 0.99), independent of electricity/efficiency. Binary run/curtail decision value ≈ $0. Break-even electricity at D=1.48e14, 25 J/TH: $0.042/$0.064/$0.085 per kWh at P=60k/90k/120k. **FORECAST VALUE: modest (≈10% of VPI). REALIZED VALUE: none.**

## RED-TEAM RESULTS
LOO observer drop · era split · adjacency exclusion · leakage-safe censoring · interval conditioning · mechanism probe (late-for-h−1 nodes had seen no competitor: 0/40) · temporal holdout · **disjoint-panel replication (fatal)** · alternative-feature replication (fatal).

## FAILURES
H016, H017-as-fingerprint, cross-era propagation trend, network regimes, orphan-risk signal (retracted after passing 8 of 10 gates).

## DISCOVERIES
None promotable. Lesson recorded: *temporal holdout is not independent validation when feature and label share observers.*

## DATA DISCOVERY / GITHUB
7 listed repos inspected (license + activity verified; test dirs unverified, API rate limit). PyPI: 30 names, no bundled data. Data-need map written for 5 blocked hypotheses. No escalation.

## CURRENT CHAMPIONS
BLEND-v1 unchanged (L4 + calibrated intervals; challengers rejected: Kalman, +price, windows/EWMA, CUSUM, arrival-pace). No other champion.

## RESEARCH QUEUE (rescored; value = gain×impact×p / ((cost+data)×risk))
H020 stratum job-timing vs network state (1.6) · H019 fixed-panel propagation covariates (0.4) · H009 fees (0.8, BLOCKED) · H018c/d (BLOCKED) · H006 (0.25, BLOCKED).

## NEXT EXECUTED EXPERIMENT
H020 (executable, low expected value). Executable high-value work is now exhausted; remaining high-value items are BLOCKED on tx-level, market, or private operator data (see data_need_map.json). Escalation not yet requested.
