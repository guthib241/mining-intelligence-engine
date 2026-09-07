# Mathematical constructs — iteration 005 (formal definitions, tested or deleted)

## M1 Predictive Surprise (epoch-level)
S_e = -log N(r_e ; 0, sigma_w) where r_e = actual_e - champion_e, sigma_w = sd of residuals in the 60-epoch training window.
Assumption: residual Gaussian within window. Null: permuted residual order. Cost O(E). Test: does S_{e-1} predict |r_e| (variance clustering)? -> EXP22 variable `S_prev`.

## M2 Interval Dispersion Index (within-epoch, pre-outcome)
ID_e(k) = Var(iv_1..iv_k) / mean(iv_1..iv_k)^2 over the first k header intervals. Poisson => ID=1. ID>1 suggests within-epoch hashrate change or timestamp noise; ID<1 suggests regular timing.
Sufficiency argument: for a homogeneous Poisson process the span is sufficient for rate; ID adds information only if the process is inhomogeneous. -> EXP22 variable `disp` (this is also the information-loss audit RAW vs AGGREGATED).

## M3 Finder Lead / Network Delay (block-level, stratum-derived)
L_h = t_firstjob(h) - t_firstobs(h). If the finding pool's monitored stratum endpoint emits the next job at creation, L_h < 0 with |L_h| ~ propagation delay to observers; delay_net_h = -min(L_h, 0). Independent of observer heterogeneity because the reference is a single pool-side event. Null: L_h unimodal at 0 => construct fails. -> EXP24.

## M4 Halving Distance
HD_e = signed number of epochs from the nearest halving (epochs after halving positive). Hypothesis: hashprice halves -> marginal capacity exits -> negative adjustments in the following 1-3 epochs. n=3 events in data range. -> EXP22 variable `halv`.

## M5 Error-Complementarity
C = corr(err_A, err_B) across P1 test epochs for A=champion, B=drift, B=std-estimator, B=EWMA. Ensemble is worth testing only if C < 0.7. Measured in EXP22 bookkeeping; decision recorded.
