"""Registered experiments. Each returns a result dict with metrics/percentiles/comparison/verdict/next_action."""
import numpy as np

from ..economics import MiningEconomics, breakeven_E
from ..evaluation import compare, promotion_gates, protocol_frame, score
from ..forecasting import walk_forward
from ..latent_state import kalman_forecast
from ..models import DriftModel, KalmanHashrate, NaiveZero, OLSBlend, RidgeBlend
from ..optimization import value_of_information_table
from .runner import register


def _champion_models():
    return {"naive": lambda: NaiveZero(), "drift": lambda: DriftModel(), "BLEND-v1": lambda: OLSBlend(features=["extrap", "mom"])}
@register("REPRO-BLEND-v1", "Champion BLEND-v1 reproduces its recorded P1 scores", ["BTCHDR-2025-12-14"], primary=False)
def repro_champion(cfg):
    ks = cfg.get("horizons", [0, 300, 500, 1000, 1500])
    out = {}
    for k in ks:
        fr, _ = protocol_frame("P1", k)
        models = _champion_models()
        if k < 30:
            models["BLEND-v1"] = lambda: OLSBlend(features=["mom"])
        p = walk_forward(fr, models)
        m = np.isfinite(p["BLEND-v1"])
        out[f"k{k}"] = {n: score(fr.actual.values[m], p[n][m], p[n + "_q"][m]) for n in models}
    return {"metrics": {k: {n: round(v["MAE"], 2) for n, v in d.items()} for k, d in out.items()}, "full": out, "verdict": "REPRODUCTION", "next_action": "none"}
@register("CHALLENGE", "Candidate model vs BLEND-v1 under P1 with strict comparison + gates", ["BTCHDR-2025-12-14"], primary=True)
def challenge(cfg):
    """cfg: {'candidate': 'kalman'|'ols', 'features': [...], 'k': 500, 'q': 3e-3}"""
    k = cfg.get("k", 500)
    cand = cfg.get("candidate", "ols")
    fr, raw = protocol_frame("P1", k)
    if cand == "kalman":
        fr["kalman"] = [kalman_forecast(raw["t"], raw["D"], k, e, q=cfg.get("q", 3e-3)) for e in fr.e]
        factory = lambda: KalmanHashrate(q=cfg.get("q", 3e-3))  # noqa: E731
    else:
        feats = cfg.get("features", ["extrap", "mom"])
        factory = lambda: OLSBlend(features=feats)  # noqa: E731
    models = _champion_models()
    models["candidate"] = factory
    p = walk_forward(fr, models)
    eco = MiningEconomics()
    m = np.isfinite(p["candidate"]) & np.isfinite(p["BLEND-v1"])
    voi = value_of_information_table(fr.D_now.values[m], fr.actual.values[m], {"champion": p["BLEND-v1"][m], "candidate": p["candidate"][m], "baseline": p["naive"][m]}, eco)
    econ = {"table": voi, "not_worse": voi["candidate"]["forward_mispricing"] <= voi["champion"]["forward_mispricing"] * 1.02}
    cmpx = {"candidate": factory().complexity(), "champion": OLSBlend().complexity()}
    cmpx["acceptable"] = cmpx["candidate"]["n_params"] <= 3 * max(1, cmpx["champion"]["n_params"])
    c = compare(fr, p, {"baseline": "naive", "champion": "BLEND-v1", "candidate": "candidate"}, complexity=cmpx, econ=econ)
    g = promotion_gates(c)
    return {"comparison": c, "gates": g, "metrics": {r: c["table"][r]["MAE"] for r in c["table"]}, "percentiles": c["table"]["candidate"]["pct"], "verdict": g["verdict"],
            "next_action": "promote via research/promote.py only if all gates pass and an independent rerun agrees" if g["all_passed"] else "record and move on"}


@register("EXP26-SLOPE-RIDGE", "Within-epoch pace slope + ridge shrinkage vs BLEND-v1 under P1", ["BTCHDR-2025-12-14"], primary=True)
def exp26_slope_ridge(cfg):
    """H023: the champion prices the epoch remainder at the from-start mean pace, discarding any TREND in that
    pace. If hashrate drift inside the observed window persists into the unobserved remainder, a slope term
    carries information that 'extrap' averages away. Distinct from EXP8 (win/EWMA REPLACED the level; this adds
    a trend alongside it). Ridge is motivated by the 60-row training window, not by capacity.

    Stage 1 tunes alpha on epochs <330 ONLY (protocol P1 'tuning_allowed_on'). Stage 2 evaluates the locked spec
    on the untouched test epochs 350-458 with the strict comparison + promotion gates.
    """
    from ..data import load
    from ..features import build_frame
    ks = cfg.get("horizons", [0, 300, 500, 1000, 1500])
    alphas = cfg.get("alphas", [0.0, 0.3, 1.0, 3.0, 10.0, 30.0])
    feats = cfg.get("features", ["extrap", "mom", "slope"])
    z = load("BTCHDR-2025-12-14")
    t = z["time"].astype(np.int64)
    D = z["difficulty"]
    out = {}
    for k in ks:
        kf = [f for f in feats if not (k < 30 and f in ("extrap", "slope")) and not (k < 120 and f == "slope")]
        ch_f = ["mom"] if k < 30 else ["extrap", "mom"]
        # ---- Stage 1: tuning, epochs <330 only ----
        tf = build_frame(t, D, k, 100)
        tf = tf[tf.e < 330].reset_index(drop=True)
        tuning = {}
        for a in alphas:
            tp = walk_forward(tf, {"cand": lambda a=a, kf=kf: RidgeBlend(features=kf, alpha=a)})
            mt = np.isfinite(tp["cand"])
            tuning[str(a)] = round(score(tf.actual.values[mt], tp["cand"][mt])["MAE"], 4) if mt.sum() else None
        valid = {a: v for a, v in tuning.items() if v is not None}
        best_alpha = float(min(valid, key=valid.get)) if valid else 1.0
        # ---- Stage 2: locked evaluation on P1 test epochs ----
        fr, _ = protocol_frame("P1", k)
        models = _champion_models()
        if k < 30:
            models["BLEND-v1"] = lambda: OLSBlend(features=["mom"])
        models["candidate"] = lambda kf=kf, a=best_alpha: RidgeBlend(features=kf, alpha=a)
        p = walk_forward(fr, models)
        cmpx = {"candidate": RidgeBlend(features=kf, alpha=best_alpha).complexity(), "champion": OLSBlend(features=ch_f).complexity()}
        cmpx["acceptable"] = cmpx["candidate"]["n_params"] <= 3 * max(1, cmpx["champion"]["n_params"])
        c = compare(fr, p, {"baseline": "naive", "champion": "BLEND-v1", "candidate": "candidate"}, complexity=cmpx)
        g = promotion_gates(c)
        out[f"k{k}"] = {"features": kf, "tuning_lt330": tuning, "alpha_selected": best_alpha,
                        "MAE": {r: round(c["table"][r]["MAE"], 4) for r in c["table"]},
                        "paired": c["paired"], "regimes": c["regimes"], "gates": g["gates"], "verdict": g["verdict"]}
    promoted = [k for k, v in out.items() if v["verdict"] == "CHAMPION CANDIDATE"]
    return {"metrics": {k: {"champion": v["MAE"]["champion"], "candidate": v["MAE"]["candidate"], "verdict": v["verdict"]} for k, v in out.items()},
            "full": out, "verdict": "CHAMPION CANDIDATE" if promoted else "NO MATERIAL DIFFERENCE",
            "next_action": f"independent rerun then research/promote.py for {promoted}" if promoted else "record negative result; champion unchanged"}


@register("EXP28-CURTAIL-MAP", "Where in the electricity-price / efficiency space does the difficulty forecast change a run-or-curtail decision, and what is it worth?", ["BTCHDR-2025-12-14", "synthetic: parametric energy-price scenarios"], primary=True)
def exp28_curtail_map(cfg):
    """F4. The engine cannot mine, so its only route to value is deciding WHEN to mine. FINDINGS records the
    binary run/stop decision value as ~$0, but that was measured at ONE scenario (E=$0.06/kWh) where the miner
    is profitable no matter what difficulty does, so the decision never flips and no forecast can matter.

    This maps the decision across the electricity-price / hardware-efficiency plane and finds the band where the
    decision is actually live. Per-cell it reports how often the forecast flips the decision, the dollars that
    flipping earns over the naive baseline, and the perfect-foresight ceiling. Economics are parametric; only the
    difficulty forecasts are empirical (champion vs naive on the P1 test epochs).
    """
    k = cfg.get("k", 500)
    Es = cfg.get("electricity", [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12])
    etas = cfg.get("efficiency", [17.0, 21.0, 25.0, 30.0])
    P = cfg.get("btc_price", 90e3)
    H = cfg.get("H_ths", 1000.0)
    fr, _ = protocol_frame("P1", k)
    p = walk_forward(fr, _champion_models())
    m = np.isfinite(p["BLEND-v1"]) & np.isfinite(p["naive"])
    D_now, actual = fr.D_now.values[m], fr.actual.values[m]
    champ, naive = p["BLEND-v1"][m], p["naive"][m]

    grid, live = {}, []
    for eta in etas:
        for E in Es:
            eco = MiningEconomics(H_ths=H, P=P, eta=eta, E=E)
            run_c = eco.profit(D_now * np.exp(champ)) > 0
            run_n = eco.profit(D_now * np.exp(naive)) > 0
            run_p = eco.profit(D_now * np.exp(actual)) > 0
            v = {name: eco.decision_value(D_now, actual, f) for name, f in (("champion", champ), ("naive", naive), ("perfect", actual))}
            cell = {"eta_J_per_TH": eta, "E_usd_kWh": E,
                    "breakeven_E_usd_kWh": round(float(breakeven_E(float(np.median(D_now)), P, eco.p["R"], eta, eco.p["pool_fee"])), 5),
                    "epochs_run_perfect": int(run_p.sum()), "epochs_run_champion": int(run_c.sum()),
                    "decision_flips_vs_naive": int((run_c != run_n).sum()),
                    "wrong_calls_champion": int((run_c != run_p).sum()),
                    "value_champion_usd": round(v["champion"], 2), "value_naive_usd": round(v["naive"], 2),
                    "value_perfect_usd": round(v["perfect"], 2),
                    "forecast_gain_vs_naive_usd": round(v["champion"] - v["naive"], 2),
                    "perfect_gain_vs_naive_usd": round(v["perfect"] - v["naive"], 2)}
            cell["captured_pct"] = round(100 * cell["forecast_gain_vs_naive_usd"] / cell["perfect_gain_vs_naive_usd"], 1) if cell["perfect_gain_vs_naive_usd"] else None
            grid[f"eta{eta}_E{E}"] = cell
            if cell["decision_flips_vs_naive"] > 0:
                live.append(cell)

    best = max(live, key=lambda c: c["forecast_gain_vs_naive_usd"], default=None)
    return {"scenario": {"k": k, "H_ths": H, "btc_price": P, "n_epochs": int(m.sum()), "days_per_epoch": 14},
            "grid": grid,
            "live_band": {"n_cells_with_any_flip": len(live), "n_cells_total": len(grid),
                          "cells": sorted(live, key=lambda c: -c["forecast_gain_vs_naive_usd"])[:12]},
            "best_cell": best,
            "metrics": {"cells_where_decision_is_live": len(live), "of_total": len(grid),
                        "best_forecast_gain_usd_per_epoch": best["forecast_gain_vs_naive_usd"] if best else 0.0,
                        "best_cell_E_usd_kWh": best["E_usd_kWh"] if best else None,
                        "best_cell_eta": best["eta_J_per_TH"] if best else None},
            "verdict": "DECISION BAND FOUND" if live else "DECISION NEVER LIVE IN THIS GRID",
            "next_action": "read the live band as the operating region where this forecast is worth anything; outside it the forecast is worth $0 regardless of accuracy"}


@register("EXP-SHA-LADDER", "SHA-256 output bits are learnable from input bits (round-reduced ladder + differential analysis)", ["synthetic: generated SHA-256 pairs"], primary=True)
def sha_ladder(cfg):
    from .sha_ladder import run
    return run(cfg)
