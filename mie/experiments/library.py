"""Registered experiments. Each returns a result dict with metrics/percentiles/comparison/verdict/next_action."""
import numpy as np

from ..economics import MiningEconomics
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


@register("EXP-SHA-LADDER", "SHA-256 output bits are learnable from input bits (round-reduced ladder + differential analysis)", ["synthetic: generated SHA-256 pairs"], primary=True)
def sha_ladder(cfg):
    from .sha_ladder import run
    return run(cfg)
