"""Registered experiments. Each returns a result dict with metrics/percentiles/comparison/verdict/next_action."""
import numpy as np

from ..economics import MiningEconomics
from ..evaluation import compare, promotion_gates, protocol_frame, score
from ..forecasting import walk_forward
from ..latent_state import kalman_forecast
from ..models import DriftModel, KalmanHashrate, NaiveZero, OLSBlend
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


@register("EXP-SHA-LADDER", "SHA-256 output bits are learnable from input bits (round-reduced ladder + differential analysis)", ["synthetic: generated SHA-256 pairs"], primary=True)
def sha_ladder(cfg):
    from .sha_ladder import run
    return run(cfg)
