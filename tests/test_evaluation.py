import numpy as np

from mie.evaluation import compare, promotion_gates, score
from mie.forecasting import walk_forward
from mie.models import DriftModel, NaiveZero, OLSBlend


def test_walk_forward_no_lookahead(synth_frame):
    out = walk_forward(synth_frame, {"m": lambda: OLSBlend()}, window=60, min_train=20)
    assert np.isnan(out["m"][:20]).all() and np.isfinite(out["m"][20:]).all() and out["m_q"].shape[1] == 9
def test_compare_and_gates_detect_real_improvement(synth_frame):
    p = walk_forward(synth_frame, {"naive": lambda: NaiveZero(), "drift": lambda: DriftModel(), "cand": lambda: OLSBlend(features=["extrap"])})
    c = compare(synth_frame, p, {"baseline": "naive", "champion": "drift", "candidate": "cand"})
    assert set(c["table"]) == {"baseline", "champion", "candidate", "null"}
    assert c["delta_candidate_vs_champion"]["MAE"]["abs"] > 0 and c["paired"]["p_wilcoxon"] < 0.01 and c["regimes"]
    g = promotion_gates(c)
    assert g["gates"]["beats_null"] and g["verdict"] in ("CHAMPION CANDIDATE", "PROMISING")
def test_gates_reject_noise_candidate(synth_frame):
    f = synth_frame.copy()
    f["junk"] = np.random.default_rng(0).normal(size=len(f))
    p = walk_forward(f, {"naive": lambda: NaiveZero(), "drift": lambda: DriftModel(), "cand": lambda: OLSBlend(features=["junk"])})
    g = promotion_gates(compare(f, p, {"baseline": "naive", "champion": "drift", "candidate": "cand"}))
    assert g["verdict"] in ("REJECT", "NO MATERIAL DIFFERENCE") and not g["all_passed"]
def test_score_percentiles_and_coverage():
    y = np.random.default_rng(0).normal(size=500)
    pred = np.zeros(500)
    q = np.percentile(y, [1,5,10,25,50,75,90,95,99])[None, :].repeat(500, 0)
    s = score(y, pred, q)
    assert abs(s["cov90"] - 0.9) < 0.03 and abs(s["cov50"] - 0.5) < 0.03 and "P99" in s["pct"]
