"""Permanent synthetic/null regression suite. The pipeline must recover signal, reject noise, and flag leakage."""
import numpy as np
import pandas as pd

from ..forecasting import walk_forward
from ..models import DriftModel, OLSBlend
from ..statistics import paired_delta


def _detect(df, cols):
    out = walk_forward(df, {"base": lambda: DriftModel(), "model": lambda: OLSBlend(features=cols)})
    m = np.isfinite(out["model"]) & np.isfinite(out["base"])
    y = df.actual.values[m]
    eb = np.abs(y - out["base"][m])
    em = np.abs(y - out["model"][m])
    pd_ = paired_delta(eb, em)
    return {"mae_base": float(eb.mean()), "mae_model": float(em.mean()), "p": pd_["p_wilcoxon"], "detected": bool(pd_["p_wilcoxon"] < 0.01 and em.mean() < 0.95 * eb.mean())}
def scenarios(N=400, seed=42):
    r = np.random.default_rng(seed)
    x = r.normal(size=N)
    n = r.normal(size=N)
    S = {}
    S["A_pure_noise"] = (pd.DataFrame({"actual": n, "x": x}), ["x"], False)
    S["B_known_signal"] = (pd.DataFrame({"actual": 0.7 * x + n, "x": x}), ["x"], True)
    S["B2_weak_signal"] = (pd.DataFrame({"actual": 0.3 * x + n, "x": x}), ["x"], None)     # power-limited: reported, not asserted
    S["C_nonlinear"] = (pd.DataFrame({"actual": 0.6 * x ** 2 - 0.6 + n, "x": x, "x2": x ** 2}), ["x", "x2"], True)
    xl = np.roll(x, 2)
    S["D_delayed"] = (pd.DataFrame({"actual": 0.5 * xl + n, "x_lag2": xl}), ["x_lag2"], True)
    reg = (np.arange(N) // 100) % 2
    S["E_regime_switch"] = (pd.DataFrame({"actual": np.where(reg == 1, 0.5, -0.5) * x + n, "x": x}), ["x"], None)
    S["F_structural_break"] = (pd.DataFrame({"actual": np.where(np.arange(N) < 250, 0.7 * x, 0.0) + n, "x": x}), ["x"], None)
    xm = x.copy()
    xm[r.random(N) < 0.15] = np.nan
    S["G_missing_data"] = (pd.DataFrame({"actual": 0.7 * x + n, "x": xm}), ["x"], True)
    F = r.normal(size=(N, 50))
    yI = r.normal(size=N)
    best = int(np.argmax([abs(np.corrcoef(F[:150, j], yI[:150])[0, 1]) for j in range(50)]))
    S["I_multiple_testing_trap"] = (pd.DataFrame({"actual": yI, "fbest": F[:, best]}), ["fbest"], False)
    yJ = r.normal(size=N)
    S["J_future_leak_trap"] = (pd.DataFrame({"actual": yJ, "leak": np.roll(yJ, -1)}), ["leak"], "leak")
    return S
def leak_check(df, col):
    a = df[col].values[:-1]
    b = df.actual.values[1:]
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])
def run_lab(N=400, seed=42):
    res = {}
    for k, (df, cols, truth) in scenarios(N, seed).items():
        r = _detect(df, cols)
        r["expected"] = truth
        r["leak_r_future"] = leak_check(df, cols[0])
        res[k] = r
    fdr_ok = (not res["A_pure_noise"]["detected"]) and (not res["I_multiple_testing_trap"]["detected"]) and (not res["J_future_leak_trap"]["detected"])
    power_ok = res["B_known_signal"]["detected"] and res["C_nonlinear"]["detected"] and res["D_delayed"]["detected"] and res["G_missing_data"]["detected"]
    leak_flag_ok = res["J_future_leak_trap"]["leak_r_future"] > 0.9
    return {"scenarios": res, "false_discovery_control": fdr_ok, "power_known_signals": power_ok, "leak_detector": leak_flag_ok, "PASS": fdr_ok and power_ok and leak_flag_ok}
