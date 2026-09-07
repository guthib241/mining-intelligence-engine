"""Strict Result-Comparison Protocol as code: baseline / champion / candidate / null on identical rows."""
import numpy as np

from ..statistics import paired_delta
from .metrics import score


def compare(frame, preds: dict, roles: dict, regime_col="year", null_seed=0, complexity=None, econ=None):
    """roles: {'baseline':name,'champion':name,'candidate':name}. preds from walk_forward. Null = candidate predictions permuted across rows."""
    names = [roles[r] for r in ("baseline", "champion", "candidate")]
    m = np.all([np.isfinite(preds[n]) for n in names], axis=0)
    y = frame.actual.values
    rng = np.random.default_rng(null_seed)
    null_pred = preds[roles["candidate"]].copy()
    null_pred[m] = rng.permutation(null_pred[m])
    table = {}
    for r, n in zip(("baseline", "champion", "candidate"), names):
        qn = preds.get(n + "_q")
        table[r] = score(y[m], preds[n][m], qn[m] if qn is not None else None)
    table["null"] = score(y[m], null_pred[m])
    ch, ca = table["champion"], table["candidate"]
    deltas = {met: {"abs": ch[met] - ca[met], "rel_pct": (ch[met] - ca[met]) / abs(ch[met]) * 100 if ch[met] else 0.0} for met in ("MAE", "RMSE")}
    paired = paired_delta(np.abs(y[m] - preds[roles["champion"]][m]) * 100, np.abs(y[m] - preds[roles["candidate"]][m]) * 100)
    regimes = {}
    if regime_col in frame:
        for g in sorted(set(frame[regime_col].values[m])):
            mm = m & (frame[regime_col].values == g)
            regimes[str(g)] = {r: round(float(np.abs(y[mm] - preds[n][mm]).mean() * 100), 3) for r, n in zip(("baseline", "champion", "candidate"), names)}
    same_rows = {"n": int(m.sum()), "row_ids": [int(x) for x in frame.e.values[m][:3]] + ["..."]}
    return {"roles": roles, "rows": same_rows, "table": table, "delta_candidate_vs_champion": deltas, "paired": paired, "regimes": regimes,
            "complexity": complexity or {}, "economics": econ or {}}
