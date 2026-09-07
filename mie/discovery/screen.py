"""Lead/lag screen with multiple-testing control. Never promotes; returns candidates for pre-registered experiments."""
import numpy as np
from scipy import stats


def leadlag_screen(features: dict, target: np.ndarray, lags=(-1, 0, 1, 2)):
    rows = []
    for name, x in features.items():
        x = np.asarray(x, float)
        for lag in lags:
            if lag >= 0:
                a, b = x[: len(x) - lag], target[lag:]
            else:
                a, b = x[-lag:], target[: len(target) + lag]
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            r, p = stats.spearmanr(a[m], b[m])
            rows.append({"feature": name, "lag": lag, "n": int(m.sum()), "rho": float(r), "p": float(p)})
    thr = 0.05 / max(1, len(rows))
    for r in rows:
        r["bonferroni_thr"] = thr
        r["survives"] = bool(r["p"] < thr)
    return sorted(rows, key=lambda r: r["p"])
