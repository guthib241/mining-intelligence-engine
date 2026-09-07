import numpy as np

from ..models.base import QS


def score(y, pred, q=None):
    m = np.isfinite(pred) & np.isfinite(y)
    y = y[m]
    p = pred[m]
    err = (y - p) * 100
    out = {"n": int(m.sum()), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "mean_err": float(err.mean()), "median_err": float(np.median(err)), "sd_err": float(err.std()),
           "pct": {f"P{k}": float(np.percentile(err, k)) for k in QS}, "worst_abs": float(np.abs(err).max())}
    if q is not None:
        Q = q[m]
        out.update(cov50=float(np.mean((y >= Q[:, 3]) & (y <= Q[:, 5]))), cov90=float(np.mean((y >= Q[:, 1]) & (y <= Q[:, 7]))), width90=float(np.mean(Q[:, 7] - Q[:, 1]) * 100),
                             pinball=float(np.mean([np.mean(np.maximum(QS[j] / 100 * (y - Q[:, j]), (QS[j] / 100 - 1) * (y - Q[:, j]))) for j in range(len(QS))]) * 100))
        out["calibration_gap"] = float(abs(out["cov90"] - 0.9) + abs(out["cov50"] - 0.5))
    return out
