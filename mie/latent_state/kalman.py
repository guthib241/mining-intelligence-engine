"""Local-level+drift Kalman filter on log-hashrate with analytic Poisson observation noise (EXP4; rejected as champion, kept as a model family)."""
import numpy as np

from ..features import EPOCH

S_EPOCH = 1 / np.sqrt(EPOCH - 1)
def epoch_hash_obs(t, D):
    E = len(t) // EPOCH
    ep = np.arange(E)
    span = (t[ep * EPOCH + EPOCH - 1] - t[ep * EPOCH]).astype(float)
    return E, np.log(D[ep * EPOCH]) + np.log((EPOCH - 1) * 600 / span)
def kalman_forecast(t, D, k, e, q=3e-3, drift_win=26):
    E, logH = epoch_hash_obs(t, D)
    d = float(np.mean(np.diff(logH[max(0, e - drift_win):e])))
    x, P = logH[0], S_EPOCH ** 2
    for i in range(1, e):
        xp, Pp = x + d, P + q
        K = Pp / (Pp + S_EPOCH ** 2)
        x = xp + K * (logH[i] - xp)
        P = (1 - K) * Pp
    x, P = x + d, P + q
    base = e * EPOCH
    if k >= 30:
        elapsed = float(t[base + k] - t[base])
        z = np.log(D[base]) + np.log(k * 600 / elapsed)
        s2 = 1 / k
        K = P / (P + s2)
        x = x + K * (z - x)
    else:
        elapsed = 0.0
    rem = ((EPOCH - 1 - k) if k >= 30 else EPOCH - 1) * 600 * D[base] / np.exp(x)
    return float(np.log(np.clip(EPOCH * 600 / (elapsed + rem), 0.25, 4)))
