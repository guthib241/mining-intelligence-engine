"""Difficulty-epoch features. Features at horizon k use only blocks <= e*2016+k (leakage-safe by construction)."""
import numpy as np
import pandas as pd

EPOCH = 2016
def epoch_table(t, D):
    E = len(t) // EPOCH
    ep = np.arange(E)
    D_e = D[ep * EPOCH]
    return E, t[ep * EPOCH], np.log(D_e[1:] / D_e[:-1])
def pace_extrap(t, e, k, win=None, halflife=None):
    base = e * EPOCH
    if k < 30:
        return np.nan
    if halflife:
        iv = np.diff(t[base:base + k + 1]).astype(float)
        w = 0.5 ** (np.arange(k)[::-1] / halflife)
        pace = float(np.sum(w * iv) / np.sum(w))
    elif win and k > win:
        pace = (t[base + k] - t[base + k - win]) / win
    else:
        pace = (t[base + k] - t[base]) / k
    elapsed = t[base + k] - t[base]
    return float(np.log(np.clip(EPOCH * 600 / (elapsed + (EPOCH - 1 - k) * pace), 0.25, 4)))
def build_frame(t, D, k, start, extra=None):
    E, _, lr = epoch_table(t, D)
    rows = []
    for e in range(start, E - 1):
        r = {"e": e, "actual": lr[e], "extrap": pace_extrap(t, e, k), "mom": lr[e - 1], "t0": int(t[e * EPOCH])}
        for name, fn in (extra or {}).items():
            r[name] = fn(e)
        rows.append(r)
    return pd.DataFrame(rows)
