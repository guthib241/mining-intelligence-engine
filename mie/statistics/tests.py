import numpy as np
from scipy import stats


def paired_delta(err_a, err_b):
    """delta = err_a - err_b (positive => b better). Returns mean, CI95, wilcoxon p, win-rate."""
    a = np.asarray(err_a, float)
    b = np.asarray(err_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    if n < 3:
        return {"n": int(n), "mean": float(np.nan), "ci95": [np.nan, np.nan], "p_wilcoxon": np.nan, "win_rate": np.nan}
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
    p = float(stats.wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0
    return {"n": int(n), "mean": float(d.mean()), "ci95": [float(ci[0]), float(ci[1])], "p_wilcoxon": p, "win_rate": float(np.mean(b < a))}
def block_bootstrap_ci(x, stat, block=14, B=500, seed=0, q=(2.5, 97.5)):
    r = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in r.integers(0, n, n // block + 1)])[:n]
        out.append(stat(x[idx]))
    return [float(v) for v in np.percentile(out, q)]
def mi_binned(x, y, bx=8, by=8):
    qx = np.digitize(x, np.quantile(x, np.linspace(0, 1, bx + 1)[1:-1]))
    qy = np.digitize(y, np.quantile(y, np.linspace(0, 1, by + 1)[1:-1]))
    j = np.zeros((bx, by))
    np.add.at(j, (qx, qy), 1)
    p = j / j.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
def mi_null(x, y, n=200, seed=0, **kw):
    r = np.random.default_rng(seed)
    obs = mi_binned(x, y, **kw)
    null = np.array([mi_binned(x, r.permutation(y), **kw) for _ in range(n)])
    return {"mi": obs, "p": float((null >= obs).mean()), "null_mean": float(null.mean()), "null_sd": float(null.std())}
