"""Evaluation protocols: registered, versioned, reproducible. P1 is the official difficulty-forecast protocol."""
import numpy as np

from ..data import load
from ..features import build_frame

PROTOCOLS = {
    "P1": {"dataset": "BTCHDR-2025-12-14", "target": "log(D_next/D_cur)", "start_epoch": 330, "warmup_epochs": 20, "window": 60, "min_train": 20,
           "test_epochs": "350-458 (n=109)", "horizons_k": [0, 100, 300, 500, 1000, 1500], "metric": "MAE %", "tuning_allowed_on": "epochs <330",
           "regimes": {"year": "calendar year of epoch start"}}
}
def protocol_frame(pid, k, extra=None):
    p = PROTOCOLS[pid]
    z = load(p["dataset"])
    t = z["time"].astype(np.int64)
    D = z["difficulty"]
    fr = build_frame(t, D, k, p["start_epoch"], extra=extra)
    fr["year"] = np.array([int(str(np.datetime64(int(x), "s"))[:4]) for x in fr.t0])
    fr["D_now"] = D[fr.e.values * 2016]
    return fr, dict(t=t, D=D)
