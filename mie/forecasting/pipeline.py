"""Walk-forward forecasting: strict chronological training windows, identical rows for every model."""
import numpy as np
import pandas as pd

from ..infrastructure import ModelFailure
from ..models.base import QS


def walk_forward(frame: pd.DataFrame, models: dict, window=60, min_train=20):
    """models: name -> factory() returning an unfitted BaseModel. Returns dict name->pred, name+'_q'->quantiles (n x 9)."""
    out: dict = {}
    for name, factory in models.items():
        pred = np.full(len(frame), np.nan)
        q = np.full((len(frame), len(QS)), np.nan)
        for i in range(len(frame)):
            tr = frame.iloc[max(0, i - window):i]
            if len(tr) < min_train:
                continue
            m = factory()
            try:
                m.fit(tr)
            except ModelFailure:
                continue
            row = frame.iloc[[i]]
            p = m.predict(row)[0]
            if np.isfinite(p):
                pred[i] = p
                q[i] = m.predict_distribution(row)[0]
        out[name] = pred
        out[name + "_q"] = q
    return out
