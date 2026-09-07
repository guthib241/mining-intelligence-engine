"""Common model interface: fit / predict / predict_distribution / evaluate / serialize / load / complexity."""
import json

import numpy as np
import pandas as pd

from ..infrastructure import ModelFailure

QS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
class BaseModel:
    name = "base"
    def __init__(self, **params):
        self.params = params
        self._fitted = False
        self._resid = None
        self.features = []
    def fit(self, train: pd.DataFrame):
        raise NotImplementedError
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    def _check(self):
        if not self._fitted:
            raise ModelFailure(f"{self.name}: predict before fit")
    def predict_distribution(self, X, qs=QS):
        self._check()
        p = self.predict(X)
        if self._resid is None or len(self._resid) < 5:
            raise ModelFailure(f"{self.name}: no residuals for distribution")
        return p[:, None] + np.percentile(self._resid, qs)[None, :]
    def evaluate(self, X, y):
        p = self.predict(X)
        err = y - p
        return {"MAE": float(np.nanmean(np.abs(err))), "RMSE": float(np.sqrt(np.nanmean(err ** 2))), "n": int(np.isfinite(err).sum())}
    def serialize(self):
        return {"name": self.name, "params": self.params, "state": self._state()}
    def _state(self):
        return {}
    @classmethod
    def load(cls, blob):
        m = cls(**blob.get("params", {}))
        m._set_state(blob.get("state", {}))
        m._fitted = True
        return m
    def _set_state(self, s):
        pass
    def complexity(self):
        return {"n_params": 0, "family": self.name}
    def to_json(self):
        return json.dumps(self.serialize(), default=float)
