import numpy as np

from ..infrastructure import ModelFailure
from .base import BaseModel


class NaiveZero(BaseModel):
    name = "naive_zero"
    def fit(self, train):
        self._resid = train.actual.values.copy()
        self._fitted = True
        return self
    def predict(self, X):
        return np.zeros(len(X))
class OLSBlend(BaseModel):
    """Walk-forward OLS. BLEND-v1 == OLSBlend(features=['extrap','mom']); Drift == OLSBlend(features=[])."""
    name = "ols_blend"
    def __init__(self, features=("extrap", "mom"), **p):
        super().__init__(features=list(features), **p)
        self.features = list(features)
        self.beta = None
    def fit(self, train):
        tr = train.dropna(subset=self.features + ["actual"]) if self.features else train
        if len(tr) < 5:
            raise ModelFailure("ols_blend: too few training rows")
        X = np.c_[np.ones(len(tr)), tr[self.features].values] if self.features else np.ones((len(tr), 1))
        self.beta = np.linalg.lstsq(X, tr.actual.values, rcond=None)[0]
        self._resid = tr.actual.values - X @ self.beta
        self._fitted = True
        return self
    def predict(self, X):
        self._check()
        if self.features:
            F = X[self.features].values.astype(float)
            out = np.c_[np.ones(len(X)), F] @ self.beta
            out[np.isnan(F).any(1)] = np.nan
            return out
        return np.full(len(X), self.beta[0])
    def _state(self):
        return {"beta": [float(b) for b in self.beta]}
    def _set_state(self, s):
        self.beta = np.array(s["beta"])
    def complexity(self):
        return {"n_params": len(self.features) + 1, "family": "linear"}
class DriftModel(OLSBlend):
    name = "drift"
    def __init__(self, **p):
        super().__init__(features=(), **p)
class RidgeBlend(OLSBlend):
    """Walk-forward ridge on standardized features. Motivated by the 60-row training window (P1), where plain
    OLS coefficients are noisy. Intercept is never penalized; standardization uses training-window stats only,
    so no test information reaches the fit. alpha is tuned on epochs <330 only (protocol P1)."""
    name = "ridge_blend"
    def __init__(self, features=("extrap", "mom", "slope"), alpha=1.0, **p):
        super().__init__(features=features, alpha=alpha, **p)
        self.alpha = float(alpha)
        self.mu = None
        self.sd = None
    def fit(self, train):
        tr = train.dropna(subset=self.features + ["actual"])
        if len(tr) < 5:
            raise ModelFailure("ridge_blend: too few training rows")
        F = tr[self.features].values.astype(float)
        self.mu = F.mean(0)
        self.sd = F.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        X = np.c_[np.ones(len(tr)), (F - self.mu) / self.sd]
        y = tr.actual.values
        P = np.eye(X.shape[1]) * self.alpha
        P[0, 0] = 0.0
        self.beta = np.linalg.solve(X.T @ X + P, X.T @ y)
        self._resid = y - X @ self.beta
        self._fitted = True
        return self
    def predict(self, X):
        self._check()
        F = X[self.features].values.astype(float)
        out = np.c_[np.ones(len(X)), (F - self.mu) / self.sd] @ self.beta
        out[np.isnan(F).any(1)] = np.nan
        return out
    def _state(self):
        return {"beta": [float(b) for b in self.beta], "mu": [float(v) for v in self.mu], "sd": [float(v) for v in self.sd]}
    def _set_state(self, s):
        self.beta = np.array(s["beta"])
        self.mu = np.array(s["mu"])
        self.sd = np.array(s["sd"])
    def complexity(self):
        return {"n_params": len(self.features) + 1, "family": "linear-l2"}
class KalmanHashrate(BaseModel):
    """Consumes a precomputed 'kalman' feature column (latent_state.kalman)."""
    name = "kalman"
    def __init__(self, q=3e-3, **p):
        super().__init__(q=q, **p)
        self.q = q
    def fit(self, train):
        self._resid = (train.actual - train["kalman"]).dropna().values
        self._fitted = True
        return self
    def predict(self, X):
        self._check()
        return X["kalman"].values.astype(float)
    def complexity(self):
        return {"n_params": 1, "family": "state-space"}
REGISTRY = {"naive_zero": NaiveZero, "drift": DriftModel, "ols_blend": OLSBlend, "ridge_blend": RidgeBlend, "kalman": KalmanHashrate}
