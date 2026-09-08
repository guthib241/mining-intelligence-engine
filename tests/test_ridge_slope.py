"""RidgeBlend + pace_slope (EXP26). Covers the model interface and the leakage-safety of the new feature."""
import numpy as np
import pandas as pd
import pytest

from mie.features.difficulty import EPOCH, pace_extrap, pace_slope
from mie.infrastructure import ModelFailure
from mie.models import OLSBlend, RidgeBlend


@pytest.fixture
def slope_frame(synth_frame):
    r = np.random.default_rng(7)
    f = synth_frame.copy()
    f["slope"] = r.normal(size=len(f)) * 0.05
    return f


def test_ridge_roundtrip_and_nan(slope_frame):
    m = RidgeBlend(features=["extrap", "mom", "slope"], alpha=1.0).fit(slope_frame.iloc[:100])
    p = m.predict(slope_frame.iloc[100:110])
    assert p.shape == (10,) and np.isfinite(p).all()
    q = m.predict_distribution(slope_frame.iloc[100:110])
    assert q.shape == (10, 9) and (np.diff(q, axis=1) >= 0).all()
    m2 = RidgeBlend.load(m.serialize())
    assert np.allclose(m2.predict(slope_frame.iloc[100:110]), p)
    f = slope_frame.copy()
    f.loc[105, "slope"] = np.nan
    assert np.isnan(m.predict(f.iloc[[105]])[0])


def test_ridge_alpha_zero_matches_ols(slope_frame):
    """alpha=0 is plain OLS: standardization must not change the fitted predictions."""
    tr, te = slope_frame.iloc[:120], slope_frame.iloc[120:130]
    a = RidgeBlend(features=["extrap", "mom"], alpha=0.0).fit(tr).predict(te)
    b = OLSBlend(features=["extrap", "mom"]).fit(tr).predict(te)
    assert np.allclose(a, b, atol=1e-8)


def test_ridge_shrinks_coefficients(slope_frame):
    tr = slope_frame.iloc[:80]
    lo = RidgeBlend(features=["extrap", "mom", "slope"], alpha=0.0).fit(tr).beta[1:]
    hi = RidgeBlend(features=["extrap", "mom", "slope"], alpha=100.0).fit(tr).beta[1:]
    assert np.abs(hi).sum() < np.abs(lo).sum()


def test_ridge_too_few_rows_raises(slope_frame):
    with pytest.raises(ModelFailure):
        RidgeBlend().fit(slope_frame.iloc[:3].assign(slope=0.0))


def test_pace_slope_is_leakage_safe():
    """The feature at horizon k must depend only on blocks <= e*EPOCH+k. Mutating the epoch REMAINDER
    (the part a real forecaster cannot see) must leave the value bit-identical."""
    r = np.random.default_rng(3)
    n = EPOCH * 6
    t = np.cumsum(r.integers(300, 900, size=n)).astype(np.int64) + 1_600_000_000
    e, k = 2, 500
    before = pace_slope(t, e, k)
    t2 = t.copy()
    t2[e * EPOCH + k + 1 :] += 100_000  # rewrite only the unobserved remainder
    assert pace_slope(t2, e, k) == before
    assert pace_extrap(t2, e, k) == pace_extrap(t, e, k)
    # and it must react to a change INSIDE the observed window
    t3 = t.copy()
    t3[e * EPOCH + k // 2 : e * EPOCH + k + 1] += 5_000
    assert pace_slope(t3, e, k) != before


def test_pace_slope_undefined_for_short_windows():
    t = np.arange(EPOCH * 4, dtype=np.int64) * 600
    assert np.isnan(pace_slope(t, 1, 0))
    assert np.isnan(pace_slope(t, 1, 119))
    assert np.isfinite(pace_slope(t, 1, 120))


def test_pace_slope_sign():
    """Positive slope == blocks slowing down across the observed window."""
    iv = np.r_[np.full(300, 500), np.full(300, 700)]  # second half slower
    t = np.zeros(EPOCH * 3, dtype=np.int64)
    t[EPOCH : EPOCH + 600] = np.cumsum(iv)
    t[EPOCH + 600 :] = t[EPOCH + 599] + np.arange(1, len(t) - EPOCH - 599) * 600
    assert pace_slope(t, 1, 600) > 0
    assert pace_slope(t[::1] * 0 + np.arange(len(t)) * 600, 1, 600) == pytest.approx(0.0, abs=1e-12)


def test_build_frame_exposes_slope():
    from mie.features import build_frame
    r = np.random.default_rng(5)
    n = EPOCH * 8
    t = np.cumsum(r.integers(400, 800, size=n)).astype(np.int64) + 1_600_000_000
    D = np.repeat(np.linspace(1e13, 2e13, n // EPOCH + 1)[: n // EPOCH], EPOCH)[:n]
    fr = build_frame(t, D, 500, 2)
    assert "slope" in fr.columns and pd.notna(fr.slope).any()


def test_directional_accuracy_metric():
    """dir_acc must be reported with its base rate: the metric is meaningless alone when classes are skewed."""
    from mie.evaluation import score
    y = np.array([0.05, 0.02, 0.03, -0.01, 0.04])       # 4 up, 1 down -> base rate 0.8
    s = score(y, np.array([0.04, 0.01, 0.02, -0.02, 0.03]))
    assert s["dir_acc"] == 1.0 and s["dir_base_rate"] == pytest.approx(0.8)
    assert s["dir_edge"] == pytest.approx(0.2)
    # a model that always says "up" scores the base rate exactly, and shows zero edge
    s2 = score(y, np.full(5, 0.01))
    assert s2["dir_acc"] == pytest.approx(0.8) and s2["dir_edge"] == pytest.approx(0.0)
