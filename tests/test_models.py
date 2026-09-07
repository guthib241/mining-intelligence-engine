import numpy as np
import pytest

from mie.infrastructure import ModelFailure
from mie.models import DriftModel, NaiveZero, OLSBlend


def test_interface_roundtrip(synth_frame):
    m = OLSBlend(features=["extrap", "mom"]).fit(synth_frame.iloc[:100])
    p = m.predict(synth_frame.iloc[100:110])
    assert p.shape == (10,) and np.isfinite(p).all()
    q = m.predict_distribution(synth_frame.iloc[100:110])
    assert q.shape == (10, 9) and (np.diff(q, axis=1) >= 0).all()
    blob = m.serialize()
    m2 = OLSBlend.load(blob)
    assert np.allclose(m2.predict(synth_frame.iloc[100:110]), p)
    ev = m.evaluate(synth_frame.iloc[100:200], synth_frame.actual.values[100:200])
    assert 0 < ev["MAE"] < 1
def test_predict_before_fit_raises(synth_frame):
    with pytest.raises(ModelFailure):
        OLSBlend().predict(synth_frame)
def test_baselines(synth_frame):
    assert (NaiveZero().fit(synth_frame).predict(synth_frame) == 0).all()
    d = DriftModel().fit(synth_frame)
    assert abs(d.predict(synth_frame)[0] - synth_frame.actual.mean()) < 1e-9
def test_nan_feature_yields_nan_prediction(synth_frame):
    f = synth_frame.copy()
    f.loc[5, "extrap"] = np.nan
    m = OLSBlend().fit(synth_frame)
    p = m.predict(f.iloc[[5]])
    assert np.isnan(p[0])
