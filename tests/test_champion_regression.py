"""Permanent regression: the champion must reproduce its recorded P1 scores from the repository."""
import json

import numpy as np
import pytest

from mie.evaluation import protocol_frame, score
from mie.forecasting import walk_forward
from mie.infrastructure import PATHS
from mie.models import NaiveZero, OLSBlend


@pytest.mark.slow
@pytest.mark.parametrize("k,expected", [(500, 2.95), (1000, 2.06)])
def test_blend_v1_p1(k, expected):
    fr, _ = protocol_frame("P1", k)
    p = walk_forward(fr, {"c": lambda: OLSBlend(), "n": lambda: NaiveZero()})
    m = np.isfinite(p["c"])
    s = score(fr.actual.values[m], p["c"][m])
    assert round(s["MAE"], 2) == expected and s["n"] == 109
    rec = json.load(open(PATHS["state"] / "current_champions.json"))["difficulty_forecast"]["scores_MAE_pct"][f"k{k}"]
    assert rec == expected
