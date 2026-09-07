import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_frame():
    r = np.random.default_rng(1)
    n = 300
    x = r.normal(size=n)
    mom = r.normal(size=n) * 0.1
    return pd.DataFrame({"e": np.arange(n), "actual": 0.6 * x + 0.02 + r.normal(size=n) * 0.5, "extrap": x, "mom": mom, "year": 2020 + np.arange(n) // 75, "D_now": 1e14, "t0": 1.6e9 + np.arange(n) * 1.2e6})
