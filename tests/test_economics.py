import numpy as np

from mie.economics import MiningEconomics, breakeven_E


def test_simulator_outputs():
    eco = MiningEconomics(H_ths=1000, P=90e3, R=3.275, eta=25, E=0.06)
    r = eco.from_quantiles(1.48e14, np.array([-0.08, -0.06, -0.05, -0.03, 0.0, 0.03, 0.05, 0.06, 0.09]))
    for k in ("expected_revenue", "expected_cost", "expected_profit", "P_loss", "P_profit_gt_threshold", "profit_pct", "drawdown_worst", "breakeven_E", "breakeven_P"):
        assert k in r
    assert 0 <= r["P_loss"] <= 1 and r["profit_pct"]["P1"] <= r["profit_pct"]["P99"]
    assert abs(r["breakeven_E"] - breakeven_E(1.48e14, 90e3, 3.275, 25)) < 1e-9
def test_voi_perfect_is_zero_mispricing():
    eco = MiningEconomics()
    a = np.array([0.02, -0.03, 0.01])
    assert eco.linear_exposure_value(np.full(3, 1e14), a, a) == 0.0
    assert eco.linear_exposure_value(np.full(3, 1e14), a, np.zeros(3)) > 0
