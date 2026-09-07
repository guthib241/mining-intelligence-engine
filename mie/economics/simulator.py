"""Parametric mining economics. Accepts forecast distributions (quantiles) and returns profit distribution, P(loss), break-evens, VOI."""
import numpy as np

from ..models.base import QS


def rev_per_day(H_ths, D, P, R):
    return 144 * R * P * (H_ths * 1e12 / (D * 2 ** 32 / 600))
def cost_per_day(H_ths, eta, E, uptime=1.0):
    return H_ths * eta * 24 / 1000 * E * uptime
def profit_per_day(H_ths, D, P, R, eta, E, uptime=1.0, pool_fee=0.02):
    return rev_per_day(H_ths, D, P, R) * uptime * (1 - pool_fee) - cost_per_day(H_ths, eta, E, uptime)
def breakeven_E(D, P, R, eta, pool_fee=0.02):
    return rev_per_day(1, D, P, R) * (1 - pool_fee) / (eta * 24 / 1000)
def breakeven_P(D, R, eta, E, pool_fee=0.02):
    return cost_per_day(1, eta, E) / (rev_per_day(1, D, 1, R) * (1 - pool_fee))
class MiningEconomics:
    """Scenario: hashrate H (TH/s), price P, reward R (subsidy+fees), efficiency eta (J/TH), electricity E ($/kWh), uptime, pool_fee, days."""
    def __init__(self, H_ths=1000, P=90e3, R=3.275, eta=25, E=0.06, uptime=0.97, pool_fee=0.02, days=14, threshold=0.0):
        self.p = dict(H_ths=H_ths, P=P, R=R, eta=eta, E=E, uptime=uptime, pool_fee=pool_fee, days=days, threshold=threshold)
    def profit(self, D):
        p = self.p
        return profit_per_day(p["H_ths"], D, p["P"], p["R"], p["eta"], p["E"], p["uptime"], p["pool_fee"]) * p["days"]
    def from_quantiles(self, D_now, adj_q_log, qs=QS, n=20000, seed=0):
        """adj_q_log: log-adjustment quantiles at percentiles qs. Samples D_next by inverse-CDF interpolation."""
        r = np.random.default_rng(seed)
        ps = np.array(qs) / 100
        u = r.uniform(ps[0], ps[-1], n)
        D_next = D_now * np.exp(np.interp(u, ps, np.asarray(adj_q_log)))
        prof = self.profit(D_next)
        rev = rev_per_day(self.p["H_ths"], D_next, self.p["P"], self.p["R"]) * self.p["days"]
        return {"expected_revenue": float(rev.mean()), "expected_cost": float(cost_per_day(self.p["H_ths"], self.p["eta"], self.p["E"], self.p["uptime"]) * self.p["days"]),
                "expected_profit": float(prof.mean()), "P_loss": float((prof < 0).mean()), "P_profit_gt_threshold": float((prof > self.p["threshold"]).mean()),
                "profit_pct": {f"P{q}": float(np.percentile(prof, q)) for q in QS}, "drawdown_worst": float(prof.min()),
                "breakeven_E": float(breakeven_E(D_now, self.p["P"], self.p["R"], self.p["eta"], self.p["pool_fee"])),
                "breakeven_P": float(breakeven_P(D_now, self.p["R"], self.p["eta"], self.p["E"], self.p["pool_fee"]))}
    def linear_exposure_value(self, D_now, actual_log, pred_log):
        """Forward-selling mispricing $ = revenue x |exp(actual-pred)-1| per epoch; VOI vs a competing forecast = difference of means."""
        rev = rev_per_day(self.p["H_ths"], D_now * np.exp(actual_log), self.p["P"], self.p["R"]) * self.p["days"]
        return float(np.nanmean(rev * np.abs(np.exp(actual_log - pred_log) - 1)))
    def decision_value(self, D_now, actual_log, pred_log):
        """Binary run/curtail per epoch decided on forecast; realised with actual."""
        run = self.profit(D_now * np.exp(pred_log)) > 0
        real = self.profit(D_now * np.exp(actual_log))
        return float(np.nanmean(np.where(run, real, 0.0)))
