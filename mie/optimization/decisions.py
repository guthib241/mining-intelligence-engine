"""Decision-level value of information: compares forecasts inside the same economic scenario (forecast value vs decision value)."""
import numpy as np

from ..economics import MiningEconomics


def value_of_information_table(D_now, actual_log, preds: dict, scenario: MiningEconomics | None = None):
    eco = scenario or MiningEconomics()
    out = {}
    perfect = eco.linear_exposure_value(D_now, actual_log, actual_log)
    for name, p in preds.items():
        out[name] = {"forward_mispricing": eco.linear_exposure_value(D_now, actual_log, np.asarray(p)), "decision_realised": eco.decision_value(D_now, actual_log, np.asarray(p))}
    out["_perfect"] = {"forward_mispricing": perfect, "decision_realised": eco.decision_value(D_now, actual_log, actual_log)}
    return out
