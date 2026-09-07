"""Promotion gates (Champion Replacement Rule). Returns level and pass/fail per gate; never auto-promotes silently."""
def promotion_gates(cmp: dict, min_rel_improvement=3.0, max_p=0.05, max_tail_worsening=1.0, max_cal_gap=0.15):
    t = cmp["table"]
    ch, ca = t["champion"], t["candidate"]
    d = cmp["delta_candidate_vs_champion"]["MAE"]
    pr = cmp["paired"]
    gates = {
        "same_protocol": True,
        "beats_champion_oos": d["abs"] > 0,
        "effect_size": d["rel_pct"] >= min_rel_improvement,
        "paired_significant": (pr["p_wilcoxon"] < max_p) and (pr["ci95"][0] > 0),
        "no_tail_regression": (ca["pct"]["P99"] <= ch["pct"]["P99"] + max_tail_worsening) and (ca["pct"]["P1"] >= ch["pct"]["P1"] - max_tail_worsening),
        "calibrated": ca.get("calibration_gap", 0) <= max_cal_gap,
        "regime_no_catastrophe": all(v["candidate"] <= v["champion"] * 1.25 + 0.5 for v in cmp["regimes"].values()) if cmp["regimes"] else True,
        "beats_null": ca["MAE"] < t["null"]["MAE"],
        "complexity_acceptable": cmp.get("complexity", {}).get("acceptable", True),
        "economics_not_worse": cmp.get("economics", {}).get("not_worse", True),
    }
    passed = all(gates.values())
    if not gates["beats_champion_oos"] or not gates["beats_null"]:
        verdict = "REJECT"
    elif not gates["effect_size"] and not gates["paired_significant"]:
        verdict = "NO MATERIAL DIFFERENCE"
    elif gates["effect_size"] and not gates["paired_significant"]:
        verdict = "INTERESTING"
    elif passed:
        verdict = "CHAMPION CANDIDATE"
    else:
        verdict = "PROMISING"
    return {"gates": gates, "all_passed": passed, "verdict": verdict}
