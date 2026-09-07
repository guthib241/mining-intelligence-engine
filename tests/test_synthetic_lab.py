from mie.experiments import run_lab


def test_lab_regression():
    r = run_lab(N=400, seed=42)
    assert r["false_discovery_control"], r["scenarios"]["A_pure_noise"]
    assert r["power_known_signals"], {k: v["detected"] for k, v in r["scenarios"].items()}
    assert r["leak_detector"] and r["PASS"]
