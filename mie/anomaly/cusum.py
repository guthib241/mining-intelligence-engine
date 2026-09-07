def cusum_score(z, drift=0.0):
    """Two-sided CUSUM max over a standardized sequence z (EXP9; no early-warning value found, kept as a construct)."""
    sp = sn = mx = 0.0
    for v in z:
        sp = max(0.0, sp + v - drift)
        sn = max(0.0, sn - v - drift)
        mx = max(mx, sp, sn)
    return float(mx)
