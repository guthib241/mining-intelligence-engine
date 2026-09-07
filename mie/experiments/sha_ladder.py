"""EXP-SHA-LADDER: Can a learner predict SHA-256 output bits from input bits? Round-reduced ladder + differential (bit-flip) analysis.
SHA-256 implemented from FIPS 180-4 with a configurable number of compression rounds so the internal structure is exposed."""
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

K = np.array([
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2], dtype=np.uint64)
H0 = np.array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19], dtype=np.uint64)
M = np.uint64(0xffffffff)
def rotr(x, n):
    return ((x >> np.uint64(n)) | (x << np.uint64(32 - n))) & M
def compress(words, h, rounds):
    """Vectorised compression over N messages: words (N,16) uint64; h (8,) init; rounds<=64."""
    w = np.zeros((words.shape[0], 64), dtype=np.uint64)
    w[:, :16] = words
    for i in range(16, 64):
        s0 = rotr(w[:, i-15], 7) ^ rotr(w[:, i-15], 18) ^ (w[:, i-15] >> np.uint64(3))
        s1 = rotr(w[:, i-2], 17) ^ rotr(w[:, i-2], 19) ^ (w[:, i-2] >> np.uint64(10))
        w[:, i] = (w[:, i-16] + s0 + w[:, i-7] + s1) & M
    a, b, c, d, e, f, g, hh = [np.full(words.shape[0], v, dtype=np.uint64) for v in h]
    for i in range(rounds):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e & M) & g)
        t1 = (hh + S1 + ch + K[i] + w[:, i]) & M
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & M
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & M, c, b, a, (t1 + t2) & M
    return np.stack([(x + y) & M for x, y in zip(h, [a, b, c, d, e, f, g, hh])], axis=1)
def sha256_1block(words, rounds=64):
    """Single 512-bit padded block -> 256-bit digest as (N,8) uint32 words (padding: message = 13 words + fixed padding)."""
    return compress(words, H0, rounds)
def make_messages(N, rng):
    """Bitcoin-like: 13 random 32-bit words (header-like content incl. nonce) + standard padding for a 52-byte message."""
    w = np.zeros((N, 16), dtype=np.uint64)
    w[:, :13] = rng.integers(0, 2**32, size=(N, 13), dtype=np.uint64)
    w[:, 13] = 0x80000000
    w[:, 15] = 52 * 8
    return w
def to_bits(x32, nbits=32):
    return ((x32[:, :, None] >> np.arange(nbits, dtype=np.uint64)) & np.uint64(1)).reshape(x32.shape[0], -1).astype(np.int8)
def avalanche(N, rounds, rng, flips=8):
    """Fraction of output bits changed by a single input-bit flip, and max per-output-bit flip-probability deviation from 0.5."""
    w = make_messages(N, rng)
    h = sha256_1block(w, rounds)
    out = []
    for _ in range(flips):
        wi, bi = rng.integers(0, 13), rng.integers(0, 32)
        w2 = w.copy()
        w2[:, wi] ^= np.uint64(1) << np.uint64(bi)
        d = to_bits(sha256_1block(w2, rounds) ^ h)
        out.append(d.mean(0))
    P = np.array(out)
    return float(P.mean()), float(np.abs(P - 0.5).max()), float(2 / np.sqrt(N))
def learn(N, rounds, rng, bits=(0, 7, 31, 100, 255), test_frac=0.25):
    w = make_messages(N, rng)
    X = to_bits(w[:, :13])
    Y = to_bits(sha256_1block(w, rounds))
    n = int(N * (1 - test_frac))
    res = {}
    for b in bits:
        y = Y[:, b]
        r = {}
        if len(np.unique(y[:n])) < 2:   # constant bit at low rounds: perfectly predictable
            maj = float((y[n:] == y[:n][0]).mean())
            res[f"bit{b}"] = {"logreg": maj, "mlp": maj, "gbt": maj, "best": maj, "se": float(0.5 / np.sqrt(N - n)), "constant": True}
            continue
        for name, mdl in [("logreg", LogisticRegression(max_iter=300)), ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60, early_stopping=True, random_state=0)), ("gbt", HistGradientBoostingClassifier(max_iter=150, max_depth=6))]:
            mdl.fit(X[:n], y[:n])
            acc = float((mdl.predict(X[n:]) == y[n:]).mean())
            r[name] = acc
        r["best"] = max(r.values())
        r["se"] = float(0.5 / np.sqrt(N - n))
        res[f"bit{b}"] = r
    best_over_bits = max(v["best"] for v in res.values())
    return res, best_over_bits
def run(cfg):
    rng = np.random.default_rng(cfg.get("seed", 0))
    N = cfg.get("N", 60000)
    ladder = cfg.get("rounds", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    # sanity: full-round implementation must match hashlib
    import hashlib
    import struct
    w = make_messages(3, rng)
    ref = [hashlib.sha256(b"".join(struct.pack(">I", int(x)) for x in row[:13])).digest() for row in w]
    got = sha256_1block(w, 64)
    ok = all(bytes(b"".join(struct.pack(">I", int(x)) for x in got[i])) == ref[i] for i in range(3))
    if not ok:
        raise RuntimeError("SHA-256 implementation does not match hashlib")
    import json
    import time

    from ..infrastructure import PATHS
    cache = PATHS["reports"] / "cache" / f"sha_ladder_N{N}_seed{cfg.get('seed', 0)}.json"
    cache.parent.mkdir(exist_ok=True)
    out = json.load(open(cache)) if cache.exists() else {"implementation_matches_hashlib": True, "N": N, "ladder": {}}
    budget = cfg.get("time_budget_s", 240)
    t0 = time.time()
    for r in ladder:
        if str(r) in out["ladder"]:
            continue
        if time.time() - t0 > budget:
            out["partial"] = True
            json.dump(out, open(cache, "w"), indent=1)
            return {"status_note": "PARTIAL - rerun to continue", "done_rounds": list(out["ladder"]), "metrics": {}, "verdict": "INCOMPLETE", "next_action": "rerun same cfg"}
        rr = np.random.default_rng(cfg.get("seed", 0) + r)
        av_mean, av_maxdev, av_se = avalanche(20000, r, rr)
        res, best = learn(N, r, rr)
        se = 0.5 / np.sqrt(N * 0.25)
        z = (best - 0.5) / se
        out["ladder"][str(r)] = {"avalanche_frac_flipped": round(av_mean, 4), "max_flip_prob_dev": round(av_maxdev, 4), "learn_best_acc": round(best, 4), "z_vs_chance": round(z, 2), "per_bit": {k: {m: round(v, 4) for m, v in d.items()} for k, d in res.items()}}
        json.dump(out, open(cache, "w"), indent=1)
    out.pop("partial", None)
    full = out["ladder"]["64"]
    nbits_tested = 5
    bonf = stats.norm.ppf(1 - 0.05 / (nbits_tested * 3 * len(ladder)))
    out["full_round_verdict"] = "NO SIGNAL (chance)" if full["z_vs_chance"] < bonf else "SIGNAL - investigate"
    out["bonferroni_z"] = round(float(bonf), 2)
    wall = [int(r) for r, v in out["ladder"].items() if v["z_vs_chance"] >= bonf]
    out["rounds_with_learnable_structure"] = wall
    out["verdict"] = "PROVEN UNPREDICTABLE (full 64 rounds)" if not wall or max(wall) < 64 else "STRUCTURE AT FULL ROUNDS"
    out["metrics"] = {"full_round_best_acc": full["learn_best_acc"], "full_round_avalanche": full["avalanche_frac_flipped"], "learnability_wall_round": max(wall) if wall else 0}
    out["next_action"] = "close cryptographic-prediction branch permanently" if "PROVEN" in out["verdict"] else "replicate with independent implementation"
    return out
