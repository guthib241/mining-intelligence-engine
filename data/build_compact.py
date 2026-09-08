"""Build data/processed/headers_compact.npz — the array the dataset registry loads — from the raw 80-byte
header archives, verifying against consensus rules on the way.

parse_headers.py emits headers.pkl (a wide DataFrame, useful for ad-hoc work); the registry
(mie/data/registry.py:_load_headers) loads the compact npz. This script produces that artefact.

    git clone https://github.com/nip-333/btc-archive
    git clone https://github.com/nip-333/btc-current
    python data/build_compact.py [--root DIR]

Verification is not optional: a mismatch raises rather than writing a corrupt dataset.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

GENESIS = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
MAX_BITS = 0x1D00FFFF


def bits_to_target(b):
    e, m = b >> 24, b & 0xFFFFFF
    return m << (8 * (e - 3)) if e > 3 else m >> (8 * (3 - e))


def dsha(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def build(root, out):
    paths = [os.path.join(root, "btc-archive", "btc.archive.bin")]
    cur_dir = os.path.join(root, "btc-current")
    if os.path.isdir(cur_dir):
        paths += [os.path.join(cur_dir, f) for f in sorted(os.listdir(cur_dir)) if f.endswith(".bin")]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing raw header archives: {missing}\nclone nip-333/btc-archive and nip-333/btc-current first")

    raw = b"".join(open(p, "rb").read() for p in paths)
    n = len(raw) // 80
    a = np.frombuffer(raw[: n * 80], dtype=np.uint8).reshape(n, 80)
    print(f"inputs: {paths}\nheaders: {n}")

    hashes = np.empty((n, 32), dtype=np.uint8)
    for i in range(n):
        hashes[i] = np.frombuffer(dsha(raw[i * 80 : (i + 1) * 80]), dtype=np.uint8)

    genesis = hashes[0][::-1].tobytes().hex()
    if genesis != GENESIS:
        raise SystemExit(f"genesis mismatch: {genesis}")
    bad_links = int((~(a[:, 4:36][1:] == hashes[:-1]).all(axis=1)).sum())
    if bad_links:
        raise SystemExit(f"{bad_links} broken prev-hash links")

    bits = a[:, 72:76].copy().view("<u4").ravel()
    bad_pow = sum(1 for i in range(n) if hashes[i][::-1].tobytes() > bits_to_target(int(bits[i])).to_bytes(32, "big"))
    if bad_pow:
        raise SystemExit(f"{bad_pow} PoW failures")
    print(f"genesis: {genesis}\nbad prev-hash links: 0\npow failures: 0")

    target = np.array([float(bits_to_target(int(b))) for b in bits], dtype=np.float64)
    ts = a[:, 68:72].copy().view("<u4").ravel()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        height=np.arange(n, dtype=np.int32),
        time=ts.astype(np.int64),
        bits=bits.astype(np.uint32),
        nonce=a[:, 76:80].copy().view("<u4").ravel().astype(np.uint32),
        version=a[:, 0:4].copy().view("<i4").ravel().astype(np.int32),
        difficulty=float(bits_to_target(MAX_BITS)) / target,
    )
    print(f"wrote {out} ({os.path.getsize(out)} bytes); epochs {n // 2016}; last block {np.datetime64(int(ts[-1]), 's')}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(here)), help="directory holding the btc-archive/ and btc-current/ clones")
    ap.add_argument("--out", default=os.path.join(here, "processed", "headers_compact.npz"))
    args = ap.parse_args()
    sys.exit(build(args.root, args.out))
