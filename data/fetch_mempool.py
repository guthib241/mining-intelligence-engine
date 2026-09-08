"""Fetch historical mempool + fee data from mempool.space. RUN THIS ON A MACHINE WITH INTERNET ACCESS.

The research sandbox reaches GitHub/PyPI only, so this is the hand-off script: run it, commit what it writes
under data/external/, and the engine can use it.

Two different things are fetched, and the distinction matters:

  * statistics/<window>  -> MEMPOOL BACKLOG over time (vsize by fee level). This is the PREDICTOR for
    fee-regime forecasting (F5) and the repo currently has NONE of it. This is the valuable one.
  * mining/blocks/fees   -> REALIZED fees per block. This is the LABEL. The repo has a 402-row transcribed
    subsample at ~2-day resolution (MEMPOOL-FEES-3Y-S4, authority MEDIUM); this refetches it at full
    resolution, first-hand, which also upgrades its authority.

Usage:
    python data/fetch_mempool.py                 # 3y of both, the default
    python data/fetch_mempool.py --windows 3y 1y 24h
    python data/fetch_mempool.py --out data/external

Only stdlib. Writes raw JSON (never edited by hand) plus a sha256 manifest, so the data is verifiable later.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://mempool.space/api/v1"
UA = "mining-intelligence-engine/0.1 (research; https://github.com/guthib241/mining-intelligence-engine)"


def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            wait = 2 ** (i + 1)
            print(f"  {type(e).__name__}: {e} -- retry {i+1}/{retries} in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"failed after {retries} attempts: {url}")


def save(obj, path):
    raw = json.dumps(obj, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(raw)
    return {"file": os.path.basename(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(obj) if isinstance(obj, list) else None}


def describe(obj, name):
    """Print the shape so the parser can be written against reality, not against a guess."""
    if isinstance(obj, list) and obj:
        k = obj[0]
        print(f"  {name}: {len(obj)} records; first record keys: {sorted(k)[:14] if isinstance(k, dict) else type(k).__name__}")
        if isinstance(k, dict):
            for key in sorted(k)[:14]:
                v = k[key]
                v = f"list[{len(v)}]" if isinstance(v, list) else repr(v)[:40]
                print(f"      {key}: {v}")
    else:
        print(f"  {name}: {type(obj).__name__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["3y"], help="2h 24h 1w 1m 3m 6m 1y 2y 3y (longer window = coarser buckets)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "external"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {"source": BASE, "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "license": "mempool.space public API (rate-limited, no stated license)", "files": []}
    for w in args.windows:
        for kind, path in (("mempool_backlog", f"statistics/{w}"), ("block_fees", f"mining/blocks/fees/{w}")):
            url = f"{BASE}/{path}"
            print(f"GET {url}")
            obj = get(url)
            entry = save(obj, os.path.join(args.out, f"{kind}_{w}.json"))
            entry.update(kind=kind, window=w, url=url)
            describe(obj, entry["file"])
            manifest["files"].append(entry)
            time.sleep(2)  # be polite to a free public API

    mpath = os.path.join(args.out, "mempool_fetch_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"\nwrote {len(manifest['files'])} files + {mpath}")
    print("\nNext: commit data/external/*.json and push. Raw JSON is kept verbatim so the parse step is\n"
          "reproducible and the sha256s in the manifest stay meaningful.")
