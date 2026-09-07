"""Dataset registry. Raw data is immutable (external clones + data/raw); processed artefacts live in data/processed.
Each dataset: id, loader, manifest (schema/coverage/resolution/source/license/checksum/validation/limitations)."""
import json

import numpy as np
import pandas as pd

from ..infrastructure import PATHS, ROOT, DataFailure


def _load_headers():
    p = PATHS["processed"] / "headers_compact.npz"
    if not p.exists():
        raise DataFailure(f"headers_compact.npz missing at {p}; rebuild with data/parse_headers.py")
    z = np.load(p)
    return {k: z[k] for k in z.files}
def _load_price():
    p = ROOT / "btc" / "btc_prices.csv"
    if not p.exists():
        raise DataFailure("btc/btc_prices.csv missing (clone jptrustlearning/btc)")
    px = pd.read_csv(p)
    px.columns = ["date", "o", "h", "l", "c", "v"]
    px["date"] = pd.to_datetime(px.date)
    return px.set_index("date").sort_index()
def _load_fees():
    p = PATHS["external"] / "mempool_fees_3y_stride4.csv"
    if not p.exists():
        raise DataFailure("mempool fee file missing")
    f = pd.read_csv(p)
    f["date"] = pd.to_datetime(f.ts, unit="s")
    return f
def _load_arrivals():
    p = PATHS["processed"] / "arrivals.pkl"
    if not p.exists():
        raise DataFailure("arrivals.pkl missing; rebuild with data/ingest_arrivals.py")
    return pd.read_pickle(p)

DATASETS = {
    "BTCHDR-2025-12-14": {"loader": _load_headers, "manifest": "dataset_manifest.json", "manifest_key": 0, "schema": {"height": "int32", "time": "int64", "bits": "uint32", "nonce": "uint32", "version": "int32", "difficulty": "float64"}},
    "BTCUSD-BINANCE-DAILY": {"loader": _load_price, "manifest": "dataset_manifest.json", "manifest_key": 1, "schema": {"o,h,l,c,v": "float"}},
    "MEMPOOL-FEES-3Y-S4": {"loader": _load_fees, "manifest": "manifest_fees.json", "manifest_key": None, "schema": {"ts": "int", "avgFees_sat": "int", "usd": "int"}},
    "BLOCK-ARRIVALS-CC0": {"loader": _load_arrivals, "manifest": "manifest_arrivals.json", "manifest_key": None, "schema": {"index": "height", "first,last,med": "unix s", "n_src": "int"}},
}
def manifest(ds_id):
    d = DATASETS[ds_id]
    m = json.load(open(PATHS["state"] / d["manifest"]))
    m = m[d["manifest_key"]] if d["manifest_key"] is not None else m
    m = dict(m)
    m["schema"] = d["schema"]
    return m
def load(ds_id):
    if ds_id not in DATASETS:
        raise DataFailure(f"unknown dataset {ds_id}")
    return DATASETS[ds_id]["loader"]()
def validate_all():
    out = {}
    for k in DATASETS:
        try:
            d = load(k)
            r = {"status": "OK"}
            if k.startswith("BTCHDR"):
                r.update(n=int(len(d["time"])), epochs=int(len(d["time"]) // 2016))
            elif k.startswith("BTCUSD"):
                r.update(n=int(len(d)), gaps=int((d.index.to_series().diff().dt.days > 1).sum()), nonpositive=int((d.c <= 0).sum()))
            elif k.startswith("MEMPOOL"):
                r.update(n=int(len(d)), monotone=bool((d.ts.diff().dropna() > 0).all()))
            elif k.startswith("BLOCK-ARR"):
                r.update(n=int(len(d)), multi_observer=int((d.n_src >= 3).sum()))
            out[k] = r
        except DataFailure as e:
            out[k] = {"status": "DATA_FAILURE", "detail": str(e)}
    return out
