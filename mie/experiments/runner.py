"""Experiment engine. An experiment = registered callable(cfg)->result dict. Runner records provenance, resources, failure class, and appends to the ledger."""
import datetime as dt
import json
import traceback

import numpy as np

from ..infrastructure import PATHS, ResourceTracker, classify_exception, git_commit, git_dirty
from ..research.governor import Governor
from ..research.ledger import append

EXPERIMENT_REGISTRY: dict = {}
def register(exp_id, hypothesis, datasets, primary=True):
    def deco(fn):
        EXPERIMENT_REGISTRY[exp_id] = {"fn": fn, "hypothesis": hypothesis, "datasets": datasets, "primary": primary}
        return fn
    return deco
def run_experiment(exp_id, cfg=None, seed=0, force=False):
    cfg = dict(cfg or {})
    spec = EXPERIMENT_REGISTRY[exp_id]
    gov = Governor()
    if spec["primary"] and not force and not gov.can_run():
        rec = {"experiment_id": exp_id, "status": "BLOCKED", "failure_class": "BLOCKED", "reason": f"governor: {gov.used()}/{gov.cap} primary experiments used this iteration"}
        append("experiments", rec)
        return rec
    np.random.seed(seed)
    rec = {"experiment_id": exp_id, "hypothesis": spec["hypothesis"], "datasets": spec["datasets"], "git_commit": git_commit(), "git_dirty": git_dirty(),
           "config": cfg, "seed": seed, "started": dt.datetime.now(dt.timezone.utc).isoformat() + "Z", "primary": spec["primary"]}
    try:
        with ResourceTracker() as rt:
            result = spec["fn"](cfg)
        rec.update(status="COMPLETED", resources=rt.as_dict(), **result)
        if spec["primary"]:
            gov.consume(exp_id)
    except BaseException as e:
        rec.update(status="FAILED", failure_class=classify_exception(e), error=str(e)[:500], traceback=traceback.format_exc()[-1500:])
        rec["verdict"] = "NOT A SCIENTIFIC RESULT (run failed)"
    out = PATHS["runs"] / f"{exp_id}_{rec['started'][:19].replace(':','')}.json"
    json.dump(rec, open(out, "w"), indent=1, default=_ser)
    rec["run_file"] = str(out.relative_to(PATHS["runs"].parents[1]))
    append("experiments", {k: v for k, v in rec.items() if k not in ("traceback",)})
    return rec
def _ser(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
