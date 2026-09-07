"""Single machine-readable project state, rebuilt from the repository (a fresh session needs no chat history)."""
import glob
import json
import os

from ..infrastructure import PATHS, git_commit, git_dirty
from .governor import Governor
from .ledger import read
from .queue import ResearchQueue


def build_state(write=True):
    exps = read("experiments")
    hyps = read("hypotheses")
    status = {}
    for h in hyps:
        if "id" in h:
            status[h["id"]] = h.get("status", "?")
    q = ResearchQueue()
    champions = json.load(open(PATHS["state"] / "current_champions.json")) if (PATHS["state"] / "current_champions.json").exists() else {}
    manifests = {}
    for f in ("dataset_manifest.json", "manifest_arrivals.json", "manifest_fees.json"):
        p = PATHS["state"] / f
        if p.exists():
            m = json.load(open(p))
            manifests[f] = [x.get("dataset_id") for x in m] if isinstance(m, list) else m.get("dataset_id")
    reports = sorted(glob.glob(str(PATHS["reports"] / "ITERATION_*_FINAL.md")))
    state = {
        "git_commit": git_commit(), "git_dirty": git_dirty(),
        "champion": champions.get("difficulty_forecast", {}),
        "candidates": [it for it in q.items if it["status"] in ("PROMISING", "REPLICATING", "VALIDATING")],
        "hypotheses": {"active": [k for k, v in status.items() if v.startswith(("UNTESTED", "ACTIVE", "PROMISING", "REPLICAT", "VALIDAT", "TARGET"))],
                       "rejected": [k for k, v in status.items() if v.startswith("REJECT")],
                       "blocked": [k for k, v in status.items() if v.startswith("BLOCK")] + [it["id"] for it in q.blocked()]},
        "experiments": {"completed": [e.get("experiment_id") for e in exps if e.get("status", "COMPLETED") == "COMPLETED"],
                        "failed": [{"id": e.get("experiment_id"), "class": e.get("failure_class")} for e in exps if e.get("status") == "FAILED"],
                        "count": len(exps)},
        "queue_top": [{"id": it["id"], "priority": it["priority"], "status": it["status"]} for it in sorted(q.items, key=lambda x: -x["priority"])[:6]],
        "datasets": manifests, "governor": Governor().status(),
        "latest_report": os.path.basename(reports[-1]) if reports else None, "code_version": "mie " + __import__("mie").__version__,
    }
    if write:
        json.dump(state, open(PATHS["state"] / "project_state.json", "w"), indent=1, default=str)
    return state
