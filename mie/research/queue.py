"""Executable research queue. Each item: id, question, hypothesis, data_requirements, status, info_gain, econ_value, p_success, compute_cost, data_cost, fdr_risk, experiment (registered id or None)."""
import json

from ..infrastructure import PATHS

QUEUE_FILE = PATHS["state"] / "research_queue.json"
ACTIVE = ("UNTESTED", "ACTIVE", "PROMISING", "REPLICATING", "VALIDATING")
def priority(it):
    gain = it.get("info_gain", it.get("information_gain", 1))
    econ = it.get("econ_value", it.get("impact", 1))
    p = it.get("p_success", it.get("p_useful", 0.5))
    cost = (it.get("compute_cost", it.get("cost", 1)) + it.get("data_cost", 0))
    risk = max(1, it.get("fdr_risk", it.get("risk", 1))) * max(1, it.get("complexity", 1))
    return round(gain * econ * p / max(1e-9, cost * risk), 3)
class ResearchQueue:
    def __init__(self, path=QUEUE_FILE):
        self.path = path
        self.items = json.load(open(path)) if path.exists() else []
        for it in self.items:
            it["priority"] = priority(it)
    def save(self):
        json.dump(sorted(self.items, key=lambda x: -x["priority"]), open(self.path, "w"), indent=1)
    def add(self, **it):
        assert "id" in it and "hypothesis" in it
        it.setdefault("status", "UNTESTED")
        it["priority"] = priority(it)
        self.items.append(it)
        self.save()
        return it
    def set_status(self, id_, status, note=None):
        for it in self.items:
            if it["id"] == id_:
                it["status"] = status
                if note:
                    it["note"] = note
        self.save()
    def executable(self):
        return [it for it in self.items if any(it["status"].startswith(s) for s in ACTIVE) and it.get("experiment")]
    def next(self, n=3):
        return sorted(self.executable(), key=lambda x: -x["priority"])[:n]
    def blocked(self):
        return [it for it in self.items if it["status"].startswith("BLOCKED")]
