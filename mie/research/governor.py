"""Three-primary-experiment governor. State persisted per iteration in research_state/governor.json."""
import datetime as dt
import json

from ..infrastructure import PATHS

GOV = PATHS["state"] / "governor.json"
class Governor:
    cap = 3
    def __init__(self):
        self.s = json.load(open(GOV)) if GOV.exists() else {"iteration": 1, "used": [], "opened": dt.datetime.now(dt.timezone.utc).isoformat()}
    def _save(self):
        json.dump(self.s, open(GOV, "w"), indent=1)
    def used(self):
        return len(self.s["used"])
    def can_run(self):
        return self.used() < self.cap
    def consume(self, exp_id):
        self.s["used"].append({"id": exp_id, "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        self._save()
    def new_iteration(self, note=""):
        self.s = {"iteration": self.s["iteration"] + 1, "used": [], "opened": dt.datetime.now(dt.timezone.utc).isoformat(), "note": note}
        self._save()
        return self.s["iteration"]
    def status(self):
        return {"iteration": self.s["iteration"], "used": self.used(), "cap": self.cap, "remaining": self.cap - self.used(), "ids": [u["id"] for u in self.s["used"]]}
