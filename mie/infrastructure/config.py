from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "raw": ROOT / "data" / "raw",
    "processed": ROOT / "data" / "processed",
    "external": ROOT / "data" / "external",
    "state": ROOT / "research_state",
    "ledger": ROOT / "research_ledger",
    "reports": ROOT / "reports",
    "runs": ROOT / "reports" / "runs",
}
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)
