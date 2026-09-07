from mie.infrastructure import Blocked, DataFailure, ModelFailure, classify_exception
from mie.research import ResearchQueue, build_state


def test_failure_taxonomy():
    assert classify_exception(DataFailure("x")) == "DATA_FAILURE" and classify_exception(ModelFailure("x")) == "MODEL_FAILURE"
    assert classify_exception(Blocked("x")) == "BLOCKED" and classify_exception(ValueError("x")) == "CODE_FAILURE" and classify_exception(FileNotFoundError("d")) == "DATA_FAILURE"
def test_governor_cap(tmp_path, monkeypatch):
    import mie.research.governor as g
    monkeypatch.setattr(g, "GOV", tmp_path / "gov.json")
    gov = g.Governor()
    for i in range(3):
        assert gov.can_run()
        gov.consume(f"E{i}")
    assert not gov.can_run() and gov.status()["remaining"] == 0
    gov.new_iteration("next")
    assert g.Governor().can_run()
def test_queue_priority_and_next():
    q = ResearchQueue()
    assert all("priority" in it for it in q.items)
    top = sorted(q.items, key=lambda x: -x["priority"])
    assert top[0]["priority"] >= top[-1]["priority"]
def test_project_state_reconstructs():
    s = build_state(write=False)
    assert s["champion"]["model_id"] == "BLEND-v1" and "P1" in s["champion"]["protocol"] and s["experiments"]["count"] > 20 and "governor" in s
