"""Append-only research ledger (jsonl). Streams: experiments, hypotheses, failures, discoveries, lessons, decisions."""
import json

from ..infrastructure import PATHS

STREAMS = ("experiments", "hypotheses", "failures", "discoveries", "lessons", "decisions")
def append(stream, rec):
    assert stream in STREAMS, stream
    with open(PATHS["ledger"] / f"{stream}.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
def read(stream):
    p = PATHS["ledger"] / f"{stream}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_unparsed": line[:200]})
    return out
