import subprocess

from .config import ROOT


def _run(args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""
def git_commit() -> str:
    return _run(["rev-parse", "--short", "HEAD"]) or "nogit"
def git_dirty() -> bool:
    return bool(_run(["status", "--porcelain", "--untracked-files=no"]))
