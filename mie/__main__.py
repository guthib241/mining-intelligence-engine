"""CLI: python -m mie <state|validate|lab|next|run EXP_ID|compare|new-iteration>"""
import json
import sys


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "state"
    if cmd == "state":
        from .research import build_state
        print(json.dumps(build_state(), indent=1, default=str))
    elif cmd == "validate":
        from .data import validate_all
        print(json.dumps(validate_all(), indent=1))
    elif cmd == "lab":
        from .experiments import run_lab
        r = run_lab()
        print(json.dumps({k: v for k, v in r.items() if k != "scenarios"}, indent=1))
        sys.exit(0 if r["PASS"] else 1)
    elif cmd == "next":
        from .research import ResearchQueue
        print(json.dumps(ResearchQueue().next(), indent=1))
    elif cmd == "run":
        import mie.experiments.library  # noqa: F401  registers experiments

        from .experiments import run_experiment
        cfg = json.loads(argv[3]) if len(argv) > 3 else {}
        r = run_experiment(argv[2], cfg)
        print(json.dumps({k: v for k, v in r.items() if k not in ("traceback",)}, indent=1, default=str))
    elif cmd == "new-iteration":
        from .research import Governor
        print("iteration", Governor().new_iteration(" ".join(argv[2:])))
    else:
        print(__doc__)
if __name__ == "__main__":
    main(sys.argv)
