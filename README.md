# Mining Intelligence Engine (MIE)

Reproducible research machine for discovering real, robust, economically meaningful advantages in cryptocurrency mining.

```
make ci            # lint + types + tests + synthetic/null lab
python -m mie state          # rebuild machine-readable project state from the repo
python -m mie validate       # dataset validation
python -m mie next           # next executable experiments from the research queue
python -m mie run CHALLENGE '{"candidate":"kalman","k":500}'   # governed primary experiment
python -m mie run REPRO-BLEND-v1                               # champion reproduction (non-primary)
python -m mie new-iteration "note"                             # reset the 3-experiment governor
```
Data setup (not in git): `git clone nip-333/btc-archive btc-archive; ... python data/parse_headers.py` → `data/processed/headers_compact.npz`; `git clone jptrustlearning/btc btc`; `git clone bitcoin-data/block-arrival-times; python data/ingest_arrivals.py`.

Read `docs/ARCHITECTURE.md`, `research_state/project_state.json`, `research_state/research_frontier.json`, and the latest `reports/ITERATION_*_FINAL.md` before doing anything. Legacy scripts in `experiments/exp*.py` and `lib/` are retained for provenance of EXP1–EXP25; new work uses `mie`.
