.PHONY: test lint type lab validate state ci
test: ; python3 -m pytest -q -p no:warnings
lint: ; ruff check mie tests
type: ; mypy mie --ignore-missing-imports
lab: ; python3 -m mie lab
validate: ; python3 -m mie validate
state: ; python3 -m mie state
ci: lint type test lab
