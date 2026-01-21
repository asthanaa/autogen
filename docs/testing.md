# Testing

## Fast tests (default)

Runs the quick unit + smoke tests:

- `conda run -n autogen pytest -q`

## Slow tests

PySCF-backed CCSD checks (now parameterized over several molecules) are marked
slow and skipped by default. Enable them with:

- `RUN_SLOW=1 conda run -n autogen pytest -q`

Or just that subset:

- `RUN_SLOW=1 conda run -n autogen pytest -q -k ccsd`
- `RUN_SLOW=1 conda run -n autogen pytest -q -k eom_ccsd`
