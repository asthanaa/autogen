# Testing

## Fast tests (default)

Runs the quick unit + smoke tests:

- `conda run -n autogen pytest -q`

## Slow tests

The CCSD amplitude generation test is marked slow and skipped by default.
Enable it with:

- `RUN_SLOW=1 conda run -n autogen pytest -q`

Or just that subset:

- `RUN_SLOW=1 conda run -n autogen pytest -q -k ccsd`
