# Testing and validation

## Default local suite

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The default marker expression excludes `molecular` and `remote` tests. It is safe for a
local laptop and exercises symbolic algebra, generated kernels on synthetic tensors,
solver contracts, projection contracts, repository layout, and compact result fixtures.

## Molecular suite

Molecular integral generation, CASSCF, QPCCSD/PAV calculations, and plot-producing runs
must execute on Medora, Talon, or an authorized remote desktop—not on the laptop and not
on NDSU systems.

```bash
python -m pip install -e ".[molecular,test]"
python -m pytest -m "molecular and remote"
```

The checked-in one-geometry N2 record is a regression anchor. Updating it requires a
new approved-host calculation with input/result hashes and numerical certification; a
local test failure is not a reason to replace it.

## Repository contract

```bash
python scripts/validate_repository.py
python scripts/check_repository_blobs.py --limit-mib 50
```

The first command validates the canonical method hierarchy, CLI entry point, JSON
package-data declaration, archive guards, and excluded-plan provenance. The second
checks candidate worktree files and reachable Git blobs; CI applies a 50 MiB limit.

## Distribution contract

```bash
python -m build
python scripts/validate_clean_install.py dist/autogen_wick-*.whl
```

The validator installs the wheel in a temporary environment, changes outside the source
checkout, imports all public method packages, verifies the `qpccsd` console script, and
loads both QPCCSD canonical-term JSON resources. This catches dependencies on repository
root paths that editable installs can hide.
