# Autogen

[![PyPI](https://img.shields.io/pypi/v/autogen-wick.svg)](https://pypi.org/project/autogen-wick/)
[![Python](https://img.shields.io/pypi/pyversions/autogen-wick.svg)](https://pypi.org/project/autogen-wick/)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://asthanaa.github.io/autogen/)

Autogen combines a Wick-theorem expression generator with reviewed coupled-cluster
method implementations. The installable library is rooted at `src/autogen`; generated
runtime code, derivation inputs, and tests are organized by method rather than kept in a
second source tree.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install autogen-wick
```

For molecular reference construction and regressions, install PySCF explicitly:

```bash
python -m pip install "autogen-wick[molecular]"
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The default test selection never starts molecular calculations. Those calculations are
restricted to Medora, Talon, or an authorized remote desktop.

## Canonical APIs

Symbolic Wick-algebra utilities remain available under:

- `autogen.library`
- `autogen.main_tools`
- `autogen.pkg`

Reviewed electronic-structure methods live under:

- `autogen.methods.ccsd`
- `autogen.methods.eom_ccsd`
- `autogen.methods.qpccsd`

The QPCCSD command-line interface is installed as `qpccsd`:

```bash
qpccsd --help
qpccsd show-defaults
qpccsd run configs/qpccsd/n2_sto3g.toml
```

## Default QPCCSD contract

The public QPCCSD route is deliberately narrow:

1. fit a projected AGP reference to the active natural occupations and active
   pair-transfer 2-RDM block;
2. solve direct QPCCSD in the full symmetry-allowed pair/quadruple coordinate space,
   including pure-active T1/T2 coordinates;
3. report the direct quasiparticle Hamiltonian energy—never CASSCF plus a QP correction;
4. evaluate particle-number PAV with the converged amplitudes held fixed and validate it
   on a doubled midpoint grid.

PN-OAP, masked excitation spaces, CASSCF-plus-delta energies, cubic extensions, and
alternative references are experimental or archived routes. They are not selected by
the default import or CLI.

## Repository map

```text
src/autogen/methods/       canonical method packages
tests/methods/             method-specific tests and compact fixtures
configs/                   reviewed example configurations
docs/methods/              scientific definitions and certification contracts
provenance/                machine-readable source/artifact records
scripts/                   generation, validation, and remote launch helpers
```

Large generated plans, checkpoints, logs, raw results, campaign copies, and plot build
trees are excluded from Git and wheels. See the
[repository layout](docs/repository_layout.md) and [provenance policy](docs/provenance.md).

## Validate a checkout

```bash
python scripts/validate_repository.py
python scripts/check_repository_blobs.py --limit-mib 50
python -m pytest
python -m build
python scripts/validate_clean_install.py dist/autogen_wick-*.whl
```

The clean-install check creates an isolated environment, imports all public method
packages outside the checkout, verifies the QPCCSD entry point, and confirms that both
canonical-term JSON manifests were packaged.

Documentation starts at [docs/index.md](docs/index.md).

Citation metadata is provided in [CITATION.cff](CITATION.cff).
