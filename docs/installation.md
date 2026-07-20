# Installation

## Quick install (PyPI)

To install the latest release directly from PyPI:

```bash
pip install autogen-wick
```

Python 3.10 or newer is required. The base installation includes NumPy, SciPy,
Matplotlib, and thread-pool controls. You can then use it in Python:

```python
import autogen
```

Install the molecular extra when PySCF-backed references or regressions are needed:

```bash
pip install "autogen-wick[molecular]"
```

Or with conda:

```bash
conda install -c conda-forge pyscf
```

## Recommended: conda environment

From the repo root:

```bash
conda env create -f environment.yml
conda activate autogen
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate autogen
```

This environment installs the project in editable mode (`-e .`), so imports work from anywhere.

For a lightweight virtual-environment development setup instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Molecular calculations are excluded from the default test selection. Run them only on
Medora, Talon, or an authorized remote desktop with the `molecular` extra installed.

## Build artifacts (sdist + wheel)

```bash
conda run -n autogen python -m build
python scripts/validate_clean_install.py dist/autogen_wick-*.whl
```

Outputs go into `dist/`.

Next: see [testing.md](testing.md) and [usage.md](usage.md).
