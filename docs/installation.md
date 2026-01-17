# Installation

## Quick install (PyPI)

To install the latest release directly from PyPI:

```bash
pip install autogen-wick
```

This will install the package and all dependencies. You can then use it in Python:

```python
import autogen
```

Optional: install PySCF if you want to run the generated integral/einsum scripts:

```bash
pip install pyscf
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

## Build artifacts (sdist + wheel)

```bash
conda run -n autogen python -m build
```

Outputs go into `dist/`.

Next: see [testing.md](testing.md) and [usage.md](usage.md).
