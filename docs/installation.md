# Installation

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