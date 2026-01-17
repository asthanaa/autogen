
# Autogen

<!-- Badges -->
<p align="left">
	<a href="https://pypi.org/project/autogen-wick/"><img alt="PyPI" src="https://img.shields.io/pypi/v/autogen-wick.svg"></a>
	<a href="https://pypi.org/project/autogen-wick/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autogen-wick.svg"></a>
	<a href="https://github.com/ayushasthana/autogen/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/ayushasthana/autogen"></a>
</p>
	<a href="https://ayushasthana.github.io/autogen/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue.svg"></a>
</p>
</p>

Autogen is an automatic expression generator for second-quantized many-body expressions using Wick’s theorem, aimed at quantum chemistry derivations (including UCC-style algebra).

Most documentation lives in [docs/index.md](docs/index.md).


## Installation

Install from PyPI:

```bash
pip install autogen-wick
```

Project name on PyPI: **autogen-wick**
Python import: `import autogen`

Optional: install PySCF (needed for generated integral/einsum scripts):

```bash
pip install pyscf
```

Or with conda:

```bash
conda install -c conda-forge pyscf
```

Or use the conda environment for development:

```bash
conda env create -f environment.yml
conda activate autogen
pytest -q
```

## Canonical imports

Use these package paths:

- `autogen.library`
- `autogen.main_tools`
- `autogen.pkg`

## Common workflows

- Debug workflow (writes LaTeX-ish output to `latex_output.txt` by default):
	- `python debug.py`
- Run fast tests:
	- `pytest`
- Run the slow CCSD integration test:
	- `RUN_SLOW=1 pytest -k ccsd`

## Build

```bash
conda run -n autogen python -m build
```

## Where to read next

- Docs home: [docs/index.md](docs/index.md)
- Concepts/definitions: [docs/concepts.md](docs/concepts.md)
- API guide: [docs/api.md](docs/api.md)
- Usage examples: [docs/usage.md](docs/usage.md)
