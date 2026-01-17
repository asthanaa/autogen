# Usage

## Create environment

From repo root:

- `conda env create -f environment.yml` (first time)
- `conda env update -f environment.yml --prune` (update)
- `conda activate autogen`

## Typical operations

### Commutator

```python
from autogen.main_tools.commutator import comm

terms = comm(['V2'], ['T2'], 1)
```

### Filtering fully contracted terms

```python
from autogen.library.full_con import full_terms

contracted = full_terms(terms)
```

### Debug script

- `python debug.py`

This uses the implementation in `autogen.debug` and writes to `latex_output.txt` by default.
