# Input and output

Autogen is a library. Most “inputs” are Python scripts that call the APIs and then print terms to a file.

## Output file

Many workflows print LaTeX-friendly output to a text file (commonly `latex_output.txt`).

- The package does not require a specific output file name, but many scripts default to `latex_output.txt`.
- The debug workflow defaults to `latex_output.txt` as well.

## Printing terms

```python
from autogen.library.print_terms import print_terms

print_terms(terms, 'latex_output.txt')
```

## Debug workflow

```bash
python debug.py
```

This calls `autogen.debug.run_debug()` and writes to `latex_output.txt` by default.

## Example scripts

You may see historical scripts named `input.py` under various folders (e.g. older experiments under `backup/`). Treat those as runnable examples rather than a single canonical input format.

## Method specs and generated code layout

- Method input specs live under `method_inputs/<method>/` (for example, `method_inputs/ccsd/ccsd_spec.py`).
- Generated code is written under `generated_code/methods/<method>/` and includes `__init__.py` files so it can be imported as a package.
- Slow integration test molecule fixtures live under `tests/molecules/`.

Next: see [usage.md](usage.md) for example pipelines.
