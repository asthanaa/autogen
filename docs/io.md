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

## Method inputs and generated code

Canonical method content lives below `src/autogen/methods/<method>`. Derivation inputs
and conventions are under `derivation`, compact generated kernels/manifests under
`generated`, and handwritten CCSD/EOM runtimes under `runtime`. QPCCSD uses its reviewed
`production` runtime layer.

Generated JSON manifests required by runtime or algebra audits are included in the
wheel. Raw calculation results, checkpoints, logs, and large code-generation plans are
external artifacts and must not be loaded implicitly from a checkout-relative path.

Next: see [usage.md](usage.md) for example pipelines.
