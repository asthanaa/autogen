# Repository layout

## One installable source tree

`src/autogen` is the only Python library source root. A method is organized by role:

```text
src/autogen/methods/<method>/
  derivation/      symbolic specifications and conventions, when applicable
  generated/       checked, reproducible runtime kernels and compact manifests
  runtime/         handwritten CCSD/EOM-CCSD solvers and adapters
  production/      reviewed QPCCSD runtime (used instead of runtime/)
```

Public `__init__.py` modules are facades. Callers should not import repository-relative
`generated_code` or `method_inputs` paths. Generated modules use package-relative
imports so the installed wheel behaves the same way as the checkout.

Tests mirror the method names under `tests/methods`. Repository-level packaging and Git
contract tests live under `tests/repository`; they are not part of a method's scientific
test suite.

## Generated data

Compact JSON manifests needed to reproduce or audit runtime algebra are package data.
They ship with the wheel beside the generated QPCCSD modules. Large diagnostic plans,
rendered reports, and full campaign outputs are not runtime data and do not ship.

The historical `projected_codegen_plan.json` was 387,064,269 bytes. Its identity and
regeneration command are retained in
`provenance/excluded_large_objects.json`, while `.gitignore` prevents the object from
returning to the repository.

## Configurations, scripts, and documentation

- `configs/<method>` contains reviewed, human-readable example inputs.
- `scripts` contains generation, validation, and remote launch helpers—not library code.
- `docs/methods/<method>` records equations, conventions, and certification criteria.
- `provenance` contains compact machine-readable records for excluded artifacts and
  consolidation sources.

## Archive boundary

`archive`, `artifacts`, `results`, checkpoint, log, raw-plot, and scratch directories are
ignored. Archives may preserve exact historical snapshots outside the repository, but
they are never imported, discovered by pytest, included in a wheel, or treated as a
second implementation. Scientific results promoted to tests must be reduced to compact,
provenance-stamped fixtures under `tests/methods/<method>`.
