# Production API

## Supported surface

The stable production surface is `autogen.methods.qpccsd`:

```python
from autogen.methods.qpccsd import (
    ProductionConfig,
    ProductionResult,
    prepare_projected_agp_reference,
    build_full_active_qp_space,
    solve_direct_qpccsd,
    evaluate_direct_pav,
    run_qpccsd_pav,
)
```

`run_qpccsd_pav` performs the stages in order and returns a structured
`ProductionResult`. The lower-level stage functions exist for restart,
profiling, and audit workflows; using them does not change the production
contracts.

Scientific method identifiers and the direct-energy convention live in
`autogen.methods.qpccsd.production.contracts`. They are constants rather than user-selectable
configuration fields.

## Configuration

The QPCCSD CLI reads the following TOML sections and constructs a
`ProductionConfig`. The Python dataclass itself is numerical-only and does not
provide a `from_toml` method; Python callers prepare the molecular reference
explicitly and instantiate `ProductionConfig` directly.

### `[system]`

| Field | Meaning |
|---|---|
| `molecule` | molecular helper preset, `"n2"` or `"h2"` |
| `basis` | PySCF basis name |
| `distance_angstrom` | internuclear separation in angstrom |
| `charge` | molecular charge |
| `spin` | `N_alpha - N_beta` |
| `symmetry` | enable molecular spatial symmetry |

### `[active_space]`

| Field | Meaning |
|---|---|
| `orbitals` | number of active spatial orbitals |
| `electrons` | number of active electrons |
| `frozen_n1s` | freeze both nitrogen 1s orbitals |

### `[solver]`

This section accepts the reviewed numerical controls exposed by
`SolverOptions`, including the iteration limits, residual/energy tolerances,
step radii, denominator floor, and Newton-Krylov limits. Regularization fields
affect the path to a root only. Certification reevaluates the unshifted
equations.

### `[projection]`

| Field | Meaning |
|---|---|
| `grid_size` | base midpoint grid; validation uses twice this size |
| `validation_tolerance` | maximum base/doubled-grid energy difference |
| `cache_mib` | projection cache limit in MiB |
| `parallel_mode` | `serial`, `blas`, `pipeline`, or `auto` |
| `workers` | worker count when applicable |
| `blas_threads` | BLAS thread count per process |
| `maximum_imaginary_energy` | certification bound for raw and PAV imaginary energies |
| `project_uncertified_finite_amplitudes` | allow explicitly diagnostic PAV from a finite uncertified root |
| `compute_residual_diagnostic` | opt in to the expensive projected-residual diagnostic |

The target number is derived from the correlated molecular problem. The
projection backend is fixed to midpoint Ser2/W2 PAV and cannot be changed in
this section.

### `[output]`

`path` is the JSON result path. Checkpoints and heavy diagnostics, when
requested by a remote driver, belong in the same ignored run directory.

See the repository's
[STO-3G](https://github.com/asthanaa/autogen/blob/master/configs/qpccsd/n2_sto3g.toml)
and
[6-311G](https://github.com/asthanaa/autogen/blob/master/configs/qpccsd/n2_6311g.toml)
examples.

## Command line

```text
python -m autogen.methods.qpccsd.cli show-defaults
python -m autogen.methods.qpccsd.cli run CONFIG.toml
qpccsd validate RESULT.json [--anchor CERTIFIED_ANCHOR.json]
```

- `show-defaults` prints the immutable method identifiers and numerical
  defaults.
- `run` performs the full reference, direct QPCCSD, certification, and PAV
  workflow, writes a structured JSON result and a pickle-free compressed
  amplitude checkpoint, and prints the result JSON.
- `validate` checks a serialized result’s production method contract. With an
  anchor, it also compares QPCCSD/PAV energies, coordinate counts, the raw
  residual, and the PAV grid error with that certified record.

`validate` does not rerun contractions or independently certify a calculation;
it audits fields already present in the result. TOML keys are parsed and
validated at the beginning of `run`.

`run` also accepts `--reference-checkpoint` and `--initial-amplitudes` for
audited restarts. A reference checkpoint is accepted only when its molecular
composition, geometry, basis, active electron/orbital counts, frozen-core
choice, projected-AGP source protocol, fit diagnostics, and canonicality all
match the configuration. Older checkpoints without those fields fail closed;
they are never silently promoted to the production protocol.

The default CLI does not expose PN-OAP, CAS-plus-delta, masked spaces,
alternative reference models, or alternative disentanglement backends.

When `--reference-checkpoint` is supplied, the loader verifies the atomic
composition, distance, basis, active orbital/electron counts, frozen-core
indices, and physical/correlated particle targets against the TOML request
before restoring molecular integrals. Legacy checkpoints without the exact
source-RDM protocol and projected-AGP fit provenance fail closed and must be
rebuilt; their method is never guessed from a filename.

## Result contract

`ProductionResult.to_dict()` separates values from certification. Its
serialized form
contains:

- schema and immutable method identifiers;
- host, Python, NumPy, and caller-supplied provenance information;
- CASSCF/reference diagnostics, active natural occupations, relative and
  number-scaled signed geminals, the global number-setting scale, and compact
  SHA-256 fingerprints of the active RDMs and molecular-orbital coefficients;
- excitation-space counts;
- direct QPCCSD energy, residual, and solver diagnostics;
- base/doubled-grid PAV energies and projection diagnostics; and
- component and overall certification states.

The CLI additionally stores the configuration SHA-256 and the output
amplitude-file SHA-256. The amplitude checkpoint uses NumPy NPZ with pickle
disabled.

Consumers must use the explicit `energy_convention` and certification fields.
They must not infer a method from filenames or treat a finite value as
certified.
