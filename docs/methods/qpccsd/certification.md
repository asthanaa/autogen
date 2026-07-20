# Certification and result states

## Fail-closed principle

The workflow distinguishes four questions:

1. Was the projected-AGP reference built consistently?
2. Is the saved QPCCSD state a root of the original equations?
3. Is the fixed-amplitude PAV quadrature numerically stable?
4. For a continued curve, has an independent branch-closure check been done?

A finite energy answers none of these questions by itself. Missing evidence is
reported as missing, not silently promoted to success.

## Reference gates

The production reference must have finite RDMs and fit diagnostics, the
requested CAS electron/orbital counts, exact orbital partition coverage, the
correct target particle number, and Bogoliubov canonical defects below
`1e-12`. The occupation and pair-transfer fit errors are retained as model
diagnostics. They are not hard convergence gates unless a campaign explicitly
declares additional scientific acceptance criteria.

Certification also requires the exact versioned source-RDM conventions,
successful finite projected-AGP fit diagnostics, finite relative/scaled
geminals, an explicit global number-setting scale, and two Bogoliubov
canonical defects strictly below `1e-12`. Missing legacy metadata is a hard
failure rather than a serialization default.

## QPCCSD root gates

The terminal amplitudes are evaluated once more using the original,
unshifted Hamiltonian and residual equations. Certification requires strict
inequalities:

```text
max(abs(R_pair), abs(R_quadruple)) < 1e-8
abs(Im(E_QPCCSD))                  < 1e-8 Eh.
```

Denominator floors, trust radii, line searches, and Newton shifts may be used
to propose iterative steps. They do not appear in the final equations or
reported energy.

For a potential-energy curve, a stronger branch certificate may additionally
require independent forward/reverse continuation closure:

```text
abs(Delta E)            <= 1e-8 Eh
max(abs(Delta amplitude)) <= 1e-6.
```

That campaign-level closure state is kept distinct from single-point root
convergence.

## PAV gates

PAV reuses an immutable copy of the terminal unprojected amplitudes. The
implementation verifies that projection did not alter them. The base grid is
the reported value; the doubled grid is an independent discretization check.

```text
abs(E_2L - E_L) < 1e-8 Eh
abs(Im(E_L))    < 1e-8 Eh.
```

The projected denominator must also remain finite and above the configured
overlap safety threshold. No Richardson-extrapolated energy is reported.

## Labels

Use these semantic states in tables and plots:

| State | Meaning | Plotting guidance |
|---|---|---|
| `certified` | all required reference, root, and PAV gates pass | filled marker |
| `root_certified` | single-point root passes; campaign closure not requested or not yet supplied | record closure separately |
| `diagnostic_only` | PAV was evaluated from a finite but uncertified source root, or a required PAV gate failed | open marker; exclude from certified statistics |
| `failed` | no finite or contract-valid value is available | omit energy; report failure |

If a curve requires independent branch closure, an otherwise valid root may
remain an open marker until that closure is available. The reason must be
stored explicitly so it is not confused with a residual or quadrature
failure.

## One-geometry molecular regression

The repository’s remote molecular test executes one frozen-core N2/STO-3G
geometry through both QPCCSD and PAV. It compares the fresh values with a
checked-in, provenance-stamped regression record and verifies the method
identifiers, coordinate counts, residual, direct-energy convention, frozen
amplitudes, doubled-grid difference, and imaginary-energy gates.

The certified Medora anchor at `R = 1.10 angstrom` is:

| Quantity | Expected value |
|---|---:|
| direct QPCCSD energy | `-107.65269549964435 Eh` |
| fresh unshifted residual maximum | `9.660306061257431e-10` |
| fixed-amplitude Ser2 PAV energy | `-107.65060143314642 Eh` |
| 9/18-grid difference | `3.47388253863627e-18 Eh` |
| PAV imaginary-energy magnitude | `7.422711739519054e-18 Eh` |

The machine-readable record is
`tests/methods/qpccsd/fixtures/n2_sto3g_cas66_r1p10_certified.json`; its source result,
amplitude, and reference SHA-256 values are part of that fixture. The fresh
clean-repository run must agree within `1e-8 Eh` while independently passing
all numerical gates.

The final clean-repository validation rebuilt CASSCF and the projected-AGP
reference from scratch on Medora and started QPCCSD from zero amplitudes. It
obtained `-107.65269549939507 Eh` for direct QPCCSD with a fresh unshifted
residual of `9.33506029421327e-10`, and `-107.65060143315307 Eh` for PAV.
The differences from the certified anchor were `2.493e-10 Eh` and
`6.651e-12 Eh`, respectively; all 54 tests passed. The complete receipt is
[`tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json`](https://github.com/asthanaa/autogen/blob/master/tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json).

Updating that regression reference requires an independent rerun on an
approved host and a documented scientific reason. A test failure must not be
resolved by simply replacing expected energies.
