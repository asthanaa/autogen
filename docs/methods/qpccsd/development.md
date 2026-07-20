# Development and testing

## Boundaries

Production changes must preserve four invariants:

1. projected-AGP fit to natural occupations plus the active pair-transfer
   2-RDM block;
2. all symmetry-allowed coordinates, including pure-active T1/T2;
3. direct molecular-Hamiltonian energy with no CASSCF addition; and
4. fixed-amplitude midpoint Ser2/W2 PAV.

A change to any of these defines an experimental route and belongs under
`experiments/` or an explicitly experimental namespace.

## Test groups

Run fast synthetic and contract tests locally:

```bash
python -m pytest -m "not molecular and not remote" -q
```

On Medora, Talon, or an authorized remote desktop, run the complete suite:

```bash
python -m pytest -q
```

Important regression groups cover:

- signed-AGP subset sums, fitting, number scaling, and canonicality;
- coordinate construction and the N2 STO-3G/6-311G counts;
- generated Wick kernels and analytic Jacobian-vector products;
- direct energy bookkeeping at zero and nonzero amplitudes;
- solver convergence and fresh unshifted residual evaluation;
- PAV immutability, Ser2 base/doubled grids, and imaginary-energy gates;
- production import/CLI isolation from experimental routes; and
- one molecular N2 geometry compared with the checked-in regression record.

## Molecular regression policy

The one-geometry test is marked both `molecular` and `remote`. It audits an
already calculated result and never starts a molecular calculation itself; if
the result is absent, it skips. A new expected energy is accepted only with a
result file containing the immutable method identifiers, input and source
hashes, approved-host provenance, residual and grid gates, and a short
explanation in the change record.

The checked-in compact record for the reference geometry is
[`tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json`](https://github.com/asthanaa/autogen/blob/master/tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json).
It records the remote environment, input and output hashes, exact energy
comparison, certification gates, timings, and complete remote test outcome.

Use an energy tolerance tight enough to catch bookkeeping or kernel changes
but broad enough for documented BLAS/PySCF roundoff. Never weaken the residual
or projection certification gates to make a regression pass.

## Packaging checks

Before a release or commit intended for handoff:

```bash
python -m build
python -m pip install --force-reinstall dist/*.whl
python -m autogen.methods.qpccsd.cli show-defaults
python -m pytest -m "not molecular and not remote" -q
```

Inspect the wheel to confirm the canonical-term JSON files are included and
campaign data, checkpoints, logs, PDFs, and absolute machine paths are not.

## Review checklist

- Does every total energy state the direct-energy convention?
- Can an experimental route be reached from the default CLI or wildcard
  production import? If so, move it out of the public surface.
- Are PAV input amplitudes byte-for-byte unchanged?
- Are base and doubled PAV values retained separately?
- Are failed or incomplete gates represented explicitly?
- Were molecular calculations run only on approved remote hosts?
- Are all new generated kernels paired with algebraic and numerical tests?
