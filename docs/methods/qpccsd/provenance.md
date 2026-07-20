# Provenance

## Consolidation baseline

This method package was imported from the reviewed
`qpccsd_agp2rdm_pav_clean` export. That export was consolidated on 2026-07-20
from the existing `autogen2` worktree at commit:

```text
970dc4ac6291048c85bfd3a198196422ee928d42
implement scalable projected-metric PN-OAP QPCCSD
```

The newest solver-rescue implementation was selected from the isolated
`projected_agp_2rdm_raw_qpccsd_rescue_20260719/solver_rescue/` snapshot. The
projected-AGP reference, active-coordinate driver logic, PAV campaign audits,
and manuscript definitions were cross-checked against the sibling
`reference_campaign/`, `active_t12_qpccsd/`, `pav_campaign/`, and
`plots/inputs/manuscript/` records.

The consolidation copies only runtime code, generated kernels required by
that code, compact regression evidence, tests, and documentation. Campaign
outputs, restarts, duplicate source trees, temporary files, plot builds, and
machine-specific launch state are not part of this repository.

## Scientific decisions made during consolidation

- The projected-AGP fit to active natural occupations and the pair-transfer
  2-RDM block is the only production reference.
- Pure-active T1/T2 coordinates are enabled unconditionally in the production
  space.
- The production energy is the direct quasiparticle molecular-Hamiltonian
  energy. Legacy automatic CASSCF-plus-delta bookkeeping is not used.
- Production projection is fixed-amplitude PAV using midpoint Ser2/W2 with
  `W3 = 0` and a doubled-grid validation.
- PN-OAP and other alternative theories are experimental.
- Generated canonical-term JSON manifests remain package data because they
  are part of runtime/algebra provenance rather than campaign output.

## Reproducibility record

The consolidation commit should be treated as the code identity for the first
clean molecular regression. Its regression JSON must additionally store the
exact file SHA-256 values, Python and dependency versions, host, configuration
hash, and the remote job receipt. This document records origin; it does not
substitute for the machine-readable result provenance.

The compact Medora receipt is stored at
[`tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json`](https://github.com/asthanaa/autogen/blob/master/tests/methods/qpccsd/validation/n2_sto3g_cas66_r1p10_medora_evidence.json).
