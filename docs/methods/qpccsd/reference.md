# Projected-AGP reference

## CAS source

Let `|Psi_CAS>` be a state-specific singlet CASSCF solution. The spin-summed
spatial one-particle RDM is

```text
gamma[p,q] = <Psi_CAS| sum_sigma a[q,sigma]^dagger a[p,sigma] |Psi_CAS>.
```

Within each spatial-symmetry block, active orbitals are rotated so that
`gamma[p,q] = n[p] delta[p,q]`. The active 2-RDM is transformed by the same
rotation. In the spin-free convention used by the implementation, the fit
target is the symmetrized pair-transfer block

```text
Pi[p,q] = 1/2 (Gamma[pq,pq] + Gamma[qp,qp]).
```

This exact index convention must be recorded with every reference artifact.

## Fixed-number AGP fit

For `n = N_active / 2` active electron pairs and pair creator
`P[p]^dagger = a[p,alpha]^dagger a[p,beta]^dagger`, the fitted state is

```text
|AGP_Nactive> proportional to
P_Nactive product_p (1 + g[p] P[p]^dagger) |vacuum>.
```

For an `n`-orbital subset `I`, define `C_I = product_(p in I) g[p]` and
`Z_n = sum_(|I|=n) C_I^2`. Exact subset sums give the AGP occupations and
pair transfers:

```text
n_AGP[p] = (2/Z_n) sum_(I contains p) C_I^2

Pi_AGP[p,q] = (2/Z_n) sum_(I contains q, p not in I) C_(I-q+p) C_I,
Pi_AGP[p,p] = n_AGP[p].
```

Relative signed geminals minimize the equal-weight least-squares differences
between `n_AGP` and `n_CAS`, and between `Pi_AGP` and `Pi_CAS`. Initial
magnitudes follow `sqrt(n[p]/(2-n[p]))`; relative signs come from the dominant
eigenvector of the target pair-transfer matrix. Logarithmic magnitude ratios
remove the irrelevant common scale of fixed-number AGP.

No energy, FCI result, CC result, or later geometry is used to select the fit.

## Unprojected Bogoliubov vacuum

The common AGP scale cancels in the fixed-number state but matters in the
unprojected quasiparticle vacuum. A positive scale `lambda` is therefore
chosen to satisfy

```text
sum_(p in active) 2 (lambda g[p])^2 / (1 + (lambda g[p])^2) = N_active.
```

Then

```text
u[p] = 1 / sqrt(1 + (lambda g[p])^2)
v[p] = lambda g[p] u[p].
```

Inactive correlated orbitals use `(u,v)=(0,1)` and external orbitals use
`(u,v)=(1,0)`. The resulting matrices must satisfy

```text
U^dagger U + V^dagger V = I
U^T V + V^T U = 0.
```

The Gaussian Bogoliubov vacuum used in Wick contractions and its
target-number AGP component are related, but they are not the same object.

## What information is and is not retained

The reference uses the diagonal active 1-RDM and one pair-transfer block of
the active 2-RDM. It does not use every element of the active 2-RDM, nor the
active 3- or 4-RDM. A generic CAS wavefunction cannot be reconstructed from
this information. The compression error is a reference-model limitation, not
a QPCCSD convergence error, and is never converted into an energy correction.

Each result must retain at least:

- CASSCF convergence and energy as provenance only;
- active electron and orbital counts;
- frozen, active, and external orbital identities;
- natural occupations and orbital-symmetry labels;
- signed relative geminals and the global number-setting scale;
- maximum occupation-fit and pair-transfer-fit errors;
- normalized pair-subspace fidelity;
- target mean active number; and
- both Bogoliubov canonical defects.

The production serializer retains these compactly as active natural
occupations, relative and number-scaled signed geminals, and the explicit
global number-setting scale. It also records deterministic SHA-256
fingerprints of the active 1-RDM, active 2-RDM, and correlated MO coefficient
array without embedding those full arrays in result JSON.

The source-RDM record is fail-closed. It must state exactly the spin-summed
spatial 1-RDM convention, the PySCF spin-free 2-RDM convention, the
`active_rdm2[p,q,p,q]` pair-transfer indexing, the versioned projected-AGP fit
protocol, and `complete_active_rdm2_reconstructed = false`. A checkpoint that
omits any of these fields is not inferred or upgraded during serialization.

The serialized production result also retains compact SHA-256 fingerprints
of the active 1-RDM, active 2-RDM, and correlated molecular-orbital
coefficients. These fingerprints identify the numerical reference inputs
without expanding the result JSON with full dense arrays.

For a potential-energy curve, orbital/block alignment, CI phase continuity,
and signed-geminal continuity should also be recorded. They diagnose branch
continuity but do not replace the fit-error fields.
