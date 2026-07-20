# Production method

## Scope

The production route is one named scientific method:

**projected-AGP-referenced, direct full-coordinate QPCCSD followed by
fixed-amplitude particle-number PAV with the Ser2/W2 closure.**

The reference model, coordinate definition, energy bookkeeping, and
projection schedule are invariants in the production API. They cannot be
changed through a configuration file.

## Quasiparticle reference

A state-specific singlet CASSCF calculation supplies the active one-particle
RDM and active two-particle RDM. Within each spatial-symmetry block, the
active orbitals are rotated to diagonalize the spin-summed 1-RDM. A signed
projected AGP is then fitted to:

- the resulting natural occupations; and
- the symmetrized pair-transfer block of the active 2-RDM.

The relative geminal signs retain pair-phase information that natural
occupations alone cannot provide. A final common scale sets the mean particle
number of the unprojected paired Bogoliubov vacuum. Inactive correlated
orbitals are occupied and external orbitals are empty in that vacuum.

The fit does **not** reconstruct the complete CAS 2-RDM. See
[reference.md](reference.md) for the definition and mandatory diagnostics.

## Hamiltonian and cluster operator

The molecular Hamiltonian is transformed over every correlated spin orbital
and normal ordered with respect to the canonical Bogoliubov vacuum
`|Phi>`. QPCCSD uses the pure-creation cluster operator

```text
T = T1 + T2
T1 = (1/2!) t[pq] beta[p]^dagger beta[q]^dagger
T2 = (1/4!) t[pqrs] beta[p]^dagger beta[q]^dagger
                   beta[r]^dagger beta[s]^dagger.
```

The connected Baker-Campbell-Hausdorff expansion terminates after the fourth
nested commutator for a two-body Hamiltonian. Generated, antisymmetry-reduced
Wick contractions evaluate the scalar, two-creation, and four-creation
components of `exp(-T) H exp(T)`.

## Full symmetry-allowed coordinate space

The residual equations are solved in the total-singlet, totally symmetric
coordinate space. All allowed indices span the complete correlated orbital
space, including pair and quadruple coordinates wholly internal to the
active orbitals. There is no active-space excitation mask.

“Full” therefore means all coordinates allowed by spin and molecular
symmetry. It does not discard those symmetries. The frozen-core N2 CAS(6,6)
regression counts are 8 pair plus 57 quadruple coordinates in STO-3G and 32
pair plus 1641 quadruple coordinates in 6-311G.

## Direct energy contract

For the converged unprojected amplitudes `T*`,

```text
E_raw = <Phi| exp(-T*) H exp(T*) |Phi>.
```

This scalar is the reported QPCCSD total energy. The CASSCF RDMs influence the
reference, but the CASSCF energy is not added. The following expressions
define different, experimental methods and are prohibited in production:

```text
E_CASSCF + E_QP(T*)
E_CASSCF + E_QP(T*) - E_QP(0)
E_CASSCF + Delta_QPCCSD.
```

The result record stores the literal energy convention `direct` together with
the production method identifiers, so downstream plotting cannot confuse the
methods.

## Unprojected optimization and rescue

The production solver combines a diagonal quasiparticle preconditioner,
limited-history secant updates, an analytic Jacobian-vector product, and a
right-preconditioned Newton-Krylov hookstep when ordinary iterations stall.
A denominator floor may regularize only the preconditioner used to propose a
step. It never shifts the Hamiltonian, the residual equations, or the energy.

Certification always reevaluates the original unshifted equations from the
saved terminal amplitudes. For difficult potential-energy curves, geometry
continuation and forward/reverse closure are separate campaign operations;
they do not change the single-point method.

## Particle-number projection after variation

For target correlated electron number `N`,

```text
P_N = (1/2 pi) integral d(phi) exp(i phi (Nhat - N)).
```

The PAV energy is evaluated at the fixed unprojected amplitudes:

```text
E_PAV^N = <Phi| P_N H exp(T*) |Phi>
          --------------------------------.
          <Phi| P_N   exp(T*) |Phi>
```

No projected residual is solved. The amplitudes and orbitals are not relaxed
in the presence of the projector. This is a coupled-cluster projective
energy, not a Hermitian variational quotient; it need not be lower than the
unprojected value and is not an upper bound to FCI.

Production evaluates the U(1) integral on a shifted midpoint grid. At each
angle, the rotated correlated state is represented by a scalar correlated
norm and disentangled `W1` and `W2` clusters. The Ser2/W2 closure retains all
components of `W1` and `W2`, including pure-active components, and sets
`W3 = 0`. “Ser2” denotes this closure, not a projected reoptimization.

The base grid supplies the reported energy. A second calculation on the
doubled grid is a validation, not a Richardson extrapolation. The N2 defaults
are 9/18 grid points for STO-3G and 25/50 for 6-311G.

## Scaling and limitations

With `M` quasiparticle spin orbitals and `L` gauge points, the dominant
rank-four contractions scale as `O(L M^6)`. The implementation does not form
state vectors or runtime tensors above rank four.

The main controlled approximations are the compression of a general CAS
state into a paired Gaussian reference, the QPCCSD cluster truncation, and
the Ser2 condition `W3 = 0`. Particle-number PAV removes wrong-number
components but does not repair all three approximations. All reference-fit
and projection diagnostics must therefore accompany energies used in a
scientific claim.
