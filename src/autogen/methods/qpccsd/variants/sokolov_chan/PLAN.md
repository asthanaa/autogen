# Plan: Sokolov-Chan-style CASSCF + QPCCSD N2 comparison

## Status

Planned only. This route is not implemented, validated, or callable from the
production CLI.

## Question

Measure how a Sokolov-Chan-style active-space reference and explicit
CASSCF-plus-dynamic-QPCCSD energy partition perform on the N2 potential-energy
curve, using the same molecular Hamiltonians and comparison protocol as the
production projected-AGP study.

## Proposed method contract

1. Generate a state-specific CASSCF(6,6) reference with both nitrogen 1s
   orbitals frozen.
2. Construct the quasiparticle reference from the active one-particle density
   matrix following the selected Sokolov-Chan convention. Record exactly
   which density eigenvectors, occupations, phases, and inactive/external
   rules enter the Bogoliubov transformation.
3. Exclude correlation already assigned to the active solver according to the
   published excitation-space partition. Do not reuse the production
   full-active coordinate claim for this route.
4. Solve the corresponding unprojected QPCCSD amplitude equations.
5. Report the explicitly labelled energy

   ```text
   E_SC-QPCCSD = E_CASSCF + E_QP(T*) - E_QP(0).
   ```

   This is a CAS-plus-dynamic correction and must never be labelled as the
   production direct QPCCSD energy.
6. Do not add PAV in the first implementation. Establish and validate the
   unprojected energy partition before defining any projected counterpart.

## Implementation isolation

- Place implementation under an experimental namespace.
- Use distinct configuration and result schemas.
- Reject production result identifiers at load time.
- Reuse only algebraically common tensor kernels and molecular inputs.
- Add tests proving that production imports and commands cannot select this
  route.

## N2 comparison

Start with STO-3G and 6-311G at the same geometries, frozen-core convention,
active-space orbitals, and CASSCF convergence tolerances as the production
study. For every point retain CASSCF, zero-amplitude quasiparticle, dynamic
correction, and assembled total energy separately. Compare with FCI only on
an identical frozen-core Hamiltonian.

Minimum pilot geometries are 1.10 and 2.00 angstrom. Expand to the full PES
only after both points pass reference, energy-partition, residual, and
continuity audits.

## Validation gates

- reproduce zero-correlation and noninteracting limits;
- prove numerically that the assembled total equals
  `E_CASSCF + E_QP(T*) - E_QP(0)`;
- verify that active/external excitation masks match the published method;
- compare analytic kernels with a small exact-sector oracle;
- require a fresh unshifted residual below `1e-8`;
- verify orbital and root continuity in forward/reverse scans; and
- keep all molecular calculations on Medora, Talon, or the authorized remote
  desktop.

## Deliverables

- a self-contained method note with equations and conventions;
- experimental source and unit tests;
- two-geometry pilot result/audit records;
- a same-Hamiltonian N2 PES table and paper-style comparison plot; and
- a written decision on whether the route warrants projection or cubic terms.
