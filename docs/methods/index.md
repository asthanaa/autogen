# Methods

Every reviewed method has one namespace below `autogen.methods` and one corresponding
test tree below `tests/methods`.

| Method | Public namespace | Runtime organization |
|---|---|---|
| CCSD | `autogen.methods.ccsd` | derivation, generated kernels, runtime |
| EOM-CCSD | `autogen.methods.eom_ccsd` | derivation, generated kernels, runtime |
| QPCCSD/PAV | `autogen.methods.qpccsd` | generated kernels and reviewed `production` runtime |

The generated layers contain reproducible contractions and compact algebra manifests.
Handwritten solvers and adapters live in `runtime` for CCSD/EOM-CCSD. QPCCSD calls its
reviewed runtime layer `production` to distinguish it from experimental routes; it does
not have a redundant `runtime` alias.

## Production QPCCSD

The default route is projected-AGP-referenced, full-coordinate, direct-energy QPCCSD
followed by fixed-amplitude particle-number PAV. Its reference definition, energy
bookkeeping, excitation space, projection closure, and certification thresholds are
scientific contracts rather than interchangeable command-line options.

Read the detailed guides:

- [Method definition](qpccsd/method.md)
- [Projected-AGP reference](qpccsd/reference.md)
- [API and CLI](qpccsd/api.md)
- [Certification](qpccsd/certification.md)
- [Remote execution](qpccsd/remote_execution.md)

Alternative QP references, PN-OAP, cubic cluster operators, and CASSCF-plus-correction
energies must remain outside the public production facade.
