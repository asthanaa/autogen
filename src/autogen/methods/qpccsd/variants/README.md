# Experimental and test routes

This directory is a quarantine boundary for scientific alternatives. Nothing
described here is selected by the production configuration, default CLI, or
the canonical `autogen.methods.qpccsd` API. The legacy
`autogen.qpccsd.workflow` module is only a compatibility forwarder to that
production API.

The entries below are a research inventory, not a claim that every route has
an executable implementation. In particular, the Sokolov--Chan namespace has
only a reference-construction helper and plan; its CASSCF-plus-QPCCSD energy
route is not implemented. The cubic namespace retains documentation and an
isolation-policy snapshot only; its external implementation and campaign data
are not vendored here.

Routes retained for controlled comparison or testing include:

- a Sokolov-style linear 1-RDM reference and CASSCF-plus-dynamic-QPCCSD
  bookkeeping;
- explicit CASSCF-plus-delta energy conventions;
- excitation spaces that mask pure-active coordinates;
- projected-amplitude optimization (PN-OAP);
- ODE2, midpoint-Richardson, and Ser3 projection backends;
- exact-sector Fock-space oracles for small systems;
- level-shift, denominator, and Jacobian diagnostics;
- CAS-contracted and HFB reference variants; and
- cubic quasiparticle transformations.

An experimental result must use a distinct method identifier and result
schema. It must never overwrite a production result or be merged into a
production PES under the same legend. Useful oracle code may live in the test
suite, but it remains outside default imports.

See [the Sokolov-Chan N2 plan](sokolov_chan/PLAN.md) for the first isolated
comparison proposed after this cleanup.
