# Cubic QPCCSD research implementation

> **Consolidation note.** This document is retained for research provenance.
> The independent cubic package, generated artifacts, tests, and campaign data
> described below are not vendored in this repository. The local
> `autogen.methods.qpccsd.variants.cubic` namespace is an unimplemented,
> disabled boundary and cannot run a cubic-QPCCSD calculation. The accompanying
> `isolation_policy.json` is a historical snapshot rather than a policy applied
> to the consolidated source tree.

> **Method boundary.** The original `unprojected-cqpccsd(4)` implementation
> below is retained as a readable legacy diagnostic. It is not the repaired
> method. The repair introduces three immutable identities:
> `sokolov-cubic-fb1-qpccsd`, `literal-cubic-r12-audit`, and
> `exact-active-unitary-oracle`. See the
> [order-consistent appendix](docs/repaired_cubic_appendix.md). No repaired
> 6-311G curve is released until every fail-closed acceptance gate passes.

This directory is a standalone implementation of an **unprojected** cubic
quasiparticle CCSD method. The Python package is `cubic_qpccsd`; the method is
reported as `CQPCCSD(4)`, where `(4)` means that the completely normal-ordered
cubic-transformed Hamiltonian is closed at total quasiparticle rank four.
Retaining ranks 6--12 would define a separate future method, require newly
derived residual equations, and would not inherit the formal \(N^6\) residual-
scaling statement below.

The implementation does not call particle-number projection, integrate gauge
angles, or import the parent repository's linear-QPCCSD implementation. It
contains its own molecular loader, cubic reference contracts, Hamiltonian
builder, rank-four residual kernel, excitation spaces, solver, command-line
entry point, tests, and independent SeQuant check.

## Scientific definition

The implemented full cubic inverse is

\[
c_p = U_{pa}\beta_a + V^*_{pa}\beta_a^\dagger
+ \frac1{3!}D_{p;abc}\beta_c\beta_b\beta_a
+ \frac12 X_{p;a\mid bc}\beta_a^\dagger\beta_c\beta_b
+ \frac12 Y_{p;ab\mid c}\beta_a^\dagger\beta_b^\dagger\beta_c
+ \frac1{3!}C_{p;abc}\beta_a^\dagger\beta_b^\dagger\beta_c^\dagger .
\]

This inverse is a definition of this project, motivated by the higher-order
polynomial hierarchy discussed by Sokolov and Chan. Their Eq. (13) is a
forward polynomial transformation, and their numerical work uses the linear
Bogoliubov transformation. They do not publish this `C/D/X/Y` inverse convention,
CQPCCSD(4), or its certification procedure; those are extensions developed
here.

`C` and `D` are fully antisymmetric in their three QP indices; `X` is
antisymmetric in `b,c`; `Y` is antisymmetric in `a,b`. In molecular
runs the cubic tensors are active-space tensors, while `U` and `V` retain the
full correlated spin-orbital dimension. The creator field is obtained only by
Hermitian adjunction and is not fitted independently.

The physical Hamiltonian is

\[
H=E+h_{pq}c_p^\dagger c_q
+\frac14\bar v_{pqrs}c_p^\dagger c_q^\dagger c_s c_r.
\]

All transformed field products are expanded and fermionically normal ordered
before the rank closure is made. The untruncated two-body substitution reaches
rank 12. `CQPCCSD(4)` retains only

\[
H^{(4)}=H^{00}+H^{11}+H^{20}+H^{02}+H^{22}
+H^{31}+H^{13}+H^{40}+H^{04}.
\]

Ranks 6, 8, 10, and 12 are discarded only after all possible contractions.
The recorded `by_cubic_insertions = 0,...,4` label counts cubic fields used in
a substitution branch; it is not an operator rank or perturbation order.

This is not the linear Hamiltonian with high-rank arrays merely omitted.
Contractions involving `C,D,X,Y` contribute to the retained scalar, rank-two,
and rank-four blocks, so their numerical values change whenever the cubic
coefficients are nonzero. The result becomes exactly the linear Bogoliubov
Hamiltonian only at `C = D = X = Y = 0`.

After rank-four closure, the pair and quadruple cluster-amplitude definitions
and connected QPCCSD residual topology are unchanged. The vendored residual
kernel has a maximum formal iterative scaling of \(O(N^6)\). That statement
applies to residual evaluation, not automatically to construction of the
cubic-transformed Hamiltonian. The current active-domain Hamiltonian builder
uses greedy tensor-network paths with an explicit \(N^4\)-element temporary
ceiling and records conservative topology envelopes. The ceiling is no larger
than one full rank-four block already allocated by the method; it permits
small active rank-six intermediates when they avoid a catastrophically worse
all-at-once path, but it never materializes a rank-6--rank-12 Hamiltonian.
This remains an exact retained-rank prototype, not a screened large-basis
contraction planner.

Here the historical QPCCSD names mean quasiparticle pairs rather than ordinary
particle-hole ranks:

\[
T_1=\frac1{2!}t_{pq}\beta_p^\dagger\beta_q^\dagger,
\qquad
T_2=\frac1{4!}t_{pqrs}
\beta_p^\dagger\beta_q^\dagger\beta_r^\dagger\beta_s^\dagger.
\]

## Implemented backends

- `hamiltonian_engine.py` is an exact sparse-CAR oracle for small systems. It
  constructs the full transformed polynomial, normal orders it, and then
  splits retained and discarded ranks.
- `scalable_hamiltonian.py` constructs only the retained rank-four molecular
  blocks with full `U,V` and active-only `C,D,X,Y`. It never allocates dense
  rank-6--rank-12 Hamiltonian tensors. Grade-resolved rank-four blocks can be
  streamed to avoid five extra copies of every \(N^4\) array. Contraction
  temporaries are capped at \(N^4\) elements and the deterministic thread
  reducer uses a bounded two-wave queue.
- `generated_rank4_kernel.py` is the isolated 84-term QPCCSD(4) residual plan
  (42 nonzero executable contractions) with formal maximum residual scaling
  \(O(N^6)\).
- `coordinate_rank4_kernel.py` evaluates those same 42 equations only on the
  pair and quadruple coordinates requested by a restricted excitation space.
  It deletes no terms and is parity-tested against the dense generated kernel.
- `rank6_factorized_ir.py` records the 128 `g2 L0 L0 L0 Q3` source networks,
  lowers them onto all 20 new connected rank-six QPCCSD signatures, and stores
  symbolic CSE and analytic-JVP replacement metadata. The bounded molecular
  path executor now covers all 20 signatures and matches the complete sparse
  oracle. Exact antisymmetry/dummy-index canonicalization reduces 1,644,728
  labelled Wick paths to 9,364 contractions. A tempting 440-route
  one-representative-per-profile reduction is demonstrably inexact and is
  rejected fail-closed; exact three-linear/exterior-power batching is still
  required before an N=48 production solve or scaling claim.
- `full_fb1_diagnostic.py` adds the certified 9,364-path rank-six energy,
  pair/quartet residual, and analytic JVP to the existing rank-four evaluator
  under a distinct immutable diagnostic identity. Its small dense-Newton
  solver is limited to at most eight modes. `forward_full_fb1_diagnostic.py`
  constructs the source-bound molecular facade but fails closed on rejected
  references unless the caller explicitly requests a T=0 initial-residual
  timing audit. See
  [`docs/full_fb1_diagnostic.md`](docs/full_fb1_diagnostic.md).
- `product_rdm_oracle.py` evaluates full inactive-product/active-CAS/external-
  vacuum expectations through six fields from only the active 1/2/3-RDM.  It
  avoids the 195.7 GB dense full-space 3-RDM and is parity-tested against
  explicit small Fock states.
- `mixed_q3_refit.py` performs a distinct 45-coordinate, matrix-free reference
  diagnostic: all 33 active canonical directions plus the complete 12-member
  active/occupied-inactive spectator shell. At N2/6-311G, 1.10 Angstrom the
  mixed gradient and update are exactly zero, and a product-sector screen
  excludes first-order descent in all 2,286 two-active-role coordinates. It
  remains reference-only and cannot expose an energy. See
  [`docs/mixed_q3_refit.md`](docs/mixed_q3_refit.md).
- `canonical_completion.py` integrates a compatible forward cubic tangent
  into anti-Hermitian `K2+K4` and constructs its second-BCH field.  The current
  1.10 Angstrom FB2 result is a direct-CI reference diagnostic, not a molecular
  energy or production checkpoint.
- `active_k24_exponential_diagnostic.py` optimizes a separately named,
  spin-scalar active `exp(K4)` reference and, only when requested or required
  by failed K4 gates, a six-angle singlet-pair `K2+K4` extension. Sparse exact
  exponential actions, certified Frechet gradients, multiple starts, and
  direct 1/2/3-RDM checks are active-space diagnostics only; they cannot be
  mixed with the finite cubic/FB1 Hamiltonian or promoted to an energy. See
  [`docs/exact_active_k24_exponential.md`](docs/exact_active_k24_exponential.md).
- `hamiltonian_checkpoint.py` stores exact, pickle-free, source/transform/
  build-option-bound rank-four blocks so solver controls can change without
  rebuilding the cubic Hamiltonian.
- `sequant/` independently substitutes the same full cubic fields into the
  physical Hamiltonian and performs fermionic normal ordering with pinned
  SeQuant revision `db5dd8dae6408c764a84c2b165cd9e413ab91ac6`.

The exact and scalable Python Hamiltonian builders are parity-tested on small
systems. The independent SeQuant v2 expansion contains 11,102 complete terms:
3,514 retained through rank four and 7,588 discarded terms through rank 12.

## Reference fitting and the nonzero-cubic gate

The production candidate is parameterized by a real anti-Hermitian quartic
generator (K_4). The active annihilators and inverse are truncated at the
same first BCH order,

\[
b_p=\beta_p+[K_4,\beta_p],\qquad
c_p(b)=c_p(\beta)+[c_p(\beta),K_4].
\]

Fitting minimizes the direct-CI annihilation residual. An independent oracle
evaluates the same quadratic objective with the phase-aligned 1-, 2-, and
3-RDM; no 4-RDM is used. The molecular basis conserves (M_S) and the Abelian
point-group irrep, reducing the N2 CAS(6,6) problem from 5,280 to 249 real
directions. It is explicitly not claimed to be an (S^2)-adapted generator.

The certificate reconstructs the normalized first-order state
((1+K_4)|0_\beta\rangle), compares its 1/2/3-RDMs to the CAS targets, checks
the annihilation objective, and records rank-resolved CAR and composition
metrics. When exact CAS-scale polynomial expansion is too costly, rigorous
finite coefficient-norm upper bounds are recorded and the certificate fails
closed. A nonzero full cubic transform is production-eligible only when the
`FirstBCHGeneratorCertificate`:

1. passes the declared 1/2/3-RDM, annihilation, CAR, composition, and
   number-variance tolerances;
2. declares that no particle-number projection was used; and
3. is SHA-256-bound to active `U,V,C,D,X,Y`.

The molecular consumer requires two distinct typed contracts: the active fit
certificate and a `MolecularEmbeddingCertificate`. The latter binds the active
certificate fingerprint, active and full transform hashes, molecular source
hash, active/external coupling, external canonicality, and combined full-space
CAR/composition metrics. Replacing an active digest with a full-space digest is
not certification. Missing, rejected, active-only, or stale certificates fail
before the Hamiltonian is built.

Rejected transforms can be run only through an explicit diagnostic override.
Their reports use `campaign_eligibility=uncertified-diagnostic-only`, and the
plotter uses open energy markers only for converged, uncertified solutions.
Nonconverged endpoints remain in the JSON audit data but appear only as
axis-anchored solver-status marks, with a distinct mark when the recorded
amplitude ceiling was engaged. This path never converts a failed generator fit
into an accepted molecular result.

The legacy Hamiltonian derivation is in
[`docs/full_cubic_hamiltonian.md`](docs/full_cubic_hamiltonian.md). The repaired
forward/inverse derivation, full-rank convention, exact active oracle, and
acceptance boundary are in
[`docs/repaired_cubic_appendix.md`](docs/repaired_cubic_appendix.md).

## Molecular point calculation

Install the isolated package from this directory:

```bash
python -m pip install -e '.[molecular,dev]'
```

Run a strict zero-cubic control from a portable molecular continuation file:

```bash
python scripts/run_molecular_cqpccsd4_point.py SOURCE.npz \
  --output artifacts/results/point.cqpccsd4.json \
  --amplitudes-output artifacts/results/point.cqpccsd4.amplitudes.npz
```

Run a certified nonzero cubic point by adding:

```bash
--cubic-reference artifacts/references/point.cubic-reference.npz
```

Build the independent 0.05 Å source continuation, run the six display points,
and merge them with the read-only archived 6-311G figure data with:

```bash
python scripts/build_n2_6311g_cqpccsd4_sources.py
python scripts/run_n2_6311g_full_cubic_campaign.py \
  --hamiltonian-workers 8 \
  --residual-backend excitation-coordinates \
  --damping 0.10 --level-shift 1.0
python scripts/plot_n2_6311g_cqpccsd4.py \
  ../artifacts/figures/n2_6311g_cas66_fci_rcc_qp_current.json \
  artifacts/results/n2_6311g_full_cubic/r*/result.json
```

The campaign and single-point molecular commands also expose the exact
analytic-JVP Newton--Krylov solver and all of its globalization controls.  For
example, replace the damped-DIIS controls above with:

```bash
--nonlinear-solver newton-krylov \
  --krylov-relative-tolerance 1e-3 --krylov-dimension 20 \
  --line-search-reduction 0.5 --line-search-minimum-step 1e-4 \
  --maximum-amplitude 5.0
```

These are numerical solver controls only; they do not change the unprojected
CQPCCSD(4) equations or enable a particle-number projection.

The complete rank4+rank6 FB1 equations currently have a separate diagnostic
constructor. For a rejected forward checkpoint it requires an explicit
initial-residual-only scope and never writes an energy:

```bash
python scripts/run_forward_full_fb1_diagnostic.py SOURCE.npz \
  --forward-checkpoint FORWARD_FIT.npz \
  --allow-rejected-initial-residual-timing \
  --output artifacts/results/full_fb1.initial_residual.json
```

There is intentionally no molecular solve option in this command. The
canonical rank-six schedule is correctness certified but not yet batched or
N=48 equation/JVP benchmarked; the rejected 440-profile schedule cannot be selected.
The real r1.10 construction-only smoke test took 65.9228 s for 10,581
comparison coordinates and found an initial residual norm of 0.06959739. The
reference remains rejected, so the resulting audit contains no energy and ran
no equation evaluation or solver iteration. See
[`docs/full_fb1_diagnostic.md`](docs/full_fb1_diagnostic.md).

The progressive 6-311G renderer ingests this construction audit together with
the restricted mixed-Q3 audit under strict report/source/checkpoint/transform
hash binding. The mixed-Q3 RMS appears only on the reference-quality panel and
the full-FB1 construction appears only as a T=0 status row; neither enters the
accepted-energy axis. Rebuild the PNG/PDF/SVG/JSON bundle with
`.venv-arm64/bin/python scripts/plot_n2_6311g_progressive.py`; see
[`docs/n2_6311g_progressive_comparison.md`](docs/n2_6311g_progressive_comparison.md).

Default output names include either `linear-control` or the leading full
transform hash, so control and cubic runs for one source do not overwrite each
other. Reports expose \(E_{\mathrm{QP}}(0)\),
\(E_{\mathrm{QP}}(T)\), their correlation difference, the
\(E_{\mathrm{QP}}(0)-E_{\mathrm{CASSCF}}\) gap, and the reported
\(E_{\mathrm{CASSCF}}+\Delta E_{\mathrm{QP}}\) total separately.

The command-line interface has no projection or gauge-angle option and always
rejects a projection-derived source or a source with unknown projection
provenance. `particle_number_projection=False` describes what the loader and
method do; it does not erase how imported `U,V` or RDMs were produced. No
\(P_N\), gauge-angle quadrature, PAV/OAP energy, or projected residual is
evaluated. A fixed-particle-number CAS RDM used as fitting data is not itself a
projection operation.

Programmatic callers can opt into reading an old or unknown-provenance source
only for a legacy regression. Such a result is labeled
`campaign_eligibility=legacy-provenance-audit-only`, still performs no
projection, and must not be pooled with the no-projection production curve.
Strict reports/checkpoints record source and transform SHA-256 values, source
`reference_mode`, provenance status, loader/method projection flags,
certificate binding, campaign eligibility, and the excitation-space hash.

The default molecular comparison coordinate space is named
`legacy-ms0-totally-symmetric`. It reproduces the determinant filter used for
the recorded linear N2 plot: the STO-3G CAS(6,6) point has 6 pair and 79
quadruple coordinates after pure-active coordinates are removed. The separate
`total-singlet-totally-symmetric` space constructs total-\(S^2\) singlet block
linear combinations and has 2 pair plus 36 quadruple coordinates at that
point; it is not silently substituted for the 6/79 plot comparison.

`full-spin-orbital` contains every pair and quadruple and is the general solver
default. `external-all-index` removes only wholly active coordinates. The
legacy space then applies determinant-level \(M_S=0\) and total-irrep filters;
it is not an \(S^2\) eigenbasis. The total-singlet mode instead uses normalized
\(M_S=0\) combinations annihilated by \(S_+\). The 6/79 and 2/36 counts apply
only to the stated STO-3G CAS(6,6) point. Active-only support for `C/D/X/Y` is a
separate choice from the CC excitation manifold.

As a regression audit only, the archived STO-3G zero-cubic amplitudes reproduce
the recorded linear value `-107.65860904153334 Eh` with residual
`1.2334109885e-9`. That old source is projection-derived and is therefore
rejected by the strict new campaign. The STO-3G and 6-311G campaign definitions
are in `configs/`. Rejected but finite 6-311G values are reported only as open
diagnostic energy markers when their nonlinear equations converged, never as a
certified nonzero cubic curve.

The displayed full-cubic campaign uses the exact coordinate residual backend.
Its damping and level shift are solver controls selected by an explicit
stability screen; they do not modify a converged root. If a point does not
converge, however, its finite iteration endpoint is solver-dependent and is
therefore retained only in the JSON audit fields, not on the energy axis. The
figure instead shows a geometry-aligned solver-status mark; amplitude-capped
trajectories are distinguished explicitly. Each point writes a source- and
transform-bound `hamiltonian.npz` before nonlinear iterations.

The exact six-point residuals, certificate errors, and bounded Newton--Krylov
checks are recorded in
[`docs/n2_6311g_campaign_results.md`](docs/n2_6311g_campaign_results.md).

The optional level shift is a positive additive shift to each quasiparticle
excitation denominator. It can stabilize iterations but does not repair
particle number, change the reference, or act as a projection.

## Independent SeQuant check

Configure, build, and run entirely inside `sequant/`:

```bash
cd sequant
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../../../qpccsd-sequant/cmake/toolchains/macos-homebrew-gcc15.cmake
cmake --build build --target generate_cubic_hamiltonian -j2
./build/generate_cubic_hamiltonian --output-dir generated/full_cubic
./build/generate_cubic_hamiltonian \
  --linear-only --output-dir generated/linear
```

The executable restricts output to relative, non-symlink paths below
`sequant/`. The original SeQuant build tree and its generated multi-gigabyte
artifacts are intentionally excluded from this compact experimental snapshot;
regenerate them in a separate isolated checkout before using this route.

## Hard isolation contract

Everything owned by this method stays below `cubic_qpccsd/`. In particular,
the implementation must not edit or generate into these parent linear paths:

- `../src/autogen/qpccsd/`
- `../method_inputs/qp_ccsd/`
- `../generated_code/methods/qp_ccsd/`

Runtime code must not import `autogen.qpccsd`,
`generated_code.methods.qp_ccsd`, or any particle-number-projection module.
Every project-owned output passes through `cubic_qpccsd.paths.guard_write_path`
or a guarded atomic-write helper. The guard rejects path escapes and symlink
components.

Capture and verify the protected parent trees with:

```bash
python scripts/audit_isolation.py capture
python scripts/audit_isolation.py verify
```

Run the isolated test suite with:

```bash
python -m pytest -o addopts='' -q
```

From the parent repository the equivalent explicit command is:

```bash
python -m pytest -c cubic_qpccsd/pyproject.toml cubic_qpccsd/tests \
  -o addopts='' -q
```
