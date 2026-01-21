# Work log

This file summarizes the recent performance and correctness work on Autogen.

## Changes implemented

- Compare path: added fast structural keys, two-stage reduction, lazy matrix build,
  and a lightweight `_CoeffTerm` to avoid deep copies in `src/autogen/library/compare.py`
  and `src/autogen/library/compare_utils.py`.
- Compare level-5: cache coefficient index graphs on juggled terms to avoid
  rebuilding `ind` objects; add a guarded permutation check for short operator
  names and an optional `AUTOGEN_COMPARE_LEVEL5=matrix` mode.
- Generated code layout: method outputs now live under
  `generated_code/methods/<method>`, with input specs in
  `method_inputs/<method>` and molecule fixtures in `tests/molecules`.
- EE-EOM-CCSD: added R1/R2 operator support, a spin-summed EOM spec, a
  Davidson solver emitter, and PySCF-based regression tests.
- EE-EOM-CCSD generation: switched to exact BCH nested-commutator expansion
  (H̅) plus hash-first term reduction to cut contraction and compare cost.
- EE-EOM-CCSD spin adaptation: default to the spin-orbital singlet wrapper for
  `SPIN_ADAPTED` specs and map R1/R2 via PySCF singlet transforms to match
  `eom_rccsd.EOMEESinglet`.
- EE-EOM-CCSD direct mode: treat `AUTOGEN_SPIN_SUMMED_MODE=direct` as an alias
  to the singlet wrapper to keep results consistent with PySCF.
- EOM index handling: tokenized indexed labels (e.g., `i1`, `a1`) during
  canonicalization and intermediate selection, and made `view_tensor`/output
  labeling robust to numeric suffixes.
- Special conditions: avoid exiting on non-indexed operator names and default
  to position 1 instead.
- Contraction engine: rewrote `make_c` pairing enumeration to avoid deep copies,
  added pattern-level matching caches, and added a pure-Python DFS fallback that
  can be cached and reused across index renamings.
- Multi-operator contraction: added LRU caching in `src/autogen/main_tools/multi_cont.py`.
- Parity: cached index maps in `src/autogen/pkg/parity.py`.
- Term objects: added `copy_inputs=False` for `class_term.term` and used it in
  `src/autogen/library/change_terms.py`.
- Spin-summed option: gated spin-loop factors and cross-spin 1/2 factors behind
  `AUTOGEN_SPIN_SUMMED` in `src/autogen/pkg/fix_uv.py` (default on).
- Spin-summed CCSD residuals: generate directly from Wicks terms when
  `AUTOGEN_SPIN_SUMMED=1`. The spin-orbital wrapper path is kept as an optional
  fallback via `AUTOGEN_SPIN_SUMMED_MODE=spinorb`.
- CCSD solver energy: updated the generated solver to use the PySCF RHF
  CCSD energy formula with raw integrals (`g_raw`) for numerical agreement.
- Numba support: optional contraction enumeration via `AUTOGEN_NUMBA=1`.
- Fixed warnings and correctness: indentation in `src/autogen/library/class_term.py`,
  raw-string fixes in `src/autogen/pkg/fix_uv.py` and `src/autogen/pkg/fewt.py`,
  and `pytest`/`debug.py` path fixes for local imports.

## New or updated scripts

- `scripts/time_ccsd_terms.py`: time each CCSD amplitude term and print slowest terms.
- `scripts/time_ccsd_full.py`: run full CCSD amplitude and print elapsed time.
- `scripts/profile_term_breakdown.py`: measure `make_op`, `multi_cont`, `full_con`,
  `change_terms`, `compare+reduce`.
- `scripts/profile_compare_hotspots.py`: report per-function compare timings.
- `tests/test_ccsd_pyscf.py`: slow PySCF-backed verification of energy and residuals.
- `tests/test_eom_ccsd_pyscf.py`: slow PySCF-backed EE-EOM-CCSD checks.

## Environment variables

- `AUTOGEN_COMPARE_MODE=fast|full|check`
- `AUTOGEN_COMPARE_LEVEL5=cached|matrix`
- `AUTOGEN_QUIET=1`
- `AUTOGEN_CACHE=0`
- `AUTOGEN_MULTI_CONT_CACHE=0`, `AUTOGEN_MULTI_CONT_CACHE_SIZE=...`
- `AUTOGEN_MATCHING_CACHE=0`, `AUTOGEN_MATCHING_CACHE_SIZE=...`
- `AUTOGEN_NUMBA=1`
- `AUTOGEN_NUMBA_CANDS_CACHE=0`, `AUTOGEN_NUMBA_CANDS_CACHE_SIZE=...`
- `AUTOGEN_SPIN_SUMMED=1` (default on)
- `AUTOGEN_SPIN_SUMMED_MODE=spinorb` (legacy wrapper path)
- `SPIN_ADAPTED=True` in a spec for RHF spin-summed CCSD residual generation

For details, see `docs/usage.md` and `docs/performance.md`.

## Performance snapshots

Benchmarked on the slowest CCSD term
`['X2','V2','T2','T21']` using a custom breakdown:

- Before cached level-5: `multi_cont` ~4.30s, `compare+reduce` ~21.16s.
- After cached level-5: `multi_cont` ~4.68s, `compare+reduce` ~11.35s.

Per-term CCSD timing (`scripts/time_ccsd_terms.py`):
- Total 33.883s → 27.295s.
- Slowest term 18.651s → 11.805s.

Full CCSD amplitude timing (`scripts/time_ccsd_full.py`):
- 34.199s → 26.582s.

## Intermediate residuals

- Added `--intermediates` mode in `scripts/gen_einsum.py` to emit reusable pair
  intermediates and grouped `einsum` calls with `optimize=True`.
- Generated CCSD residuals now expose `AUTOGEN_INTERMEDIATES` and
  `compute_r1_r2` for shared intermediate reuse across R1/R2.
- Added `--spec` support to generate residual code from a user-provided Python
  spec (custom term lists with optional intermediate emission).

## Tests

- `pytest -q` (1 skipped)
- `RUN_SLOW=1 pytest -q -k pyscf` for PySCF numeric checks
- `RUN_SLOW=1 pytest -q -k eom_ccsd` for EE-EOM-CCSD checks

## Numba install note (macOS)

Pip builds can fail due to missing LLVM. Prefer conda-forge:

```bash
conda activate autogen
python -m pip uninstall -y numba llvmlite
conda install -c conda-forge numba llvmlite
```
