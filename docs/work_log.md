# Work log

This file summarizes the recent performance and correctness work on Autogen.

## Changes implemented

- Compare path: added fast structural keys, two-stage reduction, lazy matrix build,
  and a lightweight `_CoeffTerm` to avoid deep copies in `src/autogen/library/compare.py`
  and `src/autogen/library/compare_utils.py`.
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

## Environment variables

- `AUTOGEN_COMPARE_MODE=fast|full|check`
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

Benchmarked on the heaviest CCSD amplitude term
`['X2','V2','T1','T11','T12']` using `scripts/profile_term_breakdown.py`:

- Before compare optimization: `compare+reduce` ~18.8s, `multi_cont` ~3.3s.
- After compare optimization: `compare+reduce` ~0.20s, `multi_cont` ~3.3s.
- After `make_c`/`fix_con` rewrite and matching caches: `multi_cont` ~0.78s,
  `compare+reduce` ~0.18s (total ~0.97s).

Compare hotspot timing (`scripts/profile_compare_hotspots.py`) on the same term:
- `compare` ~20s → ~0.21s, dominated by `level4`/`level5` before the rewrite.

Full CCSD amplitude timing (`scripts/time_ccsd_full.py`) with Numba available:
- ~39.3s (Numba-enabled run in this repo). If Numba is not installed, the run
  falls back to the Python path and can be significantly slower.

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

## Numba install note (macOS)

Pip builds can fail due to missing LLVM. Prefer conda-forge:

```bash
conda activate autogen
python -m pip uninstall -y numba llvmlite
conda install -c conda-forge numba llvmlite
```
