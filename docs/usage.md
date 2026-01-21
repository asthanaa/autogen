# Usage

## Create environment

From repo root:

- `conda env create -f environment.yml` (first time)
- `conda env update -f environment.yml --prune` (update)
- `conda activate autogen`

## Typical operations

### Commutator

```python
from autogen.main_tools.commutator import comm

terms = comm(['V2'], ['T2'], 1)
```

### Filtering fully contracted terms

```python
from autogen.library.full_con import full_terms

contracted = full_terms(terms)
```

### Debug script

- `python debug.py`

This uses the implementation in `autogen.debug` and writes to `latex_output.txt` by default.

### Performance / compare modes

When reducing equivalent terms, the compare layer supports an opt-in mode switch:

- `AUTOGEN_COMPARE_MODE=fast` (default) uses faster comparison paths when safe.
- `AUTOGEN_COMPARE_MODE=full` forces the original compare logic.
- `AUTOGEN_COMPARE_MODE=check` runs both and warns on any mismatch.
- `AUTOGEN_QUIET=1` suppresses verbose term/contraction prints.
- `AUTOGEN_CACHE=0` disables contraction prefix caching (debug only).
- `AUTOGEN_MULTI_CONT_CACHE=0` disables multi-operator contraction caching.
- `AUTOGEN_MULTI_CONT_CACHE_SIZE=256` sets the multi-operator cache size (LRU).
- `AUTOGEN_SPIN_SUMMED=1` emits spin-summed residuals (recommended for RHF).
- `AUTOGEN_SPIN_SUMMED_MODE=spinorb` switches to the legacy spin-orbital wrapper path.
- `AUTOGEN_INTERMEDIATE_MIN=3` sets the minimum reuse count for CCSD intermediates.
- `AUTOGEN_INTERMEDIATE_MAX=80` caps the number of CCSD intermediates (0 = no cap).
- `AUTOGEN_MATCHING_CACHE=0` disables pattern-level contraction match caching in `make_c`.
- `AUTOGEN_MATCHING_CACHE_SIZE=128` sets the pattern cache size (LRU).
- `AUTOGEN_NUMBA=1` enables Numba-based contraction enumeration (optional).
- `AUTOGEN_NUMBA_CANDS_CACHE=0` disables caching of typed candidate lists for Numba.
- `AUTOGEN_NUMBA_CANDS_CACHE_SIZE=64` sets the typed-candidate cache size (LRU).

Example:

```bash
AUTOGEN_COMPARE_MODE=check python debug.py
```

Benchmark the compare-heavy workflows:

```bash
python scripts/bench_compare.py --repeat 3 --warmup 1
```

### Generated einsum scripts

The generated einsum examples (e.g., under `generated_code/`) rely on PySCF for
integrals. Install it before running those scripts:

```bash
pip install pyscf
```

Or with conda:

```bash
conda install -c conda-forge pyscf
```

Generate einsum scripts:

```bash
python scripts/gen_einsum.py V2 T1 T1
python scripts/gen_einsum.py V2 T2
python scripts/gen_einsum.py F1 T1
python scripts/gen_einsum.py CCSD_ENERGY
python scripts/gen_einsum.py CCSD_AMPLITUDE
python scripts/gen_einsum.py CCSD_AMPLITUDE --full --quiet
python scripts/gen_einsum.py CCSD_AMPLITUDE --intermediates --quiet
python scripts/gen_einsum.py --spec path/to/spec.py --full --quiet
python scripts/gen_einsum.py --spec path/to/spec.py --intermediates --quiet
python scripts/gen_einsum.py --spec method_inputs/ccsd/ccsd_spec.py --intermediates --quiet
python scripts/gen_einsum.py --spec method_inputs/eom_ccsd/ee_eom_ccsd_spec.py --intermediates --quiet
```

Notes:
- The CCSD energy script uses PySCF CCSD amplitudes and Fock matrix in the MO basis.
- The default molecule for generated scripts is H2O/6-31G in `generated_code/pyscf_integrals.py`.
- The iterative CCSD solver is generated under `generated_code/methods/ccsd/ccsd_amplitude/`.
- Use `--full` to emit explicit residual terms and `--intermediates` to emit reusable intermediates.
- Use `--spec` to drive custom term lists; the output defaults to `generated_code/methods/<spec-stem>/residuals.py`.
- Method input specs live under `method_inputs/<method>/`.
- For RHF spin-summed CCSD, generate with `AUTOGEN_SPIN_SUMMED=1` (optionally keep `SPIN_ADAPTED=True` in the spec).
- Spin-summed residuals are generated directly from the Wicks terms. Set `AUTOGEN_SPIN_SUMMED_MODE=spinorb` only if you need the spin-orbital wrapper mapping.
- EE-EOM-CCSD uses the same spin-summed RHF pathway and emits `eom_solver.py` plus `eom_pyscf_test.py` under `generated_code/methods/eom_ccsd/`.
- The EOM spec defaults to the spin-orbital singlet wrapper when `SPIN_ADAPTED=True`; override with `AUTOGEN_SPIN_SUMMED_MODE=direct` if you need the direct Wicks spin-sum. In spin-orbital mode, the generator emits `residuals_spinorb.py` plus a `residuals.py` wrapper that maps singlet amplitudes in/out.
- The EE-EOM-CCSD spec defaults to `EOM_BCH=True`, which builds H̅ via nested commutators (exact for CCSD, fewer terms than direct T-expansion).
