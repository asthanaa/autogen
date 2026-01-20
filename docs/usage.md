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
- `AUTOGEN_SPIN_SUMMED=1` enables spin-summed coefficients for spatial-orbital integrals (default).
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
```

Notes:
- The CCSD energy script uses PySCF CCSD amplitudes and Fock matrix in the MO basis.
- The default molecule for generated scripts is H2O/6-31G in `generated_code/pyscf_integrals.py`.
- The iterative CCSD solver is generated under `generated_code/ccsd_amplitude/`.
