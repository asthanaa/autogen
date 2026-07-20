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

### Regenerating method kernels

The checked-in kernels under `autogen.methods.<method>.generated` are the reviewed
runtime artifacts. Regeneration always targets a temporary candidate directory; it does
not overwrite the installed implementation.

```python
from autogen.methods.ccsd.derivation.emitters.regenerate import regenerate as regenerate_ccsd
from autogen.methods.eom_ccsd.derivation.emitters.regenerate import regenerate as regenerate_eom

regenerate_ccsd("tmp/ccsd-candidate")
regenerate_eom("tmp/eom-ccsd-candidate")
```

Compare a candidate with the canonical generated package and run the method parity tests
before deliberately synchronizing it. Generator options, spin conventions, and source
specifications are kept in each method's `derivation` layer so output-directory names do
not select scientific behavior.

PySCF is needed only for molecular adapters and numerical molecular regressions:

```bash
python -m pip install -e ".[molecular,test]"
```

Run those calculations only on an approved remote host. Synthetic generation/parity
tests remain part of the normal local suite.
