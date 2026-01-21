# Performance notes

This page summarizes the performance-related changes and how to control them.

## Compare reduction

The term reduction logic has a fast path that groups terms by a canonical hash
and only falls back to full structural comparison when needed.

- `AUTOGEN_COMPARE_MODE=fast` (default) uses the fast compare path.
- `AUTOGEN_COMPARE_MODE=full` forces the original compare path.
- `AUTOGEN_COMPARE_MODE=check` runs both and warns on mismatches.

The reducer now uses a two-stage bucketization:
1) coarse key (term signature + index incidence + structural key), then
2) matrix signature inside buckets with >1 term.
This shrinks compare-heavy buckets before invoking full comparisons.

## Contraction caching

Two caching layers reduce repeated contraction work:

- Prefix caching in `driv3` (useful for commutator/product flows).
  - Disable with `AUTOGEN_CACHE=0`.
- LRU caching for multi-operator contractions in `multi_cont`.
  - Disable with `AUTOGEN_MULTI_CONT_CACHE=0`.
  - Resize with `AUTOGEN_MULTI_CONT_CACHE_SIZE=256` (default 256).

`make_c` also memoizes contraction matchings by operator pattern so repeated
calls across different index names can reuse the same matching lists.

- Disable with `AUTOGEN_MATCHING_CACHE=0`.
- Resize with `AUTOGEN_MATCHING_CACHE_SIZE=128` (default 128).

Cached objects share underlying operator structures. Treat contraction objects
as immutable. If you need to mutate them, deep copy first.

## Copy avoidance in term objects

`class_term.term` supports `copy_inputs=False` to avoid deep copies of lists
that are already unique to the term. When used, callers must treat inputs as
immutable for the lifetime of the term.

## Contraction emission

The contraction formatter operates on lists and slices rather than deques to
reduce per-term overhead when building and emitting contracted terms.

## Numba contraction enumeration

`make_c` can use a Numba-accelerated matcher to enumerate contraction pairs.

- Enable with `AUTOGEN_NUMBA=1`.
- Requires `numba` to be installed.
- If Numba is unavailable or fails at runtime, it falls back to the original
  Python recursion.
- `AUTOGEN_NUMBA_CANDS_CACHE=0` disables caching the typed candidate list.
- `AUTOGEN_NUMBA_CANDS_CACHE_SIZE=64` sets the typed candidate cache size.

Note: the order of emitted terms can differ when Numba is enabled, but the
resulting set of terms and constants should be equivalent.

## CCSD intermediates

`scripts/gen_einsum.py CCSD_AMPLITUDE --intermediates` emits reusable pair
intermediates and groups repeated `einsum` calls.

- `AUTOGEN_INTERMEDIATE_MIN=3` sets the minimum reuse count to materialize.
- `AUTOGEN_INTERMEDIATE_MAX=80` caps the number of intermediates (0 = no cap).

## Parity helpers

The parity computation caches the index map for `full_pos` to avoid rebuilding
the same dictionary for each term. This is transparent to callers.

## Benchmarking and checks

Compare-heavy benchmarks:

```bash
python scripts/bench_compare.py --repeat 3 --warmup 1
```

Quick correctness check with both compare paths:

```bash
AUTOGEN_COMPARE_MODE=check python debug.py
```

When debugging mismatches, disable caches and Numba:

```bash
AUTOGEN_COMPARE_MODE=full AUTOGEN_CACHE=0 AUTOGEN_MULTI_CONT_CACHE=0 AUTOGEN_NUMBA=0 python debug.py
```
