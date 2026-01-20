"""Benchmark compare-heavy equation generation paths.

Usage:
  python scripts/bench_compare.py --repeat 3 --warmup 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autogen.library import compare_utils
from autogen.library.full_con import full_terms
from autogen.main_tools.commutator import comm
from autogen.main_tools.product import prod


def _run_case(name, func, repeat, warmup):
    for _ in range(warmup):
        func()

    timings = []
    compare_utils.reset_compare_stats()
    for _ in range(repeat):
        compare_utils.reset_compare_stats()
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        timings.append(t1 - t0)
        _ = result

    stats = compare_utils.get_compare_stats()
    avg = sum(timings) / len(timings)
    print(
        f"{name}: avg={avg:.4f}s min={min(timings):.4f}s max={max(timings):.4f}s"
    )
    if stats is not None:
        print(
            "  compare_calls={calls} merges={merges} buckets={buckets} "
            "bucket_pairs={pairs} max_bucket={max_bucket} reduce_calls={rc} "
            "compare_time={elapsed:.4f}s".format(
                calls=stats.compare_calls,
                merges=stats.merges,
                buckets=stats.buckets,
                pairs=stats.bucket_pairs,
                max_bucket=stats.max_bucket,
                rc=stats.reduce_calls,
                elapsed=stats.elapsed,
            )
        )


def case_v2_t2():
    return comm(["V2"], ["T2"], last=1)


def case_nested_t():
    inner = comm(["V2"], ["T1"], last=0)
    return comm(inner, ["T11"], last=1)


def case_v2_d1():
    return comm(["V2"], ["D1"], last=1)


def case_x1_v2_d1():
    vd1 = comm(["V2"], ["D1"], last=0)
    return prod(["X1"], vd1, last=1)


def case_full_contracted():
    terms = comm(["V2"], ["T2"], last=1)
    return full_terms(terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()

    compare_utils.enable_compare_stats()

    _run_case("[V2, T2]", case_v2_t2, args.repeat, args.warmup)
    _run_case("[[V2, T1], T11]", case_nested_t, args.repeat, args.warmup)
    _run_case("[V2, D1]", case_v2_d1, args.repeat, args.warmup)
    _run_case("X1 * [V2, D1]", case_x1_v2_d1, args.repeat, args.warmup)
    _run_case("[V2, T2] fully contracted", case_full_contracted, args.repeat, args.warmup)


if __name__ == "__main__":
    main()
