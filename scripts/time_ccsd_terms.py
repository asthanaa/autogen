import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autogen.main_tools import driv3 as driv2


def main():
    from autogen.pkg import numba_contractions

    orig = driv2.driver
    timings = []

    def timed_driver(fc, ops, quiet=None):
        start = time.perf_counter()
        out = orig(fc, ops, quiet=quiet)
        elapsed = time.perf_counter() - start
        timings.append((elapsed, ops, fc))
        print(f"{ops} (fc={fc}): {elapsed:.3f}s")
        return out

    driv2.driver = timed_driver

    print("AUTOGEN_NUMBA=", os.getenv("AUTOGEN_NUMBA", ""))
    print("NUMBA_AVAILABLE=", numba_contractions.HAS_NUMBA)

    from tests import ccsd_amplitude as ccsd

    start = time.perf_counter()
    ccsd.amplitude()
    total = time.perf_counter() - start

    print("\n=== Slowest terms ===")
    for elapsed, ops, fc in sorted(timings, reverse=True)[:10]:
        print(f"{elapsed:.3f}s  ops={ops} fc={fc}")
    print(f"\nTotal: {total:.3f}s")


if __name__ == "__main__":
    main()
