import os
import sys
import time
from pathlib import Path


def _ensure_paths():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    for path in (str(src), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def main():
    _ensure_paths()
    from autogen.pkg import numba_contractions
    from tests import ccsd_amplitude as ccsd

    print("AUTOGEN_NUMBA=", os.getenv("AUTOGEN_NUMBA", ""))
    print("NUMBA_AVAILABLE=", numba_contractions.HAS_NUMBA)
    start = time.perf_counter()
    ccsd.amplitude()
    elapsed = time.perf_counter() - start
    print(f"CCSD amplitude elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
