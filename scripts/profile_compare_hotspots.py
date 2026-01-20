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


timings = {}
counts = {}


def wrap_fn(module, name, label=None):
    if not hasattr(module, name):
        return
    orig = getattr(module, name)
    if getattr(orig, "_wrapped", False):
        return
    key = label or f"{module.__name__}.{name}"

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = orig(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timings[key] = timings.get(key, 0.0) + elapsed
        counts[key] = counts.get(key, 0) + 1
        return out

    wrapper._wrapped = True
    setattr(module, name, wrapper)


def install_wrappers():
    import autogen.library.compare as cmp_full
    import autogen.library.compare_test as cmp_test
    import autogen.library.compare_functions as cf
    import autogen.library.compare_functions2 as cf2

    # compare_test fast path
    wrap_fn(cmp_test, "compare")
    wrap_fn(cmp_test, "level1")
    wrap_fn(cmp_test, "create_matrices")

    # full compare path
    wrap_fn(cmp_full, "compare")
    wrap_fn(cmp_full, "level1")
    wrap_fn(cmp_full, "level2")
    wrap_fn(cmp_full, "level3")
    wrap_fn(cmp_full, "level4")
    wrap_fn(cmp_full, "level5")
    wrap_fn(cmp_full, "arrowwork")
    wrap_fn(cmp_full, "go_forward")
    wrap_fn(cmp_full, "go_find")
    wrap_fn(cmp_full, "match")
    wrap_fn(cmp_full, "pick")

    # compare helper functions
    wrap_fn(cf, "permutation_check")
    wrap_fn(cf, "permute_matrix_check")
    wrap_fn(cf, "level2")
    wrap_fn(cf2, "positionchange")
    wrap_fn(cf2, "permute_matrix_check")


def main():
    ops = ["X2", "V2", "T1", "T11", "T12"]
    fc = 1.0 / 6.0
    install_wrappers()
    driv2.driver(fc, ops, quiet=True)

    print("\n=== Compare timing breakdown ===")
    for key, elapsed in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{elapsed:.3f}s  calls={counts.get(key, 0)}  {key}")


if __name__ == "__main__":
    main()
