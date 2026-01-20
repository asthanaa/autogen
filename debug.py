"""Debug harness.

This file is intended to be a *raw, editable* place to run the core APIs
while you debug the algebra/contraction logic.

It writes human-readable LaTeX-ish output to `latex_output.txt` in the
current working directory.

Notes:
- This is not a unit test. It is a manual debugging script.
- Keep it simple and explicit: the goal is to have the exact lines you want
  to run right in front of you.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autogen.library.full_con import full_terms
from autogen.library.print_terms import print_terms
from autogen.main_tools.commutator import comm
from autogen.main_tools.product import prod


def _reset_output(output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")


def _section(output_file: str, title: str) -> None:
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\\n% ==== {title} ====\\n")


def debug_run(output_file: str = "latex_output.txt") -> None:
    _reset_output(output_file)

    # ---------------------------------------------------------------------
    # Case 1: Basic commutator [V2, T2]
    # ---------------------------------------------------------------------
    _section(output_file, "[V2, T2] (fully contracted)")
    terms = comm(["V2"], ["T2"], last=1)
    assert isinstance(terms, list) and len(terms) > 0
    terms_fc = full_terms(terms)
    print_terms(terms_fc, output_file)

    # ---------------------------------------------------------------------
    # Case 2: Nested commutator [[V2, T1], T11]
    # NOTE: Use distinct labels like T11/T12 to represent multiple same-type
    # operators in some workflows.
    # ---------------------------------------------------------------------
    _section(output_file, "[[V2, T1], T11] (fully contracted)")
    inner = comm(["V2"], ["T1"], last=0)
    outer = comm(inner, ["T11"], last=1)
    assert isinstance(outer, list) and len(outer) > 0
    outer_fc = full_terms(outer)
    print_terms(outer_fc, output_file)

    # ---------------------------------------------------------------------
    # Case 3: De-excitation operators (D1/D2)
    # ---------------------------------------------------------------------
    _section(output_file, "[V2, D1] (raw)")
    vd1 = comm(["V2"], ["D1"], last=1)
    assert isinstance(vd1, list) and len(vd1) > 0
    print_terms(vd1, output_file)

    _section(output_file, "[V2, D2] (raw)")
    vd2 = comm(["V2"], ["D2"], last=1)
    assert isinstance(vd2, list) and len(vd2) > 0
    print_terms(vd2, output_file)

    _section(output_file, "[[V2, D1], T1] (raw)")
    inner = comm(["V2"], ["D1"], last=0)
    expr = comm(inner, ["T1"], last=1)
    assert isinstance(expr, list) and len(expr) > 0
    print_terms(expr, output_file)

    # ---------------------------------------------------------------------
    # Case 4: Compose product with commutator: X1 * [V2, D1]
    # ---------------------------------------------------------------------
    _section(output_file, "X1 * [V2, D1] (raw)")
    vd1 = comm(["V2"], ["D1"], last=0)
    x1_vd1 = prod(["X1"], vd1, last=1)
    assert isinstance(x1_vd1, list) and len(x1_vd1) > 0
    print_terms(x1_vd1, output_file)


if __name__ == "__main__":
    debug_run()
