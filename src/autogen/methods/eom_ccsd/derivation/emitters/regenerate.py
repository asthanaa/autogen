"""Regenerate EE-EOM-CCSD candidates through the shared Autogen emitter."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

from autogen.codegen import cli

SPEC_PATH = Path(__file__).parents[1] / "specs" / "ee_eom_ccsd_spec.py"


@contextmanager
def _argv(arguments: list[str]):
    previous = sys.argv
    sys.argv = ["autogen-eom-ccsd-regenerate", *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def regenerate(output_dir: str | Path, *, quiet: bool = True) -> Path:
    """Emit an EE-EOM-CCSD candidate tree for parity validation."""

    destination = Path(output_dir).resolve()
    arguments = [
        "--spec",
        str(SPEC_PATH),
        "--out",
        str(destination),
        "--full",
        "--tasks",
        "eom_solver",
    ]
    if quiet:
        arguments.append("--quiet")
    with _argv(arguments):
        cli.main()
    return destination
