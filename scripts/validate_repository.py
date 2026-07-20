#!/usr/bin/env python3
"""Validate the canonical repository layout and packaging contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
LARGE_PLAN = "generated_code/methods/qp_ccsd/projected_codegen_plan.json"
EXPECTED_SIZE = 387064269
EXPECTED_SHA256 = "636b05aad1003e8c5d985712eee49bb13e6718ef6cffbdf0595d24bd978f59a0"


def _require(path: str, failures: list[str]) -> None:
    if not (ROOT / path).exists():
        failures.append(f"missing required path: {path}")


def main() -> int:
    failures: list[str] = []
    required_paths = [
        "src/autogen/methods/__init__.py",
        "src/autogen/methods/ccsd/derivation",
        "src/autogen/methods/ccsd/generated",
        "src/autogen/methods/ccsd/runtime",
        "src/autogen/methods/eom_ccsd/derivation",
        "src/autogen/methods/eom_ccsd/generated",
        "src/autogen/methods/eom_ccsd/runtime",
        "src/autogen/methods/qpccsd/cli.py",
        "src/autogen/methods/qpccsd/derivation",
        "src/autogen/methods/qpccsd/generated",
        "src/autogen/methods/qpccsd/production",
        "tests/methods/ccsd",
        "tests/methods/eom_ccsd",
        "tests/methods/qpccsd",
        "docs/methods/qpccsd",
        "configs/qpccsd",
    ]
    for path in required_paths:
        _require(path, failures)

    forbidden = [
        "qpccsd_agp2rdm_pav_clean",
        "cubic_qpccsd",
        LARGE_PLAN,
    ]
    for path in forbidden:
        if (ROOT / path).exists():
            failures.append(f"forbidden repository path is present: {path}")

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    entry_point = project.get("project", {}).get("scripts", {}).get("qpccsd")
    expected_entry_point = "autogen.methods.qpccsd.cli:main"
    if entry_point != expected_entry_point:
        failures.append(f"qpccsd entry point must be {expected_entry_point!r}")
    dependencies = project.get("project", {}).get("dependencies", [])
    for prefix in ("numpy", "scipy", "threadpoolctl"):
        if not any(str(item).startswith(prefix) for item in dependencies):
            failures.append(f"missing runtime dependency: {prefix}")
    package_data = project.get("tool", {}).get("setuptools", {}).get("package-data", {})
    if "*.json" not in package_data.get("*", []):
        failures.append("JSON method manifests are not declared as package data")

    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for guard in (
        "/generated_code/methods/qp_ccsd/projected_codegen_plan.json",
        "**/projected_codegen_plan.json",
        "/checkpoints/",
        "/logs/",
        "/results/",
        "/raw_plots/",
    ):
        if guard not in ignore_text:
            failures.append(f"missing .gitignore guard: {guard}")

    provenance_path = ROOT / "provenance/excluded_large_objects.json"
    if not provenance_path.is_file():
        failures.append("missing excluded-large-object provenance manifest")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        objects = provenance.get("objects", [])
        match = next((item for item in objects if item.get("former_path") == LARGE_PLAN), None)
        if match is None:
            failures.append("large-plan provenance record is missing")
        else:
            if match.get("size_bytes") != EXPECTED_SIZE:
                failures.append("large-plan provenance size is incorrect")
            if match.get("sha256") != EXPECTED_SHA256:
                failures.append("large-plan provenance SHA-256 is incorrect")
            if not match.get("regeneration", {}).get("command"):
                failures.append("large-plan regeneration command is missing")

    if failures:
        print("repository validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("repository layout and packaging contract validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
