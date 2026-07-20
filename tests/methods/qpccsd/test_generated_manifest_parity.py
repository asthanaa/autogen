from __future__ import annotations

import json
from pathlib import Path

import pytest

import autogen.methods.qpccsd as qpccsd
from autogen.methods.qpccsd.derivation.emitters.qpccsd_projected_moments import (
    build_projected_moment_artifact,
)


@pytest.mark.parametrize(
    "name",
    [
        "projected_moment_canonical_terms.json",
        "qpccsd_canonical_terms.json",
    ],
)
def test_runtime_manifest_snapshot_matches_derivation_source(name: str) -> None:
    package_root = Path(qpccsd.__file__).resolve().parent
    runtime_snapshot = package_root / "generated" / name
    derivation_source = package_root / "derivation" / "equations" / name

    assert runtime_snapshot.read_bytes() == derivation_source.read_bytes()


def test_projected_moment_emitter_reproduces_the_canonical_manifest() -> None:
    package_root = Path(qpccsd.__file__).resolve().parent
    canonical_path = (
        package_root
        / "derivation"
        / "equations"
        / "projected_moment_canonical_terms.json"
    )

    assert build_projected_moment_artifact() == json.loads(
        canonical_path.read_text(encoding="utf-8")
    )
