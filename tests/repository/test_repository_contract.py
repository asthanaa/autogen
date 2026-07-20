from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_repository_layout_and_packaging_contract() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_repository.py")],
        check=True,
        cwd=ROOT,
    )


def test_repository_has_no_large_tracked_or_candidate_files() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_repository_blobs.py"),
            "--root",
            str(ROOT),
            "--limit-mib",
            "50",
        ],
        check=True,
        cwd=ROOT,
    )


def test_excluded_plan_provenance_is_machine_readable() -> None:
    path = ROOT / "provenance/excluded_large_objects.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "autogen-excluded-large-object-provenance-v1"
    record = payload["objects"][0]
    assert record["size_bytes"] == 387064269
    assert record["sha256"] == (
        "636b05aad1003e8c5d985712eee49bb13e6718ef6cffbdf0595d24bd978f59a0"
    )
    assert record["source"]["commit"] == "0a851d3469a8476aa36d01b3437021c9c2815ace"
    assert record["regeneration"]["command"].startswith(
        "AUTOGEN_QP_Z_CONTRACTION=1 PYTHONPATH=src python -m autogen.codegen.cli"
    )


def test_citation_describes_the_consolidated_release() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'title: "AutoGen-Wick: generated coupled-cluster methods"' in citation
    assert "version: 0.2.0" in citation
    assert "date-released: 2026-07-20" in citation
    assert 'repository-code: "https://github.com/asthanaa/autogen"' in citation
