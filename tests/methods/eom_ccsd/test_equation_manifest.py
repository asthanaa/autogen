from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from autogen.methods.eom_ccsd import generated


METHOD_ROOT = Path(generated.__file__).resolve().parent.parent
MANIFEST_PATH = METHOD_ROOT / "derivation" / "equations" / "equation_manifest.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_equation_manifest_tracks_every_frozen_artifact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["schema"] == "autogen-equation-manifest-v1"
    assert manifest["method"] == "ee-eom-ccsd"
    assert manifest["status"] == "frozen-validated-baseline"
    assert manifest["regeneration_policy"] == (
        "replace_only_after_equation_numeric_pyscf_parity"
    )

    tracked = [manifest["source_spec"], *manifest["generated_artifacts"]]
    tracked.append(manifest["emitter"])
    for artifact in tracked:
        path = METHOD_ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert _digest(path) == artifact["sha256"], artifact["path"]

    installed = {
        path.relative_to(METHOD_ROOT).as_posix()
        for path in (METHOD_ROOT / "generated").rglob("*.py")
        if path.name != "__init__.py"
    }
    assert installed == {item["path"] for item in manifest["generated_artifacts"]}
