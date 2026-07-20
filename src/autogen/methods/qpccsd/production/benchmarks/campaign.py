from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..excitation_space import BlockQPExcitationSpace, QPExcitationSpace
from ..models import QPAmplitudes
from ...generated.generated_projected_moments import (
    PROJECTOR_ORDERING,
    PROJECTION_EQUATION_SCHEMA,
)
from ..projection import PROJECTION_SOLVER_SCHEMA


CAMPAIGN_SCHEMA = "n2-molecular-bcc-pes-v6"
AMPLITUDE_CHECKPOINT_SCHEMA = "n2-molecular-bcc-amplitudes-v6"
SCIENTIFIC_TITLE = (
    "Molecular Bogoliubov Coupled-Cluster Theory with Particle-Number Projection"
)
REQUIRED_CHECKPOINT_KEYS = (
    "basis",
    "distance_angstrom",
    "cas_norb",
    "cas_nelec",
    "reference_mode",
    "method",
    "source_hash",
    "excitation_space_hash",
    "grid_size",
    "ode_substeps",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2) + "\n")
    temporary.replace(path)
    return path


def excitation_space_hash(space: QPExcitationSpace | BlockQPExcitationSpace) -> str:
    digest = sha256()
    digest.update(np.asarray([space.nspin], dtype=np.int64).tobytes())
    for keys, rank in (
        (space.active_spin_indices, 1),
        (space.pair_indices, 2),
        (space.quadruple_indices, 4),
    ):
        values = np.asarray(keys, dtype=np.int64)
        digest.update(np.asarray([len(keys), rank], dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    for attribute in ("pair_expansions", "quadruple_expansions"):
        expansions = getattr(space, attribute, ())
        digest.update(attribute.encode("ascii"))
        for expansion in expansions:
            for key, coefficient in expansion:
                digest.update(np.asarray(key, dtype=np.int64).tobytes())
                digest.update(
                    np.asarray(
                        [complex(coefficient).real, complex(coefficient).imag],
                        dtype=np.float64,
                    ).tobytes()
                )
            digest.update(b"\0")
    return digest.hexdigest()


def source_tree_hash(root: str | Path) -> str:
    root = Path(root).resolve()
    files: list[Path] = []
    for relative in ("src", "scripts", "tests"):
        directory = root / relative
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    files.extend(
        path for path in (root / "pyproject.toml", root / "pytest.ini") if path.is_file()
    )
    digest = sha256()
    for path in sorted(files):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_metadata(
    *,
    basis: str,
    distance_angstrom: float,
    cas_norb: int,
    cas_nelec: int,
    reference_mode: str,
    method: str,
    source_hash: str,
    excitation_hash: str,
    grid_size: int,
    ode_substeps: int,
) -> dict[str, Any]:
    return {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "checkpoint_schema": AMPLITUDE_CHECKPOINT_SCHEMA,
        "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
        "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
        "projector_ordering": PROJECTOR_ORDERING,
        "basis": str(basis).lower(),
        "distance_angstrom": float(distance_angstrom),
        "cas_norb": int(cas_norb),
        "cas_nelec": int(cas_nelec),
        "reference_mode": str(reference_mode),
        "method": str(method),
        "source_hash": str(source_hash),
        "excitation_space_hash": str(excitation_hash),
        "grid_size": int(grid_size),
        "ode_substeps": int(ode_substeps),
    }


def save_amplitude_checkpoint(
    path: str | Path,
    amplitudes: QPAmplitudes,
    metadata: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"amplitude checkpoint metadata is missing {missing}")
    merged = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "checkpoint_schema": AMPLITUDE_CHECKPOINT_SCHEMA,
        "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
        "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
        "projector_ordering": PROJECTOR_ORDERING,
        **dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            t1=np.asarray(amplitudes.t1),
            t2=np.asarray(amplitudes.t2),
            metadata_json=np.asarray(json.dumps(_json_value(merged))),
            state_json=np.asarray(json.dumps(_json_value(state))),
        )
    temporary.replace(path)
    return path


def load_amplitude_checkpoint(
    path: str | Path,
    expected: Mapping[str, Any],
) -> tuple[QPAmplitudes, dict[str, Any], dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"].item()))
        state = json.loads(str(checkpoint["state_json"].item()))
        required = {
            "campaign_schema": CAMPAIGN_SCHEMA,
            "checkpoint_schema": AMPLITUDE_CHECKPOINT_SCHEMA,
            "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
            "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
            "projector_ordering": PROJECTOR_ORDERING,
            **dict(expected),
        }
        mismatches: dict[str, tuple[Any, Any]] = {}
        for key, required_value in required.items():
            actual = metadata.get(key)
            if key == "distance_angstrom":
                matches = actual is not None and abs(float(actual) - float(required_value)) < 1.0e-12
            else:
                matches = actual == required_value
            if not matches:
                mismatches[key] = (actual, required_value)
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (required {required!r})"
                for key, (actual, required) in mismatches.items()
            )
            raise ValueError(f"incompatible amplitude checkpoint: {details}")
        amplitudes = QPAmplitudes(
            t1=np.asarray(checkpoint["t1"]),
            t2=np.asarray(checkpoint["t2"]),
        )
    return amplitudes, metadata, state


class AtomicAmplitudeCheckpointer:
    def __init__(
        self,
        path: str | Path,
        metadata: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata)

    def __call__(self, amplitudes: QPAmplitudes, state: dict[str, object]) -> None:
        save_amplitude_checkpoint(self.path, amplitudes, self.metadata, state)


__all__ = [
    "AMPLITUDE_CHECKPOINT_SCHEMA",
    "AtomicAmplitudeCheckpointer",
    "CAMPAIGN_SCHEMA",
    "SCIENTIFIC_TITLE",
    "checkpoint_metadata",
    "excitation_space_hash",
    "load_amplitude_checkpoint",
    "save_amplitude_checkpoint",
    "source_tree_hash",
    "write_json_atomic",
]
