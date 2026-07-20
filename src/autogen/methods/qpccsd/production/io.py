"""Pickle-free, atomic I/O for production results and amplitudes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .models import QPAmplitudes


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON document must contain an object")
    return payload


def load_amplitudes(path: str | Path, *, nspin: int) -> QPAmplitudes:
    """Load any pickle-free checkpoint containing compatible t1/t2 arrays."""

    with np.load(Path(path), allow_pickle=False) as checkpoint:
        missing = {"t1", "t2"}.difference(checkpoint.files)
        if missing:
            raise ValueError(f"amplitude checkpoint is missing {sorted(missing)}")
        amplitudes = QPAmplitudes(
            np.asarray(checkpoint["t1"]).copy(),
            np.asarray(checkpoint["t2"]).copy(),
        )
    if amplitudes.t1.shape != (nspin, nspin):
        raise ValueError("amplitude checkpoint has the wrong pair dimension")
    if amplitudes.t2.shape != (nspin,) * 4:
        raise ValueError("amplitude checkpoint has the wrong quadruple dimension")
    return amplitudes


def save_amplitudes(
    path: str | Path,
    amplitudes: QPAmplitudes,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("projected-agp-fullspace-qpccsd-amplitudes-v1"),
            t1=np.asarray(amplitudes.t1),
            t2=np.asarray(amplitudes.t2),
            metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
        )
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


__all__ = [
    "file_sha256",
    "load_amplitudes",
    "read_json",
    "save_amplitudes",
    "write_json_atomic",
]
