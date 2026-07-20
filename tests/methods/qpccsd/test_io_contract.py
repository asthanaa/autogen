from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autogen.methods.qpccsd.production.io import (
    file_sha256,
    load_amplitudes,
    read_json,
    save_amplitudes,
    write_json_atomic,
)
from autogen.methods.qpccsd.production.models import QPAmplitudes


def test_amplitude_checkpoint_round_trips_without_pickle(tmp_path: Path) -> None:
    nspin = 4
    amplitudes = QPAmplitudes(
        t1=np.arange(nspin**2).reshape(nspin, nspin) * (1.0 + 0.25j),
        t2=np.arange(nspin**4).reshape((nspin,) * 4) * (0.001 - 0.002j),
    )
    path = tmp_path / "amplitudes.npz"

    save_amplitudes(path, amplitudes, metadata={"energy_convention": "direct"})
    restored = load_amplitudes(path, nspin=nspin)

    np.testing.assert_array_equal(restored.t1, amplitudes.t1)
    np.testing.assert_array_equal(restored.t2, amplitudes.t2)
    with np.load(path, allow_pickle=False) as checkpoint:
        assert checkpoint["schema"].item() == (
            "projected-agp-fullspace-qpccsd-amplitudes-v1"
        )
        assert json.loads(checkpoint["metadata_json"].item()) == {
            "energy_convention": "direct"
        }
    assert len(file_sha256(path)) == 64


def test_amplitude_checkpoint_dimension_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "amplitudes.npz"
    save_amplitudes(path, QPAmplitudes.zeros(4))

    with pytest.raises(ValueError, match="wrong pair dimension"):
        load_amplitudes(path, nspin=6)


def test_json_result_is_strict_and_round_trips_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    payload = {"schema": "test", "energy_eh": -1.25, "certified": True}

    assert write_json_atomic(path, payload) == path
    assert read_json(path) == payload
    assert path.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(ValueError, match="JSON document must contain an object"):
        scalar = tmp_path / "scalar.json"
        scalar.write_text("[]\n", encoding="utf-8")
        read_json(scalar)
