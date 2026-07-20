from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "qpccsd"


@pytest.mark.parametrize(
    ("filename", "basis", "grid_size"),
    [
        ("n2_sto3g.toml", "sto-3g", 9),
        ("n2_6311g.toml", "6-311g", 25),
    ],
)
def test_example_config_uses_reviewed_production_defaults(
    filename: str,
    basis: str,
    grid_size: int,
) -> None:
    payload = tomllib.loads((CONFIG_ROOT / filename).read_text(encoding="utf-8"))

    assert set(payload) == {"system", "active_space", "solver", "projection", "output"}
    assert payload["system"]["molecule"] == "n2"
    assert payload["system"]["basis"].lower() == basis
    assert payload["active_space"] == {
        "orbitals": 6,
        "electrons": 6,
        "frozen_n1s": True,
    }
    assert payload["projection"]["grid_size"] == grid_size
    assert payload["projection"]["validation_tolerance"] == 1.0e-8
    assert not Path(payload["output"]["path"]).is_absolute()


def test_method_identity_cannot_be_overridden_in_example_configs() -> None:
    forbidden = {
        "reference_mode",
        "include_active_t1_t2",
        "energy_convention",
        "cas_plus_delta",
        "projection_mode",
        "projected_residual_optimized",
        "disentanglement_backend",
        "gauge_quadrature",
        "richardson",
        "ser3",
        "oap",
    }
    for path in sorted(CONFIG_ROOT.glob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        configured_keys = {
            key
            for section in payload.values()
            if isinstance(section, dict)
            for key in section
        }
        assert configured_keys.isdisjoint(forbidden), path.name
