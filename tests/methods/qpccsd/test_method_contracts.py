from __future__ import annotations

import inspect

import pytest

from autogen.methods.qpccsd.production.contracts import (
    EnergyConvention,
    PRODUCTION_EXCITATION_SPACE,
    PRODUCTION_PROJECTION,
    PRODUCTION_REFERENCE_MODE,
    PRODUCTION_RESULT_SCHEMA,
    normalize_energy_convention,
)


def test_production_identity_is_explicit_and_stable() -> None:
    assert PRODUCTION_RESULT_SCHEMA == "projected-agp-fullspace-qpccsd-pav-v1"
    assert PRODUCTION_REFERENCE_MODE == "projected_agp_2rdm"
    assert PRODUCTION_EXCITATION_SPACE == "full-symmetry-adapted-active-t1-t2"
    assert PRODUCTION_PROJECTION == "fixed-amplitude-pn-pav-ser2-w2"


def test_direct_energy_is_the_normalized_default() -> None:
    assert normalize_energy_convention("direct") is EnergyConvention.DIRECT
    assert normalize_energy_convention(EnergyConvention.DIRECT) is EnergyConvention.DIRECT
    assert EnergyConvention.DIRECT.description == (
        "E_QP(T) in the correlated molecular Hamiltonian"
    )


def test_unknown_energy_convention_fails_closed() -> None:
    with pytest.raises(ValueError, match="energy_convention must be one of"):
        normalize_energy_convention("automatic")


def test_low_level_solvers_do_not_infer_energy_from_reference_type() -> None:
    from autogen.methods.qpccsd.production.production import solve_qpccsd
    from autogen.methods.qpccsd.production.projection import evaluate_pav_qpccsd

    qp_default = inspect.signature(solve_qpccsd).parameters["energy_convention"].default
    pav_default = inspect.signature(evaluate_pav_qpccsd).parameters[
        "energy_convention"
    ].default
    assert qp_default in ("direct", EnergyConvention.DIRECT)
    assert pav_default in ("direct", EnergyConvention.DIRECT)
