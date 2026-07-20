from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from autogen.methods.qpccsd.production.cas_reference import build_projected_agp_reference
from autogen.methods.qpccsd.production.cli import validate_payload
from autogen.methods.qpccsd.production.models import QPHamiltonian, SolverOptions
from autogen.methods.qpccsd.production.workflow import ProductionConfig, run_qpccsd_pav


def _reference():
    return build_projected_agp_reference(
        casscf_energy=-8.75,
        mo_coeff=np.eye(2),
        active_rdm1=np.eye(2),
        active_rdm2=np.zeros((2,) * 4),
        active_spatial_indices=(0, 1),
        inactive_spatial_indices=(),
        external_spatial_indices=(),
        correlated_target_number=2,
    )


def _hamiltonian(*, source_residual: float = 0.0) -> QPHamiltonian:
    nspin = 4
    h20 = np.zeros((nspin, nspin))
    h20[0, 1] = source_residual
    h20[1, 0] = -source_residual
    rank4 = np.zeros((nspin,) * 4)
    return QPHamiltonian(
        constant=-3.25,
        h11=np.diag(np.linspace(0.5, 1.0, nspin)),
        h20=h20,
        h02=h20.copy(),
        h22=rank4,
        h31=rank4.copy(),
        h13=rank4.copy(),
        h40=rank4.copy(),
        h04=rank4.copy(),
    )


def test_config_locks_the_scientific_method() -> None:
    configurable = {item.name for item in fields(ProductionConfig)}
    forbidden_method_switches = {
        "reference_mode",
        "include_active_t1_t2",
        "energy_convention",
        "cas_plus_delta",
        "projection_mode",
        "projected_residual_optimized",
        "disentanglement_backend",
        "gauge_quadrature",
        "richardson",
        "oap",
    }
    assert configurable.isdisjoint(forbidden_method_switches)


def test_projection_defaults_are_basis_size_aware_and_locked() -> None:
    config = ProductionConfig()
    sto3g = config.projection_options(SimpleNamespace(nspin=16, target_number=10))
    triple_zeta = config.projection_options(
        SimpleNamespace(nspin=48, target_number=10)
    )

    assert sto3g.grid_size == 9
    assert triple_zeta.grid_size == 25
    for options in (sto3g, triple_zeta):
        assert options.target_number == 10
        assert options.parity == "even"
        assert options.gauge_quadrature == "midpoint"
        assert options.ode_substeps == 1
        assert options.auto_select_ode is False
        assert options.max_grid_refinements == 0
        assert options.disentanglement_backend == "ser2"
        assert options.validation_tolerance == 1.0e-8


def test_end_to_end_synthetic_workflow_serializes_the_production_contract() -> None:
    result = run_qpccsd_pav(
        _reference(),
        hamiltonian=_hamiltonian(),
        config=ProductionConfig(
            solver=SolverOptions(max_iterations=0),
            projection_grid_size=5,
        ),
        provenance={"test_case": "synthetic-stationary-root"},
    )
    payload = result.to_dict()

    assert set(payload) == {
        "schema",
        "status",
        "method",
        "system",
        "reference",
        "excitation_space",
        "energies_eh",
        "qpccsd",
        "pav",
        "certification",
        "timings_s",
        "provenance",
    }
    assert payload["schema"] == "projected-agp-fullspace-qpccsd-pav-v1"
    assert payload["status"] == "certified"
    assert payload["method"]["reference"]["mode"] == "projected_agp_2rdm"
    assert payload["method"]["reference"]["protocol"] == (
        "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
    )
    assert payload["method"]["reference"]["complete_active_rdm2_reconstructed"] is False
    assert payload["method"]["excitation_space"] == {
        "mode": "full-symmetry-adapted-active-t1-t2",
        "include_active_t1_t2": True,
        "symmetry": "total-singlet and totally symmetric",
    }
    assert payload["method"]["energy"]["convention"] == "direct"
    assert payload["method"]["energy"]["casscf_energy_added"] is False
    assert payload["method"]["energy"]["cas_plus_delta_applied"] is False
    assert payload["method"]["projection"]["amplitudes_fixed"] is True
    assert payload["method"]["projection"]["projected_residual_optimized"] is False
    assert payload["method"]["projection"]["closure"] == "W1/W2 with W3=0"
    assert payload["qpccsd"]["energy_eh"] == -3.25
    assert payload["qpccsd"]["energy_convention"] == "direct"
    assert payload["pav"]["evaluated"] is True
    np.testing.assert_allclose(
        payload["pav"]["energy_eh"],
        -3.25,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert payload["pav"]["energy_convention"] == "direct"
    assert payload["pav"]["cas_plus_delta_applied"] is False
    assert payload["pav"]["amplitudes_fixed"] is True
    assert payload["pav"]["projected_residual_optimized"] is False
    assert payload["pav"]["baseline_grid_size"] == 5
    assert payload["pav"]["validation_grid_size"] == 10
    reference = payload["reference"]
    assert len(reference["active_natural_occupations"]) == 2
    assert len(reference["relative_signed_geminals"]) == 2
    assert len(reference["scaled_signed_geminals"]) == 2
    assert np.isfinite(reference["global_number_setting_scale"])
    assert "signed_geminals" not in reference
    assert reference["source_rdm_metadata"]["reference_protocol"] == (
        "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
    )
    for fingerprint in reference["array_fingerprints_sha256"].values():
        assert len(fingerprint) == 64
        int(fingerprint, 16)
    assert validate_payload(payload) == {
        "valid": True,
        "failures": [],
        "comparison": {},
    }
    assert payload["certification"] == {
        "raw_certified": True,
        "pav_numerically_validated": True,
        "certified": True,
        "diagnostic_only": False,
        "pav_from_uncertified_amplitudes": False,
        "reference_finite_and_canonical": True,
        "reference_canonical_tolerance": 1.0e-12,
        "fresh_terminal_unshifted_residual": True,
    }


def test_finite_pav_from_an_uncertified_root_is_diagnostic_only() -> None:
    result = run_qpccsd_pav(
        _reference(),
        hamiltonian=_hamiltonian(source_residual=0.01),
        config=ProductionConfig(
            solver=SolverOptions(max_iterations=0),
            projection_grid_size=5,
        ),
    )
    payload = result.to_dict()

    assert payload["status"] == "diagnostic_only"
    assert payload["qpccsd"]["converged"] is False
    assert payload["qpccsd"]["residual_norm"] == 0.01
    assert payload["pav"]["evaluated"] is True
    assert payload["pav"]["validation_passed"] is True
    assert payload["certification"] == {
        "raw_certified": False,
        "pav_numerically_validated": True,
        "certified": False,
        "diagnostic_only": True,
        "pav_from_uncertified_amplitudes": True,
        "reference_finite_and_canonical": True,
        "reference_canonical_tolerance": 1.0e-12,
        "fresh_terminal_unshifted_residual": True,
    }


def test_nonfinite_result_is_failed_and_not_diagnostic_only() -> None:
    hamiltonian = _hamiltonian()
    hamiltonian.constant = complex(float("nan"))

    result = run_qpccsd_pav(
        _reference(),
        hamiltonian=hamiltonian,
        config=ProductionConfig(
            solver=SolverOptions(max_iterations=0),
            projection_grid_size=5,
        ),
    )

    assert result.status == "failed"
    assert result.certified is False
    assert result.diagnostic_only is False
    assert result.pav is None
    assert result.to_dict()["certification"]["diagnostic_only"] is False
