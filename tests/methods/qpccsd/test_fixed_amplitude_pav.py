from __future__ import annotations

import numpy as np

from autogen.methods.qpccsd.production.cas_reference import build_cas_qp_reference_from_rdms
from autogen.methods.qpccsd.production.excitation_space import build_symmetry_adapted_qp_space
from autogen.methods.qpccsd.production.models import ProjectionOptions, QPHamiltonian
from autogen.methods.qpccsd.production.projection import evaluate_pav_qpccsd


def _reference_with_distinct_casscf_energy():
    return build_cas_qp_reference_from_rdms(
        casscf_energy=-8.75,
        mo_coeff=np.eye(2),
        active_rdm1=np.eye(2),
        active_rdm2=np.zeros((2,) * 4),
        active_spatial_indices=(0, 1),
        inactive_spatial_indices=(),
        external_spatial_indices=(),
        correlated_target_number=2,
    )


def _stationary_hamiltonian(nspin: int, constant: float) -> QPHamiltonian:
    pair = np.zeros((nspin, nspin))
    rank4 = np.zeros((nspin,) * 4)
    return QPHamiltonian(
        constant=constant,
        h11=np.diag(np.linspace(0.5, 1.0, nspin)),
        h20=pair,
        h02=pair.copy(),
        h22=rank4,
        h31=rank4.copy(),
        h13=rank4.copy(),
        h40=rank4.copy(),
        h04=rank4.copy(),
    )


def test_pav_uses_fixed_unprojected_amplitudes() -> None:
    reference = _reference_with_distinct_casscf_energy()
    hamiltonian = _stationary_hamiltonian(reference.nspin, constant=-3.25)
    space = build_symmetry_adapted_qp_space(
        reference, include_active_t1_t2=True
    )
    amplitudes = space.unpack(np.linspace(-0.003, 0.004, space.coordinate_count))
    t1_before = amplitudes.t1.copy()
    t2_before = amplitudes.t2.copy()

    projection = evaluate_pav_qpccsd(
        hamiltonian,
        reference,
        amplitudes,
        excitation_space=space,
        options=ProjectionOptions(
            target_number=reference.target_number,
            grid_size=5,
            gauge_quadrature="midpoint",
            ode_substeps=1,
            auto_select_ode=False,
            disentanglement_backend="ser2",
        ),
    )

    # PAV is an energy evaluation, not a projected residual optimization.
    np.testing.assert_array_equal(amplitudes.t1, t1_before)
    np.testing.assert_array_equal(amplitudes.t2, t2_before)
    np.testing.assert_allclose(projection.amplitudes.t1, t1_before, atol=1.0e-15)
    np.testing.assert_allclose(projection.amplitudes.t2, t2_before, atol=1.0e-15)
    assert projection.diagnostics["projected_residual_optimized"] is False
    assert projection.diagnostics["amplitudes_optimized_for"] == "unprojected_qpccsd"
    assert projection.diagnostics["residual_diagnostic_computed"] is False
    assert projection.diagnostics["full_residual_evaluations"] == 0


def test_pav_is_direct_ser2_midpoint_with_doubled_validation() -> None:
    reference = _reference_with_distinct_casscf_energy()
    hamiltonian = _stationary_hamiltonian(reference.nspin, constant=-3.25)
    space = build_symmetry_adapted_qp_space(
        reference, include_active_t1_t2=True
    )
    amplitudes = space.unpack(np.zeros(space.coordinate_count))

    projection = evaluate_pav_qpccsd(
        hamiltonian,
        reference,
        amplitudes,
        excitation_space=space,
        options=ProjectionOptions(
            target_number=reference.target_number,
            grid_size=5,
            gauge_quadrature="midpoint",
            ode_substeps=1,
            auto_select_ode=False,
            disentanglement_backend="ser2",
        ),
    )

    np.testing.assert_allclose(
        complex(projection.total_energy).real,
        -3.25,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert projection.diagnostics["energy_convention"] == "direct"
    assert projection.diagnostics["casscf_energy_added"] is False
    assert projection.diagnostics["cas_plus_delta_applied"] is False
    assert projection.diagnostics["disentanglement_backend"] == "ser2"
    assert projection.diagnostics["gauge_quadrature"] == "midpoint"
    assert projection.diagnostics["baseline_grid_size"] == 5
    assert projection.diagnostics["validation_grid_size"] == 10
    assert projection.diagnostics["baseline_ode_substeps"] == 1
    assert projection.diagnostics["validation_ode_substeps"] == 2
    assert projection.validation_passed is True
