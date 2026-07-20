from __future__ import annotations

import numpy as np
import pytest

from autogen.methods.qpccsd.production.cas_reference import build_cas_qp_reference_from_rdms
from autogen.methods.qpccsd.production.excitation_space import build_symmetry_adapted_qp_space
from autogen.methods.qpccsd.production.models import QPHamiltonian, SolverOptions
from autogen.methods.qpccsd.production.production import solve_qpccsd


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


def test_cas_reference_does_not_implicitly_add_casscf_energy() -> None:
    reference = _reference_with_distinct_casscf_energy()
    hamiltonian = _stationary_hamiltonian(reference.nspin, constant=-3.25)
    space = build_symmetry_adapted_qp_space(
        reference, include_active_t1_t2=True
    )

    result = solve_qpccsd(
        hamiltonian,
        reference,
        excitation_space=space,
        options=SolverOptions(max_iterations=0),
    )

    assert result.converged is True
    assert complex(result.total_energy).real == pytest.approx(-3.25, abs=1.0e-14)
    assert result.diagnostics["energy_convention"] == "direct"
    assert result.diagnostics["casscf_energy_added"] is False
    assert result.diagnostics["cas_plus_delta_applied"] is False
    assert result.diagnostics["casscf_reference_energy_diagnostic"] == pytest.approx(
        -8.75
    )


def test_casscf_plus_delta_requires_an_explicit_experimental_choice() -> None:
    reference = _reference_with_distinct_casscf_energy()
    hamiltonian = _stationary_hamiltonian(reference.nspin, constant=-3.25)
    space = build_symmetry_adapted_qp_space(
        reference, include_active_t1_t2=True
    )

    result = solve_qpccsd(
        hamiltonian,
        reference,
        excitation_space=space,
        options=SolverOptions(max_iterations=0),
        energy_convention="casscf_plus_delta",
    )

    assert complex(result.total_energy).real == pytest.approx(-8.75, abs=1.0e-14)
    assert result.diagnostics["energy_convention"] == "casscf_plus_delta"
    assert result.diagnostics["casscf_energy_added"] is True
    assert result.diagnostics["cas_plus_delta_applied"] is True
