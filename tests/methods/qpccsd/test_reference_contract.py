from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autogen.methods.qpccsd.production.cas_reference import build_projected_agp_reference
from autogen.methods.qpccsd.production.workflow import (
    _validate_checkpoint_request,
    build_full_active_qp_space,
)


def _reference():
    rdm1 = np.diag([1.4, 0.6])
    rdm2 = np.zeros((2,) * 4)
    rdm2[0, 0, 0, 0] = 0.8
    rdm2[1, 1, 1, 1] = 0.2
    rdm2[0, 1, 0, 1] = 0.3
    rdm2[1, 0, 1, 0] = 0.3
    return build_projected_agp_reference(
        casscf_energy=-2.0,
        mo_coeff=np.eye(2),
        active_rdm1=rdm1,
        active_rdm2=rdm2,
        active_spatial_indices=(0, 1),
        inactive_spatial_indices=(),
        external_spatial_indices=(),
        correlated_target_number=2,
    )


def test_reference_uses_natural_occupations_and_pair_transfer_2rdm() -> None:
    reference = _reference()

    assert reference.reference_mode == "projected_agp_2rdm"
    assert reference.signed_geminals is not None
    assert reference.signed_geminals.shape == (2,)
    assert reference.source_rdm_metadata["rdm1"] == "spin-summed spatial <p^dagger q>"
    assert (
        reference.source_rdm_metadata["pair_transfer_block"]
        == "active_rdm2[p,q,p,q]"
    )
    assert reference.source_rdm_metadata["complete_active_rdm2_reconstructed"] is False
    assert reference.source_rdm_metadata["reference_protocol"] == (
        "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
    )
    assert np.isfinite(reference.bogoliubov.metadata["global_number_scale"])
    assert reference.bogoliubov.metadata["global_number_scale"] >= 0.0
    np.testing.assert_allclose(
        reference.bogoliubov.metadata["active_natural_occupations"],
        reference.spatial_occupations[list(reference.active_spatial_indices)],
        rtol=0.0,
        atol=1.0e-12,
    )
    normal_error, anomalous_error = reference.bogoliubov.canonical_errors()
    assert normal_error < 1.0e-12
    assert anomalous_error < 1.0e-12


def test_nonfinite_reference_data_fail_before_a_production_solve() -> None:
    reference = _reference()
    reference.active_rdm1[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_full_active_qp_space(reference)


def test_noncanonical_bogoliubov_reference_fails_before_a_production_solve() -> None:
    reference = _reference()
    reference.bogoliubov.U[0, 0] += 0.1

    with pytest.raises(ValueError, match="canonical"):
        build_full_active_qp_space(reference)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda reference: reference.source_rdm_metadata.clear(), "source-RDM"),
        (
            lambda reference: reference.reconstruction_metrics.pop("fit_cost"),
            "fit diagnostics",
        ),
        (
            lambda reference: reference.bogoliubov.metadata.pop("global_number_scale"),
            "number-setting scale",
        ),
    ],
)
def test_incomplete_projected_agp_provenance_fails_closed(mutation, message: str) -> None:
    reference = _reference()
    mutation(reference)

    with pytest.raises(ValueError, match=message):
        build_full_active_qp_space(reference)


class _CheckpointMolecule:
    charge = 0
    spin = 0
    symmetry = True
    basis = "sto-3g"
    nelectron = 14

    def atom_charges(self) -> np.ndarray:
        return np.asarray([7, 7])

    def atom_coords(self, *, unit: str) -> np.ndarray:
        assert unit == "Angstrom"
        return np.asarray([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])


def _checkpoint():
    reference = _reference()
    reference.frozen_spatial_indices = (0, 1)
    reference.physical_target_number = 14
    reference.bogoliubov.target_number = 10
    return SimpleNamespace(molecule=_CheckpointMolecule(), reference=reference)


def test_checkpoint_request_identity_happy_path() -> None:
    _validate_checkpoint_request(
        _checkpoint(),
        molecule="n2",
        distance_angstrom=1.10,
        basis="STO-3G",
        cas_norb=2,
        cas_nelec=2,
        frozen_n1s=True,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"molecule": "h2", "frozen_n1s": False}, "molecule mismatch"),
        ({"distance_angstrom": 1.20}, "geometry mismatch"),
        ({"basis": "6-311g"}, "basis mismatch"),
        ({"cas_norb": 3}, "orbital-count mismatch"),
        ({"cas_nelec": 4}, "electron-count mismatch"),
        ({"frozen_n1s": False}, "frozen-core mismatch"),
    ],
)
def test_checkpoint_request_mismatches_fail_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    request = {
        "molecule": "n2",
        "distance_angstrom": 1.10,
        "basis": "sto-3g",
        "cas_norb": 2,
        "cas_nelec": 2,
        "frozen_n1s": True,
    }
    request.update(overrides)

    with pytest.raises(ValueError, match=message):
        _validate_checkpoint_request(_checkpoint(), **request)
