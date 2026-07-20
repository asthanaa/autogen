from __future__ import annotations

import numpy as np

from autogen.methods.qpccsd.production.cas_reference import build_cas_qp_reference_from_rdms
from autogen.methods.qpccsd.production.workflow import build_full_active_qp_space


def _synthetic_sto3g_reference():
    reference = build_cas_qp_reference_from_rdms(
        casscf_energy=-100.0,
        mo_coeff=np.eye(8),
        active_rdm1=np.eye(6),
        active_rdm2=np.zeros((6,) * 4),
        active_spatial_indices=(2, 3, 4, 5, 6, 7),
        inactive_spatial_indices=(0, 1),
        external_spatial_indices=(),
        correlated_target_number=10,
    )
    # PySCF Dooh irreps for A1g, A1u, E1uy, E1ux, A1g, E1gx, E1gy, A1u.
    reference.metadata["orbital_irrep_ids"] = np.asarray(
        [0, 5, 6, 7, 0, 2, 3, 5], dtype=np.int64
    )
    return reference


def _synthetic_6311g_reference():
    reference = build_cas_qp_reference_from_rdms(
        casscf_energy=-100.0,
        mo_coeff=np.eye(24),
        active_rdm1=np.eye(6),
        active_rdm2=np.zeros((6,) * 4),
        active_spatial_indices=(2, 3, 4, 5, 6, 7),
        inactive_spatial_indices=(0, 1),
        external_spatial_indices=tuple(range(8, 24)),
        correlated_target_number=10,
    )
    reference.metadata["orbital_irrep_ids"] = np.asarray(
        [
            0,
            5,
            6,
            7,
            0,
            2,
            3,
            5,
            0,
            0,
            0,
            0,
            2,
            2,
            3,
            3,
            5,
            5,
            5,
            5,
            6,
            6,
            7,
            7,
        ],
        dtype=np.int64,
    )
    return reference


def test_default_full_active_space_has_the_certified_sto3g_coordinates() -> None:
    space = build_full_active_qp_space(_synthetic_sto3g_reference())
    diagnostics = space.diagnostics()

    assert space.include_active_t1_t2 is True
    assert space.internal_coordinate_count == 0
    assert space.pair_count == 8
    assert space.quadruple_count == 57
    assert space.coordinate_count == 65
    assert diagnostics["pair_block_counts"] == {"ix": 2, "xy": 6}
    assert diagnostics["quadruple_block_counts"] == {
        "ijxy": 18,
        "ixyz": 18,
        "xyzw": 21,
    }


def test_full_active_coordinates_round_trip_without_masking() -> None:
    space = build_full_active_qp_space(_synthetic_sto3g_reference())
    vector = np.linspace(-0.05, 0.07, space.coordinate_count)
    amplitudes = space.unpack(vector)

    np.testing.assert_allclose(space.pack(amplitudes), vector, atol=1.0e-14)
    assert space.forbidden_amplitude_norm(amplitudes) == 0.0
    assert np.max(np.abs(amplitudes.t1[4:16, 4:16])) > 0.0
    assert np.max(np.abs(amplitudes.t2[4:16, 4:16, 4:16, 4:16])) > 0.0


def test_full_active_space_has_the_certified_6311g_coordinate_count() -> None:
    space = build_full_active_qp_space(_synthetic_6311g_reference())
    diagnostics = space.diagnostics()

    assert space.include_active_t1_t2 is True
    assert space.internal_coordinate_count == 0
    assert space.pair_count == 32
    assert space.quadruple_count == 1641
    assert space.coordinate_count == 1673
    assert diagnostics["pair_block_counts"]["xy"] == 6
    assert diagnostics["quadruple_block_counts"]["xyzw"] == 21
