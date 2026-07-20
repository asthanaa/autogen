from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ANCHOR_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "n2_sto3g_cas66_r1p10_certified.json"
)
DEFAULT_RESULT_PATH = (
    REPOSITORY_ROOT / "results" / "n2_sto3g_r1.10" / "production_result.json"
)


def _result_path() -> Path:
    override = os.environ.get("QPCCSD_N2_VALIDATION_RESULT")
    return DEFAULT_RESULT_PATH if override is None else Path(override).expanduser()


@pytest.mark.molecular
@pytest.mark.remote
@pytest.mark.slow
def test_n2_sto3g_qpccsd_and_pav_match_the_certified_calculation() -> None:
    result_path = _result_path()
    if not result_path.is_file():
        pytest.skip(
            "run the N2/STO-3G validation on Medora/Talon/authorized desktop "
            "and set QPCCSD_N2_VALIDATION_RESULT"
        )

    anchor = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    actual = json.loads(result_path.read_text(encoding="utf-8"))
    expected = anchor["expected"]
    tolerances = anchor["acceptance_tolerances"]
    method = anchor["method_contract"]
    coordinates = anchor["coordinate_contract"]

    assert actual["schema"] == "projected-agp-fullspace-qpccsd-pav-v1"
    assert actual["status"] == "certified"
    assert actual["method"]["reference"]["mode"] == method["reference"]
    assert actual["method"]["reference"]["protocol"] == (
        "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
    )
    assert actual["method"]["reference"]["complete_active_rdm2_reconstructed"] is False
    assert actual["method"]["excitation_space"]["include_active_t1_t2"] is True
    assert actual["method"]["energy"]["convention"] == "direct"
    assert actual["method"]["energy"]["casscf_energy_added"] is False
    assert actual["method"]["energy"]["cas_plus_delta_applied"] is False
    assert actual["method"]["projection"]["workflow"] == "projection after variation"
    assert actual["method"]["projection"]["amplitudes_fixed"] is True
    assert actual["method"]["projection"]["projected_residual_optimized"] is False
    assert actual["method"]["projection"]["disentanglement_backend"] == "ser2"
    assert actual["method"]["projection"]["closure"] == "W1/W2 with W3=0"

    reference = actual["reference"]
    assert reference["source_rdm_metadata"] == {
        "rdm1": "spin-summed spatial <p^dagger q>",
        "rdm2": "PySCF spin-free convention",
        "pair_transfer_block": "active_rdm2[p,q,p,q]",
        "reference_protocol": (
            "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
        ),
        "complete_active_rdm2_reconstructed": False,
    }
    for key in (
        "active_natural_occupations",
        "relative_signed_geminals",
        "scaled_signed_geminals",
    ):
        assert len(reference[key]) == 6
        assert all(float("-inf") < float(value) < float("inf") for value in reference[key])
    assert 0.0 <= reference["global_number_setting_scale"] < float("inf")
    for fingerprint in reference["array_fingerprints_sha256"].values():
        assert len(fingerprint) == 64
        int(fingerprint, 16)
    assert reference["reconstruction_metrics"]["fit_success"] is True
    assert len(reference["bogoliubov_canonical_errors"]) == 2
    assert max(reference["bogoliubov_canonical_errors"]) < 1.0e-12

    assert actual["system"]["nspin"] == coordinates["nspin"]
    assert actual["system"]["correlated_target_number"] == 10
    assert actual["system"]["physical_target_number"] == 14
    assert len(actual["system"]["frozen_spatial_indices"]) == 2
    assert actual["excitation_space"]["allowed_pairs"] == coordinates["pair_count"]
    assert (
        actual["excitation_space"]["allowed_quadruples"]
        == coordinates["quadruple_count"]
    )
    assert actual["excitation_space"]["coordinate_count"] == coordinates[
        "coordinate_count"
    ]
    assert actual["excitation_space"]["active_pair_coordinates"] == coordinates[
        "active_pair_count"
    ]
    assert actual["excitation_space"]["active_quadruple_coordinates"] == coordinates[
        "active_quadruple_count"
    ]
    assert actual["excitation_space"]["internal_coordinates_excluded"] == 0

    energy_tolerance = tolerances["energy_absolute_eh"]
    assert actual["qpccsd"]["converged"] is True
    assert actual["qpccsd"]["energy_convention"] == "direct"
    assert actual["qpccsd"]["terminal_unshifted_audit"] is True
    assert actual["qpccsd"]["energy_eh"] == pytest.approx(
        expected["qpccsd_energy_eh"], abs=energy_tolerance
    )
    assert actual["qpccsd"]["residual_norm"] <= tolerances["residual_maximum"]

    assert actual["pav"]["evaluated"] is True
    assert actual["pav"]["energy_convention"] == "direct"
    assert actual["pav"]["cas_plus_delta_applied"] is False
    assert actual["pav"]["amplitudes_fixed"] is True
    assert actual["pav"]["projected_residual_optimized"] is False
    assert actual["pav"]["energy_eh"] == pytest.approx(
        expected["pav_qpccsd_energy_eh"], abs=energy_tolerance
    )
    assert actual["pav"]["baseline_grid_size"] == method["grid_size"]
    assert actual["pav"]["validation_grid_size"] == method["validation_grid_size"]
    assert actual["pav"]["validation_passed"] is True
    assert (
        actual["pav"]["raw_grid_error_eh"]
        <= tolerances["pav_grid_error_maximum_eh"]
    )
    assert (
        actual["pav"]["raw_imaginary_energy_error_eh"]
        <= tolerances["pav_imaginary_energy_maximum_eh"]
    )
    assert abs(actual["pav"]["energy_imaginary_eh"]) <= tolerances[
        "pav_imaginary_energy_maximum_eh"
    ]

    assert actual["certification"] == {
        "raw_certified": True,
        "pav_numerically_validated": True,
        "certified": True,
        "diagnostic_only": False,
        "pav_from_uncertified_amplitudes": False,
        "reference_finite_and_canonical": True,
        "reference_canonical_tolerance": 1.0e-12,
        "fresh_terminal_unshifted_residual": True,
    }

    site = str(
        actual["provenance"].get("site", actual["provenance"].get("hostname", ""))
    ).lower()
    assert any(allowed in site for allowed in ("medora", "talon", "desktop"))
    assert "ndsu" not in site
