from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogen.methods.qpccsd.production.cli import validate_payload


TEST_ROOT = Path(__file__).resolve().parent


def test_certified_anchor_is_the_production_route(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    contract = n2_sto3g_r1p10_anchor["method_contract"]
    assert contract == {
        "reference": "projected_agp_2rdm",
        "reference_fit_data": [
            "active_natural_occupations",
            "active_pair_transfer_2rdm",
        ],
        "excitation_space": "spatial-total-singlet-block",
        "include_active_t1_t2": True,
        "energy_convention": "direct",
        "cas_energy_added": False,
        "cas_plus_delta_applied": False,
        "projection": "particle_number_projection_after_variation",
        "projected_residual_optimized": False,
        "disentanglement": "ser2",
        "w3_included": False,
        "gauge_quadrature": "midpoint",
        "grid_size": 9,
        "validation_grid_size": 18,
        "ode_substeps": 1,
        "validation_ode_substeps": 2,
        "contour_radius": 1.0,
    }


def test_certified_anchor_has_full_active_sto3g_coordinates(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    coordinates = n2_sto3g_r1p10_anchor["coordinate_contract"]
    assert coordinates["pair_count"] == 8
    assert coordinates["quadruple_count"] == 57
    assert coordinates["coordinate_count"] == 65
    assert coordinates["active_pair_count"] == 6
    assert coordinates["active_quadruple_count"] == 21
    assert coordinates["pair_block_counts"]["xy"] == 6
    assert coordinates["quadruple_block_counts"]["xyzw"] == 21


def test_certified_anchor_passes_its_declared_numerical_gates(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    expected = n2_sto3g_r1p10_anchor["expected"]
    tolerances = n2_sto3g_r1p10_anchor["acceptance_tolerances"]
    assert expected["qpccsd_converged"] is True
    assert expected["pav_validation_passed"] is True
    assert expected["certified"] is True
    assert expected["qpccsd_residual_norm"] <= tolerances["residual_maximum"]
    assert expected["pav_grid_error_eh"] <= tolerances["pav_grid_error_maximum_eh"]
    assert (
        expected["pav_imaginary_energy_error_eh"]
        <= tolerances["pav_imaginary_energy_maximum_eh"]
    )
    assert expected["qpccsd_energy_eh"] != pytest.approx(
        expected["pav_qpccsd_energy_eh"], abs=1.0e-6
    )


def test_remote_validation_request_is_locked_to_the_anchor() -> None:
    request_path = TEST_ROOT / "validation" / "n2_sto3g_cas66_r1p10_run.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    fixture_path = (request_path.parent / request["expected_fixture"]).resolve()
    anchor = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert request["case_id"] == anchor["case_id"]
    assert request["production_config"]["energy_convention"] == "direct"
    assert request["production_config"]["include_active_t1_t2"] is True
    assert request["production_config"]["run_pav"] is True
    assert request["allowed_sites"] == ["medora", "talon", "authorized_desktop"]
    assert request["forbidden_sites"] == ["laptop", "ndsu"]


def test_checked_in_medora_qpccsd_and_pav_result_matches_the_anchor() -> None:
    anchor_path = TEST_ROOT / "fixtures" / "n2_sto3g_cas66_r1p10_certified.json"
    result_path = TEST_ROOT / "validation" / "n2_sto3g_cas66_r1p10_result.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    report = validate_payload(result, anchor=anchor)

    assert report["valid"] is True, report["failures"]
    assert report["failures"] == []
    assert report["comparison"]["qpccsd_energy_error_eh"] < 1.0e-8
    assert report["comparison"]["pav_energy_error_eh"] < 1.0e-8
