from __future__ import annotations

import copy
import json

import pytest

from autogen.methods.qpccsd.production.cli import _production_config, main, validate_payload
from autogen.methods.qpccsd.production.contracts import (
    PRODUCTION_PAIR_TRANSFER_CONVENTION,
    PRODUCTION_RDM1_CONVENTION,
    PRODUCTION_RDM2_CONVENTION,
    PRODUCTION_REFERENCE_PROTOCOL,
)


def _anchor_shaped_result(anchor: dict[str, object]) -> dict[str, object]:
    expected = anchor["expected"]
    coordinates = anchor["coordinate_contract"]
    return {
        "schema": "projected-agp-fullspace-qpccsd-pav-v1",
        "status": "certified",
        "method": {
            "reference": {
                "mode": "projected_agp_2rdm",
                "protocol": PRODUCTION_REFERENCE_PROTOCOL,
                "complete_active_rdm2_reconstructed": False,
            },
            "excitation_space": {
                "mode": "full-symmetry-adapted-active-t1-t2",
                "include_active_t1_t2": True,
            },
            "energy": {
                "convention": "direct",
                "casscf_energy_added": False,
                "cas_plus_delta_applied": False,
            },
            "projection": {
                "amplitudes_fixed": True,
                "projected_residual_optimized": False,
                "disentanglement_backend": "ser2",
                "gauge_quadrature": "midpoint",
                "mode": "fixed-amplitude-pn-pav-ser2-w2",
            },
        },
        "reference": {
            "reference_mode": "projected_agp_2rdm",
            "active_natural_occupations": [1.0, 1.0],
            "relative_signed_geminals": [1.0, -1.0],
            "scaled_signed_geminals": [0.5, -0.5],
            "global_number_setting_scale": 0.5,
            "array_fingerprints_sha256": {
                "active_rdm1": "1" * 64,
                "active_rdm2": "2" * 64,
                "mo_coeff": "3" * 64,
            },
            "reconstruction_metrics": {
                "fit_success": True,
                "fit_message": "synthetic anchor-shaped result",
                "fit_cost": 0.0,
                "rdm1_max_abs_error": 0.0,
                "pair_rdm2_max_abs_error": 0.0,
                "pair_subspace_fidelity": 1.0,
                "active_average_number_error": 0.0,
            },
            "source_rdm_metadata": {
                "rdm1": PRODUCTION_RDM1_CONVENTION,
                "rdm2": PRODUCTION_RDM2_CONVENTION,
                "pair_transfer_block": PRODUCTION_PAIR_TRANSFER_CONVENTION,
                "reference_protocol": PRODUCTION_REFERENCE_PROTOCOL,
                "complete_active_rdm2_reconstructed": False,
            },
            "bogoliubov_canonical_errors": [0.0, 0.0],
        },
        "excitation_space": {
            "allowed_pairs": coordinates["pair_count"],
            "allowed_quadruples": coordinates["quadruple_count"],
        },
        "qpccsd": {
            "energy_eh": expected["qpccsd_energy_eh"],
            "energy_imaginary_eh": 0.0,
            "residual_norm": expected["qpccsd_residual_norm"],
            "converged": True,
            "energy_convention": "direct",
            "terminal_unshifted_audit": True,
            "terminal_unshifted_energy_difference_eh": 0.0,
        },
        "pav": {
            "evaluated": True,
            "energy_eh": expected["pav_qpccsd_energy_eh"],
            "energy_imaginary_eh": 0.0,
            "grid_error_eh": expected["pav_grid_error_eh"],
            "raw_imaginary_energy_error_eh": expected[
                "pav_imaginary_energy_error_eh"
            ],
            "validation_passed": True,
            "energy_convention": "direct",
            "cas_plus_delta_applied": False,
            "projected_residual_optimized": False,
            "amplitudes_fixed": True,
        },
        "certification": {
            "raw_certified": True,
            "pav_numerically_validated": True,
            "certified": True,
            "diagnostic_only": False,
            "reference_finite_and_canonical": True,
            "reference_canonical_tolerance": 1.0e-12,
            "fresh_terminal_unshifted_residual": True,
            "pav_from_uncertified_amplitudes": False,
        },
    }


def test_show_defaults_reports_the_locked_route(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show-defaults"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["scientific_contract"] == {
        "reference_mode": "projected_agp_2rdm",
        "include_active_t1_t2": True,
        "energy_convention": "direct",
        "projection": "fixed-amplitude Ser2 PAV",
    }
    assert payload["projection_grid_size"] is None
    assert payload["projection_validation_tolerance"] == 1.0e-8
    assert payload["project_uncertified_finite_amplitudes"] is True


def test_config_parser_rejects_an_experimental_projection_switch() -> None:
    with pytest.raises(ValueError, match=r"unknown \[projection\] keys"):
        _production_config(
            {
                "projection": {
                    "grid_size": 9,
                    "disentanglement_backend": "ser3",
                }
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"experimental": {}},
        {"system": {"energy_convention": "casscf_plus_delta"}},
        {"active_space": {"reference_mode": "sokolov_linear_1rdm"}},
        {"output": {"path": "result.json", "publish": True}},
    ],
)
def test_config_parser_rejects_unknown_or_misplaced_keys(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        _production_config(payload)


def test_anchor_validator_accepts_only_the_matching_production_route(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    payload = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    report = validate_payload(payload, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is True
    assert report["failures"] == []
    assert report["comparison"]["qpccsd_energy_error_eh"] == 0.0
    assert report["comparison"]["pav_energy_error_eh"] == 0.0

    wrong_energy = copy.deepcopy(payload)
    wrong_energy["qpccsd"]["energy_eh"] += 2.0e-8
    report = validate_payload(wrong_energy, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is False
    assert "QPCCSD anchor energy" in report["failures"]

    wrong_method = copy.deepcopy(payload)
    wrong_method["method"]["energy"]["cas_plus_delta_applied"] = True
    report = validate_payload(wrong_method, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is False
    assert any("cas_plus_delta_applied" in item for item in report["failures"])


def test_anchor_validator_fails_closed_on_diagnostic_or_malformed_results(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    payload = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    payload["status"] = "diagnostic_only"
    payload["certification"] = {
        "raw_certified": False,
        "pav_numerically_validated": True,
        "certified": False,
        "diagnostic_only": True,
    }
    report = validate_payload(payload, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is False
    assert any("certif" in reason.lower() for reason in report["failures"])

    malformed = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    malformed["pav"]["energy_eh"] = None
    report = validate_payload(malformed, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is False
    assert report["failures"]


def test_validator_rejects_status_and_certification_inconsistency(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    payload = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    payload["certification"]["certified"] = False
    payload["certification"]["diagnostic_only"] = True

    report = validate_payload(payload)

    assert report["valid"] is False
    assert "status/certified consistency" in report["failures"]
    assert "status/diagnostic consistency" in report["failures"]


def test_validator_rejects_missing_reference_provenance(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    payload = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    payload["reference"]["source_rdm_metadata"] = {}

    report = validate_payload(payload)

    assert report["valid"] is False
    assert any("source-RDM" in reason for reason in report["failures"])


def test_validator_rechecks_reference_and_certification_gates(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    noncanonical = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    noncanonical["reference"]["bogoliubov_canonical_errors"] = [2.0e-12, 0.0]
    report = validate_payload(noncanonical)
    assert "reference Bogoliubov canonical errors" in report["failures"]

    false_reference_flag = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    false_reference_flag["certification"]["reference_finite_and_canonical"] = False
    report = validate_payload(false_reference_flag)
    assert "certified reference finite/canonical flag" in report["failures"]

    stale_pav_source = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    stale_pav_source["certification"]["pav_from_uncertified_amplitudes"] = True
    report = validate_payload(stale_pav_source)
    assert "PAV uncertified-source consistency" in report["failures"]


def test_anchor_validator_applies_imaginary_energy_gates(
    n2_sto3g_r1p10_anchor: dict[str, object],
) -> None:
    payload = _anchor_shaped_result(n2_sto3g_r1p10_anchor)
    payload["status"] = "certified"
    payload["certification"] = {
        "raw_certified": True,
        "pav_numerically_validated": True,
        "certified": True,
        "diagnostic_only": False,
    }
    payload["qpccsd"]["energy_imaginary_eh"] = 2.0e-8
    payload["pav"].update(
        {
            "energy_imaginary_eh": 0.0,
            "raw_imaginary_energy_error_eh": 0.0,
        }
    )
    report = validate_payload(payload, anchor=n2_sto3g_r1p10_anchor)
    assert report["valid"] is False
    assert any("imaginary" in reason.lower() for reason in report["failures"])
