"""Command-line interface for the reviewed production workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - selected by the Python runtime
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .contracts import (
    PRODUCTION_PAIR_TRANSFER_CONVENTION,
    PRODUCTION_RDM1_CONVENTION,
    PRODUCTION_RDM2_CONVENTION,
    PRODUCTION_REFERENCE_FIT_DIAGNOSTICS,
    PRODUCTION_REFERENCE_PROTOCOL,
    PRODUCTION_RESULT_SCHEMA,
)
from .io import (
    file_sha256,
    load_amplitudes,
    read_json,
    save_amplitudes,
    write_json_atomic,
)
from .models import ExecutionOptions, SolverOptions
from .workflow import (
    ProductionConfig,
    prepare_projected_agp_reference,
    run_qpccsd_pav,
)


def _reject_unknown(
    values: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(sorted(unknown))}")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration must contain a TOML table")
    _reject_unknown(
        payload,
        {"system", "active_space", "solver", "projection", "output"},
        "top-level configuration",
    )
    return payload


def _production_config(payload: dict[str, Any]) -> ProductionConfig:
    _reject_unknown(
        payload,
        {"system", "active_space", "solver", "projection", "output"},
        "top-level configuration",
    )
    tables = {
        name: payload.get(name, {})
        for name in ("system", "active_space", "solver", "projection", "output")
    }
    for name, table in tables.items():
        if not isinstance(table, dict):
            raise ValueError(f"[{name}] must be a TOML table")
    _reject_unknown(
        tables["system"],
        {
            "molecule",
            "basis",
            "distance_angstrom",
            "charge",
            "spin",
            "symmetry",
        },
        "[system]",
    )
    _reject_unknown(
        tables["active_space"],
        {"orbitals", "electrons", "frozen_n1s"},
        "[active_space]",
    )
    _reject_unknown(tables["output"], {"path"}, "[output]")
    solver_data = dict(payload.get("solver", {}))
    projection_data = dict(payload.get("projection", {}))
    solver = SolverOptions(**solver_data)
    cache_mib = int(projection_data.pop("cache_mib", 0))
    grid_size = projection_data.pop("grid_size", None)
    validation_tolerance = float(
        projection_data.pop("validation_tolerance", 1.0e-8)
    )
    parallel_mode = str(projection_data.pop("parallel_mode", "serial"))
    workers = projection_data.pop("workers", 1)
    blas_threads = projection_data.pop("blas_threads", 1)
    maximum_imaginary_energy = float(
        projection_data.pop("maximum_imaginary_energy", 1.0e-8)
    )
    project_uncertified = bool(
        projection_data.pop("project_uncertified_finite_amplitudes", True)
    )
    residual_diagnostic = bool(
        projection_data.pop("compute_residual_diagnostic", False)
    )
    if projection_data:
        raise ValueError(
            "unknown [projection] keys: " + ", ".join(sorted(projection_data))
        )
    execution = ExecutionOptions(
        parallel_mode=parallel_mode,
        workers=None if workers is None else int(workers),
        blas_threads=None if blas_threads is None else int(blas_threads),
        max_workspace_bytes=max(cache_mib, 1) * 1024**2,
    )
    return ProductionConfig(
        solver=solver,
        execution=execution,
        projection_grid_size=None if grid_size is None else int(grid_size),
        projection_validation_tolerance=validation_tolerance,
        projection_cache_bytes=cache_mib * 1024**2,
        maximum_imaginary_energy=maximum_imaginary_energy,
        project_uncertified_finite_amplitudes=project_uncertified,
        compute_pav_residual_diagnostic=residual_diagnostic,
    )


def _prepare_reference(
    payload: dict[str, Any],
    checkpoint: Path | None,
):
    system = dict(payload.get("system", {}))
    active = dict(payload.get("active_space", {}))
    _reject_unknown(
        system,
        {
            "molecule",
            "basis",
            "distance_angstrom",
            "charge",
            "spin",
            "symmetry",
        },
        "[system]",
    )
    _reject_unknown(
        active,
        {"orbitals", "electrons", "frozen_n1s"},
        "[active_space]",
    )
    if int(system.get("charge", 0)) != 0 or int(system.get("spin", 0)) != 0:
        raise ValueError("production molecular helper requires a neutral singlet")
    if not bool(system.get("symmetry", True)):
        raise ValueError("production molecular helper requires molecular symmetry")
    return prepare_projected_agp_reference(
        molecule=str(system.get("molecule", "n2")),
        distance_angstrom=float(system["distance_angstrom"]),
        basis=str(system.get("basis", "sto-3g")),
        cas_norb=int(active.get("orbitals", 6)),
        cas_nelec=int(active.get("electrons", 6)),
        frozen_n1s=bool(active.get("frozen_n1s", True)),
        checkpoint=None if checkpoint is None else str(checkpoint),
    )


def _run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    payload = _load_toml(config_path)
    config = _production_config(payload)
    reference_path = None if args.reference_checkpoint is None else args.reference_checkpoint.resolve()
    reference = _prepare_reference(payload, reference_path)
    initial_path = None if args.initial_amplitudes is None else args.initial_amplitudes.resolve()
    initial = (
        None
        if initial_path is None
        else load_amplitudes(initial_path, nspin=reference.nspin)
    )
    output_config = dict(payload.get("output", {}))
    _reject_unknown(output_config, {"path"}, "[output]")
    output = (
        args.output.resolve()
        if args.output is not None
        else Path(output_config.get("path", "results/production_result.json")).resolve()
    )
    amplitude_output = (
        args.amplitudes_output.resolve()
        if args.amplitudes_output is not None
        else output.with_name(output.stem + ".amplitudes.npz")
    )
    provenance: dict[str, Any] = {
        "config_file": config_path.name,
        "config_sha256": file_sha256(config_path),
    }
    if reference_path is not None:
        provenance["reference_checkpoint"] = {
            "name": reference_path.name,
            "sha256": file_sha256(reference_path),
        }
    if initial_path is not None:
        provenance["initial_amplitudes"] = {
            "name": initial_path.name,
            "sha256": file_sha256(initial_path),
            "role": "initial_guess_only",
        }
    result = run_qpccsd_pav(
        reference,
        config=config,
        initial_amplitudes=initial,
        provenance=provenance,
    )
    save_amplitudes(
        amplitude_output,
        result.qpccsd.amplitudes,
        metadata={
            "result_schema": PRODUCTION_RESULT_SCHEMA,
            "reference_mode": reference.reference_mode,
            "include_active_t1_t2": True,
            "energy_convention": "direct",
        },
    )
    serialized = result.to_dict()
    serialized["provenance"]["output_amplitudes"] = {
        "name": amplitude_output.name,
        "sha256": file_sha256(amplitude_output),
    }
    write_json_atomic(output, serialized)
    print(json.dumps(serialized, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.certified else 2


def validate_payload(
    payload: dict[str, Any],
    *,
    anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if payload.get("schema") != PRODUCTION_RESULT_SCHEMA:
        failures.append("result schema")
    if payload.get("status") not in {"certified", "diagnostic_only", "failed"}:
        failures.append("result status")
    method = payload.get("method", {})
    if not isinstance(method, dict):
        method = {}
        failures.append("method object")
    required = {
        ("reference", "mode"): "projected_agp_2rdm",
        ("reference", "protocol"): PRODUCTION_REFERENCE_PROTOCOL,
        ("reference", "complete_active_rdm2_reconstructed"): False,
        ("excitation_space", "mode"): "full-symmetry-adapted-active-t1-t2",
        ("excitation_space", "include_active_t1_t2"): True,
        ("energy", "convention"): "direct",
        ("energy", "casscf_energy_added"): False,
        ("energy", "cas_plus_delta_applied"): False,
        ("projection", "amplitudes_fixed"): True,
        ("projection", "projected_residual_optimized"): False,
        ("projection", "disentanglement_backend"): "ser2",
        ("projection", "gauge_quadrature"): "midpoint",
        ("projection", "mode"): "fixed-amplitude-pn-pav-ser2-w2",
    }
    for (section, key), expected in required.items():
        subsection = method.get(section, {})
        actual = subsection.get(key) if isinstance(subsection, dict) else None
        if actual != expected:
            failures.append(f"method.{section}.{key}={actual!r}")
    qpccsd = payload.get("qpccsd", {})
    pav = payload.get("pav", {})
    certification = payload.get("certification", {})
    excitation_space = payload.get("excitation_space", {})
    reference = payload.get("reference", {})
    for name, section in (
        ("qpccsd", qpccsd),
        ("pav", pav),
        ("certification", certification),
        ("excitation_space", excitation_space),
        ("reference", reference),
    ):
        if not isinstance(section, dict):
            failures.append(f"{name} object")
    qpccsd = qpccsd if isinstance(qpccsd, dict) else {}
    pav = pav if isinstance(pav, dict) else {}
    certification = certification if isinstance(certification, dict) else {}
    excitation_space = excitation_space if isinstance(excitation_space, dict) else {}
    reference = reference if isinstance(reference, dict) else {}

    if reference.get("reference_mode") != "projected_agp_2rdm":
        failures.append("reference mode")
    source_metadata = reference.get("source_rdm_metadata", {})
    if not isinstance(source_metadata, dict):
        failures.append("reference source-RDM metadata object")
        source_metadata = {}
    expected_source = {
        "rdm1": PRODUCTION_RDM1_CONVENTION,
        "rdm2": PRODUCTION_RDM2_CONVENTION,
        "pair_transfer_block": PRODUCTION_PAIR_TRANSFER_CONVENTION,
        "reference_protocol": PRODUCTION_REFERENCE_PROTOCOL,
        "complete_active_rdm2_reconstructed": False,
    }
    for key, expected in expected_source.items():
        if source_metadata.get(key) != expected:
            failures.append(f"reference source-RDM {key}")
    reconstruction = reference.get("reconstruction_metrics", {})
    if not isinstance(reconstruction, dict):
        failures.append("reference reconstruction metrics object")
        reconstruction = {}
    for key in PRODUCTION_REFERENCE_FIT_DIAGNOSTICS:
        if key not in reconstruction:
            failures.append(f"reference fit diagnostic {key}")
    if reconstruction.get("fit_success") is not True:
        failures.append("reference fit success")
    if not isinstance(reconstruction.get("fit_message"), str) or not str(
        reconstruction.get("fit_message", "")
    ).strip():
        failures.append("reference fit message")
    for key in (
        "fit_cost",
        "rdm1_max_abs_error",
        "pair_rdm2_max_abs_error",
        "active_average_number_error",
    ):
        value = reconstruction.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            failures.append(f"reference fit diagnostic {key}")
    fidelity = reconstruction.get("pair_subspace_fidelity")
    if (
        isinstance(fidelity, bool)
        or not isinstance(fidelity, (int, float))
        or not np.isfinite(float(fidelity))
        or not -1.0e-12 <= float(fidelity) <= 1.0 + 1.0e-12
    ):
        failures.append("reference fit diagnostic pair_subspace_fidelity")

    provenance_vectors: list[list[Any]] = []
    for key in (
        "active_natural_occupations",
        "relative_signed_geminals",
        "scaled_signed_geminals",
    ):
        values = reference.get(key)
        if not isinstance(values, list) or not values:
            failures.append(f"reference {key}")
            continue
        try:
            finite = all(np.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            failures.append(f"reference {key}")
        provenance_vectors.append(values)
    if provenance_vectors and len({len(values) for values in provenance_vectors}) != 1:
        failures.append("reference provenance vector dimensions")
    number_scale = reference.get("global_number_setting_scale")
    if (
        isinstance(number_scale, bool)
        or not isinstance(number_scale, (int, float))
        or not np.isfinite(float(number_scale))
        or float(number_scale) < 0.0
    ):
        failures.append("reference global number-setting scale")
    fingerprints = reference.get("array_fingerprints_sha256", {})
    if not isinstance(fingerprints, dict):
        failures.append("reference array fingerprints object")
        fingerprints = {}
    for key in ("active_rdm1", "active_rdm2", "mo_coeff"):
        value = fingerprints.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            failures.append(f"reference array fingerprint {key}")
    canonical_errors = reference.get("bogoliubov_canonical_errors")
    if not isinstance(canonical_errors, list) or len(canonical_errors) != 2:
        failures.append("reference Bogoliubov canonical errors")
    else:
        try:
            canonical_values = tuple(float(value) for value in canonical_errors)
        except (TypeError, ValueError):
            canonical_values = (float("nan"), float("nan"))
        if not all(
            np.isfinite(value) and value < 1.0e-12 for value in canonical_values
        ):
            failures.append("reference Bogoliubov canonical errors")

    def finite_number(section: dict[str, Any], key: str, label: str) -> float | None:
        value = section.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            failures.append(label)
            return None
        if not np.isfinite(number):
            failures.append(label)
            return None
        return number

    raw_energy = finite_number(qpccsd, "energy_eh", "QPCCSD energy")
    raw_imaginary = finite_number(
        qpccsd, "energy_imaginary_eh", "QPCCSD imaginary energy"
    )
    residual = finite_number(qpccsd, "residual_norm", "QPCCSD residual")
    pav_energy = finite_number(pav, "energy_eh", "PAV energy")
    pav_imaginary = finite_number(
        pav, "energy_imaginary_eh", "PAV imaginary energy"
    )
    grid_error = finite_number(pav, "grid_error_eh", "PAV grid error")
    imaginary_error = finite_number(
        pav,
        "raw_imaginary_energy_error_eh",
        "PAV imaginary-energy validation",
    )
    if qpccsd.get("energy_convention") != "direct":
        failures.append("QPCCSD direct-energy contract")
    if pav.get("energy_convention") != "direct":
        failures.append("PAV direct-energy contract")
    if pav.get("cas_plus_delta_applied") is not False:
        failures.append("PAV CAS-plus-delta contract")
    if pav.get("projected_residual_optimized") is not False:
        failures.append("PAV fixed-amplitude contract")
    if pav.get("evaluated") is not True:
        failures.append("PAV evaluated flag")
    for key in (
        "raw_certified",
        "pav_numerically_validated",
        "certified",
        "diagnostic_only",
        "reference_finite_and_canonical",
        "fresh_terminal_unshifted_residual",
    ):
        if not isinstance(certification.get(key), bool):
            failures.append(f"certification.{key}")
    status = payload.get("status")
    raw_flag = certification.get("raw_certified")
    pav_flag = certification.get("pav_numerically_validated")
    certified_flag = certification.get("certified")
    diagnostic_flag = certification.get("diagnostic_only")
    if all(isinstance(value, bool) for value in (raw_flag, pav_flag, certified_flag)):
        if certified_flag is not bool(raw_flag and pav_flag):
            failures.append("certification aggregate consistency")
    expected_state_flags = {
        "certified": (True, False),
        "diagnostic_only": (False, True),
        "failed": (False, False),
    }
    if status in expected_state_flags and isinstance(certified_flag, bool) and isinstance(
        diagnostic_flag, bool
    ):
        expected_certified, expected_diagnostic = expected_state_flags[status]
        if certified_flag is not expected_certified:
            failures.append("status/certified consistency")
        if diagnostic_flag is not expected_diagnostic:
            failures.append("status/diagnostic consistency")
    if status == "failed" and (raw_flag is not False or pav_flag is not False):
        failures.append("failed component-certification consistency")
    canonical_tolerance = certification.get("reference_canonical_tolerance")
    if (
        isinstance(canonical_tolerance, bool)
        or not isinstance(canonical_tolerance, (int, float))
        or float(canonical_tolerance) != 1.0e-12
    ):
        failures.append("reference canonical tolerance")
    pav_from_uncertified = certification.get("pav_from_uncertified_amplitudes")
    if pav_from_uncertified is not None:
        if not isinstance(pav_from_uncertified, bool):
            failures.append("certification.pav_from_uncertified_amplitudes")
        elif pav_from_uncertified is not bool(
            pav.get("evaluated") is True and raw_flag is False
        ):
            failures.append("PAV uncertified-source consistency")

    claims_certified = certified_flag is True
    if claims_certified:
        certified_gates = {
            "certified status": payload.get("status") == "certified",
            "certified raw flag": certification.get("raw_certified") is True,
            "certified PAV flag": (
                certification.get("pav_numerically_validated") is True
            ),
            "certified diagnostic flag": certification.get("diagnostic_only") is False,
            "QPCCSD converged": qpccsd.get("converged") is True,
            "fresh terminal unshifted audit": (
                qpccsd.get("terminal_unshifted_audit") is True
            ),
            "certified reference finite/canonical flag": (
                certification.get("reference_finite_and_canonical") is True
            ),
            "certified fresh-terminal flag": (
                certification.get("fresh_terminal_unshifted_residual") is True
            ),
            "PAV validated": pav.get("validation_passed") is True,
            "QPCCSD residual gate": residual is not None and residual < 1.0e-8,
            "QPCCSD imaginary gate": (
                raw_imaginary is not None and abs(raw_imaginary) < 1.0e-8
            ),
            "PAV imaginary gate": (
                pav_imaginary is not None and abs(pav_imaginary) < 1.0e-8
            ),
            "PAV grid gate": grid_error is not None and grid_error < 1.0e-8,
            "PAV imaginary validation gate": (
                imaginary_error is not None and imaginary_error < 1.0e-8
            ),
        }
        failures.extend(label for label, passed in certified_gates.items() if not passed)
    comparison: dict[str, Any] = {}
    if anchor is not None:
        try:
            expected = anchor["expected"]
            tolerances = anchor["acceptance_tolerances"]
            tolerance = float(tolerances["energy_absolute_eh"])
            if raw_energy is None or pav_energy is None:
                raise ValueError("result energies are unavailable")
            raw_error = abs(raw_energy - float(expected["qpccsd_energy_eh"]))
            pav_error = abs(pav_energy - float(expected["pav_qpccsd_energy_eh"]))
            comparison = {
                "qpccsd_energy_error_eh": raw_error,
                "pav_energy_error_eh": pav_error,
                "energy_tolerance_eh": tolerance,
            }
            if raw_error > tolerance:
                failures.append("QPCCSD anchor energy")
            if pav_error > tolerance:
                failures.append("PAV anchor energy")
            coordinate = anchor["coordinate_contract"]
            if excitation_space.get("allowed_pairs") != coordinate["pair_count"]:
                failures.append("pair coordinate count")
            if (
                excitation_space.get("allowed_quadruples")
                != coordinate["quadruple_count"]
            ):
                failures.append("quadruple coordinate count")
            if residual is None or residual >= float(tolerances["residual_maximum"]):
                failures.append("QPCCSD residual")
            if grid_error is None or grid_error >= float(
                tolerances["pav_grid_error_maximum_eh"]
            ):
                failures.append("PAV grid validation")
            if bool(expected.get("certified", False)) and not claims_certified:
                failures.append("anchor requires a certified result")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"anchor comparison unavailable: {error}")
    return {
        "valid": not failures,
        "failures": failures,
        "comparison": comparison,
    }


def _validate(args: argparse.Namespace) -> int:
    payload = read_json(args.result)
    anchor = None if args.anchor is None else read_json(args.anchor)
    report = validate_payload(payload, anchor=anchor)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["valid"] else 2


def _show_defaults(_args: argparse.Namespace) -> int:
    config = ProductionConfig()
    payload = {
        "scientific_contract": {
            "reference_mode": "projected_agp_2rdm",
            "include_active_t1_t2": True,
            "energy_convention": "direct",
            "projection": "fixed-amplitude Ser2 PAV",
        },
        "solver": asdict(config.solver),
        "execution": asdict(config.execution),
        "projection_grid_size": config.projection_grid_size,
        "projection_validation_tolerance": config.projection_validation_tolerance,
        "project_uncertified_finite_amplitudes": (
            config.project_uncertified_finite_amplitudes
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="qpccsd",
        description="Direct projected-AGP full-space QPCCSD plus fixed-amplitude PAV",
    )
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the reviewed production workflow")
    run.add_argument("config", type=Path)
    run.add_argument("--reference-checkpoint", type=Path)
    run.add_argument("--initial-amplitudes", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--amplitudes-output", type=Path)
    run.set_defaults(handler=_run)
    validate = commands.add_parser("validate", help="validate a result contract")
    validate.add_argument("result", type=Path)
    validate.add_argument("--anchor", type=Path)
    validate.set_defaults(handler=_validate)
    defaults = commands.add_parser("show-defaults", help="show locked defaults")
    defaults.set_defaults(handler=_show_defaults)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
