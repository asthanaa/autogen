"""Production projected-AGP full-space QPCCSD followed by PN-PAV."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import platform
import sys
import time
from typing import Any

import numpy as np

from .cas_reference import build_fullspace_qp_hamiltonian
from .contracts import (
    EnergyConvention,
    PRODUCTION_EXCITATION_SPACE,
    PRODUCTION_PAIR_TRANSFER_CONVENTION,
    PRODUCTION_PROJECTION,
    PRODUCTION_RDM1_CONVENTION,
    PRODUCTION_RDM2_CONVENTION,
    PRODUCTION_REFERENCE_FIT_DIAGNOSTICS,
    PRODUCTION_REFERENCE_MODE,
    PRODUCTION_REFERENCE_PROTOCOL,
    PRODUCTION_RESULT_SCHEMA,
)
from .excitation_space import BlockQPExcitationSpace, build_symmetry_adapted_qp_space
from .models import (
    CASQPReference,
    ExecutionOptions,
    ProjectionOptions,
    QPAmplitudes,
    QPCCSDResult,
    QPPAVEvaluation,
    QPHamiltonian,
    SolverOptions,
)
from .production import evaluate_qpccsd, solve_qpccsd
from .projection import evaluate_pav_qpccsd


@dataclass(frozen=True)
class ProductionConfig:
    """Numerical controls for the immutable production scientific route."""

    solver: SolverOptions = field(default_factory=SolverOptions)
    execution: ExecutionOptions = field(
        default_factory=lambda: ExecutionOptions(
            parallel_mode="serial",
            workers=1,
            blas_threads=1,
        )
    )
    projection_grid_size: int | None = None
    projection_validation_tolerance: float = 1.0e-8
    projection_cache_bytes: int = 0
    maximum_imaginary_energy: float = 1.0e-8
    project_uncertified_finite_amplitudes: bool = True
    compute_pav_residual_diagnostic: bool = False

    def __post_init__(self) -> None:
        if self.projection_grid_size is not None and self.projection_grid_size <= 0:
            raise ValueError("projection_grid_size must be positive")
        if self.projection_validation_tolerance <= 0.0:
            raise ValueError("projection_validation_tolerance must be positive")
        if self.projection_cache_bytes < 0:
            raise ValueError("projection_cache_bytes must be non-negative")
        if self.maximum_imaginary_energy <= 0.0:
            raise ValueError("maximum_imaginary_energy must be positive")

    def projection_options(self, reference: CASQPReference) -> ProjectionOptions:
        """Return the locked fixed-amplitude Ser2 production settings."""

        grid_size = (
            reference.nspin // 2 + 1
            if self.projection_grid_size is None
            else self.projection_grid_size
        )
        return ProjectionOptions(
            target_number=reference.target_number,
            grid_size=grid_size,
            parity="even",
            gauge_quadrature="midpoint",
            ode_substeps=1,
            validation_tolerance=self.projection_validation_tolerance,
            validation_residual_tolerance=self.projection_validation_tolerance,
            cache_bytes=self.projection_cache_bytes,
            auto_select_ode=False,
            max_grid_refinements=0,
            disentanglement_backend="ser2",
        )


@dataclass
class ProductionResult:
    """Typed result for direct QPCCSD and fixed-amplitude PN-PAV."""

    reference: CASQPReference
    excitation_space: BlockQPExcitationSpace
    qpccsd: QPCCSDResult
    pav: QPPAVEvaluation | None
    raw_certified: bool
    pav_numerically_validated: bool
    certified: bool
    diagnostic_only: bool
    timings_s: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.certified:
            return "certified"
        if self.diagnostic_only:
            return "diagnostic_only"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize without ambiguous generic total-energy fields."""

        raw_energy = complex(self.qpccsd.total_energy)
        pav_energy = None if self.pav is None else complex(self.pav.total_energy)
        space_diagnostics = self.excitation_space.diagnostics()
        pav_diagnostics = {} if self.pav is None else self.pav.diagnostics
        return _json_value(
            {
                "schema": PRODUCTION_RESULT_SCHEMA,
                "status": self.status,
                "method": {
                    "reference": {
                        "mode": PRODUCTION_REFERENCE_MODE,
                        "protocol": self.reference.source_rdm_metadata[
                            "reference_protocol"
                        ],
                        "complete_active_rdm2_reconstructed": False,
                    },
                    "excitation_space": {
                        "mode": PRODUCTION_EXCITATION_SPACE,
                        "include_active_t1_t2": True,
                        "symmetry": "total-singlet and totally symmetric",
                    },
                    "energy": {
                        "convention": EnergyConvention.DIRECT.value,
                        "definition": EnergyConvention.DIRECT.description,
                        "casscf_energy_added": False,
                        "cas_plus_delta_applied": False,
                    },
                    "projection": {
                        "mode": PRODUCTION_PROJECTION,
                        "workflow": "projection after variation",
                        "amplitudes_fixed": True,
                        "projected_residual_optimized": False,
                        "disentanglement_backend": "ser2",
                        "closure": "W1/W2 with W3=0",
                        "gauge_quadrature": "midpoint",
                        "parity": "even",
                        "doubled_grid_validation": True,
                    },
                },
                "system": {
                    "basis": self.reference.metadata.get("basis"),
                    "atom": self.reference.metadata.get("atom"),
                    "nspin": self.reference.nspin,
                    "correlated_target_number": self.reference.target_number,
                    "physical_target_number": self.reference.physical_target_number,
                    "frozen_spatial_indices": self.reference.frozen_spatial_indices,
                    "active_spatial_indices": self.reference.active_spatial_indices,
                },
                "reference": {
                    "reference_mode": self.reference.reference_mode,
                    "casscf_energy_diagnostic_eh": self.reference.casscf_energy,
                    "active_natural_occupations": self.reference.spatial_occupations[
                        list(self.reference.active_spatial_indices)
                    ],
                    "relative_signed_geminals": self.reference.bogoliubov.metadata[
                        "relative_signed_geminals"
                    ],
                    "scaled_signed_geminals": self.reference.signed_geminals,
                    "global_number_setting_scale": self.reference.bogoliubov.metadata[
                        "global_number_scale"
                    ],
                    "array_fingerprints_sha256": {
                        "active_rdm1": _array_sha256(self.reference.active_rdm1),
                        "active_rdm2": _array_sha256(self.reference.active_rdm2),
                        "mo_coeff": _array_sha256(self.reference.mo_coeff),
                    },
                    "reconstruction_metrics": self.reference.reconstruction_metrics,
                    "source_rdm_metadata": self.reference.source_rdm_metadata,
                    "bogoliubov_canonical_errors": (
                        self.reference.bogoliubov.canonical_errors()
                    ),
                },
                "excitation_space": {
                    **space_diagnostics,
                    "allowed_pairs": self.excitation_space.pair_count,
                    "allowed_quadruples": self.excitation_space.quadruple_count,
                    "active_pair_coordinates": self.excitation_space.pair_blocks.count(
                        "xy"
                    ),
                    "active_quadruple_coordinates": (
                        self.excitation_space.quadruple_blocks.count("xyzw")
                    ),
                    "internal_coordinates_excluded": (
                        self.excitation_space.internal_coordinate_count
                    ),
                },
                "energies_eh": {
                    "qpccsd": raw_energy.real,
                    "qpccsd_imaginary": raw_energy.imag,
                    "pav_qpccsd": None if pav_energy is None else pav_energy.real,
                    "pav_qpccsd_imaginary": (
                        None if pav_energy is None else pav_energy.imag
                    ),
                },
                "qpccsd": {
                    "energy_eh": raw_energy.real,
                    "energy_imaginary_eh": raw_energy.imag,
                    "converged": self.qpccsd.converged,
                    "residual_norm": self.qpccsd.residual_norm,
                    "iterations": self.qpccsd.iterations,
                    "evaluator_calls": self.qpccsd.diagnostics.get("evaluator_calls"),
                    "energy_convention": self.qpccsd.diagnostics.get(
                        "energy_convention"
                    ),
                    "terminal_unshifted_audit": self.qpccsd.diagnostics.get(
                        "terminal_unshifted_audit"
                    ),
                    "terminal_unshifted_energy_difference_eh": (
                        self.qpccsd.diagnostics.get(
                            "terminal_unshifted_energy_difference"
                        )
                    ),
                },
                "pav": {
                    "evaluated": self.pav is not None,
                    "energy_eh": None if pav_energy is None else pav_energy.real,
                    "energy_imaginary_eh": (
                        None if pav_energy is None else pav_energy.imag
                    ),
                    "validation_passed": (
                        False if self.pav is None else self.pav.validation_passed
                    ),
                    "grid_error_eh": None if self.pav is None else self.pav.grid_error,
                    "raw_grid_error_eh": (
                        None if self.pav is None else self.pav.raw_grid_error
                    ),
                    "raw_imaginary_energy_error_eh": (
                        None
                        if self.pav is None
                        else self.pav.raw_imaginary_energy_error
                    ),
                    "baseline_grid_size": pav_diagnostics.get("baseline_grid_size"),
                    "validation_grid_size": pav_diagnostics.get(
                        "validation_grid_size"
                    ),
                    "disentanglement_backend": pav_diagnostics.get(
                        "disentanglement_backend"
                    ),
                    "gauge_quadrature": pav_diagnostics.get("gauge_quadrature"),
                    "projected_residual_optimized": False,
                    "amplitudes_fixed": True,
                    "energy_convention": pav_diagnostics.get("energy_convention"),
                    "cas_plus_delta_applied": pav_diagnostics.get(
                        "cas_plus_delta_applied"
                    ),
                },
                "certification": {
                    "raw_certified": self.raw_certified,
                    "pav_numerically_validated": self.pav_numerically_validated,
                    "certified": self.certified,
                    "diagnostic_only": self.diagnostic_only,
                    "pav_from_uncertified_amplitudes": bool(
                        self.pav is not None and not self.raw_certified
                    ),
                    "reference_finite_and_canonical": True,
                    "reference_canonical_tolerance": 1.0e-12,
                    "fresh_terminal_unshifted_residual": True,
                },
                "timings_s": self.timings_s,
                "provenance": self.provenance,
            }
        )


def _array_sha256(values: np.ndarray) -> str:
    """Fingerprint a real array in a platform-independent binary convention."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    shape = np.asarray(array.shape, dtype="<i8")
    digest = sha256()
    digest.update(b"qpccsd-real-array-v1\0")
    digest.update(shape.tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_checkpoint_request(
    continuation: Any,
    *,
    molecule: str,
    distance_angstrom: float,
    basis: str,
    cas_norb: int,
    cas_nelec: int,
    frozen_n1s: bool,
) -> None:
    """Prove that a portable continuation is the requested molecular problem."""

    checkpoint_molecule = continuation.molecule
    reference = continuation.reference
    if int(checkpoint_molecule.charge) != 0 or int(checkpoint_molecule.spin) != 0:
        raise ValueError("checkpoint is not a neutral singlet")
    if not bool(checkpoint_molecule.symmetry):
        raise ValueError("checkpoint was not built with molecular symmetry")

    charges = tuple(sorted(int(value) for value in checkpoint_molecule.atom_charges()))
    expected_charges = (7, 7) if molecule == "n2" else (1, 1)
    if charges != expected_charges:
        raise ValueError(
            f"checkpoint molecule mismatch: expected {molecule}, atomic charges={charges}"
        )
    coordinates = np.asarray(
        checkpoint_molecule.atom_coords(unit="Angstrom"), dtype=float
    )
    if coordinates.shape != (2, 3):
        raise ValueError("checkpoint molecular helper requires a diatomic geometry")
    checkpoint_distance = float(np.linalg.norm(coordinates[1] - coordinates[0]))
    if not np.isclose(
        checkpoint_distance,
        float(distance_angstrom),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError(
            "checkpoint geometry mismatch: "
            f"requested {distance_angstrom:.12g} Angstrom, "
            f"checkpoint {checkpoint_distance:.12g} Angstrom"
        )

    checkpoint_basis = checkpoint_molecule.basis
    if not isinstance(checkpoint_basis, str):
        raise ValueError("checkpoint does not retain a comparable molecular basis name")
    normalized_basis = checkpoint_basis.strip().casefold().replace(" ", "")
    requested_basis = str(basis).strip().casefold().replace(" ", "")
    if normalized_basis != requested_basis:
        raise ValueError(
            f"checkpoint basis mismatch: requested {basis!r}, checkpoint {checkpoint_basis!r}"
        )

    active_orbitals = len(reference.active_spatial_indices)
    if active_orbitals != int(cas_norb):
        raise ValueError(
            "checkpoint CAS orbital-count mismatch: "
            f"requested {cas_norb}, checkpoint {active_orbitals}"
        )
    active_electrons = float(np.trace(reference.active_rdm1))
    if not np.isclose(active_electrons, float(cas_nelec), atol=1.0e-8, rtol=0.0):
        raise ValueError(
            "checkpoint CAS electron-count mismatch: "
            f"requested {cas_nelec}, checkpoint trace {active_electrons:.12g}"
        )

    expected_frozen = (0, 1) if molecule == "n2" and frozen_n1s else ()
    actual_frozen = tuple(int(index) for index in reference.frozen_spatial_indices)
    if actual_frozen != expected_frozen:
        raise ValueError(
            "checkpoint frozen-core mismatch: "
            f"requested frozen_n1s={frozen_n1s}, checkpoint indices={actual_frozen}"
        )
    physical_electrons = int(checkpoint_molecule.nelectron)
    correlated_electrons = physical_electrons - 2 * len(expected_frozen)
    if reference.physical_target_number != physical_electrons:
        raise ValueError("checkpoint physical particle target is inconsistent")
    if reference.target_number != correlated_electrons:
        raise ValueError("checkpoint correlated particle target is inconsistent")


def prepare_projected_agp_reference(
    *,
    molecule: str = "n2",
    distance_angstrom: float | None = None,
    basis: str = "sto-3g",
    cas_norb: int = 6,
    cas_nelec: int = 6,
    frozen_n1s: bool = True,
    checkpoint: str | None = None,
    verbose: int = 0,
) -> CASQPReference:
    """Build or restore the production signed projected-AGP reference."""

    from .benchmarks.pyscf_cas import (
        build_h2_casscf_qp_reference,
        build_n2_casscf_qp_reference,
        load_casscf_continuation,
        restore_continuation_integrals,
    )

    if distance_angstrom is None:
        raise ValueError("distance_angstrom is required, including for a checkpoint")
    if distance_angstrom <= 0.0:
        raise ValueError("distance_angstrom must be positive")
    key = molecule.replace("_", "").replace("-", "").lower()
    if key not in {"n2", "h2"}:
        raise ValueError("the production molecular helper supports n2 or h2")
    if key == "h2" and (cas_norb != 2 or cas_nelec != 2 or frozen_n1s):
        raise ValueError("the H2 helper requires CAS(2,2) and frozen_n1s=False")
    if checkpoint is not None:
        continuation = load_casscf_continuation(checkpoint)
        _validate_production_reference(continuation.reference)
        _validate_checkpoint_request(
            continuation,
            molecule=key,
            distance_angstrom=distance_angstrom,
            basis=basis,
            cas_norb=cas_norb,
            cas_nelec=cas_nelec,
            frozen_n1s=frozen_n1s,
        )
        reference = restore_continuation_integrals(continuation)
    else:
        if key == "n2":
            reference = build_n2_casscf_qp_reference(
                distance_angstrom,
                basis=basis,
                cas_norb=cas_norb,
                cas_nelec=cas_nelec,
                frozen_n1s=frozen_n1s,
                reference_mode=PRODUCTION_REFERENCE_MODE,
                verbose=verbose,
            )
        elif key == "h2":
            reference = build_h2_casscf_qp_reference(
                distance_angstrom,
                basis=basis,
                reference_mode=PRODUCTION_REFERENCE_MODE,
                verbose=verbose,
            )
    _validate_production_reference(reference)
    return reference


def build_full_active_qp_space(
    reference: CASQPReference,
) -> BlockQPExcitationSpace:
    """Build all symmetry-allowed coordinates, including pure-active T1/T2."""

    _validate_production_reference(reference)
    space = build_symmetry_adapted_qp_space(
        reference,
        include_active_t1_t2=True,
    )
    if not space.include_active_t1_t2 or space.internal_coordinate_count:
        raise RuntimeError("production excitation space unexpectedly masks active amplitudes")
    return space


def solve_direct_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: CASQPReference,
    *,
    config: ProductionConfig | None = None,
    excitation_space: BlockQPExcitationSpace | None = None,
    initial_amplitudes: QPAmplitudes | None = None,
) -> QPCCSDResult:
    """Solve the direct-energy full-active QPCCSD equations."""

    config = ProductionConfig() if config is None else config
    space = (
        build_full_active_qp_space(reference)
        if excitation_space is None
        else excitation_space
    )
    _validate_production_space(space)
    result = solve_qpccsd(
        hamiltonian,
        reference,
        initial_amplitudes=initial_amplitudes,
        options=config.solver,
        excitation_space=space,
        energy_convention=EnergyConvention.DIRECT,
    )
    if result.diagnostics.get("cas_plus_delta_applied") is not False:
        raise RuntimeError("production QPCCSD violated the direct-energy contract")
    return result


def evaluate_direct_pav(
    hamiltonian: QPHamiltonian,
    reference: CASQPReference,
    amplitudes: QPAmplitudes,
    *,
    config: ProductionConfig | None = None,
    excitation_space: BlockQPExcitationSpace | None = None,
) -> QPPAVEvaluation:
    """Apply fixed-amplitude Ser2 PAV with doubled-grid validation."""

    config = ProductionConfig() if config is None else config
    space = (
        build_full_active_qp_space(reference)
        if excitation_space is None
        else excitation_space
    )
    _validate_production_space(space)
    before_t1 = np.asarray(amplitudes.t1).copy()
    before_t2 = np.asarray(amplitudes.t2).copy()
    result = evaluate_pav_qpccsd(
        hamiltonian,
        reference,
        amplitudes,
        options=config.projection_options(reference),
        excitation_space=space,
        execution_options=config.execution,
        compute_residual_diagnostic=config.compute_pav_residual_diagnostic,
        energy_convention=EnergyConvention.DIRECT,
    )
    if not np.array_equal(before_t1, amplitudes.t1) or not np.array_equal(
        before_t2, amplitudes.t2
    ):
        raise RuntimeError("PAV modified the converged QPCCSD amplitudes")
    required = {
        "disentanglement_backend": "ser2",
        "gauge_quadrature": "midpoint",
        "projected_residual_optimized": False,
        "cas_plus_delta_applied": False,
    }
    mismatches = {
        key: (result.diagnostics.get(key), expected)
        for key, expected in required.items()
        if result.diagnostics.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"production PAV contract mismatch: {mismatches}")
    return result


def run_qpccsd_pav(
    reference: CASQPReference,
    *,
    hamiltonian: QPHamiltonian | None = None,
    config: ProductionConfig | None = None,
    initial_amplitudes: QPAmplitudes | None = None,
    provenance: dict[str, Any] | None = None,
) -> ProductionResult:
    """Run the immutable direct QPCCSD then fixed-amplitude PAV workflow."""

    config = ProductionConfig() if config is None else config
    _validate_production_reference(reference)
    started = time.perf_counter()
    hamiltonian_started = time.perf_counter()
    hamiltonian = (
        build_fullspace_qp_hamiltonian(reference)
        if hamiltonian is None
        else hamiltonian
    )
    hamiltonian_seconds = time.perf_counter() - hamiltonian_started
    space = build_full_active_qp_space(reference)
    solve_started = time.perf_counter()
    qpccsd = solve_direct_qpccsd(
        hamiltonian,
        reference,
        config=config,
        excitation_space=space,
        initial_amplitudes=initial_amplitudes,
    )
    solve_seconds = time.perf_counter() - solve_started
    audit_started = time.perf_counter()
    terminal = evaluate_qpccsd(
        hamiltonian,
        qpccsd.amplitudes,
        excitation_space=space,
    )
    audit_seconds = time.perf_counter() - audit_started
    solver_energy = complex(qpccsd.total_energy)
    terminal_energy = complex(terminal.total_energy)
    terminal_energy_difference = float(abs(terminal_energy - solver_energy))
    qpccsd.diagnostics.update(
        {
            "terminal_unshifted_audit": True,
            "solver_reported_energy_before_terminal_audit": np.real_if_close(
                solver_energy
            ),
            "solver_reported_residual_before_terminal_audit": qpccsd.residual_norm,
            "terminal_unshifted_energy_difference": terminal_energy_difference,
            "terminal_unshifted_residual_norm": terminal.residual_norm,
        }
    )
    qpccsd.total_energy = np.real_if_close(terminal.total_energy)
    qpccsd.correlation_energy = np.real_if_close(terminal.correlation_energy)
    qpccsd.residual_norm = terminal.residual_norm
    raw_energy = complex(qpccsd.total_energy)
    raw_finite = bool(
        np.isfinite(raw_energy.real)
        and np.isfinite(raw_energy.imag)
        and np.isfinite(qpccsd.residual_norm)
    )
    raw_certified = bool(
        raw_finite
        and qpccsd.converged
        and qpccsd.residual_norm < config.solver.residual_tolerance
        and terminal_energy_difference < config.solver.energy_tolerance
        and abs(raw_energy.imag) < config.maximum_imaginary_energy
    )
    pav = None
    pav_seconds = 0.0
    if raw_finite and (
        raw_certified or config.project_uncertified_finite_amplitudes
    ):
        pav_started = time.perf_counter()
        pav = evaluate_direct_pav(
            hamiltonian,
            reference,
            qpccsd.amplitudes,
            config=config,
            excitation_space=space,
        )
        pav_seconds = time.perf_counter() - pav_started
    pav_energy = None if pav is None else complex(pav.total_energy)
    pav_numerically_validated = bool(
        pav is not None
        and pav.validation_passed
        and pav_energy is not None
        and abs(pav_energy.imag) < config.maximum_imaginary_energy
        and (
            pav.raw_imaginary_energy_error is None
            or pav.raw_imaginary_energy_error < config.maximum_imaginary_energy
        )
    )
    certified = bool(raw_certified and pav_numerically_validated)
    run_provenance = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hostname": platform.node(),
        **({} if provenance is None else provenance),
    }
    return ProductionResult(
        reference=reference,
        excitation_space=space,
        qpccsd=qpccsd,
        pav=pav,
        raw_certified=raw_certified,
        pav_numerically_validated=pav_numerically_validated,
        certified=certified,
        diagnostic_only=bool(not certified and raw_finite),
        timings_s={
            "build_hamiltonian": hamiltonian_seconds,
            "solve_qpccsd": solve_seconds,
            "terminal_unshifted_audit": audit_seconds,
            "evaluate_pav": pav_seconds,
            "total": time.perf_counter() - started,
        },
        provenance=run_provenance,
    )


def _validate_production_reference(reference: CASQPReference) -> None:
    if not isinstance(reference, CASQPReference):
        raise TypeError("production workflow requires a CASQPReference")
    if reference.reference_mode != PRODUCTION_REFERENCE_MODE:
        raise ValueError(
            "production workflow requires reference_mode='projected_agp_2rdm'"
        )
    if reference.signed_geminals is None:
        raise ValueError("projected-AGP reference lacks signed active geminals")
    finite_arrays = {
        "molecular orbital coefficients": reference.mo_coeff,
        "spatial occupations": reference.spatial_occupations,
        "active one-body RDM": reference.active_rdm1,
        "active two-body RDM": reference.active_rdm2,
        "signed geminals": reference.signed_geminals,
        "Bogoliubov U": reference.bogoliubov.U,
        "Bogoliubov V": reference.bogoliubov.V,
    }
    for label, values in finite_arrays.items():
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError(f"production reference has nonfinite {label}")
    if not np.isfinite(reference.casscf_energy):
        raise ValueError("production reference has a nonfinite CASSCF diagnostic")

    expected_source = {
        "rdm1": PRODUCTION_RDM1_CONVENTION,
        "rdm2": PRODUCTION_RDM2_CONVENTION,
        "pair_transfer_block": PRODUCTION_PAIR_TRANSFER_CONVENTION,
        "reference_protocol": PRODUCTION_REFERENCE_PROTOCOL,
        "complete_active_rdm2_reconstructed": False,
    }
    for key, expected in expected_source.items():
        if reference.source_rdm_metadata.get(key) != expected:
            raise ValueError(
                f"production reference source-RDM contract mismatch for {key!r}"
            )

    metrics = reference.reconstruction_metrics
    missing_metrics = set(PRODUCTION_REFERENCE_FIT_DIAGNOSTICS).difference(metrics)
    if missing_metrics:
        raise ValueError(
            "production reference lacks fit diagnostics: "
            + ", ".join(sorted(missing_metrics))
        )
    if metrics["fit_success"] is not True:
        raise ValueError("production projected-AGP fit was not successful")
    if not isinstance(metrics["fit_message"], str) or not metrics["fit_message"].strip():
        raise ValueError("production projected-AGP fit lacks a diagnostic message")
    nonnegative_metrics = (
        "fit_cost",
        "rdm1_max_abs_error",
        "pair_rdm2_max_abs_error",
        "active_average_number_error",
    )
    for key in nonnegative_metrics:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"production reference has invalid fit diagnostic {key!r}")
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"production reference has invalid fit diagnostic {key!r}")
    fidelity = metrics["pair_subspace_fidelity"]
    if isinstance(fidelity, bool) or not isinstance(fidelity, (int, float, np.number)):
        raise ValueError("production reference has invalid pair-subspace fidelity")
    if not np.isfinite(float(fidelity)) or not -1.0e-12 <= float(fidelity) <= 1.0 + 1.0e-12:
        raise ValueError("production reference has invalid pair-subspace fidelity")

    active_count = len(reference.active_spatial_indices)
    relative = np.asarray(
        reference.bogoliubov.metadata.get("relative_signed_geminals", ()),
        dtype=float,
    )
    if relative.shape != (active_count,) or not np.all(np.isfinite(relative)):
        raise ValueError("production reference lacks finite relative signed geminals")
    natural_occupations = np.asarray(
        reference.bogoliubov.metadata.get("active_natural_occupations", ()),
        dtype=float,
    )
    expected_occupations = reference.spatial_occupations[
        list(reference.active_spatial_indices)
    ]
    if natural_occupations.shape != (active_count,) or not np.allclose(
        natural_occupations,
        expected_occupations,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("production reference active natural occupations are inconsistent")
    number_scale = reference.bogoliubov.metadata.get("global_number_scale")
    if (
        isinstance(number_scale, bool)
        or not isinstance(number_scale, (int, float, np.number))
        or not np.isfinite(float(number_scale))
        or float(number_scale) < 0.0
    ):
        raise ValueError("production reference lacks a finite global number-setting scale")
    normal_error, anomalous_error = reference.bogoliubov.canonical_errors()
    if normal_error >= 1.0e-12 or anomalous_error >= 1.0e-12:
        raise ValueError(
            "production reference fails Bogoliubov canonicality: "
            f"normal={normal_error:.3e}, anomalous={anomalous_error:.3e}"
        )


def _validate_production_space(space: BlockQPExcitationSpace) -> None:
    if not isinstance(space, BlockQPExcitationSpace):
        raise TypeError("production workflow requires the block spin-adapted space")
    if not space.include_active_t1_t2:
        raise ValueError("production workflow requires pure-active T1/T2 coordinates")
    if space.internal_coordinate_count:
        raise ValueError("production workflow cannot mask active coordinates")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ProductionConfig",
    "ProductionResult",
    "build_full_active_qp_space",
    "evaluate_direct_pav",
    "prepare_projected_agp_reference",
    "run_qpccsd_pav",
    "solve_direct_qpccsd",
]
