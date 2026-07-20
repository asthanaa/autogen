from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
import math
import time
from typing import Callable

import numpy as np

from ..generated import generated_wick_orbits as generated_wick
from .contracts import EnergyConvention, normalize_energy_convention
from .excitation_space import (
    BlockQPExcitationSpace,
    QPExcitationSpace,
    QPExcitationSpaceLike,
    build_block_spin_adapted_qp_space,
)
from .models import (
    BogoliubovReference,
    CASQPReference,
    GaugeModeOptions,
    KernelEvaluation,
    KernelDirectionalDerivative,
    ProjectedResidualBasis,
    QPAmplitudes,
    QPCCSDResult,
    QPHamiltonian,
    SolverOptions,
)


IterationCallback = Callable[[QPAmplitudes, dict[str, object]], None]


def _projector_active(projector: object | None) -> bool:
    if projector is None:
        return False
    active = getattr(projector, "active", None)
    if active is not None:
        return bool(active)
    return bool(getattr(projector, "gauge_rank", 0))


def pair_keys(nspin: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(nspin), 2))


def quadruple_keys(nspin: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(combinations(range(nspin), 4))


@lru_cache(maxsize=None)
def _full_excitation_space(nspin: int) -> QPExcitationSpace:
    return QPExcitationSpace.full(nspin)


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


_QUAD_PERMUTATIONS = tuple(
    (permutation, _permutation_sign(permutation)) for permutation in permutations(range(4))
)


def amplitudes_from_vector(
    vector: np.ndarray,
    nspin: int,
    excitation_space: QPExcitationSpaceLike | None = None,
) -> QPAmplitudes:
    if excitation_space is not None:
        if excitation_space.nspin != nspin:
            raise ValueError("excitation space and amplitude dimensions differ")
        return excitation_space.unpack(vector)
    vector = np.asarray(vector)
    pairs = pair_keys(nspin)
    quads = quadruple_keys(nspin)
    if vector.shape != (len(pairs) + len(quads),):
        raise ValueError("independent amplitude vector has an incompatible shape")
    t1 = np.zeros((nspin, nspin), dtype=vector.dtype)
    t2 = np.zeros((nspin,) * 4, dtype=vector.dtype)
    for value, (p, q) in zip(vector[: len(pairs)], pairs):
        t1[p, q] = value
        t1[q, p] = -value
    for value, key in zip(vector[len(pairs) :], quads):
        for permutation, sign in _QUAD_PERMUTATIONS:
            indices = tuple(key[index] for index in permutation)
            t2[indices] = sign * value
    return QPAmplitudes(t1=t1, t2=t2)


def amplitudes_to_vector(
    amplitudes: QPAmplitudes,
    excitation_space: QPExcitationSpaceLike | None = None,
) -> np.ndarray:
    if excitation_space is not None:
        return excitation_space.pack(amplitudes)
    nspin = amplitudes.t1.shape[0]
    pair_values = [amplitudes.t1[key] for key in pair_keys(nspin)]
    quad_values = [amplitudes.t2[key] for key in quadruple_keys(nspin)]
    return np.asarray(pair_values + quad_values)


def residual_to_vector(
    r1: np.ndarray,
    r2: np.ndarray,
    excitation_space: QPExcitationSpaceLike | None = None,
) -> np.ndarray:
    if excitation_space is not None:
        return excitation_space.residual_vector(r1, r2)
    nspin = r1.shape[0]
    pair_values = [r1[key] for key in pair_keys(nspin)]
    quad_values = [r2[key] for key in quadruple_keys(nspin)]
    return np.asarray(pair_values + quad_values)


def _kernel_arguments(hamiltonian: QPHamiltonian, amplitudes: QPAmplitudes) -> tuple[np.ndarray, ...]:
    return (
        amplitudes.t1,
        amplitudes.t2,
        hamiltonian.h02,
        hamiltonian.h04,
        hamiltonian.h11,
        hamiltonian.h13,
        hamiltonian.h20,
        hamiltonian.h22,
        hamiltonian.h31,
        hamiltonian.h40,
    )


def evaluate_qpccsd(
    hamiltonian: QPHamiltonian,
    amplitudes: QPAmplitudes,
    *,
    excitation_space: QPExcitationSpaceLike | None = None,
) -> KernelEvaluation:
    """Evaluate the fourth-order connected BCH energy and residual tensors."""

    if amplitudes.t1.shape != (hamiltonian.nspin, hamiltonian.nspin):
        raise ValueError("amplitude and Hamiltonian dimensions do not match")
    started = time.perf_counter()
    space = _full_excitation_space(hamiltonian.nspin) if excitation_space is None else excitation_space
    if space.nspin != hamiltonian.nspin:
        raise ValueError("excitation space and Hamiltonian dimensions do not match")
    zero_amplitude_fast_path = not np.any(amplitudes.t1) and not np.any(amplitudes.t2)
    if zero_amplitude_fast_path:
        # The connected BCH kernel terminates at its bare H20/H40 blocks when
        # T=0. Avoiding all nonlinear contractions is especially important for
        # projected CAS baselines, which evaluate this identity at every angle.
        correlation_energy = 0.0 + 0.0j
        r1 = np.array(hamiltonian.h20, copy=True)
        r2 = np.array(hamiltonian.h40, copy=True)
    else:
        outputs = generated_wick.compute_outputs(*_kernel_arguments(hamiltonian, amplitudes))
        correlation_energy = complex(outputs["energy"])
        r1 = np.asarray(outputs["r1"])
        r2 = np.asarray(outputs["r2"])
    residual = residual_to_vector(r1, r2, space)
    residual_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
    max_imaginary = float(np.max(np.abs(np.imag(residual)))) if residual.size else 0.0
    return KernelEvaluation(
        total_energy=hamiltonian.constant + correlation_energy,
        correlation_energy=correlation_energy,
        r1=np.real_if_close(r1),
        r2=np.real_if_close(r2),
        residual_norm=residual_norm,
        elapsed=time.perf_counter() - started,
        diagnostics={
            "max_imaginary_residual": max_imaginary,
            "contraction_backend": generated_wick.CONTRACTION_BACKEND,
            "contraction_count": generated_wick.CONTRACTION_COUNT,
            "max_formal_scaling": generated_wick.MAX_FORMAL_SCALING,
            "zero_amplitude_fast_path": zero_amplitude_fast_path,
            "internal_residual_norm": space.internal_residual_norm(r1, r2),
            "allowed_pair_count": space.pair_count,
            "allowed_quadruple_count": space.quadruple_count,
            "internal_coordinate_count": space.internal_coordinate_count,
            "forbidden_amplitude_norm": space.forbidden_amplitude_norm(amplitudes),
            **space.diagnostics(),
        },
    )


def evaluate_qpccsd_energy(
    hamiltonian: QPHamiltonian,
    amplitudes: QPAmplitudes,
) -> complex:
    """Evaluate only the connected BCH scalar using its generated Wick kernel."""

    if amplitudes.t1.shape != (hamiltonian.nspin, hamiltonian.nspin):
        raise ValueError("amplitude and Hamiltonian dimensions do not match")
    if amplitudes.t2.shape != (hamiltonian.nspin,) * 4:
        raise ValueError("amplitude and Hamiltonian dimensions do not match")
    if not np.any(amplitudes.t1) and not np.any(amplitudes.t2):
        return complex(hamiltonian.constant)
    correlation_energy = generated_wick.compute_energy(
        *_kernel_arguments(hamiltonian, amplitudes)
    )
    return complex(hamiltonian.constant) + complex(correlation_energy)


def evaluate_qpccsd_jvp(
    hamiltonian: QPHamiltonian,
    amplitudes: QPAmplitudes,
    direction: QPAmplitudes,
) -> KernelDirectionalDerivative:
    """Evaluate the exact directional derivative of the generated BCH kernel."""

    _evaluation, derivative = evaluate_qpccsd_with_jvp(
        hamiltonian,
        amplitudes,
        direction,
    )
    return derivative


def evaluate_qpccsd_with_jvp(
    hamiltonian: QPHamiltonian,
    amplitudes: QPAmplitudes,
    direction: QPAmplitudes,
    *,
    excitation_space: QPExcitationSpaceLike | None = None,
) -> tuple[KernelEvaluation, KernelDirectionalDerivative]:
    """Evaluate the generated BCH kernel and its exact directional derivative."""

    expected_pair = (hamiltonian.nspin, hamiltonian.nspin)
    expected_quad = (hamiltonian.nspin,) * 4
    if amplitudes.t1.shape != expected_pair or direction.t1.shape != expected_pair:
        raise ValueError("amplitude and direction pair dimensions do not match")
    if amplitudes.t2.shape != expected_quad or direction.t2.shape != expected_quad:
        raise ValueError("amplitude and direction quadruple dimensions do not match")
    started = time.perf_counter()
    space = (
        _full_excitation_space(hamiltonian.nspin)
        if excitation_space is None
        else excitation_space
    )
    arguments = _kernel_arguments(hamiltonian, amplitudes)
    outputs = generated_wick.compute_outputs_and_jvp(
        amplitudes.t1,
        amplitudes.t2,
        direction.t1,
        direction.t2,
        *arguments[2:],
    )
    elapsed = time.perf_counter() - started
    correlation_energy = complex(outputs["energy"])
    r1 = np.asarray(outputs["r1"])
    r2 = np.asarray(outputs["r2"])
    residual = residual_to_vector(r1, r2, space)
    evaluation = KernelEvaluation(
        total_energy=hamiltonian.constant + correlation_energy,
        correlation_energy=correlation_energy,
        r1=r1,
        r2=r2,
        residual_norm=float(np.max(np.abs(residual))) if residual.size else 0.0,
        elapsed=elapsed,
        diagnostics={
            "contraction_backend": generated_wick.CONTRACTION_BACKEND,
        },
    )
    derivative_energy = complex(outputs["energy_jvp"])
    derivative = KernelDirectionalDerivative(
        total_energy=derivative_energy,
        correlation_energy=derivative_energy,
        r1=np.asarray(outputs["r1_jvp"]),
        r2=np.asarray(outputs["r2_jvp"]),
        elapsed=elapsed,
        diagnostics={
            "derivative": "analytic-generated-wick-jvp",
            "contraction_backend": generated_wick.CONTRACTION_BACKEND,
        },
    )
    return evaluation, derivative


class _LimitedMemoryMultisecantBroyden:
    def __init__(self, inverse_diagonal: np.ndarray, history_size: int) -> None:
        self.inverse_diagonal = np.asarray(inverse_diagonal, dtype=float)
        self.history_size = max(0, int(history_size))
        self.steps: list[np.ndarray] = []
        self.residual_changes: list[np.ndarray] = []

    def _multisecant_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not self.steps:
            return None
        steps = np.column_stack(self.steps)
        changes = np.column_stack(self.residual_changes)
        correction = steps - self.inverse_diagonal[:, None] * changes
        gram = changes.T @ changes
        count = gram.shape[0]
        regularization = 1.0e-12 * max(
            float(np.trace(gram)) / max(count, 1),
            1.0e-30,
        )
        gram.flat[:: count + 1] += regularization
        return changes, correction, gram

    def apply(self, vector: np.ndarray) -> np.ndarray:
        result = self.inverse_diagonal * vector
        matrices = self._multisecant_matrices()
        if matrices is None:
            return result
        changes, correction, gram = matrices
        try:
            coefficients = np.linalg.solve(gram, changes.T @ vector)
        except np.linalg.LinAlgError:
            return result
        return result + correction @ coefficients

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        result = self.inverse_diagonal * vector
        matrices = self._multisecant_matrices()
        if matrices is None:
            return result
        changes, correction, gram = matrices
        try:
            coefficients = np.linalg.solve(gram, correction.T @ vector)
        except np.linalg.LinAlgError:
            return result
        return result + changes @ coefficients

    def apply_complex(self, vector: np.ndarray) -> np.ndarray:
        """Apply a complex-linear multisecant inverse.

        The nonlinear optimizer stores complex amplitudes as a doubled real
        vector.  Applying that unrestricted real update directly inside complex
        GMRES is invalid because it need not commute with multiplication by
        ``1j``.  Reconstructing complex secants and using Hermitian products
        preserves the holomorphic Krylov problem.
        """

        vector = np.asarray(vector, dtype=np.complex128)
        coordinate_count = vector.size
        if self.inverse_diagonal.size != 2 * coordinate_count:
            raise ValueError("complex multisecant dimensions do not match")
        inverse_diagonal = self.inverse_diagonal[:coordinate_count]
        result = inverse_diagonal * vector
        if not self.steps:
            return result
        steps = np.column_stack(
            [
                step[:coordinate_count] + 1.0j * step[coordinate_count:]
                for step in self.steps
            ]
        )
        changes = np.column_stack(
            [
                change[:coordinate_count] + 1.0j * change[coordinate_count:]
                for change in self.residual_changes
            ]
        )
        correction = steps - inverse_diagonal[:, None] * changes
        gram = changes.conj().T @ changes
        count = gram.shape[0]
        regularization = 1.0e-12 * max(
            float(np.real(np.trace(gram))) / max(count, 1),
            1.0e-30,
        )
        gram.flat[:: count + 1] += regularization
        try:
            coefficients = np.linalg.solve(gram, changes.conj().T @ vector)
        except np.linalg.LinAlgError:
            return result
        return result + correction @ coefficients

    def update(self, step: np.ndarray, residual_change: np.ndarray) -> bool:
        step = np.asarray(step, dtype=float)
        residual_change = np.asarray(residual_change, dtype=float)
        scale = max(1.0, float(np.linalg.norm(step)))
        if (
            not np.all(np.isfinite(step))
            or not np.all(np.isfinite(residual_change))
            or np.linalg.norm(residual_change) < 1.0e-14 * scale
        ):
            return False
        self.steps.append(step.copy())
        self.residual_changes.append(residual_change.copy())
        if len(self.steps) > self.history_size:
            self.steps.pop(0)
            self.residual_changes.pop(0)
        return True

    def update_block_scaling(
        self,
        step: np.ndarray,
        residual_change: np.ndarray,
        groups: tuple[np.ndarray, ...],
        *,
        minimum_scale: float,
        maximum_scale: float,
    ) -> tuple[float, ...]:
        """Rescale diagonal inverse-Jacobian blocks from one accepted secant."""

        step = np.asarray(step, dtype=float)
        residual_change = np.asarray(residual_change, dtype=float)
        factors: list[float] = []
        for group in groups:
            indices = np.asarray(group, dtype=np.int64)
            model = self.inverse_diagonal[indices] * residual_change[indices]
            denominator = float(np.dot(model, model))
            if denominator <= 1.0e-28:
                factors.append(1.0)
                continue
            estimate = float(np.dot(model, step[indices]) / denominator)
            if not np.isfinite(estimate) or estimate <= 0.0:
                factors.append(1.0)
                continue
            factor = float(np.clip(estimate, minimum_scale, maximum_scale))
            self.inverse_diagonal[indices] *= factor
            factors.append(factor)
        return tuple(factors)

    def restart(self) -> None:
        self.steps.clear()
        self.residual_changes.clear()

    def discard_latest(self) -> bool:
        if not self.steps:
            return False
        self.steps.pop()
        self.residual_changes.pop()
        return True


def _inverse_denominators(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | None,
    options: SolverOptions,
    excitation_space: QPExcitationSpaceLike,
) -> tuple[np.ndarray, dict[str, object]]:
    floor = options.denominator_floor
    if reference is None or reference.metadata.get("denominator_source") == "h11":
        raw_energies = np.real(np.diag(hamiltonian.h11))
        energy_source = "h11-diagonal"
    else:
        raw_energies = np.asarray(reference.quasiparticle_energies, dtype=float)
        energy_source = "reference-quasiparticle-energies"
    raw_energies = np.asarray(raw_energies, dtype=float)
    if not np.all(np.isfinite(raw_energies)):
        raise ValueError("quasiparticle preconditioner energies must be finite")

    automatic_shift = 0.0
    applied_shift = 0.0
    if options.quasiparticle_level_shift:
        # The Sokolov-Chan prescription lifts the most negative eigenvalue to
        # zero. An explicit value is treated as a lower bound on that shift.
        automatic_shift = max(0.0, -float(np.min(raw_energies)))
        requested_shift = (
            0.0
            if options.quasiparticle_level_shift_value is None
            else float(options.quasiparticle_level_shift_value)
        )
        applied_shift = max(automatic_shift, requested_shift)
        energies = raw_energies + applied_shift
        regularization = "sokolov-chan-level-shift"
    else:
        # Preserve the established default preconditioner exactly.
        energies = np.abs(raw_energies)
        regularization = "absolute-quasiparticle-energies"
    energies = np.maximum(energies, floor)
    custom_builder = getattr(excitation_space, "inverse_denominators", None)
    if custom_builder is not None:
        inverse = np.asarray(custom_builder(energies, floor), dtype=float)
    else:
        denominators = [
            energies[p] + energies[q] for p, q in excitation_space.pair_indices
        ]
        denominators.extend(
            energies[p] + energies[q] + energies[r] + energies[s]
            for p, q, r, s in excitation_space.quadruple_indices
        )
        inverse = 1.0 / np.maximum(np.asarray(denominators), floor)
    if not np.all(np.isfinite(inverse)) or np.any(inverse <= 0.0):
        raise ValueError("quasiparticle inverse denominators must be finite and positive")
    effective_denominators = 1.0 / inverse
    diagnostics: dict[str, object] = {
        "quasiparticle_level_shift_enabled": options.quasiparticle_level_shift,
        "quasiparticle_level_shift_requested": options.quasiparticle_level_shift_value,
        "quasiparticle_level_shift_automatic": automatic_shift,
        "quasiparticle_level_shift_applied": applied_shift,
        "quasiparticle_preconditioner_energy_source": energy_source,
        "quasiparticle_preconditioner_regularization": regularization,
        "minimum_raw_quasiparticle_energy": float(np.min(raw_energies)),
        "minimum_preconditioner_quasiparticle_energy": float(np.min(energies)),
        "minimum_preconditioner_denominator": (
            float(np.min(effective_denominators))
            if effective_denominators.size
            else None
        ),
        "maximum_preconditioner_denominator": (
            float(np.max(effective_denominators))
            if effective_denominators.size
            else None
        ),
    }
    return inverse, diagnostics


def _optimizer_residual(
    evaluation: KernelEvaluation,
    excitation_space: QPExcitationSpaceLike,
    *,
    complex_coordinates: bool,
    residual_basis: object | None = None,
    gauge_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    values = residual_to_vector(evaluation.r1, evaluation.r2, excitation_space)
    values = _project_residual_values(values, residual_basis, gauge_weight)
    max_imaginary = float(np.max(np.abs(np.imag(values)))) if values.size else 0.0
    if complex_coordinates:
        residual = np.concatenate((np.real(values), np.imag(values)))
    else:
        residual = np.real(values)
    return np.asarray(residual, dtype=float), max_imaginary


def _project_residual_values(
    values: np.ndarray,
    residual_basis: object | None,
    gauge_weight: float,
) -> np.ndarray:
    """Apply the identical equation projection to residuals and derivatives."""

    values = np.asarray(values, dtype=np.complex128)
    if _projector_active(residual_basis):
        physical = residual_basis.project_physical(values)
        gauge = residual_basis.project_gauge(values)
        values = physical + gauge_weight * gauge
    return np.asarray(values, dtype=np.complex128)


def _residual_component_norms(
    evaluation: KernelEvaluation,
    excitation_space: QPExcitationSpaceLike,
    residual_basis: object | None,
) -> tuple[float, float, float]:
    values = np.asarray(
        residual_to_vector(evaluation.r1, evaluation.r2, excitation_space),
        dtype=np.complex128,
    )
    full_norm = float(np.max(np.abs(values))) if values.size else 0.0
    if not _projector_active(residual_basis):
        return full_norm, 0.0, full_norm
    physical = residual_basis.project_physical(values)
    gauge = residual_basis.project_gauge(values)
    physical_norm = float(np.max(np.abs(physical))) if physical.size else 0.0
    gauge_norm = float(np.max(np.abs(gauge))) if gauge.size else 0.0
    return physical_norm, gauge_norm, full_norm


def _convergence_merit(evaluation: KernelEvaluation) -> float:
    """Use the same max-equation norm for globalization and convergence."""

    return float(evaluation.residual_norm)


def _trial_acceptance_mode(
    *,
    current_merit: float,
    current_l2: float,
    trial_merit: float,
    trial_l2: float,
    primary_limit: float,
    options: SolverOptions,
) -> str | None:
    if not np.isfinite(trial_merit) or not np.isfinite(trial_l2):
        return None
    if trial_merit <= primary_limit:
        return "maximum"
    if (
        current_merit > 10.0 * options.residual_tolerance
        and trial_merit
        <= current_merit * (1.0 + options.filter_maximum_growth)
        and trial_l2
        <= current_l2 * (1.0 - options.filter_l2_relative_decrease)
    ):
        return "l2-filter"
    return None


def _deep_line_search_exhausted(
    current_merit: float,
    best_rejected_trial: tuple[float, float, float] | None,
    backtrack_limit: int,
) -> bool:
    """Detect when another multisecant direction is unlikely to repay its cost."""

    return bool(
        backtrack_limit >= 4
        and best_rejected_trial is not None
        and best_rejected_trial[0] >= current_merit
    )


def _limit_step(step: np.ndarray, maximum: float) -> np.ndarray:
    largest = float(np.max(np.abs(step))) if step.size else 0.0
    return step if largest <= maximum else step * (maximum / largest)


def _adaptive_newton_level_shift(
    residual_merit: float,
    configured: float | None,
) -> float:
    """Return a denominator-scaled pseudo-transient shift near singular roots."""

    if configured is not None:
        return float(configured)
    if residual_merit >= 1.0e-3:
        return 0.0
    return min(1.0e-1, max(1.0e-8, math.sqrt(max(residual_merit, 0.0))))


def _levenberg_marquardt_step(
    jacobian: np.ndarray,
    rhs: np.ndarray,
    maximum: float,
) -> tuple[np.ndarray, float, float]:
    """Solve a dense Newton model with a maximum-component trust radius."""

    left, singular_values, right_adjoint = np.linalg.svd(
        np.asarray(jacobian),
        full_matrices=False,
    )
    transformed_rhs = left.conj().T @ np.asarray(rhs)

    def regularized(parameter: float) -> np.ndarray:
        if parameter == 0.0:
            cutoff = max(float(singular_values[0]), 1.0) * 1.0e-14
            factors = np.divide(
                1.0,
                singular_values,
                out=np.zeros_like(singular_values),
                where=singular_values > cutoff,
            )
        else:
            factors = singular_values / (singular_values**2 + parameter)
        return right_adjoint.conj().T @ (factors * transformed_rhs)

    raw_step = regularized(0.0)
    raw_maximum = float(np.max(np.abs(raw_step))) if raw_step.size else 0.0
    if raw_maximum <= maximum:
        return raw_step, 0.0, raw_maximum

    lower = 0.0
    upper = max(float(singular_values[0] ** 2), 1.0e-12)
    step = regularized(upper)
    while float(np.max(np.abs(step))) > maximum:
        upper *= 10.0
        step = regularized(upper)
    for _ in range(64):
        parameter = 0.5 * (lower + upper)
        step = regularized(parameter)
        if float(np.max(np.abs(step))) > maximum:
            lower = parameter
        else:
            upper = parameter
    return regularized(upper), upper, raw_maximum


@dataclass(frozen=True)
class _ArnoldiHookstepModel:
    """Compact right-preconditioned Newton model for trust-region steps."""

    residual: np.ndarray
    rhs: np.ndarray
    arnoldi_basis: np.ndarray
    physical_basis: np.ndarray
    hessenberg: np.ndarray
    relative_residual_history: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return int(self.physical_basis.shape[1])

    def _regularized_coordinates(self, parameter: float) -> np.ndarray:
        small_rhs = np.zeros(self.dimension + 1, dtype=self.hessenberg.dtype)
        small_rhs[0] = np.linalg.norm(self.rhs)
        if parameter == 0.0:
            return np.linalg.lstsq(
                self.hessenberg,
                small_rhs,
                rcond=1.0e-12,
            )[0]
        augmented = np.vstack(
            (
                self.hessenberg,
                math.sqrt(parameter) * self.physical_basis,
            )
        )
        augmented_rhs = np.concatenate(
            (
                small_rhs,
                np.zeros(self.physical_basis.shape[0], dtype=small_rhs.dtype),
            )
        )
        return np.linalg.lstsq(augmented, augmented_rhs, rcond=1.0e-12)[0]

    def solve(self, maximum: float) -> tuple[np.ndarray, dict[str, float]]:
        if maximum <= 0.0:
            raise ValueError("hookstep radius must be positive")

        raw_coordinates = self._regularized_coordinates(0.0)
        raw_step = self.physical_basis @ raw_coordinates
        raw_maximum = float(np.max(np.abs(raw_step))) if raw_step.size else 0.0
        parameter = 0.0
        coordinates = raw_coordinates
        step = raw_step
        if raw_maximum > maximum:
            h_scale = max(float(np.linalg.norm(self.hessenberg) ** 2), 1.0e-30)
            z_scale = max(float(np.linalg.norm(self.physical_basis) ** 2), 1.0e-30)
            lower = 0.0
            upper = h_scale / z_scale
            coordinates = self._regularized_coordinates(upper)
            step = self.physical_basis @ coordinates
            while float(np.max(np.abs(step))) > maximum:
                upper *= 10.0
                coordinates = self._regularized_coordinates(upper)
                step = self.physical_basis @ coordinates
            for _ in range(64):
                parameter = 0.5 * (lower + upper)
                coordinates = self._regularized_coordinates(parameter)
                step = self.physical_basis @ coordinates
                if float(np.max(np.abs(step))) > maximum:
                    lower = parameter
                else:
                    upper = parameter
            parameter = upper
            coordinates = self._regularized_coordinates(parameter)
            step = self.physical_basis @ coordinates

        arnoldi_action = (
            self.arnoldi_basis
            @ self.hessenberg
            @ coordinates
        )
        predicted_residual = self.residual + arnoldi_action
        return step, {
            "regularization": float(parameter),
            "raw_step_maximum": raw_maximum,
            "step_maximum": float(np.max(np.abs(step))) if step.size else 0.0,
            "step_l2_norm": float(np.linalg.norm(step)),
            "predicted_residual_norm": (
                float(np.max(np.abs(predicted_residual)))
                if predicted_residual.size
                else 0.0
            ),
            "predicted_residual_l2_norm": float(np.linalg.norm(predicted_residual)),
        }


def _build_right_preconditioned_arnoldi_model(
    jacobian_action: Callable[[np.ndarray], np.ndarray],
    inverse_action: Callable[[np.ndarray], np.ndarray],
    residual: np.ndarray,
    *,
    maximum_dimension: int,
    relative_tolerance: float,
) -> _ArnoldiHookstepModel:
    """Build a bounded-memory Arnoldi model without forming the Jacobian."""

    residual = np.asarray(residual)
    if residual.ndim != 1 or residual.size == 0:
        raise ValueError("the hookstep residual must be a nonempty vector")
    if maximum_dimension <= 0:
        raise ValueError("the Arnoldi dimension must be positive")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("the Arnoldi tolerance must lie in (0, 1)")
    rhs = -residual
    beta = float(np.linalg.norm(rhs))
    if beta == 0.0:
        raise ValueError("a zero residual does not require a hookstep model")

    dimension = residual.size
    maximum_dimension = min(int(maximum_dimension), dimension)
    dtype = np.result_type(residual.dtype, np.complex128 if np.iscomplexobj(residual) else float)
    basis = np.zeros((dimension, maximum_dimension + 1), dtype=dtype)
    physical = np.zeros((dimension, maximum_dimension), dtype=dtype)
    hessenberg = np.zeros((maximum_dimension + 1, maximum_dimension), dtype=dtype)
    basis[:, 0] = rhs / beta
    residual_history: list[float] = []
    built = 0

    for column in range(maximum_dimension):
        physical[:, column] = np.asarray(
            inverse_action(basis[:, column]), dtype=dtype
        )
        action = np.asarray(
            jacobian_action(physical[:, column]), dtype=dtype
        ).copy()
        if action.shape != (dimension,) or not np.all(np.isfinite(action)):
            raise ValueError("the Jacobian action returned invalid Arnoldi data")

        # Reorthogonalization is inexpensive at k <= 48 and prevents the weak
        # projected modes from being lost when the Jacobian is ill-conditioned.
        for _ in range(2):
            coefficients = basis[:, : column + 1].conj().T @ action
            hessenberg[: column + 1, column] += coefficients
            action -= basis[:, : column + 1] @ coefficients
        next_norm = float(np.linalg.norm(action))
        hessenberg[column + 1, column] = next_norm
        built = column + 1
        small_rhs = np.zeros(built + 1, dtype=dtype)
        small_rhs[0] = beta
        projected = hessenberg[: built + 1, :built]
        coordinates = np.linalg.lstsq(projected, small_rhs, rcond=1.0e-12)[0]
        relative_residual = float(
            np.linalg.norm(small_rhs - projected @ coordinates) / beta
        )
        residual_history.append(relative_residual)
        if relative_residual <= relative_tolerance or next_norm <= 1.0e-14:
            break
        basis[:, column + 1] = action / next_norm

    if built == 0:
        raise RuntimeError("failed to build an Arnoldi hookstep model")
    return _ArnoldiHookstepModel(
        residual=np.asarray(residual, dtype=dtype).copy(),
        rhs=np.asarray(rhs, dtype=dtype).copy(),
        arnoldi_basis=basis[:, : built + 1].copy(),
        physical_basis=physical[:, :built].copy(),
        hessenberg=hessenberg[: built + 1, :built].copy(),
        relative_residual_history=tuple(residual_history),
    )


def _scaled_least_squares_step(
    jacobian: np.ndarray,
    rhs: np.ndarray,
    maximum: float,
) -> tuple[np.ndarray, float]:
    """Preserve the Newton direction and leave globalization to backtracking."""

    step = np.linalg.lstsq(jacobian, rhs, rcond=1.0e-12)[0]
    raw_maximum = float(np.max(np.abs(step))) if step.size else 0.0
    return _limit_step(step, maximum), raw_maximum


def _anderson_candidate(
    vectors: list[np.ndarray],
    residuals: list[np.ndarray],
    inverse_diagonal: np.ndarray,
    current: np.ndarray,
    step_max: float,
) -> np.ndarray | None:
    if len(vectors) < 2:
        return None
    matrix = np.column_stack(residuals)
    count = matrix.shape[1]
    gram = matrix.T @ matrix
    regularization = 1.0e-10 * max(float(np.trace(gram)) / count, 1.0e-30)
    gram.flat[:: count + 1] += regularization
    augmented = np.empty((count + 1, count + 1), dtype=float)
    augmented[:count, :count] = gram
    augmented[:count, count] = 1.0
    augmented[count, :count] = 1.0
    augmented[count, count] = 0.0
    rhs = np.zeros(count + 1)
    rhs[count] = 1.0
    try:
        coefficients = np.linalg.solve(augmented, rhs)[:count]
    except np.linalg.LinAlgError:
        return None
    fixed_point_images = [
        vector - inverse_diagonal * residual
        for vector, residual in zip(vectors, residuals)
    ]
    candidate = sum(
        (
            coefficient * image
            for coefficient, image in zip(coefficients, fixed_point_images)
        ),
        np.zeros_like(current),
    )
    return current + _limit_step(candidate - current, step_max)


def _solve(
    evaluator: Callable[[QPAmplitudes], KernelEvaluation],
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | None,
    initial_amplitudes: QPAmplitudes | None,
    options: SolverOptions,
    requested_method: str,
    *,
    projected: bool,
    excitation_space: QPExcitationSpaceLike | None = None,
    iteration_callback: IterationCallback | None = None,
    jvp_evaluator: Callable[
        [QPAmplitudes, QPAmplitudes], KernelDirectionalDerivative
    ]
    | None = None,
    residual_basis: object | None = None,
    coarse_basis: ProjectedResidualBasis | None = None,
    gauge_options: GaugeModeOptions | None = None,
    step_basis_builder: Callable[[QPAmplitudes], ProjectedResidualBasis] | None = None,
) -> QPCCSDResult:
    nspin = hamiltonian.nspin
    space = _full_excitation_space(nspin) if excitation_space is None else excitation_space
    if space.nspin != nspin:
        raise ValueError("excitation space and Hamiltonian dimensions do not match")
    complex_coordinates = (
        projected if options.complex_amplitudes is None else options.complex_amplitudes
    )
    coordinate_count = space.coordinate_count
    if residual_basis is not None and residual_basis.coordinate_count != coordinate_count:
        raise ValueError("projected residual basis and excitation space differ")
    if coarse_basis is not None and coarse_basis.coordinate_count != coordinate_count:
        raise ValueError("coarse residual basis and excitation space differ")
    gauge_weight = 0.0 if gauge_options is None else gauge_options.gauge_weight

    def optimizer_amplitudes(vector: np.ndarray) -> QPAmplitudes:
        vector = np.asarray(vector, dtype=float)
        if complex_coordinates:
            if vector.shape != (2 * coordinate_count,):
                raise ValueError("complex optimizer vector has an incompatible shape")
            amplitudes = vector[:coordinate_count] + 1.0j * vector[coordinate_count:]
        else:
            if vector.shape != (coordinate_count,):
                raise ValueError("real optimizer vector has an incompatible shape")
            amplitudes = vector
        return amplitudes_from_vector(amplitudes, nspin, space)

    def optimizer_vector(amplitudes: QPAmplitudes) -> np.ndarray:
        values = np.asarray(amplitudes_to_vector(amplitudes, space))
        if complex_coordinates:
            return np.concatenate((np.real(values), np.imag(values))).astype(float)
        return np.asarray(np.real(values), dtype=float)

    def project_optimizer_vector(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        step_basis = (
            None
            if step_basis_builder is None
            else step_basis_builder(optimizer_amplitudes(current))
        )
        if not _projector_active(step_basis):
            return values
        if complex_coordinates:
            complex_values = (
                values[:coordinate_count]
                + 1.0j * values[coordinate_count:]
            )
            projected_values = step_basis.project_physical(complex_values)
            return np.concatenate(
                (np.real(projected_values), np.imag(projected_values))
            )
        projected_values = step_basis.project_physical(values.astype(np.complex128))
        return np.asarray(np.real(projected_values), dtype=float)

    if initial_amplitudes is None:
        current = np.zeros(
            2 * coordinate_count if complex_coordinates else coordinate_count
        )
    else:
        current = optimizer_vector(initial_amplitudes)
    inverse_diagonal, quasiparticle_preconditioner_diagnostics = _inverse_denominators(
        hamiltonian,
        reference,
        options,
        space,
    )
    if complex_coordinates:
        inverse_diagonal = np.concatenate((inverse_diagonal, inverse_diagonal))
    broyden = _LimitedMemoryMultisecantBroyden(
        inverse_diagonal,
        options.broyden_history,
    )
    pair_indices = np.arange(space.pair_count, dtype=np.int64)
    quadruple_indices = np.arange(
        space.pair_count, coordinate_count, dtype=np.int64
    )
    if complex_coordinates:
        pair_indices = np.concatenate((pair_indices, pair_indices + coordinate_count))
        quadruple_indices = np.concatenate(
            (quadruple_indices, quadruple_indices + coordinate_count)
        )
    preconditioner_groups = tuple(
        group for group in (pair_indices, quadruple_indices) if group.size
    )
    block_scale_history: list[dict[str, object]] = []
    evaluation = evaluator(optimizer_amplitudes(current))
    evaluator_calls = 1
    residual, max_imaginary = _optimizer_residual(
        evaluation,
        space,
        complex_coordinates=complex_coordinates,
        residual_basis=residual_basis,
        gauge_weight=gauge_weight,
    )
    residual_merit, gauge_residual_norm, full_residual_norm = _residual_component_norms(
        evaluation, space, residual_basis
    )
    residual_l2 = float(np.linalg.norm(residual))
    history: list[dict[str, object]] = []
    vectors = [current.copy()]
    residuals = [residual.copy()]
    converged = residual_merit < options.residual_tolerance
    previous_energy = complex(evaluation.total_energy)
    total_projection_time = float(evaluation.diagnostics.get("projection_time", 0.0))
    spectral_diagnostics: dict[str, object] = {
        "spectral_fallback_used": False,
    }
    jvp_calls = 0
    jvp_projection_time = 0.0
    jacobian_evaluator_calls = 0
    jacobian_projection_time = 0.0
    newton_attempts = 0
    gmres_reports: list[dict[str, object]] = []
    dense_jacobian: np.ndarray | None = None
    dense_jacobian_builds = 0
    dense_newton_reports: list[dict[str, object]] = []
    dense_step_reports: list[dict[str, object]] = []

    def notify_iteration(
        iteration: int,
        step: str,
        *,
        accepted: bool = True,
        details: dict[str, object] | None = None,
    ) -> None:
        if iteration_callback is None:
            return
        amplitudes = optimizer_amplitudes(current)
        state: dict[str, object] = {
            "iteration": int(iteration),
            "step": str(step),
            "raw_energy": complex(evaluation.total_energy),
            "residual_norm": float(residual_merit),
            "full_residual_norm": float(full_residual_norm),
            "gauge_residual_norm": float(gauge_residual_norm),
            "residual_l2_norm": float(residual_l2),
            "evaluator_calls": int(evaluator_calls),
            "projected": bool(projected),
            "converged": bool(converged),
            "accepted": bool(accepted),
        }
        if details is not None:
            state.update(details)
        iteration_callback(
            QPAmplitudes(
                t1=np.asarray(amplitudes.t1).copy(),
                t2=np.asarray(amplitudes.t2).copy(),
            ),
            state,
        )

    notify_iteration(0, "initial")

    trust_radius = options.step_max
    merit_window = [residual_merit]
    best_state = (
        current.copy(),
        evaluation,
        residual.copy(),
        max_imaginary,
        residual_merit,
        residual_l2,
    )
    best_convergence_state = best_state
    best_convergence_norm = residual_merit
    rejected_steps = 0
    consecutive_rejections = 0
    stagnation_count = 0
    convergence_stagnation_count = 0
    accepted_since_newton = options.newton_cooldown_iterations
    last_newton_state: np.ndarray | None = None
    last_coarse_state: np.ndarray | None = None
    newton_rejection_override = False
    adaptive_newton_damping = bool(
        options.newton_globalization == "line-search"
        and options.newton_level_shift is None
    )
    current_newton_level_shift = _adaptive_newton_level_shift(
        residual_merit,
        options.newton_level_shift,
    )
    newton_damping_retry = False
    newton_damping_retries = 0
    hookstep_model: _ArnoldiHookstepModel | None = None
    hookstep_model_state: np.ndarray | None = None
    hookstep_model_builds = 0
    hookstep_model_reuses = 0
    hookstep_retry = False
    hookstep_retries = 0
    current_newton_trust_radius = options.newton_step_max
    prefer_anderson = False
    dense_initial = bool(
        options.dense_newton_min_coordinates <= coordinate_count
        <= options.dense_newton_max_coordinates
        and options.dense_newton_max_builds > 0
    )
    # When explicitly requested, begin with the matrix-free Newton model
    # instead of spending several full residual calls discovering that the
    # diagonal fixed-point iteration has stalled.  This applies equally to raw
    # and number-projected equations; ``newton_initial=False`` preserves the
    # established delayed-Newton behavior.
    use_newton = bool(
        options.newton_krylov
        and jvp_evaluator is not None
        and (options.newton_initial or dense_initial)
    )

    def newton_state_has_changed() -> bool:
        return last_newton_state is None or not np.array_equal(
            current,
            last_newton_state,
        )

    def hookstep_model_is_current() -> bool:
        return bool(
            hookstep_model is not None
            and hookstep_model_state is not None
            and np.array_equal(current, hookstep_model_state)
        )

    def newton_is_ready() -> bool:
        return bool(
            options.newton_krylov
            and jvp_evaluator is not None
            and (
                newton_state_has_changed()
                or newton_damping_retry
                or hookstep_retry
            )
            and (
                last_newton_state is None
                or accepted_since_newton >= options.newton_cooldown_iterations
                or newton_rejection_override
                or newton_damping_retry
                or hookstep_retry
            )
        )

    for iteration in range(1, options.max_iterations + 1):
        if converged or evaluator_calls >= options.max_evaluator_calls:
            break

        step_kind = "broyden"
        proposed_step = None
        attempted_newton = False
        attempted_level_shift: float | None = None
        newton_report: dict[str, object] | None = None
        if use_newton and not newton_is_ready():
            use_newton = False
        coarse_ready = bool(
            use_newton
            and gauge_options is not None
            and gauge_options.coarse_newton
            and coarse_basis is not None
            and coarse_basis.gauge_rank
            and jvp_evaluator is not None
            and jvp_calls + coarse_basis.gauge_rank <= options.max_jvp_calls
            and (
                last_coarse_state is None
                or not np.array_equal(current, last_coarse_state)
            )
        )
        if coarse_ready:
            current_amplitudes = optimizer_amplitudes(current)
            full_values = residual_to_vector(evaluation.r1, evaluation.r2, space)
            columns: list[np.ndarray] = []
            for direction in coarse_basis.gauge_vectors.T:
                derivative = jvp_evaluator(
                    current_amplitudes,
                    amplitudes_from_vector(direction, nspin, space),
                )
                jvp_calls += 1
                jvp_projection_time += float(derivative.elapsed)
                columns.append(
                    np.asarray(
                        residual_to_vector(derivative.r1, derivative.r2, space),
                        dtype=np.complex128,
                    )
                )
            coarse_jacobian = np.column_stack(columns)
            coefficients, *_ = np.linalg.lstsq(
                coarse_jacobian,
                -np.asarray(full_values, dtype=np.complex128),
                rcond=max(1.0e-12, options.denominator_floor),
            )
            coarse_step = coarse_basis.gauge_vectors @ coefficients
            if complex_coordinates:
                proposed_step = np.concatenate(
                    (np.real(coarse_step), np.imag(coarse_step))
                )
            else:
                proposed_step = np.asarray(np.real(coarse_step), dtype=float)
            proposed_step = _limit_step(proposed_step, options.newton_step_max)
            step_kind = "newton-gauge-coarse"
            attempted_newton = True
            newton_attempts += 1
            last_coarse_state = current.copy()
            use_newton = False
            gmres_reports.append(
                {
                    "iteration": iteration,
                    "coordinate_system": "candidate-number-orbit-coarse-space",
                    "coarse_rank": coarse_basis.gauge_rank,
                    "jvp_calls": jvp_calls,
                    "raw_step_maximum": float(np.max(np.abs(coarse_step))),
                    "globalization": "full-residual-filter",
                }
            )
        if (
            proposed_step is None
            and use_newton
            and options.newton_krylov
            and jvp_evaluator is not None
            and (
                jvp_calls + 1 < options.max_jvp_calls
                or (
                    options.newton_globalization == "hookstep"
                    and hookstep_model_is_current()
                )
            )
        ):
            from scipy.sparse.linalg import LinearOperator, gmres

            current_amplitudes = optimizer_amplitudes(current)
            attempted_newton = True
            last_newton_state = current.copy()
            accepted_since_newton = 0
            newton_rejection_override = False
            newton_damping_retry = False
            hookstep_retry = False

            def raw_jvp(direction_amplitudes: QPAmplitudes) -> np.ndarray:
                nonlocal jvp_projection_time
                derivative = jvp_evaluator(
                    current_amplitudes,
                    direction_amplitudes,
                )
                jvp_projection_time += float(derivative.elapsed)
                values = residual_to_vector(
                    derivative.r1,
                    derivative.r2,
                    space,
                )
                return _project_residual_values(
                    values,
                    residual_basis,
                    gauge_weight,
                )

            def apply_jvp(direction_amplitudes: QPAmplitudes) -> np.ndarray:
                nonlocal jvp_calls
                if jvp_calls >= options.max_jvp_calls:
                    raise RuntimeError("analytic JVP budget exhausted")
                values = raw_jvp(direction_amplitudes)
                jvp_calls += 1
                return values

            def real_jacobian_action(direction_vector: np.ndarray) -> np.ndarray:
                direction_vector = project_optimizer_vector(direction_vector)
                values = apply_jvp(optimizer_amplitudes(direction_vector))
                if complex_coordinates:
                    return np.concatenate((np.real(values), np.imag(values)))
                return np.asarray(np.real(values), dtype=float)

            dense_eligible = (
                projected
                and options.dense_newton_min_coordinates <= coordinate_count
                <= options.dense_newton_max_coordinates
                and options.dense_newton_max_builds > 0
            )
            if complex_coordinates:
                dimension = coordinate_count
                complex_residual = (
                    residual[:coordinate_count]
                    + 1.0j * residual[coordinate_count:]
                )

                def complex_jacobian_action(direction_vector: np.ndarray) -> np.ndarray:
                    packed_direction = np.concatenate(
                        (np.real(direction_vector), np.imag(direction_vector))
                    )
                    projected_direction = project_optimizer_vector(packed_direction)
                    physical_direction = (
                        projected_direction[:coordinate_count]
                        + 1.0j * projected_direction[coordinate_count:]
                    )
                    return np.asarray(
                        apply_jvp(
                            amplitudes_from_vector(physical_direction, nspin, space)
                        ),
                        dtype=np.complex128,
                    )

                jacobian = LinearOperator(
                    (dimension, dimension),
                    matvec=complex_jacobian_action,
                    dtype=np.complex128,
                )

                inverse_jacobian = LinearOperator(
                    (dimension, dimension),
                    matvec=broyden.apply_complex,
                    dtype=np.complex128,
                )
                gmres_rhs = -complex_residual
                gmres_coordinates = "complex-holomorphic"
            else:
                dimension = residual.size
                jacobian = LinearOperator(
                    (dimension, dimension),
                    matvec=real_jacobian_action,
                    dtype=float,
                )
                inverse_jacobian = LinearOperator(
                    (dimension, dimension),
                    matvec=broyden.apply,
                    dtype=float,
                )
                gmres_rhs = -residual
                gmres_coordinates = "real"
            dense_derivative = options.dense_newton_derivative
            dense_derivative_budget_available = (
                dense_derivative == "finite-difference"
                or jvp_calls + dimension <= options.max_jvp_calls
            )
            if (
                dense_eligible
                and dense_jacobian is None
                and dense_jacobian_builds < options.dense_newton_max_builds
                and dense_derivative_budget_available
            ):
                dense_started = time.perf_counter()
                dense_jvp_time_before = jvp_projection_time
                dense_workers = min(
                    dimension,
                    options.dense_newton_workers or 1,
                )
                current_values = np.asarray(
                    amplitudes_to_vector(current_amplitudes, space)
                )

                def dense_column(index: int) -> tuple[np.ndarray, float]:
                    basis = np.zeros(dimension, dtype=jacobian.dtype)
                    basis[index] = 1.0
                    if complex_coordinates:
                        packed_basis = np.concatenate(
                            (np.real(basis), np.imag(basis))
                        )
                        projected_basis = project_optimizer_vector(packed_basis)
                        basis = (
                            projected_basis[:coordinate_count]
                            + 1.0j * projected_basis[coordinate_count:]
                        )
                    else:
                        basis = project_optimizer_vector(basis)
                    if dense_derivative == "analytic-jvp":
                        if complex_coordinates:
                            column = np.asarray(
                                raw_jvp(
                                    amplitudes_from_vector(basis, nspin, space)
                                ),
                                dtype=np.complex128,
                            )
                        else:
                            column = np.asarray(
                                np.real(raw_jvp(optimizer_amplitudes(basis))),
                                dtype=float,
                            )
                        return column, 0.0

                    finite_difference_step = (
                        options.dense_newton_finite_difference_step
                        * max(1.0, float(np.max(np.abs(current_values))))
                    )
                    trial_values = current_values + finite_difference_step * basis
                    trial_evaluation = evaluator(
                        amplitudes_from_vector(trial_values, nspin, space)
                    )
                    trial_residual = residual_to_vector(
                        trial_evaluation.r1,
                        trial_evaluation.r2,
                        space,
                    )
                    trial_residual = _project_residual_values(
                        trial_residual,
                        residual_basis,
                        gauge_weight,
                    )
                    if complex_coordinates:
                        column = np.asarray(
                            (trial_residual - complex_residual)
                            / finite_difference_step,
                            dtype=np.complex128,
                        )
                    else:
                        column = np.asarray(
                            np.real(trial_residual) - residual,
                            dtype=float,
                        ) / finite_difference_step
                    return column, float(
                        trial_evaluation.diagnostics.get("projection_time", 0.0)
                    )

                if dense_workers == 1:
                    dense_columns = tuple(
                        dense_column(index) for index in range(dimension)
                    )
                else:
                    with ThreadPoolExecutor(
                        max_workers=dense_workers,
                        thread_name_prefix="qpccsd-dense-jacobian",
                    ) as executor:
                        dense_columns = tuple(
                            executor.map(dense_column, range(dimension))
                        )
                dense_jacobian = np.column_stack(
                    tuple(column for column, _ in dense_columns)
                )
                build_projection_time = float(
                    sum(projection_time for _, projection_time in dense_columns)
                )
                if dense_derivative == "analytic-jvp":
                    build_projection_time = (
                        jvp_projection_time - dense_jvp_time_before
                    )
                    jvp_calls += dimension
                else:
                    jacobian_evaluator_calls += dimension
                    jacobian_projection_time += build_projection_time
                    total_projection_time += build_projection_time
                dense_jacobian_builds += 1
                dense_newton_reports.append(
                    {
                        "iteration": iteration,
                        "build": dense_jacobian_builds,
                        "dimension": dimension,
                        "workers": dense_workers,
                        "derivative": dense_derivative,
                        "directions_projected_to_solver_section": bool(
                            step_basis_builder is not None
                        ),
                        "finite_difference_step": (
                            options.dense_newton_finite_difference_step
                            if dense_derivative == "finite-difference"
                            else None
                        ),
                        "evaluator_calls": (
                            dimension
                            if dense_derivative == "finite-difference"
                            else 0
                        ),
                        "jvp_calls": (
                            dimension if dense_derivative == "analytic-jvp" else 0
                        ),
                        "projection_time": build_projection_time,
                        "elapsed": time.perf_counter() - dense_started,
                        "condition": float(np.linalg.cond(dense_jacobian)),
                    }
                )

            if dense_jacobian is not None:
                # A rejected multisecant direction only bounds that direction.
                # Reusing its shrunken radius here can permanently suppress a
                # well-defined Newton correction after the cheap model fails.
                dense_radius = options.dense_newton_step_max
                (
                    newton_step,
                    regularization,
                    raw_step_maximum,
                ) = _levenberg_marquardt_step(
                    dense_jacobian,
                    gmres_rhs,
                    dense_radius,
                )
                dense_step_reports.append(
                    {
                        "iteration": iteration,
                        "trust_radius": dense_radius,
                        "raw_step_maximum": raw_step_maximum,
                        "regularization": regularization,
                        "direction": "levenberg-marquardt-hookstep",
                    }
                )
                if complex_coordinates:
                    newton_step = np.concatenate(
                        (np.real(newton_step), np.imag(newton_step))
                    )
                proposed_step = newton_step
                step_kind = "dense-newton"
            else:
                newton_attempts += 1
                if options.newton_globalization == "hookstep":
                    model_reused = hookstep_model_is_current()
                    try:
                        if not model_reused:
                            krylov_dimension = min(
                                options.gmres_restart,
                                dimension,
                                options.max_jvp_calls - jvp_calls,
                            )
                            model_residual = (
                                complex_residual if complex_coordinates else residual
                            )
                            hookstep_model = _build_right_preconditioned_arnoldi_model(
                                jacobian.matvec,
                                inverse_jacobian.matvec,
                                model_residual,
                                maximum_dimension=krylov_dimension,
                                relative_tolerance=min(
                                    0.5,
                                    max(
                                        1.0e-3,
                                        math.sqrt(max(residual_merit, 0.0)),
                                    ),
                                ),
                            )
                            hookstep_model_state = current.copy()
                            hookstep_model_builds += 1
                        else:
                            assert hookstep_model is not None
                            hookstep_model_reuses += 1
                        assert hookstep_model is not None
                        newton_step, hookstep_details = hookstep_model.solve(
                            current_newton_trust_radius
                        )
                        if complex_coordinates:
                            newton_step = np.concatenate(
                                (np.real(newton_step), np.imag(newton_step))
                            )
                        proposed_step = np.asarray(newton_step, dtype=float)
                        step_kind = "newton-krylov-hookstep"
                        gmres_info: int | str = 0
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                        gmres_info = f"{type(error).__name__}: {error}"
                        hookstep_details = {}
                    newton_report = {
                        "iteration": iteration,
                        "info": gmres_info,
                        "jvp_calls": jvp_calls,
                        "coordinate_system": gmres_coordinates,
                        "preconditioning": "right",
                        "globalization": "arnoldi-hookstep",
                        "multisecant_history": len(broyden.steps),
                        "krylov_dimension": (
                            0 if hookstep_model is None else hookstep_model.dimension
                        ),
                        "krylov_residual_history": (
                            []
                            if hookstep_model is None
                            else list(hookstep_model.relative_residual_history)
                        ),
                        "hookstep_model_reused": model_reused,
                        "hookstep_trust_radius": current_newton_trust_radius,
                        **hookstep_details,
                    }
                else:
                    if adaptive_newton_damping:
                        current_newton_level_shift = max(
                            current_newton_level_shift,
                            _adaptive_newton_level_shift(residual_merit, None),
                        )
                    level_shift = current_newton_level_shift
                    attempted_level_shift = level_shift
                    right_preconditioned_jacobian = LinearOperator(
                        (dimension, dimension),
                        matvec=lambda vector: (
                            jacobian.matvec(inverse_jacobian.matvec(vector))
                            + level_shift * vector
                        ),
                        dtype=jacobian.dtype,
                    )
                    gmres_residual_history: list[float] = []
                    krylov_dimension = min(
                        options.gmres_restart,
                        dimension,
                        options.max_jvp_calls - jvp_calls - 1,
                    )
                    try:
                        krylov_step, gmres_info = gmres(
                            right_preconditioned_jacobian,
                            gmres_rhs,
                            restart=krylov_dimension,
                            maxiter=options.gmres_max_iterations,
                            rtol=min(
                                0.5,
                                max(1.0e-3, math.sqrt(max(residual_merit, 0.0))),
                            ),
                            atol=0.0,
                            callback=lambda value: gmres_residual_history.append(
                                float(value)
                            ),
                            callback_type="pr_norm",
                        )
                        newton_step = inverse_jacobian.matvec(krylov_step)
                        if np.all(np.isfinite(newton_step)):
                            if complex_coordinates:
                                newton_step = np.concatenate(
                                    (np.real(newton_step), np.imag(newton_step))
                                )
                            proposed_step = _limit_step(
                                newton_step, options.newton_step_max
                            )
                            step_kind = "newton-krylov"
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                        gmres_info = f"{type(error).__name__}: {error}"
                    newton_report = {
                        "iteration": iteration,
                        "info": gmres_info,
                        "jvp_calls": jvp_calls,
                        "coordinate_system": gmres_coordinates,
                        "preconditioning": "right",
                        "globalization": "line-search",
                        "denominator_scaled_level_shift": level_shift,
                        "multisecant_history": len(broyden.steps),
                        "krylov_dimension": krylov_dimension,
                        "krylov_residual_history": gmres_residual_history,
                    }
                gmres_reports.append(newton_report)
            use_newton = False

        if proposed_step is None:
            candidate = None
            if len(vectors) >= 2 and (
                prefer_anderson or (len(vectors) >= 4 and iteration % 4 == 0)
            ):
                candidate = _anderson_candidate(
                    vectors[-options.broyden_history :],
                    residuals[-options.broyden_history :],
                    broyden.inverse_diagonal,
                    current,
                    trust_radius,
                )
                if candidate is not None:
                    step_kind = "anderson"
            prefer_anderson = False
            proposed_step = (
                _limit_step(candidate - current, trust_radius)
                if candidate is not None
                else _limit_step(-broyden.apply(residual), trust_radius)
            )
        proposed_step = project_optimizer_vector(proposed_step)
        accepted: tuple[
            np.ndarray, KernelEvaluation, np.ndarray, float, float, float, str
        ] | None = None
        accepted_hookstep_ratio: float | None = None
        best_rejected_trial: tuple[float, float, float] | None = None
        last_hookstep_ratio: float | None = None
        scale = 1.0
        if step_kind.startswith("newton-krylov-hookstep"):
            # Re-solving the compact regularized model at a smaller radius is
            # both cheaper and more robust than scaling the same direction.
            backtrack_limit = 0
        else:
            backtrack_limit = (
                options.newton_max_backtracks
                if "newton" in step_kind
                else options.max_backtracks
            )
        for backtrack in range(backtrack_limit + 1):
            if evaluator_calls >= options.max_evaluator_calls:
                break
            trial = current + scale * proposed_step
            trial_evaluation = evaluator(optimizer_amplitudes(trial))
            evaluator_calls += 1
            total_projection_time += float(trial_evaluation.diagnostics.get("projection_time", 0.0))
            trial_residual, trial_imaginary = _optimizer_residual(
                trial_evaluation,
                space,
                complex_coordinates=complex_coordinates,
                residual_basis=residual_basis,
                gauge_weight=gauge_weight,
            )
            trial_merit, _trial_gauge_norm, _trial_full_norm = _residual_component_norms(
                trial_evaluation, space, residual_basis
            )
            trial_l2 = float(np.linalg.norm(trial_residual))
            if (
                best_rejected_trial is None
                or trial_merit < best_rejected_trial[0]
            ):
                best_rejected_trial = (trial_merit, trial_l2, scale)
            if (
                step_kind.startswith("newton-krylov-hookstep")
                and newton_report is not None
            ):
                predicted_l2 = float(
                    newton_report.get("predicted_residual_l2_norm", math.inf)
                )
                predicted_reduction = residual_l2**2 - predicted_l2**2
                actual_reduction = residual_l2**2 - trial_l2**2
                last_hookstep_ratio = (
                    actual_reduction / predicted_reduction
                    if predicted_reduction > 0.0
                    else -math.inf
                )
                maximum_allowed = residual_merit * (
                    1.0 + options.newton_hookstep_maximum_growth
                )
                acceptance_mode = (
                    "hookstep-l2"
                    if (
                        actual_reduction > 0.0
                        and last_hookstep_ratio
                        >= options.newton_hookstep_acceptance_ratio
                        and trial_merit <= maximum_allowed
                    )
                    else None
                )
            else:
                window_limit = (
                    max(merit_window[-options.nonmonotone_window :])
                    * options.nonmonotone_factor
                )
                current_limit = residual_merit * options.nonmonotone_factor
                acceptance_limit = min(window_limit, current_limit)
                acceptance_mode = _trial_acceptance_mode(
                    current_merit=residual_merit,
                    current_l2=residual_l2,
                    trial_merit=trial_merit,
                    trial_l2=trial_l2,
                    primary_limit=acceptance_limit,
                    options=options,
                )
            if acceptance_mode is not None:
                accepted_step_kind = (
                    step_kind
                    if backtrack == 0
                    else f"{step_kind}-safeguarded"
                )
                if acceptance_mode == "l2-filter":
                    accepted_step_kind = f"{accepted_step_kind}-filter"
                elif acceptance_mode == "hookstep-l2":
                    accepted_step_kind = f"{accepted_step_kind}-least-squares"
                    accepted_hookstep_ratio = last_hookstep_ratio
                accepted = (
                    trial,
                    trial_evaluation,
                    trial_residual,
                    trial_imaginary,
                    trial_merit,
                    trial_l2,
                    accepted_step_kind,
                )
                break
            scale *= options.backtrack_shrink

        if accepted is None:
            rejected_steps += 1
            if attempted_newton:
                consecutive_rejections = 0
                if step_kind.startswith("newton-krylov-hookstep"):
                    can_retry_smaller_hookstep = bool(
                        hookstep_model_is_current()
                        and current_newton_trust_radius
                        > options.newton_hookstep_min_radius * (1.0 + 1.0e-12)
                    )
                    if can_retry_smaller_hookstep:
                        current_newton_trust_radius = max(
                            options.newton_hookstep_min_radius,
                            current_newton_trust_radius * options.backtrack_shrink,
                        )
                        hookstep_retry = True
                        hookstep_retries += 1
                        use_newton = True
                        prefer_anderson = False
                    else:
                        use_newton = False
                        prefer_anderson = True
                else:
                    can_retry_with_more_damping = bool(
                        adaptive_newton_damping
                        and attempted_level_shift is not None
                        and attempted_level_shift < options.newton_level_shift_max
                        and jvp_calls + 1 < options.max_jvp_calls
                    )
                    if can_retry_with_more_damping:
                        seed_shift = (
                            attempted_level_shift * options.newton_level_shift_growth
                            if attempted_level_shift > 0.0
                            else max(math.sqrt(max(residual_merit, 0.0)), 1.0e-8)
                        )
                        current_newton_level_shift = min(
                            options.newton_level_shift_max,
                            seed_shift,
                        )
                        newton_damping_retry = True
                        newton_damping_retries += 1
                        use_newton = True
                        prefer_anderson = False
                    else:
                        # A fixed or maximally damped model must move to a new
                        # amplitude state before rebuilding the same Jacobian.
                        use_newton = False
                        prefer_anderson = True
            else:
                consecutive_rejections += 1
                if step_kind == "broyden":
                    prefer_anderson = True
                if (
                    (
                        consecutive_rejections >= options.newton_after_rejections
                        or _deep_line_search_exhausted(
                            residual_merit,
                            best_rejected_trial,
                            backtrack_limit,
                        )
                    )
                    and options.newton_krylov
                    and jvp_evaluator is not None
                    and newton_state_has_changed()
                ):
                    # An accepted Newton step changes the linearization point.
                    # If every cheap direction then fails, the cooldown cannot
                    # be satisfied and must not permanently block a fresh model.
                    newton_rejection_override = True
                    use_newton = True
                    consecutive_rejections = 0
            # A rejected step diagnoses its direction, not the size of the valid
            # trust region.  In particular, a tiny but uphill denominator step
            # must not force the following Newton step into the same tiny radius.
            if not attempted_newton:
                trust_radius = max(
                    options.minimum_step_max,
                    trust_radius * options.backtrack_shrink,
                )
            history.append(
                {
                    "iteration": iteration,
                    "energy": float(np.real(evaluation.total_energy)),
                    "residual_norm": residual_merit,
                    "full_residual_norm": full_residual_norm,
                    "gauge_residual_norm": gauge_residual_norm,
                    "residual_l2_norm": residual_l2,
                    "trust_radius": trust_radius,
                    "newton_trust_radius": current_newton_trust_radius,
                    "step": f"{step_kind}-rejected",
                    "trial_residual_norm": (
                        None
                        if best_rejected_trial is None
                        else best_rejected_trial[0]
                    ),
                    "trial_residual_l2_norm": (
                        None
                        if best_rejected_trial is None
                        else best_rejected_trial[1]
                    ),
                    "trial_step_scale": (
                        None
                        if best_rejected_trial is None
                        else best_rejected_trial[2]
                    ),
                }
            )
            if newton_report is not None:
                newton_report.update(
                    {
                        "accepted": False,
                        "nonlinear_backtracks": backtrack_limit,
                        "nonlinear_trust_ratio": last_hookstep_ratio,
                    }
                )
            rejection_details = {}
            if best_rejected_trial is not None:
                rejection_details = {
                    "trial_residual_norm": best_rejected_trial[0],
                    "trial_residual_l2_norm": best_rejected_trial[1],
                    "trial_step_scale": best_rejected_trial[2],
                }
            notify_iteration(
                iteration,
                f"{step_kind}-rejected",
                accepted=False,
                details=rejection_details,
            )
            if (
                trust_radius <= options.minimum_step_max
                and not newton_damping_retry
                and not hookstep_retry
            ):
                break
            continue

        (
            trial,
            trial_evaluation,
            trial_residual,
            trial_imaginary,
            trial_merit,
            trial_l2,
            step_kind,
        ) = accepted
        consecutive_rejections = 0
        prefer_anderson = False
        if dense_jacobian is not None:
            dense_step = trial - current
            dense_change = trial_residual - residual
            if complex_coordinates:
                dense_step = (
                    dense_step[:coordinate_count]
                    + 1.0j * dense_step[coordinate_count:]
                )
                dense_change = (
                    dense_change[:coordinate_count]
                    + 1.0j * dense_change[coordinate_count:]
                )
            denominator = np.vdot(dense_step, dense_step)
            if abs(denominator) > 1.0e-28:
                defect = dense_change - dense_jacobian @ dense_step
                dense_jacobian += np.outer(defect, dense_step.conj()) / denominator
        accepted_step = trial - current
        accepted_residual_change = trial_residual - residual
        secant_retained = broyden.update(accepted_step, accepted_residual_change)
        if (
            secant_retained
            and projected
            and options.adaptive_block_preconditioner
            and coordinate_count >= 16
        ):
            factors = broyden.update_block_scaling(
                accepted_step,
                accepted_residual_change,
                preconditioner_groups,
                minimum_scale=options.block_preconditioner_minimum_scale,
                maximum_scale=options.block_preconditioner_maximum_scale,
            )
            block_scale_history.append(
                {
                    "iteration": iteration,
                    "pair_scale": factors[0] if factors else 1.0,
                    "quadruple_scale": factors[1] if len(factors) > 1 else None,
                }
            )
        old_merit = residual_merit
        energy_change = abs(complex(trial_evaluation.total_energy) - previous_energy)
        current = trial
        evaluation = trial_evaluation
        residual = trial_residual
        residual_merit = trial_merit
        residual_merit, gauge_residual_norm, full_residual_norm = (
            _residual_component_norms(evaluation, space, residual_basis)
        )
        residual_l2 = trial_l2
        max_imaginary = trial_imaginary
        previous_energy = complex(evaluation.total_energy)
        accepted_newton_step = "newton" in step_kind
        if accepted_newton_step:
            accepted_since_newton = 0
            if step_kind.startswith("newton-krylov-hookstep"):
                if accepted_hookstep_ratio is not None:
                    step_maximum = float(
                        0.0
                        if newton_report is None
                        else newton_report.get("step_maximum", 0.0)
                    )
                    if accepted_hookstep_ratio < 0.25:
                        current_newton_trust_radius = max(
                            options.newton_hookstep_min_radius,
                            current_newton_trust_radius
                            * options.backtrack_shrink,
                        )
                    elif (
                        accepted_hookstep_ratio > 0.75
                        and step_maximum
                        >= 0.8 * current_newton_trust_radius
                    ):
                        current_newton_trust_radius = min(
                            options.newton_step_max,
                            current_newton_trust_radius * options.trust_expand,
                        )
            elif adaptive_newton_damping and attempted_level_shift is not None:
                relative_newton_progress = (
                    (old_merit - residual_merit) / max(old_merit, 1.0e-30)
                )
                base_shift = _adaptive_newton_level_shift(
                    residual_merit,
                    None,
                )
                if scale == 1.0 and relative_newton_progress >= 0.25:
                    current_newton_level_shift = max(
                        base_shift,
                        attempted_level_shift / options.newton_level_shift_growth,
                    )
                else:
                    current_newton_level_shift = max(
                        base_shift,
                        attempted_level_shift,
                    )
        else:
            accepted_since_newton += 1
        # Every accepted step changes the tangent point. Retaining only the
        # compact model for rejected same-state radius retries bounds memory.
        hookstep_model = None
        hookstep_model_state = None
        hookstep_retry = False
        vectors.append(current.copy())
        residuals.append(residual.copy())
        keep = max(2, options.broyden_history)
        vectors = vectors[-keep:]
        residuals = residuals[-keep:]
        merit_window.append(residual_merit)
        merit_window = merit_window[-options.nonmonotone_window :]
        if step_kind.startswith("newton-krylov-hookstep"):
            # A hookstep is accepted on the residual least-squares model. Its
            # temporary max-equation growth must not leave the independent
            # multisecant radius collapsed by earlier rejected directions.
            trust_radius = max(trust_radius, current_newton_trust_radius)
        elif step_kind.startswith("newton-krylov") and residual_merit < old_merit:
            trust_radius = max(trust_radius, options.newton_step_max)
        if residual_merit < 0.5 * old_merit:
            trust_radius = min(options.step_max, trust_radius * options.trust_expand)
        elif residual_merit > old_merit:
            trust_radius = max(
                options.minimum_step_max,
                trust_radius * options.backtrack_shrink,
            )
        previous_best_merit = best_state[4]
        relative_progress = (
            (previous_best_merit - residual_merit) / max(previous_best_merit, 1.0e-30)
        )
        if residual_merit < previous_best_merit:
            best_state = (
                current.copy(),
                evaluation,
                residual.copy(),
                max_imaginary,
                residual_merit,
                residual_l2,
            )
        if relative_progress >= options.stagnation_relative_improvement:
            stagnation_count = 0
        else:
            stagnation_count += 1
        previous_best_convergence = best_convergence_norm
        relative_convergence_progress = (
            (previous_best_convergence - residual_merit)
            / max(previous_best_convergence, 1.0e-30)
        )
        if residual_merit < previous_best_convergence:
            best_convergence_norm = residual_merit
            best_convergence_state = (
                current.copy(),
                evaluation,
                residual.copy(),
                max_imaginary,
                residual_merit,
                residual_l2,
            )
        if relative_convergence_progress >= options.stagnation_relative_improvement:
            convergence_stagnation_count = 0
        else:
            convergence_stagnation_count += 1
        if (
            stagnation_count >= options.stagnation_iterations
            or convergence_stagnation_count >= options.stagnation_iterations
        ):
            if newton_is_ready():
                use_newton = True
                stagnation_count = 0
                convergence_stagnation_count = 0
        if newton_report is not None:
            if accepted_newton_step:
                newton_report.update(
                    {
                        "accepted": True,
                        "nonlinear_step_scale": scale,
                        "nonlinear_backtracks": backtrack,
                        "nonlinear_relative_improvement": (
                            (old_merit - residual_merit)
                            / max(old_merit, 1.0e-30)
                        ),
                        "nonlinear_trust_ratio": accepted_hookstep_ratio,
                    }
                )
            else:
                newton_report.update(
                    {
                        "accepted": False,
                        "fallback_step": step_kind,
                    }
                )
        history.append(
            {
                "iteration": iteration,
                "energy": float(np.real(evaluation.total_energy)),
                "residual_norm": residual_merit,
                "full_residual_norm": full_residual_norm,
                "gauge_residual_norm": gauge_residual_norm,
                "residual_l2_norm": residual_l2,
                "energy_change": float(energy_change),
                "trust_radius": trust_radius,
                "newton_trust_radius": current_newton_trust_radius,
                "step": step_kind,
            }
        )
        converged = (
            residual_merit < 0.1 * options.residual_tolerance
            or (
                residual_merit < options.residual_tolerance
                and energy_change < options.energy_tolerance
            )
        )
        notify_iteration(iteration, step_kind)

    if not converged and best_convergence_norm < residual_merit:
        (
            current,
            evaluation,
            residual,
            max_imaginary,
            residual_merit,
            residual_l2,
        ) = best_convergence_state
        previous_energy = complex(evaluation.total_energy)
        residual_merit, gauge_residual_norm, full_residual_norm = (
            _residual_component_norms(evaluation, space, residual_basis)
        )
        converged = residual_merit < options.residual_tolerance

    remaining_evaluations = max(0, options.max_evaluator_calls - evaluator_calls)
    if not converged and options.spectral_fallback and remaining_evaluations > 0:
        from scipy.optimize import root

        spectral_diagnostics["spectral_fallback_used"] = True
        spectral_best = (
            current.copy(),
            evaluation,
            residual.copy(),
            max_imaginary,
            residual_merit,
            residual_l2,
        )

        def spectral_objective(vector: np.ndarray) -> np.ndarray:
            nonlocal evaluator_calls, total_projection_time, spectral_best
            trial_evaluation = evaluator(optimizer_amplitudes(vector))
            evaluator_calls += 1
            total_projection_time += float(
                trial_evaluation.diagnostics.get("projection_time", 0.0)
            )
            trial_residual, trial_imaginary = _optimizer_residual(
                trial_evaluation,
                space,
                complex_coordinates=complex_coordinates,
                residual_basis=residual_basis,
                gauge_weight=gauge_weight,
            )
            trial_merit, _trial_gauge_norm, _trial_full_norm = _residual_component_norms(
                trial_evaluation, space, residual_basis
            )
            trial_l2 = float(np.linalg.norm(trial_residual))
            if trial_merit < spectral_best[4]:
                spectral_best = (
                    np.asarray(vector, dtype=float).copy(),
                    trial_evaluation,
                    trial_residual,
                    trial_imaginary,
                    trial_merit,
                    trial_l2,
                )
            return trial_residual

        spectral = root(
            spectral_objective,
            current,
            method="df-sane",
            options={
                "fatol": 0.1 * options.residual_tolerance,
                "ftol": 1.0e-12,
                "maxfev": min(
                    options.spectral_max_evaluations,
                    remaining_evaluations,
                ),
                "line_search": "cruz",
            },
        )
        spectral_objective(np.asarray(spectral.x, dtype=float))
        if spectral_best[4] < residual_merit:
            (
                current,
                evaluation,
                residual,
                max_imaginary,
                residual_merit,
                residual_l2,
            ) = spectral_best
            previous_energy = complex(evaluation.total_energy)
            history.append(
                {
                    "iteration": len(history) + 1,
                    "energy": float(np.real(evaluation.total_energy)),
                    "residual_norm": residual_merit,
                    "residual_l2_norm": residual_l2,
                    "step": "spectral",
                }
            )
        residual_merit, gauge_residual_norm, full_residual_norm = (
            _residual_component_norms(evaluation, space, residual_basis)
        )
        converged = residual_merit < options.residual_tolerance
        notify_iteration(len(history), "spectral")
        spectral_diagnostics.update(
            {
                "spectral_fallback_success": bool(spectral.success),
                "spectral_fallback_message": str(spectral.message),
                "spectral_fallback_nfev": int(spectral.nfev),
            }
        )

    # A rejected trial does not invalidate an already converged accepted state.
    # The projected equations are defined by their physical residual; the
    # energy-change check is an early-stop safeguard, not an independent root
    # condition after the evaluator budget is exhausted.
    converged = converged or residual_merit < options.residual_tolerance
    final_amplitudes = optimizer_amplitudes(current)
    total_evaluator_calls = evaluator_calls + jacobian_evaluator_calls
    return QPCCSDResult(
        converged=converged,
        total_energy=evaluation.total_energy,
        correlation_energy=evaluation.correlation_energy,
        amplitudes=final_amplitudes,
        residual_norm=residual_merit,
        iterations=len(history),
        requested_method=requested_method,
        canonical_method="qpccsd",
        projected=projected,
        projection_time=total_projection_time + jvp_projection_time,
        hfb_pairing_collapsed=None if reference is None else reference.pairing_collapsed,
        internal_residual_norm=float(
            evaluation.diagnostics.get("internal_residual_norm", 0.0)
        ),
        allowed_pair_count=space.pair_count,
        allowed_quadruple_count=space.quadruple_count,
        forbidden_amplitude_norm=space.forbidden_amplitude_norm(
            final_amplitudes
        ),
        raw_projected_energy=evaluation.total_energy if projected else None,
        projection_equation_schema=evaluation.diagnostics.get(
            "projection_equation_schema"
        ),
        projection_solver_schema=evaluation.diagnostics.get(
            "projection_solver_schema"
        ),
        projector_ordering=evaluation.diagnostics.get("projector_ordering"),
        allowed_residual=np.asarray(
            residual_to_vector(evaluation.r1, evaluation.r2, space)
        ).copy(),
        physical_residual_norm=residual_merit,
        gauge_residual_norm=gauge_residual_norm,
        full_residual_norm=full_residual_norm,
        gauge_rank=0 if residual_basis is None else residual_basis.gauge_rank,
        gauge_identity_defect=(
            None if residual_basis is None else residual_basis.identity_defect
        ),
        history=history,
        diagnostics={
            **evaluation.diagnostics,
            "max_imaginary_residual": max_imaginary,
            "optimizer": (
                "limited-memory multisecant Broyden/Anderson with matrix-free "
                f"Newton {options.newton_globalization} globalization"
            ),
            "evaluator_calls": evaluator_calls,
            "jacobian_evaluator_calls": jacobian_evaluator_calls,
            "total_evaluator_calls": total_evaluator_calls,
            "max_evaluator_calls": options.max_evaluator_calls,
            "evaluator_budget_exhausted": bool(
                evaluator_calls >= options.max_evaluator_calls and not converged
            ),
            "rejected_steps": rejected_steps,
            "final_trust_radius": trust_radius,
            "analytic_jvp_calls": jvp_calls,
            "jvp_projection_time": jvp_projection_time,
            "residual_projection_time": total_projection_time,
            "max_jvp_calls": options.max_jvp_calls,
            "newton_krylov_attempts": newton_attempts,
            "newton_globalization": options.newton_globalization,
            "newton_cooldown_iterations": options.newton_cooldown_iterations,
            "accepted_since_newton": accepted_since_newton,
            "newton_rejection_override": newton_rejection_override,
            "adaptive_newton_damping": adaptive_newton_damping,
            "final_newton_level_shift": current_newton_level_shift,
            "newton_damping_retries": newton_damping_retries,
            "hookstep_model_builds": hookstep_model_builds,
            "hookstep_model_reuses": hookstep_model_reuses,
            "hookstep_retries": hookstep_retries,
            "final_newton_trust_radius": current_newton_trust_radius,
            "gmres_reports": gmres_reports,
            "dense_jacobian_builds": dense_jacobian_builds,
            "dense_newton_derivative": options.dense_newton_derivative,
            "jacobian_projection_time": jacobian_projection_time,
            "dense_newton_reports": dense_newton_reports,
            "dense_step_reports": dense_step_reports,
            "final_residual_l2_norm": residual_l2,
            "best_residual_norm": residual_merit,
            "physical_residual_norm": residual_merit,
            "gauge_residual_norm": gauge_residual_norm,
            "full_residual_norm": full_residual_norm,
            "gauge_rank": 0 if residual_basis is None else residual_basis.gauge_rank,
            "gauge_identity_defect": (
                None if residual_basis is None else residual_basis.identity_defect
            ),
            "gauge_basis_source": (
                None if residual_basis is None else residual_basis.source
            ),
            "gauge_singular_values": (
                []
                if residual_basis is None
                else residual_basis.singular_values.tolist()
            ),
            "gauge_weight": gauge_weight,
            "adaptive_block_preconditioner": options.adaptive_block_preconditioner,
            "block_preconditioner_scale_history": block_scale_history,
            **quasiparticle_preconditioner_diagnostics,
            "complex_amplitude_coordinates": complex_coordinates,
            **spectral_diagnostics,
            "average_projection_evaluation_time": (
                total_projection_time / total_evaluator_calls
                if projected and total_evaluator_calls
                else 0.0
            ),
        },
    )


def _resolve_reference_and_space(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference | None,
    excitation_space: QPExcitationSpaceLike | None,
) -> tuple[
    BogoliubovReference | None,
    CASQPReference | None,
    QPExcitationSpaceLike,
]:
    cas_reference = reference if isinstance(reference, CASQPReference) else None
    bogoliubov = cas_reference.bogoliubov if cas_reference is not None else reference
    if bogoliubov is not None and bogoliubov.nspin != hamiltonian.nspin:
        raise ValueError("reference and Hamiltonian dimensions do not match")
    if excitation_space is None:
        excitation_space = (
            build_block_spin_adapted_qp_space(cas_reference)
            if cas_reference is not None
            else _full_excitation_space(hamiltonian.nspin)
        )
    if excitation_space.nspin != hamiltonian.nspin:
        raise ValueError("excitation space and Hamiltonian dimensions do not match")
    return bogoliubov, cas_reference, excitation_space


def _apply_cas_energy_bookkeeping(
    result: QPCCSDResult,
    reference: CASQPReference,
    zero_amplitude_energy: complex,
) -> QPCCSDResult:
    raw_energy = complex(result.total_energy)
    dynamic_energy = raw_energy - complex(zero_amplitude_energy)
    delta_total = complex(reference.casscf_energy) + dynamic_energy
    result.reference_energy = reference.casscf_energy
    result.zero_amplitude_energy = zero_amplitude_energy
    result.dynamic_correlation_energy = np.real_if_close(dynamic_energy)
    result.raw_projected_energy = np.real_if_close(raw_energy) if result.projected else None
    result.casscf_plus_dynamic_delta = np.real_if_close(delta_total)
    result.baseline_gap = np.real_if_close(raw_energy - delta_total)
    result.total_energy = np.real_if_close(delta_total)
    result.correlation_energy = np.real_if_close(dynamic_energy)
    result.rdm_cumulant_norm = reference.rdm_cumulant_norm
    for entry in result.history:
        if "energy" in entry:
            entry["raw_qp_energy"] = entry["energy"]
            entry["energy"] = float(
                reference.casscf_energy
                + complex(entry["raw_qp_energy"] - zero_amplitude_energy).real
            )
    result.diagnostics.update(
        {
            "energy_convention": EnergyConvention.CASSCF_PLUS_DELTA.value,
            "energy_definition": EnergyConvention.CASSCF_PLUS_DELTA.description,
            "casscf_energy_added": True,
            "cas_plus_delta_applied": True,
            "raw_qp_energy": np.real_if_close(raw_energy),
            "raw_projected_energy": np.real_if_close(raw_energy)
            if result.projected
            else None,
            "casscf_plus_dynamic_delta": np.real_if_close(delta_total),
            "baseline_gap": np.real_if_close(raw_energy - delta_total),
            "casscf_reference_energy": reference.casscf_energy,
            "zero_amplitude_qp_energy": np.real_if_close(zero_amplitude_energy),
            "dynamic_correlation_energy": np.real_if_close(dynamic_energy),
            "rdm_cumulant_norm": reference.rdm_cumulant_norm,
            "physical_target_number": reference.physical_target_number,
            "correlated_target_number": reference.target_number,
            "reference_mode": reference.reference_mode,
            "reference_reconstruction_metrics": dict(
                reference.reconstruction_metrics
            ),
        }
    )
    return result


def _record_direct_energy(
    result: QPCCSDResult,
    reference: CASQPReference | None,
) -> QPCCSDResult:
    """Attach an explicit direct-energy contract without changing the energy."""

    raw_energy = np.real_if_close(complex(result.total_energy))
    result.diagnostics.update(
        {
            "energy_convention": EnergyConvention.DIRECT.value,
            "energy_definition": EnergyConvention.DIRECT.description,
            "raw_qp_energy": raw_energy,
            "casscf_energy_added": False,
            "cas_plus_delta_applied": False,
        }
    )
    if reference is not None:
        result.rdm_cumulant_norm = reference.rdm_cumulant_norm
        result.diagnostics.update(
            {
                "casscf_reference_energy_diagnostic": reference.casscf_energy,
                "physical_target_number": reference.physical_target_number,
                "correlated_target_number": reference.target_number,
                "reference_mode": reference.reference_mode,
                "reference_reconstruction_metrics": dict(
                    reference.reconstruction_metrics
                ),
            }
        )
    return result


def solve_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference | None = None,
    *,
    initial_amplitudes: QPAmplitudes | None = None,
    options: SolverOptions | None = None,
    requested_method: str = "qpccsd",
    excitation_space: QPExcitationSpaceLike | None = None,
    iteration_callback: IterationCallback | None = None,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
) -> QPCCSDResult:
    """Solve QPCCSD with an explicit, direct-by-default energy convention."""

    convention = normalize_energy_convention(energy_convention)
    options = SolverOptions() if options is None else options
    bogoliubov, cas_reference, space = _resolve_reference_and_space(
        hamiltonian, reference, excitation_space
    )
    result = _solve(
        lambda amplitudes: evaluate_qpccsd(
            hamiltonian, amplitudes, excitation_space=space
        ),
        hamiltonian,
        bogoliubov,
        initial_amplitudes,
        options,
        requested_method,
        projected=False,
        excitation_space=space,
        iteration_callback=iteration_callback,
        jvp_evaluator=lambda amplitudes, direction: evaluate_qpccsd_jvp(
            hamiltonian,
            amplitudes,
            direction,
        ),
    )
    if convention is EnergyConvention.CASSCF_PLUS_DELTA:
        if cas_reference is None:
            raise ValueError(
                "casscf_plus_delta requires a CASQPReference with a CASSCF energy"
            )
        zero = evaluate_qpccsd(
            hamiltonian,
            QPAmplitudes.zeros(hamiltonian.nspin),
            excitation_space=space,
        )
        _apply_cas_energy_bookkeeping(result, cas_reference, zero.total_energy)
    else:
        _record_direct_energy(result, cas_reference)
    return result


def solve_lbccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference | None = None,
    *,
    initial_amplitudes: QPAmplitudes | None = None,
    options: SolverOptions | None = None,
    excitation_space: QPExcitationSpaceLike | None = None,
    iteration_callback: IterationCallback | None = None,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
) -> QPCCSDResult:
    return solve_qpccsd(
        hamiltonian,
        reference,
        initial_amplitudes=initial_amplitudes,
        options=options,
        requested_method="lbccsd",
        excitation_space=excitation_space,
        iteration_callback=iteration_callback,
        energy_convention=energy_convention,
    )


solve_qp_ccsd = solve_qpccsd


class QPSolver:
    """Compatibility facade backed exclusively by generated tensor kernels."""

    def __init__(
        self,
        hamiltonian: QPHamiltonian,
        reference: BogoliubovReference | CASQPReference | None = None,
        excitation_space: QPExcitationSpaceLike | None = None,
    ) -> None:
        if not isinstance(hamiltonian, QPHamiltonian):
            raise TypeError("QPSolver requires a production QPHamiltonian")
        self.hamiltonian = hamiltonian
        self.reference = reference
        _bogoliubov, _cas, self.excitation_space = _resolve_reference_and_space(
            hamiltonian, reference, excitation_space
        )

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        return evaluate_qpccsd(
            self.hamiltonian,
            amplitudes,
            excitation_space=self.excitation_space,
        )

    compute_unprojected = evaluate

    def solve_unprojected(
        self,
        *,
        initial_amplitudes: QPAmplitudes | None = None,
        options: SolverOptions | None = None,
        energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
    ) -> QPCCSDResult:
        return solve_qpccsd(
            self.hamiltonian,
            self.reference,
            initial_amplitudes=initial_amplitudes,
            options=options,
            excitation_space=self.excitation_space,
            energy_convention=energy_convention,
        )

    def solve_projected(self, **kwargs) -> QPCCSDResult:
        if self.reference is None:
            raise ValueError("number projection requires a BogoliubovReference")
        from .projection import solve_projected_qpccsd

        return solve_projected_qpccsd(
            self.hamiltonian,
            self.reference,
            excitation_space=self.excitation_space,
            **kwargs,
        )
