from __future__ import annotations

from collections.abc import Iterator
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import os
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..generated import generated_wick_orbits
from .contracts import EnergyConvention, normalize_energy_convention
from .hamiltonian import build_qp_hamiltonian
from .excitation_space import QPExcitationSpace, QPExcitationSpaceLike
from .models import (
    BogoliubovReference,
    CASQPReference,
    ExecutionOptions,
    GaugeModeOptions,
    KernelDirectionalDerivative,
    KernelEvaluation,
    ProjectionContinuationOptions,
    ProjectionOptions,
    QPAmplitudes,
    QPCCSDResult,
    QPPAVEvaluation,
    QPPAVResult,
    QPHamiltonian,
    ProjectedResidualBasis,
    SolverOptions,
)
from .production import (
    IterationCallback,
    _apply_cas_energy_bookkeeping,
    _resolve_reference_and_space,
    _solve,
    evaluate_qpccsd,
    evaluate_qpccsd_with_jvp,
    residual_to_vector,
    solve_qpccsd,
)
from ..generated.generated_projected_moments import (
    PROJECTOR_ORDERING,
    PROJECTION_EQUATION_SCHEMA,
    projected_left_moments,
    projected_left_norm_moments,
    projected_left_vacuum_norm_moments,
)
from ..generated.generated_disentangled_series import (
    SERIES_EQUATION_SCHEMA,
    evaluate_disentangled_series,
    evaluate_disentangled_series_with_jvp,
    principal_sqrt_determinant,
)
from .projected_metric import ProjectedMetricProjector
from .wick import ScalarSimilarityTransformKernel, similarity_transform_deexcitation

_BINARY_EINSUM_PATH = ("einsum_path", (0, 1))
PROJECTION_SOLVER_SCHEMA = "pn-oap-metric-richardson-backends-v3"


def _antisymmetrize_pair(tensor: np.ndarray) -> np.ndarray:
    return 0.5 * (tensor - tensor.T)


def _transition_matrices(
    reference: BogoliubovReference,
    angle: complex,
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    gauge = np.exp(1.0j * angle)
    rotated_u = reference.U / gauge
    rotated_v = gauge * reference.V
    overlap_matrix = reference.U.conj().T @ rotated_u + reference.V.conj().T @ rotated_v
    pairing_matrix = reference.U.conj().T @ rotated_v + reference.V.conj().T @ rotated_u
    raw = np.linalg.solve(overlap_matrix.T, pairing_matrix).T
    # This sign follows beta = U^dag c + V^dag c^dag and
    # <Phi|R(angle) = <Phi|R(angle)|Phi> <Phi|exp(Z).
    z = -0.5 * (raw - raw.T)
    overlap = complex(
        np.prod(reference.u[::2] ** 2 + gauge * gauge * reference.v[::2] ** 2)
    )
    return overlap, z, overlap_matrix, gauge


def reference_thouless(
    reference: BogoliubovReference,
    angle: complex,
) -> tuple[complex, np.ndarray]:
    overlap, z, _overlap_matrix, _gauge = _transition_matrices(reference, angle)
    return overlap, z


def build_number_operator(reference: BogoliubovReference) -> QPHamiltonian:
    nspin = reference.nspin
    return build_qp_hamiltonian(
        np.eye(nspin),
        np.zeros((nspin,) * 4),
        reference,
    )


def transform_number_operator(
    reference: BogoliubovReference,
    z: np.ndarray,
    *,
    number_operator: QPHamiltonian | None = None,
) -> QPHamiltonian:
    """Evaluate A_Z = exp(Z) N exp(-Z) as a terminating polynomial."""

    bare = build_number_operator(reference) if number_operator is None else number_operator
    return similarity_transform_deexcitation(bare, z)


def _one_body_a02_after_deexcitation(
    operator: QPHamiltonian,
    z: np.ndarray,
) -> np.ndarray:
    """Return only the transformed 02 block of a one-body operator."""

    z = np.asarray(z)
    linear = np.asarray(operator.h11).T @ z
    return (
        np.asarray(operator.h02)
        + linear
        - linear.T
        - z.T @ np.asarray(operator.h20) @ z
    )


def _number_a02(
    angle: complex,
    reference: BogoliubovReference,
    number_operator: QPHamiltonian,
    a02_cache: dict[tuple[float, float], np.ndarray] | None = None,
) -> np.ndarray:
    key = (round(float(np.real(angle)), 15), round(float(np.imag(angle)), 15))
    a02 = None if a02_cache is None else a02_cache.get(key)
    if a02 is None:
        _overlap, z = reference_thouless(reference, angle)
        a02 = _one_body_a02_after_deexcitation(number_operator, z)
        if a02_cache is not None:
            a02_cache[key] = np.asarray(a02)
    return np.asarray(a02)


def _apply_one_body_to_rank4(matrix: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    """Apply one matrix to each index of a rank-four tensor.

    The disentanglement equation contains the same ``w1 @ a02`` contraction
    on all four indices of ``w2``.  Forming that matrix once and using four
    matrix products avoids four repeated three-operand einsums.
    """

    dimension = tensor.shape[0]
    rows, columns = np.nonzero(matrix)
    if rows.size < dimension * dimension // 2:
        first_index = np.zeros_like(tensor, dtype=np.result_type(matrix, tensor))
        for row, column in zip(rows, columns):
            first_index[row] += matrix[row, column] * tensor[column]
    else:
        first_index = (matrix @ tensor.reshape(dimension, -1)).reshape(tensor.shape)
    # For an antisymmetric input, actions on indices 2--4 are signed views of
    # the first-index contraction.  This is the exact four-term derivation of
    # the normalized antisymmetrizer and avoids its rank-four temporaries.
    return (
        first_index
        - first_index.transpose(1, 0, 2, 3)
        + first_index.transpose(1, 2, 0, 3)
        - first_index.transpose(1, 2, 3, 0)
    )


def _right_multiply_structurally_sparse(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Multiply by a diagonal/permutation-sparse contour matrix exactly."""

    rows, columns = np.nonzero(right)
    if (
        rows.size <= right.shape[0]
        and np.unique(rows).size == rows.size
        and np.unique(columns).size == columns.size
    ):
        result = np.zeros_like(left, dtype=np.result_type(left, right))
        result[:, columns] = left[:, rows] * right[rows, columns][None, :]
        return result
    return left @ right


def _contract_sparse_pair_rank4(pair: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(pair)
    if rows.size <= pair.shape[0]:
        if not rows.size:
            return np.zeros(tensor.shape[2:], dtype=np.result_type(pair, tensor))
        return np.einsum(
            "a,aij->ij",
            pair[rows, columns],
            tensor[rows, columns],
            optimize=_BINARY_EINSUM_PATH,
        )
    return np.einsum("ab,abij->ij", pair, tensor, optimize=_BINARY_EINSUM_PATH)


def _w2_rhs(
    angle: complex,
    correlated_norm: complex,
    w1: np.ndarray,
    w2: np.ndarray,
    reference: BogoliubovReference,
    number_operator: QPHamiltonian,
    a02_cache: dict[tuple[float, float], np.ndarray] | None = None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    a02 = _number_a02(
        angle,
        reference,
        number_operator,
        a02_cache,
    )
    scalar = 0.5 * np.einsum("ab,ab->", a02, w1, optimize=_BINARY_EINSUM_PATH)
    dw_norm = 1.0j * scalar * correlated_norm
    one_body_action = _right_multiply_structurally_sparse(w1, a02)
    dw1 = 1.0j * (
        0.5 * _contract_sparse_pair_rank4(a02, w2)
        - one_body_action @ w1.T
    )
    dw2 = 1.0j * _apply_one_body_to_rank4(one_body_action, w2)
    # These contractions are antisymmetric when a02, w1, and w2 are. Avoid
    # 24 rank-four transposes at every Runge-Kutta stage.
    return dw_norm, dw1, dw2


def _w2_rhs_with_jvp(
    angle: complex,
    correlated_norm: complex,
    w1: np.ndarray,
    w2: np.ndarray,
    direction_norm: complex,
    direction_w1: np.ndarray,
    direction_w2: np.ndarray,
    reference: BogoliubovReference,
    number_operator: QPHamiltonian,
    a02_cache: dict[tuple[float, float], np.ndarray] | None = None,
) -> tuple[
    tuple[complex, np.ndarray, np.ndarray],
    tuple[complex, np.ndarray, np.ndarray],
]:
    a02 = _number_a02(
        angle,
        reference,
        number_operator,
        a02_cache,
    )
    scalar = 0.5 * np.einsum("ab,ab->", a02, w1, optimize=_BINARY_EINSUM_PATH)
    one_body_action = _right_multiply_structurally_sparse(w1, a02)
    primal = (
        1.0j * scalar * correlated_norm,
        1.0j
        * (
            0.5
            * _contract_sparse_pair_rank4(a02, w2)
            - one_body_action @ w1.T
        ),
        1.0j * _apply_one_body_to_rank4(one_body_action, w2),
    )
    direction_scalar = 0.5 * np.einsum(
        "ab,ab->", a02, direction_w1, optimize=_BINARY_EINSUM_PATH
    )
    tangent_norm = 1.0j * (
        direction_scalar * correlated_norm + scalar * direction_norm
    )
    direction_one_body_action = _right_multiply_structurally_sparse(
        direction_w1, a02
    )
    tangent_w1 = 1.0j * (
        0.5
        * _contract_sparse_pair_rank4(a02, direction_w2)
        - direction_one_body_action @ w1.T
        - one_body_action @ direction_w1.T
    )
    tangent_w2 = 1.0j * (
        _apply_one_body_to_rank4(direction_one_body_action, w2)
        + _apply_one_body_to_rank4(one_body_action, direction_w2)
    )
    return primal, (tangent_norm, tangent_w1, tangent_w2)


def _rk4_interval(
    start: complex,
    stop: complex,
    correlated_norm: complex,
    w1: np.ndarray,
    w2: np.ndarray,
    reference: BogoliubovReference,
    number_operator: QPHamiltonian,
    substeps: int,
    a02_cache: dict[tuple[float, float], np.ndarray] | None = None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    if stop == start:
        return correlated_norm, w1, w2
    step = (stop - start) / substeps
    angle = complex(start)
    norm = complex(correlated_norm)
    pair = np.asarray(w1, dtype=np.complex128).copy()
    quad = np.asarray(w2, dtype=np.complex128).copy()
    for _ in range(substeps):
        k1 = _w2_rhs(
            angle, norm, pair, quad, reference, number_operator, a02_cache
        )
        k2 = _w2_rhs(
            angle + 0.5 * step,
            norm + 0.5 * step * k1[0],
            pair + 0.5 * step * k1[1],
            quad + 0.5 * step * k1[2],
            reference,
            number_operator,
            a02_cache,
        )
        k3 = _w2_rhs(
            angle + 0.5 * step,
            norm + 0.5 * step * k2[0],
            pair + 0.5 * step * k2[1],
            quad + 0.5 * step * k2[2],
            reference,
            number_operator,
            a02_cache,
        )
        k4 = _w2_rhs(
            angle + step,
            norm + step * k3[0],
            pair + step * k3[1],
            quad + step * k3[2],
            reference,
            number_operator,
            a02_cache,
        )
        norm += (step / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        pair += (step / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        quad += (step / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        angle += step
    return norm, pair, quad


def _rk4_interval_with_jvp(
    start: complex,
    stop: complex,
    correlated_norm: complex,
    w1: np.ndarray,
    w2: np.ndarray,
    direction_norm: complex,
    direction_w1: np.ndarray,
    direction_w2: np.ndarray,
    reference: BogoliubovReference,
    number_operator: QPHamiltonian,
    substeps: int,
    a02_cache: dict[tuple[float, float], np.ndarray] | None = None,
) -> tuple[complex, np.ndarray, np.ndarray, complex, np.ndarray, np.ndarray]:
    if stop == start:
        return (
            correlated_norm,
            w1,
            w2,
            direction_norm,
            direction_w1,
            direction_w2,
        )
    step = (stop - start) / substeps
    angle = complex(start)
    norm = complex(correlated_norm)
    pair = np.asarray(w1, dtype=np.complex128).copy()
    quad = np.asarray(w2, dtype=np.complex128).copy()
    tangent_norm = complex(direction_norm)
    tangent_pair = np.asarray(direction_w1, dtype=np.complex128).copy()
    tangent_quad = np.asarray(direction_w2, dtype=np.complex128).copy()
    for _ in range(substeps):
        k1, dk1 = _w2_rhs_with_jvp(
            angle,
            norm,
            pair,
            quad,
            tangent_norm,
            tangent_pair,
            tangent_quad,
            reference,
            number_operator,
            a02_cache,
        )
        k2, dk2 = _w2_rhs_with_jvp(
            angle + 0.5 * step,
            norm + 0.5 * step * k1[0],
            pair + 0.5 * step * k1[1],
            quad + 0.5 * step * k1[2],
            tangent_norm + 0.5 * step * dk1[0],
            tangent_pair + 0.5 * step * dk1[1],
            tangent_quad + 0.5 * step * dk1[2],
            reference,
            number_operator,
            a02_cache,
        )
        k3, dk3 = _w2_rhs_with_jvp(
            angle + 0.5 * step,
            norm + 0.5 * step * k2[0],
            pair + 0.5 * step * k2[1],
            quad + 0.5 * step * k2[2],
            tangent_norm + 0.5 * step * dk2[0],
            tangent_pair + 0.5 * step * dk2[1],
            tangent_quad + 0.5 * step * dk2[2],
            reference,
            number_operator,
            a02_cache,
        )
        k4, dk4 = _w2_rhs_with_jvp(
            angle + step,
            norm + step * k3[0],
            pair + step * k3[1],
            quad + step * k3[2],
            tangent_norm + step * dk3[0],
            tangent_pair + step * dk3[1],
            tangent_quad + step * dk3[2],
            reference,
            number_operator,
            a02_cache,
        )
        norm += (step / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        pair += (step / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        quad += (step / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        tangent_norm += (step / 6.0) * (
            dk1[0] + 2.0 * dk2[0] + 2.0 * dk3[0] + dk4[0]
        )
        tangent_pair += (step / 6.0) * (
            dk1[1] + 2.0 * dk2[1] + 2.0 * dk3[1] + dk4[1]
        )
        tangent_quad += (step / 6.0) * (
            dk1[2] + 2.0 * dk2[2] + 2.0 * dk3[2] + dk4[2]
        )
        angle += step
    return norm, pair, quad, tangent_norm, tangent_pair, tangent_quad


def _rk4_nodes(start: complex, stop: complex, substeps: int) -> tuple[complex, ...]:
    if stop == start:
        return ()
    step = (stop - start) / substeps
    nodes: list[complex] = []
    for index in range(substeps):
        angle = start + index * step
        nodes.extend((angle, angle + 0.5 * step, angle + step))
    return tuple(nodes)


def _disconnected_quadruple(pair: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,kl->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
    )


def _disconnected_quadruple_jvp(
    pair: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return (
        np.einsum("ij,kl->ijkl", direction, pair, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("ij,kl->ijkl", pair, direction, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", direction, pair, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", pair, direction, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", direction, pair, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", pair, direction, optimize=_BINARY_EINSUM_PATH)
    )


def _wedge_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,kl->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("jk,il->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("jl,ik->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("kl,ij->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
    )


def _grid_angles(options: ProjectionOptions) -> np.ndarray:
    period = math.pi if options.parity == "even" else 2.0 * math.pi
    return period * (np.arange(options.grid_size, dtype=float) + options.grid_shift) / options.grid_size


def _validate_alias_free(nspin: int, options: ProjectionOptions) -> None:
    alias_period = 2 * options.grid_size if options.parity == "even" else options.grid_size
    sectors = range(0, nspin + 1, 2) if options.parity == "even" else range(nspin + 1)
    aliases = [
        sector
        for sector in sectors
        if sector != options.target_number
        and (sector - options.target_number) % alias_period == 0
    ]
    if aliases:
        raise ValueError(
            f"grid_size={options.grid_size} aliases target N={options.target_number} "
            f"with sectors {aliases}"
        )


def _minimum_alias_free_grid_size(nspin: int, parity: str) -> int:
    return nspin // 2 + 1 if parity == "even" else nspin + 1


def _minimum_overlap_singular_value(
    reference: BogoliubovReference,
    radius: float,
    sample_count: int,
) -> float:
    radial_shift = -1.0j * math.log(radius)
    minimum = float("inf")
    for angle in np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False):
        _overlap, _z, overlap_matrix, _gauge = _transition_matrices(
            reference, complex(angle) + radial_shift
        )
        minimum = min(minimum, float(np.min(np.linalg.svd(overlap_matrix, compute_uv=False))))
    return minimum


def _select_contour_radius(
    reference: BogoliubovReference,
    options: ProjectionOptions,
) -> tuple[float, float]:
    sample_count = max(128, 16 * options.grid_size)
    if options.contour_radius is not None:
        minimum = _minimum_overlap_singular_value(reference, options.contour_radius, sample_count)
        if minimum < options.overlap_tolerance:
            raise ValueError("the requested contour crosses a Bogoliubov-overlap zero")
        return options.contour_radius, minimum
    threshold = max(
        options.contour_safety_threshold,
        1000.0 * options.overlap_tolerance,
    )
    candidates = (1.0, 0.98, 1.02, 0.95, 1.05, 0.90, 1.10, 0.85, 1.15, 0.80, 1.20)
    scored = [
        (radius, _minimum_overlap_singular_value(reference, radius, sample_count))
        for radius in candidates
    ]
    unit_radius = scored[0]
    if unit_radius[1] >= threshold:
        return unit_radius
    safe = [item for item in scored if item[1] >= threshold]
    pool = safe or scored
    return max(pool, key=lambda item: (item[1], -abs(math.log(item[0]))))


@dataclass(frozen=True)
class _ContourPoint:
    real_angle: float
    complex_angle: complex
    gauge: complex
    overlap: complex
    z: np.ndarray
    probe_annihilation: np.ndarray


@dataclass(frozen=True)
class _WeightedAngleMoments:
    denominator: complex
    energy_numerator: complex
    n2: np.ndarray
    n4: np.ndarray
    h2: np.ndarray
    h4: np.ndarray
    kernel_time: float
    transform_time: float
    elapsed: float


@dataclass(frozen=True)
class _WeightedAngleScalar:
    denominator: complex
    energy_numerator: complex
    reference_denominator: complex
    reference_energy_numerator: complex
    kernel_time: float
    transform_time: float
    elapsed: float


@dataclass(frozen=True)
class _WeightedAngleDerivativeMoments:
    denominator: complex
    direction_denominator: complex
    energy_numerator: complex
    direction_energy_numerator: complex
    n2: np.ndarray
    direction_n2: np.ndarray
    n4: np.ndarray
    direction_n4: np.ndarray
    h2: np.ndarray
    direction_h2: np.ndarray
    h4: np.ndarray
    direction_h4: np.ndarray
    kernel_time: float
    elapsed: float


@dataclass(frozen=True)
class _ProjectedRawMoments:
    denominator: complex
    energy_numerator: complex
    n2: np.ndarray
    n4: np.ndarray
    h2: np.ndarray
    h4: np.ndarray


@dataclass(frozen=True)
class _ProjectedScalarEvaluation:
    denominator: complex
    energy_numerator: complex
    total_energy: complex
    elapsed: float
    reference_denominator: complex | None = None
    reference_energy_numerator: complex | None = None
    reference_total_energy: complex | None = None
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class _ProjectedRawDerivativeMoments:
    denominator: complex
    direction_denominator: complex
    energy_numerator: complex
    direction_energy_numerator: complex
    n2: np.ndarray
    direction_n2: np.ndarray
    n4: np.ndarray
    direction_n4: np.ndarray
    h2: np.ndarray
    direction_h2: np.ndarray
    h4: np.ndarray
    direction_h4: np.ndarray


def _hamiltonian_storage_bytes(hamiltonian: QPHamiltonian) -> int:
    return int(
        sum(
            np.asarray(value).nbytes
            for key, value in hamiltonian.as_dict().items()
            if key != "constant"
        )
    )


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _kahan_add(
    total: complex | np.ndarray,
    compensation: complex | np.ndarray,
    value: complex | np.ndarray,
) -> tuple[complex | np.ndarray, complex | np.ndarray]:
    corrected = value - compensation
    updated = total + corrected
    return updated, (updated - total) - corrected


class ProjectedQPCCSDEvaluator:
    """Reusable static contour data with a full OAP evaluation per call."""

    def __init__(
        self,
        hamiltonian: QPHamiltonian,
        reference: BogoliubovReference,
        options: ProjectionOptions,
        excitation_space: QPExcitationSpaceLike | None = None,
        execution_options: ExecutionOptions | None = None,
        scalar_only: bool = False,
    ) -> None:
        if reference.nspin != hamiltonian.nspin:
            raise ValueError("reference and Hamiltonian dimensions do not match")
        if reference.target_number != options.target_number:
            raise ValueError("projection target and Bogoliubov reference target differ")
        _validate_alias_free(reference.nspin, options)
        self.hamiltonian = hamiltonian
        self.reference = reference
        self.options = options
        self.execution_options = (
            ExecutionOptions() if execution_options is None else execution_options
        )
        requested_backend = self.execution_options.integral_backend
        if requested_backend == "cholesky":
            raise NotImplementedError(
                "factorized Wick kernels are not certified; use the exact backend "
                "or auto-validated dense fallback"
            )
        self.integral_backend = "exact"
        self.integral_backend_fallback = (
            None
            if requested_backend == "exact"
            else "factorized Wick kernels unavailable; selected exact dense tensors"
        )
        self.excitation_space = (
            QPExcitationSpace.full(reference.nspin)
            if excitation_space is None
            else excitation_space
        )
        if self.excitation_space.nspin != reference.nspin:
            raise ValueError("excitation space and projected reference dimensions differ")
        self.scalar_only = bool(scalar_only)
        self.number_operator = build_number_operator(reference)
        self._a02_cache: dict[tuple[float, float], np.ndarray] = {}
        self._transformed_hamiltonians: dict[int, QPHamiltonian] = {}
        self._projected_metric_vacuum_vectors: tuple[np.ndarray, ...] | None = None
        self._transform_cache_hits = 0
        self._transform_cache_misses = 0
        setup_started = time.perf_counter()
        self.radius, self.minimum_overlap_singular_value = _select_contour_radius(reference, options)
        radial_shift = -1.0j * math.log(self.radius)
        self.points = []
        for real_angle in _grid_angles(options):
            complex_angle = complex(real_angle) + radial_shift
            overlap, z, overlap_matrix, gauge = _transition_matrices(reference, complex_angle)
            self.points.append(
                _ContourPoint(
                    float(real_angle),
                    complex_angle,
                    gauge,
                    overlap,
                    z,
                    np.linalg.inv(overlap_matrix),
                )
            )
        self._scalar_energy_kernel = ScalarSimilarityTransformKernel.from_deexcitations(
            hamiltonian,
            (point.z for point in self.points),
        )
        self.radial_steps = max(
            1,
            int(
                math.ceil(
                    options.ode_substeps
                    * abs(math.log(self.radius))
                    / max(math.pi / options.grid_size, 1.0e-12)
                )
            ),
        )
        previous = 0.0 + 0.0j
        for stop, substeps in (
            [(radial_shift, self.radial_steps)]
            + [
                (point.complex_angle, options.ode_substeps)
                for point in self.points
            ]
        ):
            for angle in _rk4_nodes(previous, stop, substeps):
                key = (round(float(np.real(angle)), 15), round(float(np.imag(angle)), 15))
                if key in self._a02_cache:
                    continue
                _overlap, z = reference_thouless(reference, angle)
                self._a02_cache[key] = np.asarray(
                    _one_body_a02_after_deexcitation(self.number_operator, z)
                )
            previous = stop

        workspace_bytes = (
            options.cache_bytes
            if self.execution_options.max_workspace_bytes is None
            else min(options.cache_bytes, self.execution_options.max_workspace_bytes)
        )
        bytes_per_hamiltonian = max(_hamiltonian_storage_bytes(hamiltonian), 1)
        self.transform_cache_capacity = (
            0
            if self.scalar_only
            else min(len(self.points), workspace_bytes // bytes_per_hamiltonian)
        )
        for index in range(self.transform_cache_capacity):
            self._transformed_hamiltonians[index] = similarity_transform_deexcitation(
                hamiltonian, self.points[index].z
            )
        self.transform_cache_bytes = int(
            sum(
                _hamiltonian_storage_bytes(value)
                for value in self._transformed_hamiltonians.values()
            )
        )
        self.contour_setup_time = time.perf_counter() - setup_started

    def _transformed_hamiltonian(self, index: int) -> tuple[QPHamiltonian, float]:
        cached = self._transformed_hamiltonians.get(index)
        if cached is not None:
            self._transform_cache_hits += 1
            return cached, 0.0
        started = time.perf_counter()
        transformed = similarity_transform_deexcitation(
            self.hamiltonian, self.points[index].z
        )
        self._transform_cache_misses += 1
        return transformed, time.perf_counter() - started

    def _execution_layout(self) -> tuple[bool, int, int, int]:
        available = _available_cpu_count()
        mode = self.execution_options.parallel_mode
        requested_workers = self.execution_options.workers
        requested_blas = self.execution_options.blas_threads
        automatic_workers = min(4, len(self.points), max(1, available - 1))

        def pipeline_layout() -> tuple[bool, int, int, int]:
            workers = min(
                requested_workers or automatic_workers,
                len(self.points),
                max(1, available - 1),
            )
            # The main thread advances the W2 ODE while gauge workers execute
            # Wick contractions.  Nested multithreaded BLAS in those concurrent
            # callers is not numerically reliable on every backend (notably the
            # macOS Accelerate stack) and also oversubscribes CPUs.  Parallelize
            # across gauge tasks and keep each BLAS call single-threaded.
            return True, workers, 1, available

        if mode == "serial" or available == 1:
            return False, 1, 1, available
        if mode == "blas":
            return False, 1, min(requested_blas or available, available), available
        if mode == "pipeline":
            return pipeline_layout()
        if self.hamiltonian.nspin < 8:
            return False, 1, min(requested_blas or 1, available), available
        return pipeline_layout()

    def _evaluate_angle(
        self,
        point_index: int,
        norm: complex,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> _WeightedAngleMoments:
        started = time.perf_counter()
        point = self.points[point_index]
        transformed_hamiltonian, transform_time = self._transformed_hamiltonian(
            point_index
        )
        connected = evaluate_qpccsd(
            transformed_hamiltonian,
            QPAmplitudes(w1, w2),
        )
        right_n2 = w1
        right_n4 = w2 + _disconnected_quadruple(w1)
        h0 = complex(connected.total_energy)
        right_h2 = np.asarray(connected.r1) + h0 * right_n2
        right_h4 = (
            np.asarray(connected.r2)
            + _wedge_pairs(np.asarray(connected.r1), right_n2)
            + h0 * right_n4
        )
        moments = projected_left_moments(
            point.probe_annihilation,
            point.z,
            n0=1.0,
            n20=right_n2,
            n40=right_n4,
            h0=h0,
            h20=right_h2,
            h40=right_h4,
        )
        scalar_norm = point.overlap * norm
        fourier_weight = (
            point.gauge ** (-self.options.target_number) / self.options.grid_size
        )
        weighted_norm = fourier_weight * scalar_norm
        return _WeightedAngleMoments(
            denominator=weighted_norm,
            energy_numerator=weighted_norm * moments.h0,
            n2=weighted_norm * moments.n20,
            n4=weighted_norm * moments.n40,
            h2=weighted_norm * moments.h20,
            h4=weighted_norm * moments.h40,
            kernel_time=connected.elapsed,
            transform_time=transform_time,
            elapsed=time.perf_counter() - started,
        )

    def _evaluate_angle_energy(
        self,
        point_index: int,
        norm: complex,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> _WeightedAngleScalar:
        started = time.perf_counter()
        point = self.points[point_index]
        kernel_started = time.perf_counter()
        energy, transformed_constant = self._scalar_energy_kernel.evaluate(
            point.z,
            w1,
            w2,
        )
        kernel_time = time.perf_counter() - kernel_started
        weighted_norm = (
            point.gauge ** (-self.options.target_number)
            * point.overlap
            * norm
            / self.options.grid_size
        )
        reference_weighted_norm = (
            point.gauge ** (-self.options.target_number)
            * point.overlap
            / self.options.grid_size
        )
        return _WeightedAngleScalar(
            denominator=weighted_norm,
            energy_numerator=weighted_norm * energy,
            reference_denominator=reference_weighted_norm,
            reference_energy_numerator=(
                reference_weighted_norm * transformed_constant
            ),
            kernel_time=kernel_time,
            transform_time=0.0,
            elapsed=time.perf_counter() - started,
        )

    def evaluate_zero_amplitude_energy(self) -> _ProjectedScalarEvaluation:
        """Evaluate the projected reference baseline without residual tensors."""

        started = time.perf_counter()
        denominator = 0.0 + 0.0j
        denominator_compensation = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        energy_compensation = 0.0 + 0.0j
        transform_time = 0.0
        kernel_time = 0.0
        reference_denominator = 0.0 + 0.0j
        reference_denominator_compensation = 0.0 + 0.0j
        reference_energy_numerator = 0.0 + 0.0j
        reference_energy_compensation = 0.0 + 0.0j
        blas_threads = self.execution_options.blas_threads or 1
        with threadpool_limits(limits=blas_threads):
            for point_index, point in enumerate(self.points):
                kernel_started = time.perf_counter()
                transformed_constant = self._scalar_energy_kernel.transformed_constant(
                    point.z
                )
                kernel_time += time.perf_counter() - kernel_started
                weighted_norm = (
                    point.gauge ** (-self.options.target_number)
                    * point.overlap
                    / self.options.grid_size
                )
                denominator, denominator_compensation = _kahan_add(
                    denominator,
                    denominator_compensation,
                    weighted_norm,
                )
                energy_numerator, energy_compensation = _kahan_add(
                    energy_numerator,
                    energy_compensation,
                    weighted_norm * transformed_constant,
                )
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the projected reference norm is numerically zero")
        return _ProjectedScalarEvaluation(
            denominator=denominator,
            energy_numerator=energy_numerator,
            total_energy=energy_numerator / denominator,
            elapsed=time.perf_counter() - started,
            reference_denominator=denominator,
            reference_energy_numerator=energy_numerator,
            reference_total_energy=energy_numerator / denominator,
            diagnostics={
                "mode": "zero-amplitude-scalar",
                "transform_time": transform_time,
                "ode_time": 0.0,
                "kernel_time": kernel_time,
                "scalar_wick_backend": self._scalar_energy_kernel.backend,
                "scalar_active_support": self._scalar_energy_kernel.support_size,
                "scalar_precomputed_bytes": self._scalar_energy_kernel.precomputed_bytes,
            },
        )

    def evaluate_energy(self, amplitudes: QPAmplitudes) -> _ProjectedScalarEvaluation:
        """Evaluate only the projected scalar energy for fixed amplitudes."""

        started = time.perf_counter()
        norm = 1.0 + 0.0j
        w1 = np.asarray(amplitudes.t1, dtype=np.complex128).copy()
        w2 = np.asarray(amplitudes.t2, dtype=np.complex128).copy()
        base_angle = -1.0j * math.log(self.radius)
        ode_started = time.perf_counter()
        norm, w1, w2 = _rk4_interval(
            0.0 + 0.0j,
            base_angle,
            norm,
            w1,
            w2,
            self.reference,
            self.number_operator,
            self.radial_steps,
            self._a02_cache,
        )
        ode_time = time.perf_counter() - ode_started
        initial_norm = norm
        initial_w1 = w1.copy()
        initial_w2 = w2.copy()
        denominator = 0.0 + 0.0j
        denominator_compensation = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        energy_compensation = 0.0 + 0.0j
        reference_denominator = 0.0 + 0.0j
        reference_denominator_compensation = 0.0 + 0.0j
        reference_energy_numerator = 0.0 + 0.0j
        reference_energy_compensation = 0.0 + 0.0j
        kernel_time = 0.0
        transform_time = 0.0
        previous = base_angle
        blas_threads = self.execution_options.blas_threads or 1
        with threadpool_limits(limits=blas_threads):
            for point_index, point in enumerate(self.points):
                ode_started = time.perf_counter()
                norm, w1, w2 = _rk4_interval(
                    previous,
                    point.complex_angle,
                    norm,
                    w1,
                    w2,
                    self.reference,
                    self.number_operator,
                    self.options.ode_substeps,
                    self._a02_cache,
                )
                ode_time += time.perf_counter() - ode_started
                previous = point.complex_angle
                value = self._evaluate_angle_energy(
                    point_index,
                    norm,
                    w1,
                    w2,
                )
                kernel_time += value.kernel_time
                transform_time += value.transform_time
                denominator, denominator_compensation = _kahan_add(
                    denominator,
                    denominator_compensation,
                    value.denominator,
                )
                energy_numerator, energy_compensation = _kahan_add(
                    energy_numerator,
                    energy_compensation,
                    value.energy_numerator,
                )
                (
                    reference_denominator,
                    reference_denominator_compensation,
                ) = _kahan_add(
                    reference_denominator,
                    reference_denominator_compensation,
                    value.reference_denominator,
                )
                (
                    reference_energy_numerator,
                    reference_energy_compensation,
                ) = _kahan_add(
                    reference_energy_numerator,
                    reference_energy_compensation,
                    value.reference_energy_numerator,
                )
        period = math.pi if self.options.parity == "even" else 2.0 * math.pi
        endpoint_started = time.perf_counter()
        endpoint_norm, endpoint_w1, endpoint_w2 = _rk4_interval(
            previous,
            base_angle + period,
            norm,
            w1,
            w2,
            self.reference,
            self.number_operator,
            self.options.ode_substeps,
            self._a02_cache,
        )
        ode_time += time.perf_counter() - endpoint_started
        contour_monodromy = {
            "correlated_norm_return_error": float(
                abs(endpoint_norm - initial_norm)
            ),
            "w1_return_error": float(
                np.max(np.abs(endpoint_w1 - initial_w1), initial=0.0)
            ),
            "w2_return_error": float(
                np.max(np.abs(endpoint_w2 - initial_w2), initial=0.0)
            ),
            "period": float(period),
        }
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        if abs(reference_denominator) < self.options.overlap_tolerance:
            raise ValueError("the projected reference norm is numerically zero")
        return _ProjectedScalarEvaluation(
            denominator=denominator,
            energy_numerator=energy_numerator,
            total_energy=energy_numerator / denominator,
            elapsed=time.perf_counter() - started,
            reference_denominator=reference_denominator,
            reference_energy_numerator=reference_energy_numerator,
            reference_total_energy=(
                reference_energy_numerator / reference_denominator
            ),
            diagnostics={
                "mode": "correlated-scalar-ode2",
                "ode_time": ode_time,
                "transform_time": transform_time,
                "kernel_time": kernel_time,
                "scalar_wick_backend": self._scalar_energy_kernel.backend,
                "scalar_active_support": self._scalar_energy_kernel.support_size,
                "scalar_precomputed_bytes": self._scalar_energy_kernel.precomputed_bytes,
                "contour_monodromy": contour_monodromy,
            },
        )

    def _evaluate_angle_jvp(
        self,
        point_index: int,
        norm: complex,
        w1: np.ndarray,
        w2: np.ndarray,
        direction_norm: complex,
        direction_w1: np.ndarray,
        direction_w2: np.ndarray,
    ) -> _WeightedAngleDerivativeMoments:
        started = time.perf_counter()
        point = self.points[point_index]
        transformed, _transform_time = self._transformed_hamiltonian(point_index)
        connected, connected_jvp = evaluate_qpccsd_with_jvp(
            transformed,
            QPAmplitudes(w1, w2),
            QPAmplitudes(direction_w1, direction_w2),
        )

        right_n2 = w1
        direction_right_n2 = direction_w1
        right_n4 = w2 + _disconnected_quadruple(w1)
        direction_right_n4 = (
            direction_w2 + _disconnected_quadruple_jvp(w1, direction_w1)
        )
        h0 = complex(connected.total_energy)
        direction_h0 = complex(connected_jvp.total_energy)
        connected_r1 = np.asarray(connected.r1)
        direction_connected_r1 = np.asarray(connected_jvp.r1)
        right_h2 = connected_r1 + h0 * right_n2
        direction_right_h2 = (
            direction_connected_r1
            + direction_h0 * right_n2
            + h0 * direction_right_n2
        )
        right_h4 = (
            np.asarray(connected.r2)
            + _wedge_pairs(connected_r1, right_n2)
            + h0 * right_n4
        )
        direction_right_h4 = (
            np.asarray(connected_jvp.r2)
            + _wedge_pairs(direction_connected_r1, right_n2)
            + _wedge_pairs(connected_r1, direction_right_n2)
            + direction_h0 * right_n4
            + h0 * direction_right_n4
        )
        moments = projected_left_moments(
            point.probe_annihilation,
            point.z,
            n0=1.0,
            n20=right_n2,
            n40=right_n4,
            h0=h0,
            h20=right_h2,
            h40=right_h4,
        )
        direction_moments = projected_left_moments(
            point.probe_annihilation,
            point.z,
            n0=0.0,
            n20=direction_right_n2,
            n40=direction_right_n4,
            h0=direction_h0,
            h20=direction_right_h2,
            h40=direction_right_h4,
        )
        base_weight = (
            point.gauge ** (-self.options.target_number)
            * point.overlap
            / self.options.grid_size
        )
        weighted_norm = base_weight * norm
        direction_weighted_norm = base_weight * direction_norm
        return _WeightedAngleDerivativeMoments(
            denominator=weighted_norm,
            direction_denominator=direction_weighted_norm,
            energy_numerator=weighted_norm * moments.h0,
            direction_energy_numerator=(
                direction_weighted_norm * moments.h0
                + weighted_norm * direction_moments.h0
            ),
            n2=weighted_norm * moments.n20,
            direction_n2=(
                direction_weighted_norm * moments.n20
                + weighted_norm * direction_moments.n20
            ),
            n4=weighted_norm * moments.n40,
            direction_n4=(
                direction_weighted_norm * moments.n40
                + weighted_norm * direction_moments.n40
            ),
            h2=weighted_norm * moments.h20,
            direction_h2=(
                direction_weighted_norm * moments.h20
                + weighted_norm * direction_moments.h20
            ),
            h4=weighted_norm * moments.h40,
            direction_h4=(
                direction_weighted_norm * moments.h40
                + weighted_norm * direction_moments.h40
            ),
            kernel_time=connected.elapsed,
            elapsed=time.perf_counter() - started,
        )

    def contour_monodromy(self, amplitudes: QPAmplitudes) -> dict[str, float]:
        """Measure the failure of the W2 flow to close after one gauge period.

        Exact disentanglement is periodic on the parity-reduced contour. A
        nonzero return error is therefore a direct diagnostic of the W3=0
        closure and predicts algebraic, rather than spectral, gauge convergence.
        """

        nspin = self.hamiltonian.nspin
        if amplitudes.t1.shape != (nspin, nspin):
            raise ValueError("monodromy pair dimensions do not match")
        if amplitudes.t2.shape != (nspin,) * 4:
            raise ValueError("monodromy quadruple dimensions do not match")
        started = time.perf_counter()
        norm = 1.0 + 0.0j
        w1 = np.asarray(amplitudes.t1, dtype=np.complex128).copy()
        w2 = np.asarray(amplitudes.t2, dtype=np.complex128).copy()
        zero_amplitude_fast_path = not np.any(w1) and not np.any(w2)
        base_angle = -1.0j * math.log(self.radius)
        if not zero_amplitude_fast_path:
            norm, w1, w2 = _rk4_interval(
                0.0 + 0.0j,
                base_angle,
                norm,
                w1,
                w2,
                self.reference,
                self.number_operator,
                self.radial_steps,
                self._a02_cache,
            )
        initial_norm = norm
        initial_w1 = w1.copy()
        initial_w2 = w2.copy()
        period = math.pi if self.options.parity == "even" else 2.0 * math.pi
        norm, w1, w2 = _rk4_interval(
            base_angle,
            base_angle + period,
            norm,
            w1,
            w2,
            self.reference,
            self.number_operator,
            self.options.grid_size * self.options.ode_substeps,
            self._a02_cache,
        )
        return {
            "correlated_norm_return_error": float(abs(norm - initial_norm)),
            "w1_return_error": float(np.max(np.abs(w1 - initial_w1))),
            "w2_return_error": float(np.max(np.abs(w2 - initial_w2))),
            "period": float(period),
            "elapsed": time.perf_counter() - started,
        }

    def build_residual_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        """Build a compact gauge basis from analytic W2 number-orbit tangents.

        Every sampled tangent is generated by the same transformed number
        operator used by the disentanglement ODE.  The resulting basis is
        independent of the Hamiltonian residual and therefore does not fit or
        remove weak physical Jacobian directions.
        """

        coordinate_count = self.excitation_space.coordinate_count
        if not options.enabled or coordinate_count == 0:
            return ProjectedResidualBasis.empty(coordinate_count)

        norm = 1.0 + 0.0j
        w1 = np.asarray(amplitudes.t1, dtype=np.complex128).copy()
        w2 = np.asarray(amplitudes.t2, dtype=np.complex128).copy()
        base_angle = -1.0j * math.log(self.radius)
        norm, w1, w2 = _rk4_interval(
            0.0 + 0.0j,
            base_angle,
            norm,
            w1,
            w2,
            self.reference,
            self.number_operator,
            self.radial_steps,
            self._a02_cache,
        )
        initial_norm = norm
        initial_w1 = w1.copy()
        initial_w2 = w2.copy()
        period = math.pi if self.options.parity == "even" else 2.0 * math.pi
        previous = base_angle
        columns: list[np.ndarray] = []
        for sample in range(options.sample_count):
            stop = base_angle + period * (sample + 0.5) / options.sample_count
            interval = stop - previous
            substeps = max(
                1,
                int(
                    math.ceil(
                        self.options.ode_substeps
                        * abs(interval)
                        / max(period / self.options.grid_size, 1.0e-12)
                    )
                ),
            )
            norm, w1, w2 = _rk4_interval(
                previous,
                stop,
                norm,
                w1,
                w2,
                self.reference,
                self.number_operator,
                substeps,
                self._a02_cache,
            )
            tangent = _w2_rhs(
                stop,
                norm,
                w1,
                w2,
                self.reference,
                self.number_operator,
                self._a02_cache,
            )
            vector = self.excitation_space.pack(QPAmplitudes(tangent[1], tangent[2]))
            length = float(np.linalg.norm(vector))
            if length > options.rank_absolute_tolerance:
                columns.append(np.asarray(vector, dtype=np.complex128) / length)
            previous = stop

        final_substeps = max(1, self.options.ode_substeps)
        norm, w1, w2 = _rk4_interval(
            previous,
            base_angle + period,
            norm,
            w1,
            w2,
            self.reference,
            self.number_operator,
            final_substeps,
            self._a02_cache,
        )
        identity_defect = max(
            float(abs(norm - initial_norm)),
            float(np.max(np.abs(w1 - initial_w1))),
            float(np.max(np.abs(w2 - initial_w2))),
        )
        if not columns:
            empty = ProjectedResidualBasis.empty(coordinate_count)
            return ProjectedResidualBasis(
                gauge_vectors=empty.gauge_vectors,
                singular_values=empty.singular_values,
                coordinate_count=coordinate_count,
                source="zero-analytic-w2-number-orbit",
                identity_defect=identity_defect,
            )

        samples = np.column_stack(columns)
        left, singular_values, _right = np.linalg.svd(samples, full_matrices=False)
        threshold = max(
            options.rank_absolute_tolerance,
            options.rank_relative_tolerance * float(singular_values[0]),
        )
        rank = min(
            options.maximum_rank,
            int(np.count_nonzero(singular_values > threshold)),
        )
        vectors = np.asarray(left[:, :rank], dtype=np.complex128).copy()
        # Fix arbitrary SVD phases so checkpoints and one-core regressions are
        # reproducible across LAPACK implementations.
        for column in range(rank):
            pivot = int(np.argmax(np.abs(vectors[:, column])))
            phase = vectors[pivot, column]
            if abs(phase) > 0.0:
                vectors[:, column] *= np.conj(phase) / abs(phase)
        return ProjectedResidualBasis(
            gauge_vectors=vectors,
            singular_values=singular_values[:rank],
            coordinate_count=coordinate_count,
            identity_defect=identity_defect,
        )

    def projected_metric_action(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``<Phi| B P_N B^dagger |Phi>`` without forming the metric."""

        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.excitation_space.coordinate_count,):
            raise ValueError("projected metric vector has an incompatible shape")
        direction = self.excitation_space.unpack(values)
        pair = np.asarray(direction.t1, dtype=np.complex128)
        quadruple = np.asarray(direction.t2, dtype=np.complex128)
        total = np.zeros(self.excitation_space.coordinate_count, dtype=np.complex128)
        compensation = np.zeros_like(total)
        base_vectors = self._metric_vacuum_vectors()
        pipeline_enabled, angle_workers, blas_threads, _available_cores = (
            self._execution_layout()
        )

        def point_contribution(point_index: int) -> np.ndarray:
            point = self.points[point_index]
            contribution = self._projected_metric_point_vector(
                point,
                pair,
                quadruple,
                base_vectors[point_index],
            )
            weight = (
                point.gauge ** (-self.options.target_number)
                * point.overlap
                / self.options.grid_size
            )
            return weight * contribution

        with threadpool_limits(limits=blas_threads):
            if pipeline_enabled:
                with ThreadPoolExecutor(
                    max_workers=angle_workers,
                    thread_name_prefix="qpccsd-metric",
                ) as executor:
                    contributions = executor.map(
                        point_contribution, range(len(self.points))
                    )
                    for contribution in contributions:
                        total, compensation = _kahan_add(
                            total, compensation, contribution
                        )
            else:
                for point_index in range(len(self.points)):
                    total, compensation = _kahan_add(
                        total, compensation, point_contribution(point_index)
                    )
        return total

    def _metric_vacuum_vectors(self) -> tuple[np.ndarray, ...]:
        """Cache compact reference-only metric moments in contour order."""

        if self._projected_metric_vacuum_vectors is not None:
            return self._projected_metric_vacuum_vectors
        pipeline_enabled, angle_workers, blas_threads, _available_cores = (
            self._execution_layout()
        )

        def build(point_index: int) -> np.ndarray:
            moments = projected_left_vacuum_norm_moments(
                self.points[point_index].z
            )
            return np.asarray(
                self.excitation_space.residual_vector(moments.n20, moments.n40),
                dtype=np.complex128,
            )

        with threadpool_limits(limits=blas_threads):
            if pipeline_enabled:
                with ThreadPoolExecutor(
                    max_workers=angle_workers,
                    thread_name_prefix="qpccsd-metric-reference",
                ) as executor:
                    vectors = tuple(executor.map(build, range(len(self.points))))
            else:
                vectors = tuple(build(index) for index in range(len(self.points)))
        self._projected_metric_vacuum_vectors = vectors
        return vectors

    def _projected_metric_point_vector(
        self,
        point: _ContourPoint,
        pair: np.ndarray,
        quadruple: np.ndarray,
        base_vector: np.ndarray,
    ) -> np.ndarray:
        contracted_quadruple = _contract_sparse_pair_rank4(point.z, quadruple)
        transformed_scalar = (
            0.5
            * np.einsum("ab,ab->", point.z, pair, optimize=_BINARY_EINSUM_PATH)
            + 0.125
            * np.einsum(
                "ab,ab->",
                point.z,
                contracted_quadruple,
                optimize=_BINARY_EINSUM_PATH,
            )
        )
        direction_moments = projected_left_norm_moments(
            point.probe_annihilation,
            point.z,
            n20=pair + 0.5 * contracted_quadruple,
            n40=quadruple,
        )
        direction_vector = np.asarray(
            self.excitation_space.residual_vector(
                direction_moments.n20,
                direction_moments.n40,
            ),
            dtype=np.complex128,
        )
        return transformed_scalar * base_vector + direction_vector

    def _projected_metric_matrix(self) -> np.ndarray:
        """Build a small metric angle-first, reusing each vacuum moment once."""

        coordinate_count = self.excitation_space.coordinate_count
        metric = np.zeros((coordinate_count, coordinate_count), dtype=np.complex128)
        compensation = np.zeros_like(metric)
        eye = np.eye(coordinate_count, dtype=np.complex128)
        base_vectors = self._metric_vacuum_vectors()
        for point_index, point in enumerate(self.points):
            weight = (
                point.gauge ** (-self.options.target_number)
                * point.overlap
                / self.options.grid_size
            )
            angle_matrix = np.empty_like(metric)
            for column in range(coordinate_count):
                direction = self.excitation_space.unpack(eye[:, column])
                angle_matrix[:, column] = self._projected_metric_point_vector(
                    point,
                    np.asarray(direction.t1, dtype=np.complex128),
                    np.asarray(direction.t2, dtype=np.complex128),
                    base_vectors[point_index],
                )
            metric, compensation = _kahan_add(
                metric,
                compensation,
                weight * angle_matrix,
            )
        return metric

    def build_metric_projector(
        self,
        options: GaugeModeOptions,
    ) -> ProjectedMetricProjector:
        return ProjectedMetricProjector(
            self.projected_metric_action,
            self.excitation_space.coordinate_count,
            explicit_matrix_builder=self._projected_metric_matrix,
            explicit_max_coordinates=options.explicit_metric_max_coordinates,
            relative_tolerance=options.metric_relative_tolerance,
            absolute_tolerance=options.metric_absolute_tolerance,
            max_iterations=options.metric_max_iterations,
            maximum_nullity=options.metric_maximum_nullity,
        )

    def build_local_gauge_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        """Return the local U(1) orbit tangent at the unrotated cluster."""

        coordinate_count = self.excitation_space.coordinate_count
        if not options.enabled or not options.local_tangent_gauge or not coordinate_count:
            return ProjectedResidualBasis.empty(coordinate_count)
        tangent = _w2_rhs(
            0.0 + 0.0j,
            1.0 + 0.0j,
            np.asarray(amplitudes.t1, dtype=np.complex128),
            np.asarray(amplitudes.t2, dtype=np.complex128),
            self.reference,
            self.number_operator,
            self._a02_cache,
        )
        vector = np.asarray(
            self.excitation_space.pack(QPAmplitudes(tangent[1], tangent[2])),
            dtype=np.complex128,
        )
        length = float(np.linalg.norm(vector))
        if length <= options.metric_absolute_tolerance:
            return ProjectedResidualBasis.empty(coordinate_count)
        vector /= length
        pivot = int(np.argmax(np.abs(vector)))
        phase = vector[pivot]
        if abs(phase):
            vector *= np.conj(phase) / abs(phase)
        return ProjectedResidualBasis(
            gauge_vectors=vector[:, None],
            singular_values=np.asarray([length]),
            coordinate_count=coordinate_count,
            source="analytic-local-u1-orbit",
            identity_defect=0.0,
        )

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        started = time.perf_counter()
        nspin = self.hamiltonian.nspin
        norm = 1.0 + 0.0j
        w1 = np.asarray(amplitudes.t1, dtype=np.complex128).copy()
        w2 = np.asarray(amplitudes.t2, dtype=np.complex128).copy()
        zero_amplitude_fast_path = not np.any(w1) and not np.any(w2)
        base_angle = -1.0j * math.log(self.radius)
        if not zero_amplitude_fast_path:
            norm, w1, w2 = _rk4_interval(
                0.0 + 0.0j,
                base_angle,
                norm,
                w1,
                w2,
                self.reference,
                self.number_operator,
                self.radial_steps,
                self._a02_cache,
            )

        denominator = 0.0 + 0.0j
        denominator_compensation = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        energy_compensation = 0.0 + 0.0j
        n2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        n2_compensation = np.zeros_like(n2_total)
        n4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        n4_compensation = np.zeros_like(n4_total)
        h2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        h2_compensation = np.zeros_like(h2_total)
        h4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        h4_compensation = np.zeros_like(h4_total)
        kernel_time = 0.0
        transform_time = 0.0
        angle_timings: list[float] = []
        ode_timings: list[float] = []
        previous = base_angle
        cache_hits_before = self._transform_cache_hits
        cache_misses_before = self._transform_cache_misses
        pipeline_enabled, angle_workers, blas_threads, available_cores = (
            self._execution_layout()
        )

        def accumulate(value: _WeightedAngleMoments) -> None:
            nonlocal denominator, denominator_compensation
            nonlocal energy_numerator, energy_compensation
            nonlocal n2_total, n2_compensation, n4_total, n4_compensation
            nonlocal h2_total, h2_compensation, h4_total, h4_compensation
            nonlocal kernel_time, transform_time
            denominator, denominator_compensation = _kahan_add(
                denominator, denominator_compensation, value.denominator
            )
            energy_numerator, energy_compensation = _kahan_add(
                energy_numerator, energy_compensation, value.energy_numerator
            )
            n2_total, n2_compensation = _kahan_add(
                n2_total, n2_compensation, value.n2
            )
            n4_total, n4_compensation = _kahan_add(
                n4_total, n4_compensation, value.n4
            )
            h2_total, h2_compensation = _kahan_add(
                h2_total, h2_compensation, value.h2
            )
            h4_total, h4_compensation = _kahan_add(
                h4_total, h4_compensation, value.h4
            )
            kernel_time += value.kernel_time
            transform_time += value.transform_time
            angle_timings.append(value.elapsed)

        with threadpool_limits(limits=blas_threads):
            if pipeline_enabled:
                pending: deque[tuple[int, Future[_WeightedAngleMoments]]] = deque()
                pending_limit = max(
                    1,
                    min(self.execution_options.pipeline_depth, angle_workers),
                )
                with ThreadPoolExecutor(
                    max_workers=angle_workers,
                    thread_name_prefix="qpccsd-gauge",
                ) as executor:
                    for point_index, point in enumerate(self.points):
                        ode_started = time.perf_counter()
                        if not zero_amplitude_fast_path:
                            norm, w1, w2 = _rk4_interval(
                                previous,
                                point.complex_angle,
                                norm,
                                w1,
                                w2,
                                self.reference,
                                self.number_operator,
                                self.options.ode_substeps,
                                self._a02_cache,
                            )
                        ode_timings.append(time.perf_counter() - ode_started)
                        previous = point.complex_angle
                        if len(pending) >= pending_limit:
                            _index, future = pending.popleft()
                            accumulate(future.result())
                        pending.append(
                            (
                                point_index,
                                executor.submit(
                                    self._evaluate_angle,
                                    point_index,
                                    norm,
                                    w1 if zero_amplitude_fast_path else w1.copy(),
                                    w2 if zero_amplitude_fast_path else w2.copy(),
                                ),
                            )
                        )
                    while pending:
                        _index, future = pending.popleft()
                        accumulate(future.result())
            else:
                for point_index, point in enumerate(self.points):
                    ode_started = time.perf_counter()
                    if not zero_amplitude_fast_path:
                        norm, w1, w2 = _rk4_interval(
                            previous,
                            point.complex_angle,
                            norm,
                            w1,
                            w2,
                            self.reference,
                            self.number_operator,
                            self.options.ode_substeps,
                            self._a02_cache,
                        )
                    ode_timings.append(time.perf_counter() - ode_started)
                    previous = point.complex_angle
                    accumulate(self._evaluate_angle(point_index, norm, w1, w2))

        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        energy = energy_numerator / denominator
        r1 = (h2_total - energy * n2_total) / denominator
        r2 = (h4_total - energy * n4_total) / denominator
        residual = residual_to_vector(r1, r2, self.excitation_space)
        residual_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        elapsed = time.perf_counter() - started
        return KernelEvaluation(
            total_energy=np.real_if_close(energy),
            correlation_energy=np.real_if_close(energy - self.hamiltonian.constant),
            r1=np.real_if_close(r1),
            r2=np.real_if_close(r2),
            residual_norm=residual_norm,
            projected_norm=denominator,
            elapsed=elapsed,
            diagnostics={
                "projection_time": elapsed,
                "kernel_time": kernel_time,
                "transform_time": transform_time,
                "contour_setup_time": self.contour_setup_time,
                "cached_number_blocks": len(self._a02_cache),
                "transform_cache_capacity": self.transform_cache_capacity,
                "transform_cache_bytes": self.transform_cache_bytes,
                "transform_cache_hits": self._transform_cache_hits - cache_hits_before,
                "transform_cache_misses": self._transform_cache_misses
                - cache_misses_before,
                "angle_timings": angle_timings,
                "ode_timings": ode_timings,
                "parallel_mode": self.execution_options.parallel_mode,
                "requested_integral_backend": self.execution_options.integral_backend,
                "selected_integral_backend": self.integral_backend,
                "integral_backend_fallback": self.integral_backend_fallback,
                "pipeline_enabled": pipeline_enabled,
                "pipeline_depth": min(
                    self.execution_options.pipeline_depth,
                    angle_workers,
                ) if pipeline_enabled else 0,
                "angle_workers": angle_workers,
                "blas_threads": blas_threads,
                "available_cores": available_cores,
                "deterministic_accumulation": True,
                "zero_amplitude_fast_path": zero_amplitude_fast_path,
                "gauge_quadrature": "midpoint",
                "grid_size": self.options.grid_size,
                "ode_substeps": self.options.ode_substeps,
                "contour_radius": self.radius,
                "contour_safety_threshold": self.options.contour_safety_threshold,
                "minimum_overlap_singular_value": self.minimum_overlap_singular_value,
                "disentanglement_backend": "ode2",
                "closure": "W2",
                "w3_omitted": True,
                "contraction_backend": generated_wick_orbits.CONTRACTION_BACKEND,
                "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
                "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
                "projector_ordering": PROJECTOR_ORDERING,
                "max_imaginary_energy": abs(float(np.imag(energy))),
                "max_imaginary_residual": float(np.max(np.abs(np.imag(residual))))
                if residual.size
                else 0.0,
                "internal_residual_norm": self.excitation_space.internal_residual_norm(
                    r1, r2
                ),
                "allowed_pair_count": self.excitation_space.pair_count,
                "allowed_quadruple_count": self.excitation_space.quadruple_count,
                "internal_coordinate_count": self.excitation_space.internal_coordinate_count,
                "forbidden_amplitude_norm": self.excitation_space.forbidden_amplitude_norm(
                    amplitudes
                ),
                **self.excitation_space.diagnostics(),
            },
            raw_moments=_ProjectedRawMoments(
                denominator=denominator,
                energy_numerator=energy_numerator,
                n2=n2_total,
                n4=n4_total,
                h2=h2_total,
                h4=h4_total,
            ),
        )

    def jvp(
        self,
        amplitudes: QPAmplitudes,
        direction: QPAmplitudes,
    ) -> KernelDirectionalDerivative:
        """Differentiate the W2 contour integration and projected Wick moments."""

        started = time.perf_counter()
        nspin = self.hamiltonian.nspin
        expected_pair = (nspin, nspin)
        expected_quad = (nspin,) * 4
        if amplitudes.t1.shape != expected_pair or direction.t1.shape != expected_pair:
            raise ValueError("projected JVP pair dimensions do not match")
        if amplitudes.t2.shape != expected_quad or direction.t2.shape != expected_quad:
            raise ValueError("projected JVP quadruple dimensions do not match")

        norm = 1.0 + 0.0j
        w1 = np.asarray(amplitudes.t1, dtype=np.complex128)
        w2 = np.asarray(amplitudes.t2, dtype=np.complex128)
        direction_norm = 0.0 + 0.0j
        direction_w1 = np.asarray(direction.t1, dtype=np.complex128)
        direction_w2 = np.asarray(direction.t2, dtype=np.complex128)
        base_angle = -1.0j * math.log(self.radius)
        (
            norm,
            w1,
            w2,
            direction_norm,
            direction_w1,
            direction_w2,
        ) = _rk4_interval_with_jvp(
            0.0 + 0.0j,
            base_angle,
            norm,
            w1,
            w2,
            direction_norm,
            direction_w1,
            direction_w2,
            self.reference,
            self.number_operator,
            self.radial_steps,
            self._a02_cache,
        )

        denominator = 0.0 + 0.0j
        direction_denominator = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        direction_energy_numerator = 0.0 + 0.0j
        n2_total = np.zeros(expected_pair, dtype=np.complex128)
        direction_n2_total = np.zeros_like(n2_total)
        h2_total = np.zeros(expected_pair, dtype=np.complex128)
        direction_h2_total = np.zeros_like(h2_total)
        n4_total = np.zeros(expected_quad, dtype=np.complex128)
        direction_n4_total = np.zeros_like(n4_total)
        h4_total = np.zeros(expected_quad, dtype=np.complex128)
        direction_h4_total = np.zeros_like(h4_total)
        kernel_time = 0.0
        ode_time = 0.0
        previous = base_angle
        pipeline_enabled, angle_workers, blas_threads, _available = (
            self._execution_layout()
        )

        def accumulate(value: _WeightedAngleDerivativeMoments) -> None:
            nonlocal denominator, direction_denominator
            nonlocal energy_numerator, direction_energy_numerator
            nonlocal n2_total, direction_n2_total, n4_total, direction_n4_total
            nonlocal h2_total, direction_h2_total, h4_total, direction_h4_total
            nonlocal kernel_time
            denominator += value.denominator
            direction_denominator += value.direction_denominator
            energy_numerator += value.energy_numerator
            direction_energy_numerator += value.direction_energy_numerator
            n2_total += value.n2
            direction_n2_total += value.direction_n2
            n4_total += value.n4
            direction_n4_total += value.direction_n4
            h2_total += value.h2
            direction_h2_total += value.direction_h2
            h4_total += value.h4
            direction_h4_total += value.direction_h4
            kernel_time += value.kernel_time

        with threadpool_limits(limits=blas_threads):
            pending: deque[
                tuple[int, Future[_WeightedAngleDerivativeMoments]]
            ] = deque()
            pending_limit = (
                max(1, min(self.execution_options.pipeline_depth, angle_workers))
                if pipeline_enabled
                else 1
            )
            executor = (
                ThreadPoolExecutor(
                    max_workers=angle_workers,
                    thread_name_prefix="qpccsd-gauge-jvp",
                )
                if pipeline_enabled
                else None
            )
            try:
                for point_index, point in enumerate(self.points):
                    ode_started = time.perf_counter()
                    (
                        norm,
                        w1,
                        w2,
                        direction_norm,
                        direction_w1,
                        direction_w2,
                    ) = _rk4_interval_with_jvp(
                        previous,
                        point.complex_angle,
                        norm,
                        w1,
                        w2,
                        direction_norm,
                        direction_w1,
                        direction_w2,
                        self.reference,
                        self.number_operator,
                        self.options.ode_substeps,
                        self._a02_cache,
                    )
                    ode_time += time.perf_counter() - ode_started
                    previous = point.complex_angle
                    if executor is None:
                        accumulate(
                            self._evaluate_angle_jvp(
                                point_index,
                                norm,
                                w1,
                                w2,
                                direction_norm,
                                direction_w1,
                                direction_w2,
                            )
                        )
                        continue
                    if len(pending) >= pending_limit:
                        _index, future = pending.popleft()
                        accumulate(future.result())
                    pending.append(
                        (
                            point_index,
                            executor.submit(
                                self._evaluate_angle_jvp,
                                point_index,
                                norm,
                                w1.copy(),
                                w2.copy(),
                                direction_norm,
                                direction_w1.copy(),
                                direction_w2.copy(),
                            ),
                        )
                    )
                while pending:
                    _index, future = pending.popleft()
                    accumulate(future.result())
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        energy = energy_numerator / denominator
        direction_energy = (
            direction_energy_numerator - energy * direction_denominator
        ) / denominator
        r1_numerator = h2_total - energy * n2_total
        r2_numerator = h4_total - energy * n4_total
        r1 = r1_numerator / denominator
        r2 = r2_numerator / denominator
        direction_r1_numerator = (
            direction_h2_total
            - direction_energy * n2_total
            - energy * direction_n2_total
        )
        direction_r2_numerator = (
            direction_h4_total
            - direction_energy * n4_total
            - energy * direction_n4_total
        )
        direction_r1 = (
            direction_r1_numerator - r1 * direction_denominator
        ) / denominator
        direction_r2 = (
            direction_r2_numerator - r2 * direction_denominator
        ) / denominator
        return KernelDirectionalDerivative(
            total_energy=np.real_if_close(direction_energy),
            correlation_energy=np.real_if_close(direction_energy),
            r1=np.real_if_close(direction_r1),
            r2=np.real_if_close(direction_r2),
            elapsed=time.perf_counter() - started,
            diagnostics={
                "derivative": "analytic-w2-projected-jvp",
                "kernel_time": kernel_time,
                "ode_time": ode_time,
                "grid_size": self.options.grid_size,
                "ode_substeps": self.options.ode_substeps,
                "pipeline_enabled": pipeline_enabled,
                "angle_workers": angle_workers,
                "blas_threads": blas_threads,
                "pipeline_depth": (
                    min(self.execution_options.pipeline_depth, angle_workers)
                    if pipeline_enabled
                    else 0
                ),
                "projector_ordering": PROJECTOR_ORDERING,
                "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
                "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
            },
            raw_moments=_ProjectedRawDerivativeMoments(
                denominator=denominator,
                direction_denominator=direction_denominator,
                energy_numerator=energy_numerator,
                direction_energy_numerator=direction_energy_numerator,
                n2=n2_total,
                direction_n2=direction_n2_total,
                n4=n4_total,
                direction_n4=direction_n4_total,
                h2=h2_total,
                direction_h2=direction_h2_total,
                h4=h4_total,
                direction_h4=direction_h4_total,
            ),
        )

    __call__ = evaluate


class SeriesProjectedQPCCSDEvaluator(ProjectedQPCCSDEvaluator):
    """Direct Ser2/Ser3 gauge evaluator with analytic tangent contractions."""

    def __init__(
        self,
        hamiltonian: QPHamiltonian,
        reference: BogoliubovReference,
        options: ProjectionOptions,
        excitation_space: QPExcitationSpaceLike | None = None,
        execution_options: ExecutionOptions | None = None,
        scalar_only: bool = False,
    ) -> None:
        if options.disentanglement_backend not in {"ser2", "ser3"}:
            raise ValueError("the series evaluator requires the ser2 or ser3 backend")
        super().__init__(
            hamiltonian,
            reference,
            options,
            excitation_space=excitation_space,
            execution_options=execution_options,
            scalar_only=scalar_only,
        )
        self.series_order = int(options.disentanglement_backend[-1])

    @staticmethod
    def _principal_gaussian_norm(t1: np.ndarray, z: np.ndarray) -> complex:
        identity = np.eye(t1.shape[0], dtype=np.complex128)
        return principal_sqrt_determinant(identity - z @ t1)

    def _gaussian_branch_signs(self, t1: np.ndarray) -> tuple[complex, ...]:
        """Continue the Onishi square root from the identity along the contour."""

        base_angle = -1.0j * math.log(self.radius)
        targets = (base_angle, *(point.complex_angle for point in self.points))
        previous_angle = 0.0 + 0.0j
        previous_value = 1.0 + 0.0j
        signs: list[complex] = []
        branch_steps = max(4, min(12, self.options.ode_substeps))
        for target_index, target in enumerate(targets):
            endpoint_principal = None
            for step in range(1, branch_steps + 1):
                angle = previous_angle + (target - previous_angle) * step / branch_steps
                _overlap, z = reference_thouless(self.reference, angle)
                principal = self._principal_gaussian_norm(t1, z)
                if abs(principal) <= self.options.overlap_tolerance:
                    raise ValueError("the Gaussian Ser norm crosses zero on the contour")
                continued = (
                    principal
                    if abs(principal - previous_value) <= abs(-principal - previous_value)
                    else -principal
                )
                previous_value = continued
                endpoint_principal = principal
            previous_angle = target
            if target_index:
                assert endpoint_principal is not None
                signs.append(
                    1.0 + 0.0j
                    if abs(previous_value - endpoint_principal)
                    <= abs(previous_value + endpoint_principal)
                    else -1.0 + 0.0j
                )
        return tuple(signs)

    def _series_states(
        self,
        amplitudes: QPAmplitudes,
    ) -> Iterator[tuple[complex, np.ndarray, np.ndarray, float, complex]]:
        t1 = np.asarray(amplitudes.t1, dtype=np.complex128)
        t2 = np.asarray(amplitudes.t2, dtype=np.complex128)
        signs = self._gaussian_branch_signs(t1)
        for point, sign in zip(self.points, signs):
            started = time.perf_counter()
            state = evaluate_disentangled_series(
                t1,
                t2,
                point.z,
                order=self.series_order,
            )
            yield (
                sign * state.norm,
                state.w1,
                state.w2,
                time.perf_counter() - started,
                state.scalar_polynomial,
            )

    def _series_states_with_jvp(
        self,
        amplitudes: QPAmplitudes,
        direction: QPAmplitudes,
    ) -> Iterator[
        tuple[
            complex,
            np.ndarray,
            np.ndarray,
            complex,
            np.ndarray,
            np.ndarray,
            float,
        ]
    ]:
        t1 = np.asarray(amplitudes.t1, dtype=np.complex128)
        t2 = np.asarray(amplitudes.t2, dtype=np.complex128)
        dt1 = np.asarray(direction.t1, dtype=np.complex128)
        dt2 = np.asarray(direction.t2, dtype=np.complex128)
        signs = self._gaussian_branch_signs(t1)
        for point, sign in zip(self.points, signs):
            started = time.perf_counter()
            state = evaluate_disentangled_series_with_jvp(
                t1,
                t2,
                point.z,
                dt1,
                dt2,
                order=self.series_order,
            )
            yield (
                sign * state.primal.norm,
                state.primal.w1,
                state.primal.w2,
                sign * state.direction_norm,
                state.direction_w1,
                state.direction_w2,
                time.perf_counter() - started,
            )

    def evaluate_energy(self, amplitudes: QPAmplitudes) -> _ProjectedScalarEvaluation:
        """Evaluate the angle-local Ser scalar without residual moments."""

        started = time.perf_counter()
        denominator = 0.0 + 0.0j
        denominator_compensation = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        energy_compensation = 0.0 + 0.0j
        reference_denominator = 0.0 + 0.0j
        reference_denominator_compensation = 0.0 + 0.0j
        reference_energy_numerator = 0.0 + 0.0j
        reference_energy_compensation = 0.0 + 0.0j
        series_time = 0.0
        kernel_time = 0.0
        transform_time = 0.0
        blas_threads = self.execution_options.blas_threads or 1
        with threadpool_limits(limits=blas_threads):
            for point_index, state in enumerate(self._series_states(amplitudes)):
                norm, w1, w2 = state[:3]
                series_time += float(state[3])
                value = self._evaluate_angle_energy(
                    point_index,
                    norm,
                    w1,
                    w2,
                )
                kernel_time += value.kernel_time
                transform_time += value.transform_time
                denominator, denominator_compensation = _kahan_add(
                    denominator,
                    denominator_compensation,
                    value.denominator,
                )
                energy_numerator, energy_compensation = _kahan_add(
                    energy_numerator,
                    energy_compensation,
                    value.energy_numerator,
                )
                (
                    reference_denominator,
                    reference_denominator_compensation,
                ) = _kahan_add(
                    reference_denominator,
                    reference_denominator_compensation,
                    value.reference_denominator,
                )
                (
                    reference_energy_numerator,
                    reference_energy_compensation,
                ) = _kahan_add(
                    reference_energy_numerator,
                    reference_energy_compensation,
                    value.reference_energy_numerator,
                )
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        if abs(reference_denominator) < self.options.overlap_tolerance:
            raise ValueError("the projected reference norm is numerically zero")
        return _ProjectedScalarEvaluation(
            denominator=denominator,
            energy_numerator=energy_numerator,
            total_energy=energy_numerator / denominator,
            elapsed=time.perf_counter() - started,
            reference_denominator=reference_denominator,
            reference_energy_numerator=reference_energy_numerator,
            reference_total_energy=(
                reference_energy_numerator / reference_denominator
            ),
            diagnostics={
                "mode": f"correlated-scalar-ser{self.series_order}",
                "series_time": series_time,
                "transform_time": transform_time,
                "kernel_time": kernel_time,
                "ode_time": 0.0,
            },
        )

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        started = time.perf_counter()
        nspin = self.hamiltonian.nspin
        states = self._series_states(amplitudes)
        denominator = 0.0 + 0.0j
        denominator_compensation = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        energy_compensation = 0.0 + 0.0j
        n2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        n2_compensation = np.zeros_like(n2_total)
        n4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        n4_compensation = np.zeros_like(n4_total)
        h2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        h2_compensation = np.zeros_like(h2_total)
        h4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        h4_compensation = np.zeros_like(h4_total)
        kernel_time = 0.0
        transform_time = 0.0
        angle_timings: list[float] = []
        series_timings: list[float] = []
        scalar_polynomials: list[complex] = []
        cache_hits_before = self._transform_cache_hits
        cache_misses_before = self._transform_cache_misses
        blas_threads = self.execution_options.blas_threads or 1

        with threadpool_limits(limits=blas_threads):
            for point_index, (norm, w1, w2, series_time, polynomial) in enumerate(states):
                value = self._evaluate_angle(point_index, norm, w1, w2)
                denominator, denominator_compensation = _kahan_add(
                    denominator, denominator_compensation, value.denominator
                )
                energy_numerator, energy_compensation = _kahan_add(
                    energy_numerator,
                    energy_compensation,
                    value.energy_numerator,
                )
                n2_total, n2_compensation = _kahan_add(
                    n2_total, n2_compensation, value.n2
                )
                n4_total, n4_compensation = _kahan_add(
                    n4_total, n4_compensation, value.n4
                )
                h2_total, h2_compensation = _kahan_add(
                    h2_total, h2_compensation, value.h2
                )
                h4_total, h4_compensation = _kahan_add(
                    h4_total, h4_compensation, value.h4
                )
                kernel_time += value.kernel_time
                transform_time += value.transform_time
                angle_timings.append(value.elapsed)
                series_timings.append(series_time)
                scalar_polynomials.append(polynomial)

        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        energy = energy_numerator / denominator
        r1 = (h2_total - energy * n2_total) / denominator
        r2 = (h4_total - energy * n4_total) / denominator
        residual = residual_to_vector(r1, r2, self.excitation_space)
        residual_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        elapsed = time.perf_counter() - started
        return KernelEvaluation(
            total_energy=np.real_if_close(energy),
            correlation_energy=np.real_if_close(energy - self.hamiltonian.constant),
            r1=np.real_if_close(r1),
            r2=np.real_if_close(r2),
            residual_norm=residual_norm,
            projected_norm=denominator,
            elapsed=elapsed,
            diagnostics={
                "projection_time": elapsed,
                "kernel_time": kernel_time,
                "transform_time": transform_time,
                "series_time": float(sum(series_timings)),
                "series_timings": series_timings,
                "angle_timings": angle_timings,
                "contour_setup_time": self.contour_setup_time,
                "transform_cache_capacity": self.transform_cache_capacity,
                "transform_cache_bytes": self.transform_cache_bytes,
                "transform_cache_hits": self._transform_cache_hits - cache_hits_before,
                "transform_cache_misses": self._transform_cache_misses
                - cache_misses_before,
                "parallel_mode": "serial-series",
                "pipeline_enabled": False,
                "angle_workers": 1,
                "blas_threads": blas_threads,
                "deterministic_accumulation": True,
                "gauge_quadrature": "midpoint",
                "grid_size": self.options.grid_size,
                "ode_substeps": 0,
                "contour_radius": self.radius,
                "contour_safety_threshold": self.options.contour_safety_threshold,
                "minimum_overlap_singular_value": self.minimum_overlap_singular_value,
                "disentanglement_backend": self.options.disentanglement_backend,
                "series_order": self.series_order,
                "series_equation_schema": SERIES_EQUATION_SCHEMA,
                "minimum_scalar_polynomial": float(
                    min(abs(value) for value in scalar_polynomials)
                ),
                "closure": f"W2-Ser{self.series_order}",
                "w3_omitted": True,
                "contraction_backend": generated_wick_orbits.CONTRACTION_BACKEND,
                "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
                "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
                "projector_ordering": PROJECTOR_ORDERING,
                "max_imaginary_energy": abs(float(np.imag(energy))),
                "max_imaginary_residual": (
                    float(np.max(np.abs(np.imag(residual)))) if residual.size else 0.0
                ),
                "internal_residual_norm": self.excitation_space.internal_residual_norm(
                    r1, r2
                ),
                "allowed_pair_count": self.excitation_space.pair_count,
                "allowed_quadruple_count": self.excitation_space.quadruple_count,
                "internal_coordinate_count": self.excitation_space.internal_coordinate_count,
                "forbidden_amplitude_norm": self.excitation_space.forbidden_amplitude_norm(
                    amplitudes
                ),
                **self.excitation_space.diagnostics(),
            },
            raw_moments=_ProjectedRawMoments(
                denominator=denominator,
                energy_numerator=energy_numerator,
                n2=n2_total,
                n4=n4_total,
                h2=h2_total,
                h4=h4_total,
            ),
        )

    def jvp(
        self,
        amplitudes: QPAmplitudes,
        direction: QPAmplitudes,
    ) -> KernelDirectionalDerivative:
        started = time.perf_counter()
        nspin = self.hamiltonian.nspin
        states = self._series_states_with_jvp(amplitudes, direction)
        denominator = 0.0 + 0.0j
        direction_denominator = 0.0 + 0.0j
        energy_numerator = 0.0 + 0.0j
        direction_energy_numerator = 0.0 + 0.0j
        n2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        direction_n2_total = np.zeros_like(n2_total)
        n4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        direction_n4_total = np.zeros_like(n4_total)
        h2_total = np.zeros((nspin, nspin), dtype=np.complex128)
        direction_h2_total = np.zeros_like(h2_total)
        h4_total = np.zeros((nspin,) * 4, dtype=np.complex128)
        direction_h4_total = np.zeros_like(h4_total)
        kernel_time = 0.0
        series_time = 0.0
        blas_threads = self.execution_options.blas_threads or 1
        with threadpool_limits(limits=blas_threads):
            for point_index, state in enumerate(states):
                value = self._evaluate_angle_jvp(point_index, *state[:6])
                denominator += value.denominator
                direction_denominator += value.direction_denominator
                energy_numerator += value.energy_numerator
                direction_energy_numerator += value.direction_energy_numerator
                n2_total += value.n2
                direction_n2_total += value.direction_n2
                n4_total += value.n4
                direction_n4_total += value.direction_n4
                h2_total += value.h2
                direction_h2_total += value.direction_h2
                h4_total += value.h4
                direction_h4_total += value.direction_h4
                kernel_time += value.kernel_time
                series_time += state[6]
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the correlated projected norm is numerically zero")
        energy = energy_numerator / denominator
        direction_energy = (
            direction_energy_numerator - energy * direction_denominator
        ) / denominator
        r1 = (h2_total - energy * n2_total) / denominator
        r2 = (h4_total - energy * n4_total) / denominator
        direction_r1 = (
            direction_h2_total
            - direction_energy * n2_total
            - energy * direction_n2_total
            - r1 * direction_denominator
        ) / denominator
        direction_r2 = (
            direction_h4_total
            - direction_energy * n4_total
            - energy * direction_n4_total
            - r2 * direction_denominator
        ) / denominator
        return KernelDirectionalDerivative(
            total_energy=np.real_if_close(direction_energy),
            correlation_energy=np.real_if_close(direction_energy),
            r1=np.real_if_close(direction_r1),
            r2=np.real_if_close(direction_r2),
            elapsed=time.perf_counter() - started,
            diagnostics={
                "derivative": f"analytic-ser{self.series_order}-projected-jvp",
                "kernel_time": kernel_time,
                "series_time": series_time,
                "grid_size": self.options.grid_size,
                "ode_substeps": 0,
                "disentanglement_backend": self.options.disentanglement_backend,
                "series_equation_schema": SERIES_EQUATION_SCHEMA,
                "pipeline_enabled": False,
                "angle_workers": 1,
                "blas_threads": blas_threads,
                "projector_ordering": PROJECTOR_ORDERING,
                "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
                "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
            },
            raw_moments=_ProjectedRawDerivativeMoments(
                denominator=denominator,
                direction_denominator=direction_denominator,
                energy_numerator=energy_numerator,
                direction_energy_numerator=direction_energy_numerator,
                n2=n2_total,
                direction_n2=direction_n2_total,
                n4=n4_total,
                direction_n4=direction_n4_total,
                h2=h2_total,
                direction_h2=direction_h2_total,
                h4=h4_total,
                direction_h4=direction_h4_total,
            ),
        )

    def contour_monodromy(self, amplitudes: QPAmplitudes) -> dict[str, float]:
        started = time.perf_counter()
        base_angle = -1.0j * math.log(self.radius)
        period = math.pi if self.options.parity == "even" else 2.0 * math.pi
        _overlap, start_z = reference_thouless(self.reference, base_angle)
        _overlap, stop_z = reference_thouless(
            self.reference, base_angle + period
        )
        start = evaluate_disentangled_series(
            amplitudes.t1, amplitudes.t2, start_z, order=self.series_order
        )
        stop = evaluate_disentangled_series(
            amplitudes.t1, amplitudes.t2, stop_z, order=self.series_order
        )
        return {
            "correlated_norm_return_error": float(abs(stop.norm - start.norm)),
            "w1_return_error": float(np.max(np.abs(stop.w1 - start.w1))),
            "w2_return_error": float(np.max(np.abs(stop.w2 - start.w2))),
            "period": float(period),
            "elapsed": time.perf_counter() - started,
        }

    def build_residual_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        raise ValueError(
            "sampled-W2 gauge bases are not defined for Ser backends; "
            "use the projected excitation metric"
        )

    def build_local_gauge_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        del amplitudes, options
        raise ValueError(
            "the ODE2 local U(1) tangent is not defined for Ser backends; "
            "use the projected excitation metric"
        )


class RichardsonProjectedQPCCSDEvaluator:
    """Fourth-order extrapolation of two alias-free midpoint gauge grids.

    Truncating the disentangled flow at W2 can make the contour moments smooth
    but nonperiodic.  The shifted midpoint projector then has a leading
    ``grid_size**-2`` error.  Combining grids L and 2L as
    ``(4 * value_2L - value_L) / 3`` removes that term without changing the
    Wick equations or the W2 ODE.
    """

    def __init__(
        self,
        hamiltonian: QPHamiltonian,
        reference: BogoliubovReference,
        options: ProjectionOptions,
        excitation_space: QPExcitationSpaceLike | None = None,
        execution_options: ExecutionOptions | None = None,
        scalar_only: bool = False,
    ) -> None:
        requested_execution = (
            ExecutionOptions() if execution_options is None else execution_options
        )
        component_execution = replace(
            requested_execution,
            max_workspace_bytes=(
                None
                if requested_execution.max_workspace_bytes is None
                else requested_execution.max_workspace_bytes // 2
            ),
        )
        midpoint_options = replace(
            options,
            gauge_quadrature="midpoint",
            auto_select_ode=False,
            cache_bytes=options.cache_bytes // 2,
        )
        component_type = (
            ProjectedQPCCSDEvaluator
            if options.disentanglement_backend == "ode2"
            else SeriesProjectedQPCCSDEvaluator
        )
        self.coarse = component_type(
            hamiltonian,
            reference,
            midpoint_options,
            excitation_space=excitation_space,
            execution_options=component_execution,
            scalar_only=scalar_only,
        )
        fine_options = replace(
            midpoint_options,
            grid_size=2 * midpoint_options.grid_size,
            contour_radius=self.coarse.radius,
        )
        self.fine = component_type(
            hamiltonian,
            reference,
            fine_options,
            excitation_space=self.coarse.excitation_space,
            execution_options=component_execution,
            scalar_only=scalar_only,
        )
        self.hamiltonian = hamiltonian
        self.reference = reference
        self.options = options
        self.execution_options = requested_execution
        self.excitation_space = self.coarse.excitation_space
        self.radius = self.coarse.radius
        self.minimum_overlap_singular_value = min(
            self.coarse.minimum_overlap_singular_value,
            self.fine.minimum_overlap_singular_value,
        )
        self.contour_setup_time = (
            self.coarse.contour_setup_time + self.fine.contour_setup_time
        )

    @staticmethod
    def _combine(coarse: np.ndarray | complex, fine: np.ndarray | complex):
        return (4.0 * fine - coarse) / 3.0

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        started = time.perf_counter()
        coarse = self.coarse.evaluate(amplitudes)
        fine = self.fine.evaluate(amplitudes)
        if coarse.raw_moments is None or fine.raw_moments is None:
            raise RuntimeError("midpoint evaluator did not retain raw projected moments")
        raw = _ProjectedRawMoments(
            denominator=self._combine(
                coarse.raw_moments.denominator, fine.raw_moments.denominator
            ),
            energy_numerator=self._combine(
                coarse.raw_moments.energy_numerator,
                fine.raw_moments.energy_numerator,
            ),
            n2=self._combine(coarse.raw_moments.n2, fine.raw_moments.n2),
            n4=self._combine(coarse.raw_moments.n4, fine.raw_moments.n4),
            h2=self._combine(coarse.raw_moments.h2, fine.raw_moments.h2),
            h4=self._combine(coarse.raw_moments.h4, fine.raw_moments.h4),
        )
        if abs(raw.denominator) < self.options.overlap_tolerance:
            raise ValueError("the Richardson-extrapolated projected norm is zero")
        energy = raw.energy_numerator / raw.denominator
        r1 = np.asarray((raw.h2 - energy * raw.n2) / raw.denominator)
        r2 = np.asarray((raw.h4 - energy * raw.n4) / raw.denominator)
        residual = residual_to_vector(r1, r2, self.excitation_space)
        residual_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        elapsed = time.perf_counter() - started
        diagnostics = dict(fine.diagnostics)
        diagnostics.update(
            {
                "projection_time": elapsed,
                "contour_setup_time": self.contour_setup_time,
                "gauge_quadrature": "midpoint-richardson",
                "richardson_order": 4,
                "richardson_combination": "raw-projected-moments",
                "coarse_grid_size": self.coarse.options.grid_size,
                "fine_grid_size": self.fine.options.grid_size,
                "coarse_projection_time": coarse.elapsed,
                "fine_projection_time": fine.elapsed,
                "coarse_energy": coarse.total_energy,
                "fine_energy": fine.total_energy,
                "coarse_residual_norm": coarse.residual_norm,
                "fine_residual_norm": fine.residual_norm,
                "internal_residual_norm": (
                    self.excitation_space.internal_residual_norm(r1, r2)
                ),
                "max_imaginary_energy": abs(float(np.imag(energy))),
                "max_imaginary_residual": (
                    float(np.max(np.abs(np.imag(residual))))
                    if residual.size
                    else 0.0
                ),
            }
        )
        return KernelEvaluation(
            total_energy=np.real_if_close(energy),
            correlation_energy=np.real_if_close(energy - self.hamiltonian.constant),
            r1=np.real_if_close(r1),
            r2=np.real_if_close(r2),
            residual_norm=residual_norm,
            projected_norm=np.real_if_close(raw.denominator),
            elapsed=elapsed,
            diagnostics=diagnostics,
            raw_moments=raw,
        )

    def evaluate_zero_amplitude_energy(self) -> _ProjectedScalarEvaluation:
        """Richardson-extrapolate only the scalar projected CAS baseline."""

        started = time.perf_counter()
        coarse = self.coarse.evaluate_zero_amplitude_energy()
        fine = self.fine.evaluate_zero_amplitude_energy()
        denominator = complex(
            self._combine(coarse.denominator, fine.denominator)
        )
        energy_numerator = complex(
            self._combine(coarse.energy_numerator, fine.energy_numerator)
        )
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the Richardson-extrapolated projected norm is zero")
        return _ProjectedScalarEvaluation(
            denominator=denominator,
            energy_numerator=energy_numerator,
            total_energy=energy_numerator / denominator,
            elapsed=time.perf_counter() - started,
            reference_denominator=denominator,
            reference_energy_numerator=energy_numerator,
            reference_total_energy=energy_numerator / denominator,
            diagnostics={
                "mode": "zero-amplitude-scalar-richardson",
                "coarse": coarse.diagnostics,
                "fine": fine.diagnostics,
            },
        )

    def evaluate_energy(self, amplitudes: QPAmplitudes) -> _ProjectedScalarEvaluation:
        """Richardson-extrapolate only the scalar projected energy."""

        started = time.perf_counter()
        coarse = self.coarse.evaluate_energy(amplitudes)
        fine = self.fine.evaluate_energy(amplitudes)
        denominator = complex(
            self._combine(coarse.denominator, fine.denominator)
        )
        energy_numerator = complex(
            self._combine(coarse.energy_numerator, fine.energy_numerator)
        )
        if (
            coarse.reference_denominator is None
            or fine.reference_denominator is None
            or coarse.reference_energy_numerator is None
            or fine.reference_energy_numerator is None
        ):
            raise RuntimeError("midpoint scalar evaluator omitted the reference baseline")
        reference_denominator = complex(
            self._combine(
                coarse.reference_denominator,
                fine.reference_denominator,
            )
        )
        reference_energy_numerator = complex(
            self._combine(
                coarse.reference_energy_numerator,
                fine.reference_energy_numerator,
            )
        )
        if abs(denominator) < self.options.overlap_tolerance:
            raise ValueError("the Richardson-extrapolated projected norm is zero")
        if abs(reference_denominator) < self.options.overlap_tolerance:
            raise ValueError("the Richardson-extrapolated reference norm is zero")
        return _ProjectedScalarEvaluation(
            denominator=denominator,
            energy_numerator=energy_numerator,
            total_energy=energy_numerator / denominator,
            elapsed=time.perf_counter() - started,
            reference_denominator=reference_denominator,
            reference_energy_numerator=reference_energy_numerator,
            reference_total_energy=(
                reference_energy_numerator / reference_denominator
            ),
            diagnostics={
                "mode": "correlated-scalar-richardson",
                "coarse": coarse.diagnostics,
                "fine": fine.diagnostics,
            },
        )

    def jvp(
        self,
        amplitudes: QPAmplitudes,
        direction: QPAmplitudes,
    ) -> KernelDirectionalDerivative:
        started = time.perf_counter()
        coarse = self.coarse.jvp(amplitudes, direction)
        fine = self.fine.jvp(amplitudes, direction)
        if coarse.raw_moments is None or fine.raw_moments is None:
            raise RuntimeError("midpoint JVP did not retain raw projected moments")
        raw = _ProjectedRawDerivativeMoments(
            **{
                name: self._combine(
                    getattr(coarse.raw_moments, name),
                    getattr(fine.raw_moments, name),
                )
                for name in _ProjectedRawDerivativeMoments.__dataclass_fields__
            }
        )
        if abs(raw.denominator) < self.options.overlap_tolerance:
            raise ValueError("the Richardson-extrapolated projected norm is zero")
        energy = raw.energy_numerator / raw.denominator
        direction_energy = (
            raw.direction_energy_numerator
            - energy * raw.direction_denominator
        ) / raw.denominator
        r1 = (raw.h2 - energy * raw.n2) / raw.denominator
        r2 = (raw.h4 - energy * raw.n4) / raw.denominator
        direction_r1 = (
            raw.direction_h2
            - direction_energy * raw.n2
            - energy * raw.direction_n2
            - r1 * raw.direction_denominator
        ) / raw.denominator
        direction_r2 = (
            raw.direction_h4
            - direction_energy * raw.n4
            - energy * raw.direction_n4
            - r2 * raw.direction_denominator
        ) / raw.denominator
        elapsed = time.perf_counter() - started
        diagnostics = dict(fine.diagnostics)
        diagnostics.update(
            {
                "derivative": "analytic-w2-projected-richardson-jvp",
                "gauge_quadrature": "midpoint-richardson",
                "richardson_order": 4,
                "richardson_combination": "raw-projected-moments",
                "coarse_grid_size": self.coarse.options.grid_size,
                "fine_grid_size": self.fine.options.grid_size,
                "coarse_projection_time": coarse.elapsed,
                "fine_projection_time": fine.elapsed,
            }
        )
        return KernelDirectionalDerivative(
            total_energy=np.real_if_close(direction_energy),
            correlation_energy=np.real_if_close(direction_energy),
            r1=np.real_if_close(direction_r1),
            r2=np.real_if_close(direction_r2),
            elapsed=elapsed,
            diagnostics=diagnostics,
            raw_moments=raw,
        )

    def contour_monodromy(self, amplitudes: QPAmplitudes) -> dict[str, float]:
        return self.fine.contour_monodromy(amplitudes)

    def build_residual_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        return self.fine.build_residual_basis(amplitudes, options)

    def build_metric_projector(
        self,
        options: GaugeModeOptions,
    ) -> ProjectedMetricProjector:
        # The excitation metric has no disentangled-cluster ODE and is a
        # finite particle-number Fourier polynomial.  One alias-free component
        # grid is exact; Richardson refinement is required only by W2 moments.
        return self.coarse.build_metric_projector(options)

    def build_local_gauge_basis(
        self,
        amplitudes: QPAmplitudes,
        options: GaugeModeOptions,
    ) -> ProjectedResidualBasis:
        return self.fine.build_local_gauge_basis(amplitudes, options)

    __call__ = evaluate


def _build_projected_evaluator(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference,
    options: ProjectionOptions,
    excitation_space: QPExcitationSpaceLike | None,
    execution_options: ExecutionOptions | None,
    scalar_only: bool = False,
) -> (
    ProjectedQPCCSDEvaluator
    | SeriesProjectedQPCCSDEvaluator
    | RichardsonProjectedQPCCSDEvaluator
):
    if options.gauge_quadrature == "midpoint-richardson":
        evaluator_type = RichardsonProjectedQPCCSDEvaluator
    elif options.disentanglement_backend == "ode2":
        evaluator_type = ProjectedQPCCSDEvaluator
    else:
        evaluator_type = SeriesProjectedQPCCSDEvaluator
    return evaluator_type(
        hamiltonian,
        reference,
        options,
        excitation_space=excitation_space,
        execution_options=execution_options,
        scalar_only=scalar_only,
    )


def _projected_component_norms(
    evaluation: KernelEvaluation,
    excitation_space: QPExcitationSpaceLike,
    basis: object,
) -> tuple[float, float, float]:
    residual = np.asarray(
        residual_to_vector(evaluation.r1, evaluation.r2, excitation_space),
        dtype=np.complex128,
    )
    full_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
    if not bool(getattr(basis, "active", getattr(basis, "gauge_rank", 0))):
        return full_norm, 0.0, full_norm
    physical = basis.project_physical(residual)
    gauge = basis.project_gauge(residual)
    return (
        float(np.max(np.abs(physical))) if physical.size else 0.0,
        float(np.max(np.abs(gauge))) if gauge.size else 0.0,
        full_norm,
    )


@dataclass(frozen=True)
class _ProjectionHomotopyEvaluator:
    projected_evaluator: ProjectedQPCCSDEvaluator | RichardsonProjectedQPCCSDEvaluator
    hamiltonian: QPHamiltonian
    excitation_space: QPExcitationSpaceLike
    strength: float

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        projected = self.projected_evaluator.evaluate(amplitudes)
        if self.strength == 1.0:
            return projected
        unprojected = evaluate_qpccsd(
            self.hamiltonian,
            amplitudes,
            excitation_space=self.excitation_space,
        )
        weight = self.strength
        energy = (1.0 - weight) * complex(unprojected.total_energy) + weight * complex(
            projected.total_energy
        )
        r1 = (1.0 - weight) * np.asarray(unprojected.r1) + weight * np.asarray(
            projected.r1
        )
        r2 = (1.0 - weight) * np.asarray(unprojected.r2) + weight * np.asarray(
            projected.r2
        )
        residual = residual_to_vector(r1, r2, self.excitation_space)
        diagnostics = dict(projected.diagnostics)
        diagnostics.update(
            {
                "projection_continuation_strength": weight,
                "projection_continuation_intermediate": True,
            }
        )
        return KernelEvaluation(
            total_energy=np.real_if_close(energy),
            correlation_energy=np.real_if_close(energy - self.hamiltonian.constant),
            r1=np.real_if_close(r1),
            r2=np.real_if_close(r2),
            residual_norm=float(np.max(np.abs(residual))) if residual.size else 0.0,
            projected_norm=projected.projected_norm,
            elapsed=projected.elapsed + unprojected.elapsed,
            diagnostics=diagnostics,
        )

    def jvp(
        self,
        amplitudes: QPAmplitudes,
        direction: QPAmplitudes,
    ) -> KernelDirectionalDerivative:
        projected = self.projected_evaluator.jvp(amplitudes, direction)
        if self.strength == 1.0:
            return projected
        _evaluation, unprojected = evaluate_qpccsd_with_jvp(
            self.hamiltonian,
            amplitudes,
            direction,
            excitation_space=self.excitation_space,
        )
        weight = self.strength
        return KernelDirectionalDerivative(
            total_energy=np.real_if_close(
                (1.0 - weight) * complex(unprojected.total_energy)
                + weight * complex(projected.total_energy)
            ),
            correlation_energy=np.real_if_close(
                (1.0 - weight) * complex(unprojected.correlation_energy)
                + weight * complex(projected.correlation_energy)
            ),
            r1=np.real_if_close(
                (1.0 - weight) * np.asarray(unprojected.r1)
                + weight * np.asarray(projected.r1)
            ),
            r2=np.real_if_close(
                (1.0 - weight) * np.asarray(unprojected.r2)
                + weight * np.asarray(projected.r2)
            ),
            elapsed=projected.elapsed + unprojected.elapsed,
            diagnostics={
                **projected.diagnostics,
                "projection_continuation_strength": weight,
                "projection_continuation_intermediate": True,
            },
        )


def _secant_continuation_predictor(
    excitation_space: QPExcitationSpaceLike,
    previous_amplitudes: QPAmplitudes,
    current_amplitudes: QPAmplitudes,
    previous_strength: float,
    current_strength: float,
    target_strength: float,
    maximum_step: float,
) -> tuple[QPAmplitudes, float]:
    """Predict the next homotopy root without constructing a dense Jacobian."""

    denominator = current_strength - previous_strength
    if denominator <= 0.0 or target_strength <= current_strength:
        raise ValueError("continuation predictor strengths must be strictly increasing")
    previous = excitation_space.pack(previous_amplitudes)
    current = excitation_space.pack(current_amplitudes)
    step = (target_strength - current_strength) * (current - previous) / denominator
    step_norm = float(np.max(np.abs(step))) if step.size else 0.0
    if step_norm > maximum_step:
        step *= maximum_step / step_norm
        step_norm = maximum_step
    return excitation_space.unpack(current + step), step_norm


def evaluate_projected_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference,
    amplitudes: QPAmplitudes,
    *,
    options: ProjectionOptions | None = None,
    excitation_space: QPExcitationSpaceLike | None = None,
    execution_options: ExecutionOptions | None = None,
    gauge_options: GaugeModeOptions | None = None,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
) -> KernelEvaluation:
    convention = normalize_energy_convention(energy_convention)
    bogoliubov, cas_reference, space = _resolve_reference_and_space(
        hamiltonian, reference, excitation_space
    )
    if convention is EnergyConvention.CASSCF_PLUS_DELTA and cas_reference is None:
        raise ValueError(
            "casscf_plus_delta requires a CASQPReference with a CASSCF energy"
        )
    assert bogoliubov is not None
    if options is None:
        options = ProjectionOptions(
            target_number=bogoliubov.target_number,
            grid_size=_minimum_alias_free_grid_size(bogoliubov.nspin, "even"),
            gauge_quadrature="midpoint-richardson",
            disentanglement_backend="ode2",
        )
    evaluator = _build_projected_evaluator(
        hamiltonian,
        bogoliubov,
        options,
        space,
        execution_options,
    )
    result = evaluator.evaluate(space.enforce(amplitudes))
    if gauge_options is not None and gauge_options.enabled:
        basis = (
            evaluator.build_metric_projector(gauge_options)
            if gauge_options.method == "projected-metric"
            else evaluator.build_residual_basis(
                space.enforce(amplitudes), gauge_options
            )
        )
        physical_norm, gauge_norm, full_norm = _projected_component_norms(
            result, space, basis
        )
        result.diagnostics.update(
            {
                "physical_residual_norm": physical_norm,
                "gauge_residual_norm": gauge_norm,
                "full_residual_norm": full_norm,
                "gauge_rank": basis.gauge_rank,
                "gauge_identity_defect": basis.identity_defect,
                "gauge_singular_values": basis.singular_values.tolist(),
                "gauge_projector_source": basis.source,
                "projected_metric": (
                    basis.diagnostics() if hasattr(basis, "diagnostics") else None
                ),
            }
        )
    if convention is EnergyConvention.CASSCF_PLUS_DELTA:
        assert cas_reference is not None
        zero = evaluator.evaluate(QPAmplitudes.zeros(hamiltonian.nspin))
        raw_energy = complex(result.total_energy)
        dynamic = raw_energy - complex(zero.total_energy)
        delta_total = complex(cas_reference.casscf_energy) + dynamic
        baseline_gap = raw_energy - delta_total
        result.total_energy = np.real_if_close(delta_total)
        result.correlation_energy = np.real_if_close(dynamic)
        result.diagnostics.update(
            {
                "energy_convention": convention.value,
                "energy_definition": convention.description,
                "casscf_energy_added": True,
                "cas_plus_delta_applied": True,
                "raw_qp_energy": np.real_if_close(raw_energy),
                "raw_projected_energy": np.real_if_close(raw_energy),
                "casscf_plus_dynamic_delta": np.real_if_close(delta_total),
                "baseline_gap": np.real_if_close(baseline_gap),
                "zero_amplitude_qp_energy": zero.total_energy,
                "casscf_reference_energy": cas_reference.casscf_energy,
                "rdm_cumulant_norm": cas_reference.rdm_cumulant_norm,
                "reference_mode": cas_reference.reference_mode,
                "reference_reconstruction_metrics": dict(
                    cas_reference.reconstruction_metrics
                ),
            }
        )
    else:
        result.diagnostics.update(
            {
                "energy_convention": convention.value,
                "energy_definition": convention.description,
                "raw_qp_energy": np.real_if_close(result.total_energy),
                "raw_projected_energy": np.real_if_close(result.total_energy),
                "casscf_energy_added": False,
                "cas_plus_delta_applied": False,
                "reference_mode": (
                    None if cas_reference is None else cas_reference.reference_mode
                ),
            }
        )
    return result


def evaluate_pav_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference,
    amplitudes: QPAmplitudes,
    *,
    options: ProjectionOptions | None = None,
    excitation_space: QPExcitationSpaceLike | None = None,
    execution_options: ExecutionOptions | None = None,
    compute_residual_diagnostic: bool = False,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
) -> QPPAVEvaluation:
    """Project fixed QPCCSD amplitudes and validate on a doubled resolution.

    This is projection after variation: no projected residual is optimized.
    The projected residual is optional and is never used to update amplitudes.
    """

    convention = normalize_energy_convention(energy_convention)
    bogoliubov, cas_reference, space = _resolve_reference_and_space(
        hamiltonian, reference, excitation_space
    )
    if convention is EnergyConvention.CASSCF_PLUS_DELTA and cas_reference is None:
        raise ValueError(
            "casscf_plus_delta requires a CASQPReference with a CASSCF energy"
        )
    apply_cas_delta = bool(
        convention is EnergyConvention.CASSCF_PLUS_DELTA
        and cas_reference is not None
    )
    assert bogoliubov is not None
    if options is None:
        options = ProjectionOptions(
            target_number=bogoliubov.target_number,
            grid_size=_minimum_alias_free_grid_size(bogoliubov.nspin, "even"),
            gauge_quadrature="midpoint",
            ode_substeps=1,
            auto_select_ode=False,
            disentanglement_backend="ser2",
        )
    locked_options = replace(options, auto_select_ode=False)
    evaluator = _build_projected_evaluator(
        hamiltonian,
        bogoliubov,
        locked_options,
        space,
        execution_options,
        scalar_only=not compute_residual_diagnostic,
    )
    fixed_amplitudes = space.enforce(amplitudes)
    projected = (
        evaluator.evaluate(fixed_amplitudes)
        if compute_residual_diagnostic
        else evaluator.evaluate_energy(fixed_amplitudes)
    )
    zero = (
        evaluator.evaluate_zero_amplitude_energy()
        if apply_cas_delta and compute_residual_diagnostic
        else None
    )
    validation_radius = evaluator.radius
    del evaluator

    validation_options = replace(
        locked_options,
        grid_size=2 * locked_options.grid_size,
        ode_substeps=2 * locked_options.ode_substeps,
        contour_radius=validation_radius,
        auto_select_ode=False,
        max_grid_refinements=0,
    )
    validation_evaluator = _build_projected_evaluator(
        hamiltonian,
        bogoliubov,
        validation_options,
        space,
        execution_options,
        scalar_only=not compute_residual_diagnostic,
    )
    validation = (
        validation_evaluator.evaluate(fixed_amplitudes)
        if compute_residual_diagnostic
        else validation_evaluator.evaluate_energy(fixed_amplitudes)
    )
    validation_zero = (
        validation_evaluator.evaluate_zero_amplitude_energy()
        if apply_cas_delta and compute_residual_diagnostic
        else None
    )

    raw_energy = complex(projected.total_energy)
    validation_raw_energy = complex(validation.total_energy)
    if not apply_cas_delta:
        total_energy = raw_energy
        validation_energy = validation_raw_energy
        dynamic_energy = None
        zero_energy = None
        baseline_gap = None
    else:
        assert cas_reference is not None
        if compute_residual_diagnostic:
            assert zero is not None and validation_zero is not None
            zero_energy = complex(zero.total_energy)
            validation_zero_energy = complex(validation_zero.total_energy)
        else:
            if (
                projected.reference_total_energy is None
                or validation.reference_total_energy is None
            ):
                raise RuntimeError("scalar PAV evaluation omitted its CAS baseline")
            zero_energy = complex(projected.reference_total_energy)
            validation_zero_energy = complex(
                validation.reference_total_energy
            )
        dynamic_energy = raw_energy - zero_energy
        validation_dynamic = validation_raw_energy - validation_zero_energy
        total_energy = complex(cas_reference.casscf_energy) + dynamic_energy
        validation_energy = (
            complex(cas_reference.casscf_energy) + validation_dynamic
        )
        baseline_gap = raw_energy - total_energy

    if compute_residual_diagnostic:
        residual = residual_to_vector(projected.r1, projected.r2, space)
        validation_residual = residual_to_vector(
            validation.r1, validation.r2, space
        )
        projected_residual_norm = (
            float(np.max(np.abs(residual))) if residual.size else 0.0
        )
        validation_residual_norm = (
            float(np.max(np.abs(validation_residual)))
            if validation_residual.size
            else 0.0
        )
        residual_grid_error = (
            float(np.max(np.abs(validation_residual - residual)))
            if residual.size
            else 0.0
        )
    else:
        projected_residual_norm = float("nan")
        validation_residual_norm = float("nan")
        residual_grid_error = float("nan")
    grid_error = float(abs(validation_energy - total_energy))
    raw_grid_error = float(abs(validation_raw_energy - raw_energy))
    imaginary_energy_error = max(
        float(abs(total_energy.imag)),
        float(abs(validation_energy.imag)),
    )
    raw_imaginary_energy_error = max(
        float(abs(raw_energy.imag)),
        float(abs(validation_raw_energy.imag)),
    )
    validation_passed = bool(
        grid_error < locked_options.validation_tolerance
        and imaginary_energy_error < locked_options.validation_tolerance
        and (
            not compute_residual_diagnostic
            or residual_grid_error
            < locked_options.validation_residual_tolerance
        )
    )
    validation_failure_reasons = []
    if grid_error >= locked_options.validation_tolerance:
        validation_failure_reasons.append("resolution_difference")
    if imaginary_energy_error >= locked_options.validation_tolerance:
        validation_failure_reasons.append("complex_energy_or_w2_contour_nonclosure")
    if (
        compute_residual_diagnostic
        and residual_grid_error >= locked_options.validation_residual_tolerance
    ):
        validation_failure_reasons.append("residual_resolution_difference")
    projection_time = float(
        projected.elapsed
        + validation.elapsed
        + (0.0 if zero is None else zero.elapsed)
        + (0.0 if validation_zero is None else validation_zero.elapsed)
    )
    gauge_components = (
        2 if locked_options.gauge_quadrature == "midpoint-richardson" else 1
    )
    return QPPAVEvaluation(
        total_energy=np.real_if_close(total_energy),
        raw_projected_energy=np.real_if_close(raw_energy),
        validation_energy=np.real_if_close(validation_energy),
        amplitudes=fixed_amplitudes,
        projected_residual_norm=projected_residual_norm,
        validation_residual_norm=validation_residual_norm,
        grid_error=grid_error,
        residual_grid_error=residual_grid_error,
        validation_passed=validation_passed,
        projection_time=projection_time,
        zero_amplitude_energy=(
            None if zero_energy is None else np.real_if_close(zero_energy)
        ),
        dynamic_correlation_energy=(
            None if dynamic_energy is None else np.real_if_close(dynamic_energy)
        ),
        baseline_gap=(
            None if baseline_gap is None else np.real_if_close(baseline_gap)
        ),
        validation_raw_projected_energy=np.real_if_close(validation_raw_energy),
        raw_grid_error=raw_grid_error,
        raw_imaginary_energy_error=raw_imaginary_energy_error,
        diagnostics={
            "method": "particle-number projection after variation",
            "energy_convention": convention.value,
            "energy_definition": convention.description,
            "casscf_energy_added": apply_cas_delta,
            "cas_plus_delta_applied": apply_cas_delta,
            "amplitudes_optimized_for": "unprojected_qpccsd",
            "projected_residual_optimized": False,
            "residual_diagnostic_computed": compute_residual_diagnostic,
            "projection_equation_schema": PROJECTION_EQUATION_SCHEMA,
            "projection_solver_schema": PROJECTION_SOLVER_SCHEMA,
            "projector_ordering": PROJECTOR_ORDERING,
            "disentanglement_backend": locked_options.disentanglement_backend,
            "gauge_quadrature": locked_options.gauge_quadrature,
            "baseline_grid_size": locked_options.grid_size,
            "baseline_ode_substeps": locked_options.ode_substeps,
            "validation_grid_size": validation_options.grid_size,
            "validation_ode_substeps": validation_options.ode_substeps,
            "contour_radius": validation_radius,
            "grid_error": grid_error,
            "raw_grid_error": raw_grid_error,
            "imaginary_energy_error": imaginary_energy_error,
            "raw_imaginary_energy_error": raw_imaginary_energy_error,
            "residual_grid_error": residual_grid_error,
            "validation_passed": validation_passed,
            "validation_failure_reasons": validation_failure_reasons,
            "projected_evaluations": (
                2
                if apply_cas_delta and compute_residual_diagnostic
                else 1
            ),
            "validation_evaluations": (
                2
                if apply_cas_delta and compute_residual_diagnostic
                else 1
            ),
            "full_residual_evaluations": (
                2 if compute_residual_diagnostic else 0
            ),
            "scalar_baseline_evaluations": (
                2
                if apply_cas_delta and compute_residual_diagnostic
                else 0
            ),
            "fused_reference_baseline": bool(
                apply_cas_delta and not compute_residual_diagnostic
            ),
            "scalar_correlated_evaluations": (
                0 if compute_residual_diagnostic else 2
            ),
            "gauge_component_evaluations": (
                (
                    4
                    if apply_cas_delta and compute_residual_diagnostic
                    else 2
                )
                * gauge_components
            ),
            **(
                {
                    "production_scalar_timings": dict(projected.diagnostics),
                    "validation_scalar_timings": dict(validation.diagnostics),
                }
                if not compute_residual_diagnostic
                else {}
            ),
        },
    )


def solve_pav_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference,
    *,
    projection_options: ProjectionOptions | None = None,
    solver_options: SolverOptions | None = None,
    initial_amplitudes: QPAmplitudes | None = None,
    excitation_space: QPExcitationSpaceLike | None = None,
    execution_options: ExecutionOptions | None = None,
    iteration_callback: IterationCallback | None = None,
    compute_residual_diagnostic: bool = False,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
) -> QPPAVResult:
    """Solve unprojected QPCCSD and project its fixed amplitudes once."""

    qpccsd = solve_qpccsd(
        hamiltonian,
        reference,
        initial_amplitudes=initial_amplitudes,
        options=solver_options,
        excitation_space=excitation_space,
        iteration_callback=iteration_callback,
        energy_convention=energy_convention,
    )
    projection = evaluate_pav_qpccsd(
        hamiltonian,
        reference,
        qpccsd.amplitudes,
        options=projection_options,
        excitation_space=excitation_space,
        execution_options=execution_options,
        compute_residual_diagnostic=compute_residual_diagnostic,
        energy_convention=energy_convention,
    )
    return QPPAVResult(
        converged=bool(qpccsd.converged and projection.validation_passed),
        qpccsd=qpccsd,
        projection=projection,
    )


def _select_ode_evaluator(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference,
    options: ProjectionOptions,
    amplitudes: QPAmplitudes,
    excitation_space: QPExcitationSpaceLike,
    execution_options: ExecutionOptions | None,
) -> tuple[
    ProjectedQPCCSDEvaluator
    | SeriesProjectedQPCCSDEvaluator
    | RichardsonProjectedQPCCSDEvaluator,
    ProjectionOptions,
    dict[str, object],
]:
    if options.disentanglement_backend != "ode2":
        locked_options = replace(options, auto_select_ode=False)
        evaluator = _build_projected_evaluator(
            hamiltonian,
            reference,
            locked_options,
            excitation_space,
            execution_options,
        )
        return evaluator, locked_options, {
            "enabled": False,
            "reason": "direct-series-backend",
            "selected_substeps": 0,
            "evaluations": 0,
            "elapsed": 0.0,
            "trials": [],
        }
    if not options.auto_select_ode:
        evaluator = _build_projected_evaluator(
            hamiltonian,
            reference,
            options,
            excitation_space,
            execution_options,
        )
        return evaluator, options, {
            "enabled": False,
            "selected_substeps": options.ode_substeps,
            "evaluations": 0,
            "elapsed": 0.0,
            "trials": [],
        }

    started = time.perf_counter()
    candidates = tuple(
        value
        for value in options.ode_substep_candidates
        if value <= options.ode_substeps
    )
    candidate_set = set(candidates)
    pairs = tuple(
        (coarse, 2 * coarse)
        for coarse in candidates
        if 2 * coarse in candidate_set
    )
    trials: list[dict[str, float | int | bool]] = []
    evaluation_count = 0
    for coarse, fine in pairs:
        coarse_options = replace(
            options,
            ode_substeps=coarse,
            auto_select_ode=False,
        )
        fine_options = replace(
            options,
            ode_substeps=fine,
            auto_select_ode=False,
        )
        coarse_evaluator = _build_projected_evaluator(
            hamiltonian,
            reference,
            coarse_options,
            excitation_space,
            execution_options,
        )
        coarse_result = coarse_evaluator.evaluate(amplitudes)
        fine_evaluator = _build_projected_evaluator(
            hamiltonian,
            reference,
            fine_options,
            excitation_space,
            execution_options,
        )
        fine_result = fine_evaluator.evaluate(amplitudes)
        evaluation_count += 2
        energy_error = float(
            abs(complex(coarse_result.total_energy) - complex(fine_result.total_energy))
        )
        coarse_residual = residual_to_vector(
            coarse_result.r1, coarse_result.r2, excitation_space
        )
        fine_residual = residual_to_vector(
            fine_result.r1, fine_result.r2, excitation_space
        )
        residual_error = float(
            np.max(np.abs(coarse_residual - fine_residual))
        ) if coarse_residual.size else 0.0
        passed = bool(
            energy_error <= options.ode_selection_energy_tolerance
            and residual_error <= options.ode_selection_residual_tolerance
        )
        trials.append(
            {
                "coarse_substeps": coarse,
                "fine_substeps": fine,
                "energy_error": energy_error,
                "residual_error": residual_error,
                "passed": passed,
            }
        )
        if passed:
            return coarse_evaluator, coarse_options, {
                "enabled": True,
                "selected_substeps": coarse,
                "evaluations": evaluation_count,
                "elapsed": time.perf_counter() - started,
                "trials": trials,
            }

    locked_options = replace(options, auto_select_ode=False)
    evaluator = _build_projected_evaluator(
        hamiltonian,
        reference,
        locked_options,
        excitation_space,
        execution_options,
    )
    return evaluator, locked_options, {
        "enabled": True,
        "selected_substeps": locked_options.ode_substeps,
        "evaluations": evaluation_count,
        "elapsed": time.perf_counter() - started,
        "trials": trials,
    }


def solve_projected_qpccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference,
    *,
    projection_options: ProjectionOptions | None = None,
    solver_options: SolverOptions | None = None,
    initial_amplitudes: QPAmplitudes | None = None,
    requested_method: str = "qpccsd",
    excitation_space: QPExcitationSpaceLike | None = None,
    iteration_callback: IterationCallback | None = None,
    execution_options: ExecutionOptions | None = None,
    gauge_options: GaugeModeOptions | None = None,
    continuation_options: ProjectionContinuationOptions | None = None,
) -> QPCCSDResult:
    bogoliubov, cas_reference, space = _resolve_reference_and_space(
        hamiltonian, reference, excitation_space
    )
    assert bogoliubov is not None
    projection_options = (
        ProjectionOptions(
            target_number=bogoliubov.target_number,
            grid_size=_minimum_alias_free_grid_size(bogoliubov.nspin, "even"),
            gauge_quadrature="midpoint-richardson",
        )
        if projection_options is None
        else projection_options
    )
    solver_options = SolverOptions() if solver_options is None else solver_options
    gauge_options = GaugeModeOptions() if gauge_options is None else gauge_options
    selection_amplitudes = space.enforce(
        QPAmplitudes.zeros(hamiltonian.nspin, dtype=np.complex128)
        if initial_amplitudes is None
        else initial_amplitudes
    )
    evaluator, locked_projection_options, ode_selection = _select_ode_evaluator(
        hamiltonian,
        bogoliubov,
        projection_options,
        selection_amplitudes,
        space,
        execution_options,
    )
    metric_setup_started = time.perf_counter()
    if not gauge_options.enabled:
        diagnostic_basis = ProjectedResidualBasis.empty(space.coordinate_count)
        coarse_basis = None
        step_basis_builder = None
    elif gauge_options.method == "projected-metric":
        diagnostic_basis = evaluator.build_metric_projector(gauge_options)
        coarse_basis = None
        step_basis_builder = (
            lambda _amplitudes: diagnostic_basis
            if gauge_options.apply_to_solver and gauge_options.project_solver_steps
            else None
        )
    else:
        diagnostic_basis = evaluator.build_residual_basis(
            selection_amplitudes,
            gauge_options,
        )
        coarse_basis = diagnostic_basis
        step_basis_builder = None
        if (
            gauge_options.apply_to_solver
            and diagnostic_basis.identity_defect > gauge_options.identity_tolerance
        ):
            raise ValueError(
                "candidate gauge basis fails the W2 contour-identity certification"
            )
    metric_setup_time = time.perf_counter() - metric_setup_started
    residual_basis = (
        diagnostic_basis
        if gauge_options.enabled and gauge_options.apply_to_solver
        else None
    )
    if continuation_options is None:
        result = _solve(
            evaluator.evaluate,
            hamiltonian,
            bogoliubov,
            initial_amplitudes,
            solver_options,
            requested_method,
            projected=True,
            excitation_space=space,
            iteration_callback=iteration_callback,
            jvp_evaluator=evaluator.jvp,
            residual_basis=residual_basis,
            coarse_basis=coarse_basis,
            gauge_options=gauge_options,
            step_basis_builder=step_basis_builder,
        )
    else:
        if sum(continuation_options.evaluator_calls) > solver_options.max_evaluator_calls:
            raise ValueError(
                "projection continuation exceeds the total evaluator-call budget"
            )
        seed = selection_amplitudes
        continuation_points: list[tuple[float, QPAmplitudes]] = []
        stage_reports: list[dict[str, object]] = []
        combined_history: list[dict[str, object]] = []
        total_projection_time = 0.0
        total_evaluator_calls = 0
        total_jvp_calls = 0
        result = None
        for strength, call_budget, stage_tolerance in zip(
            continuation_options.strengths,
            continuation_options.evaluator_calls,
            continuation_options.residual_tolerances,
        ):
            predictor_step_norm = 0.0
            if len(continuation_points) >= 2:
                previous_strength, previous_amplitudes = continuation_points[-2]
                current_strength, current_amplitudes = continuation_points[-1]
                seed, predictor_step_norm = _secant_continuation_predictor(
                    space,
                    previous_amplitudes,
                    current_amplitudes,
                    previous_strength,
                    current_strength,
                    strength,
                    continuation_options.predictor_maximum,
                )
            stage_evaluator = _ProjectionHomotopyEvaluator(
                evaluator,
                hamiltonian,
                space,
                strength,
            )
            final_stage = strength == 1.0
            stage_options = replace(
                solver_options,
                max_evaluator_calls=call_budget,
                max_iterations=min(solver_options.max_iterations, call_budget),
                residual_tolerance=(
                    solver_options.residual_tolerance
                    if final_stage
                    else max(stage_tolerance, solver_options.residual_tolerance)
                ),
                energy_tolerance=(
                    solver_options.energy_tolerance
                    if final_stage
                    else max(solver_options.energy_tolerance, 1.0e-8)
                ),
                newton_krylov=solver_options.newton_krylov if final_stage else False,
                newton_initial=False,
                spectral_fallback=False,
            )

            def stage_callback(
                amplitudes: QPAmplitudes,
                state: dict[str, object],
                *,
                _strength: float = strength,
            ) -> None:
                if iteration_callback is None:
                    return
                iteration_callback(
                    amplitudes,
                    {**state, "projection_continuation_strength": _strength},
                )

            stage_result = _solve(
                stage_evaluator.evaluate,
                hamiltonian,
                bogoliubov,
                seed,
                stage_options,
                requested_method,
                projected=True,
                excitation_space=space,
                iteration_callback=stage_callback,
                jvp_evaluator=stage_evaluator.jvp,
                residual_basis=residual_basis,
                coarse_basis=coarse_basis,
                gauge_options=gauge_options,
                step_basis_builder=step_basis_builder,
            )
            for entry in stage_result.history:
                combined_history.append(
                    {**entry, "projection_continuation_strength": strength}
                )
            stage_calls = int(stage_result.diagnostics.get("evaluator_calls", 0))
            stage_jvps = int(stage_result.diagnostics.get("analytic_jvp_calls", 0))
            stage_reports.append(
                {
                    "strength": strength,
                    "converged": stage_result.converged,
                    "residual_norm": stage_result.residual_norm,
                    "evaluator_calls": stage_calls,
                    "analytic_jvp_calls": stage_jvps,
                    "predictor_step_norm": predictor_step_norm,
                }
            )
            total_projection_time += stage_result.projection_time
            total_evaluator_calls += stage_calls
            total_jvp_calls += stage_jvps
            seed = stage_result.amplitudes
            continuation_points.append((strength, seed))
            result = stage_result
        assert result is not None
        result.history = combined_history
        result.iterations = len(combined_history)
        result.projection_time = total_projection_time
        result.diagnostics.update(
            {
                "projection_continuation": stage_reports,
                "projection_continuation_enabled": True,
                "evaluator_calls": total_evaluator_calls,
                "total_evaluator_calls": total_evaluator_calls,
                "analytic_jvp_calls": total_jvp_calls,
            }
        )
    if result.allowed_residual is not None:
        candidate_gauge = diagnostic_basis.project_gauge(result.allowed_residual)
        candidate_physical = diagnostic_basis.project_physical(result.allowed_residual)
        candidate_gauge_norm = (
            float(np.max(np.abs(candidate_gauge))) if candidate_gauge.size else 0.0
        )
        candidate_physical_norm = (
            float(np.max(np.abs(candidate_physical)))
            if candidate_physical.size
            else 0.0
        )
    else:
        candidate_gauge_norm = float("nan")
        candidate_physical_norm = float("nan")
    result.gauge_rank = diagnostic_basis.gauge_rank
    result.gauge_identity_defect = diagnostic_basis.identity_defect
    result.diagnostics.update(
        {
            "gauge_mode_treatment": (
                "equation-projected-fixed-metric-amplitude-section"
                if gauge_options.apply_to_solver
                and gauge_options.project_solver_steps
                else "equation-projected-amplitudes-unconstrained"
                if gauge_options.apply_to_solver
                else "diagnostic-only"
            ),
            "projected_metric_applied_to_amplitude_steps": bool(
                gauge_options.apply_to_solver
                and gauge_options.project_solver_steps
            ),
            "candidate_gauge_rank": diagnostic_basis.gauge_rank,
            "candidate_gauge_identity_defect": diagnostic_basis.identity_defect,
            "candidate_gauge_singular_values": (
                diagnostic_basis.singular_values.tolist()
            ),
            "candidate_gauge_residual_norm": candidate_gauge_norm,
            "candidate_physical_residual_norm": candidate_physical_norm,
            "gauge_projector_source": diagnostic_basis.source,
            "projected_metric": (
                diagnostic_basis.diagnostics()
                if hasattr(diagnostic_basis, "diagnostics")
                else None
            ),
            "projected_metric_setup_time": metric_setup_time,
        }
    )
    if not result.converged:
        zero = None
        if cas_reference is not None:
            zero = evaluator.evaluate(QPAmplitudes.zeros(hamiltonian.nspin))
            _apply_cas_energy_bookkeeping(result, cas_reference, zero.total_energy)
        result.grid_error = float("nan")
        gauge_components = (
            2
            if locked_projection_options.gauge_quadrature == "midpoint-richardson"
            else 1
        )
        total_calls = int(
            result.diagnostics.get(
                "total_evaluator_calls",
                result.diagnostics.get("evaluator_calls", 0),
            )
            + int(ode_selection["evaluations"])
            + (1 if zero is not None else 0)
        )
        result.diagnostics.update(
            {
                "validation_skipped": "amplitude_residual_not_converged",
                "validation_passed": False,
                "validation_energy": None,
                "validation_residual_norm": None,
                "validation_grid_size": None,
                "validation_ode_substeps": None,
                "validation_residual_error": float("nan"),
                "imaginary_energy_error": float("nan"),
                "validation_failure_reasons": [
                    "amplitude_residual_not_converged"
                ],
                "validation_projection_time": 0.0,
                "baseline_projection_time": (
                    0.0
                    if zero is None
                    else zero.diagnostics["projection_time"]
                ),
                "validation_baseline_projection_time": 0.0,
                "gauge_quadrature": locked_projection_options.gauge_quadrature,
                "ode_selection": ode_selection,
                "locked_ode_substeps": locked_projection_options.ode_substeps,
                "total_projected_evaluator_calls": total_calls,
                "gauge_component_evaluations_per_projected_call": gauge_components,
                "total_gauge_component_evaluator_calls": (
                    total_calls * gauge_components
                ),
            }
        )
        return result
    validation_options = replace(
        locked_projection_options,
        grid_size=2 * locked_projection_options.grid_size,
        ode_substeps=2 * locked_projection_options.ode_substeps,
        contour_radius=evaluator.radius,
        auto_select_ode=False,
    )
    validation_evaluator = _build_projected_evaluator(
        hamiltonian,
        bogoliubov,
        validation_options,
        space,
        execution_options,
    )
    validation = validation_evaluator.evaluate(result.amplitudes)
    if result.allowed_residual is None:
        raise RuntimeError("projected solver did not retain its final allowed residual")
    baseline_residual = np.asarray(result.allowed_residual)
    validation_residual = residual_to_vector(validation.r1, validation.r2, space)
    if residual_basis is not None and bool(
        getattr(residual_basis, "active", residual_basis.gauge_rank)
    ):
        validation_residual = residual_basis.project_physical(validation_residual)
        baseline_residual = residual_basis.project_physical(baseline_residual)
    validation_residual_error = float(
        np.max(np.abs(validation_residual - baseline_residual))
    ) if validation_residual.size else 0.0

    zero = None
    validation_zero = None
    if cas_reference is None:
        result.grid_error = float(
            abs(complex(validation.total_energy) - complex(result.total_energy))
        )
        reported_energy = complex(result.total_energy)
        validation_reported_energy = validation.total_energy
    elif (
        validation_residual_error
        < locked_projection_options.validation_residual_tolerance
        or not locked_projection_options.max_grid_refinements
    ):
        zero_amplitudes = QPAmplitudes.zeros(hamiltonian.nspin)
        zero = evaluator.evaluate(zero_amplitudes)
        validation_zero = validation_evaluator.evaluate(zero_amplitudes)
        dynamic = complex(result.total_energy) - complex(zero.total_energy)
        validation_dynamic = complex(validation.total_energy) - complex(
            validation_zero.total_energy
        )
        result.grid_error = float(abs(validation_dynamic - dynamic))
        reported_energy = complex(cas_reference.casscf_energy) + dynamic
        validation_reported_energy = np.real_if_close(
            cas_reference.casscf_energy + validation_dynamic
        )
    else:
        result.grid_error = float("nan")
        reported_energy = complex(result.total_energy)
        validation_reported_energy = None
    resolution_validation_passed = bool(
        np.isfinite(result.grid_error)
        and result.grid_error < locked_projection_options.validation_tolerance
        and validation_residual_error
        < locked_projection_options.validation_residual_tolerance
    )
    imaginary_energy_error = (
        float("inf")
        if validation_reported_energy is None
        else max(
            float(abs(reported_energy.imag)),
            float(abs(complex(validation_reported_energy).imag)),
        )
    )
    imaginary_validation_passed = bool(
        imaginary_energy_error < locked_projection_options.validation_tolerance
    )
    validation_passed = bool(
        resolution_validation_passed and imaginary_validation_passed
    )
    validation_failure_reasons = []
    if not resolution_validation_passed:
        validation_failure_reasons.append("resolution_difference")
    if not imaginary_validation_passed:
        validation_failure_reasons.append(
            "complex_energy_or_w2_contour_nonclosure"
        )
    if (
        not resolution_validation_passed
        and imaginary_validation_passed
        and locked_projection_options.max_grid_refinements
    ):
        prior_projected_calls = int(
            result.diagnostics.get(
                "total_evaluator_calls",
                result.diagnostics.get("evaluator_calls", 0),
            )
            + int(ode_selection["evaluations"])
            + 1
            + int(zero is not None)
            + int(validation_zero is not None)
        )
        refinement_report = {
            "grid_size": locked_projection_options.grid_size,
            "ode_substeps": locked_projection_options.ode_substeps,
            "residual_norm": result.residual_norm,
            "energy_error": result.grid_error,
            "residual_error": validation_residual_error,
            "projected_evaluator_calls": prior_projected_calls,
        }
        refined_options = replace(
            locked_projection_options,
            grid_size=2 * locked_projection_options.grid_size,
            contour_radius=evaluator.radius,
            auto_select_ode=False,
            max_grid_refinements=(
                locked_projection_options.max_grid_refinements - 1
            ),
        )
        refined = solve_projected_qpccsd(
            hamiltonian,
            reference,
            projection_options=refined_options,
            solver_options=solver_options,
            initial_amplitudes=result.amplitudes,
            requested_method=requested_method,
            excitation_space=space,
            iteration_callback=iteration_callback,
            execution_options=execution_options,
            gauge_options=gauge_options,
            continuation_options=None,
        )
        prior_history = [
            {
                **entry,
                "projection_grid_size": locked_projection_options.grid_size,
            }
            for entry in result.history
        ]
        refined_history = [
            {
                **entry,
                "projection_grid_size": refined_options.grid_size,
            }
            for entry in refined.history
        ]
        refined.history = prior_history + refined_history
        refined.iterations = len(refined.history)
        refined.projection_time += result.projection_time
        existing_reports = list(
            refined.diagnostics.get("grid_refinement_history", [])
        )
        refined.diagnostics["grid_refinement_history"] = [
            refinement_report,
            *existing_reports,
        ]
        refined.diagnostics["grid_refinement_count"] = 1 + int(
            refined.diagnostics.get("grid_refinement_count", 0)
        )
        refined.diagnostics["initial_projection_grid_size"] = (
            locked_projection_options.grid_size
        )
        refined.diagnostics["total_projected_evaluator_calls"] = int(
            refined.diagnostics.get("total_projected_evaluator_calls", 0)
            + prior_projected_calls
        )
        refined.diagnostics["total_gauge_component_evaluator_calls"] = int(
            refined.diagnostics.get("total_gauge_component_evaluator_calls", 0)
            + prior_projected_calls
            * (
                2
                if locked_projection_options.gauge_quadrature
                == "midpoint-richardson"
                else 1
            )
        )
        return refined
    result.converged = bool(
        result.converged and validation_passed
    )
    result.diagnostics.update(
        {
            "validation_energy": validation_reported_energy,
            "validation_residual_norm": validation.residual_norm,
            "validation_grid_size": validation_options.grid_size,
            "gauge_quadrature": locked_projection_options.gauge_quadrature,
            "baseline_gauge_component_grids": (
                [
                    locked_projection_options.grid_size,
                    2 * locked_projection_options.grid_size,
                ]
                if locked_projection_options.gauge_quadrature
                == "midpoint-richardson"
                else [locked_projection_options.grid_size]
            ),
            "validation_gauge_component_grids": (
                [validation_options.grid_size, 2 * validation_options.grid_size]
                if validation_options.gauge_quadrature == "midpoint-richardson"
                else [validation_options.grid_size]
            ),
            "validation_ode_substeps": validation_options.ode_substeps,
            "validation_residual_error": validation_residual_error,
            "validation_passed": validation_passed,
            "imaginary_energy_error": imaginary_energy_error,
            "validation_failure_reasons": validation_failure_reasons,
            "grid_refinement_count": 0,
            "validation_projection_time": validation.diagnostics["projection_time"],
            "baseline_projection_time": 0.0
            if zero is None
            else zero.diagnostics["projection_time"],
            "validation_baseline_projection_time": 0.0
            if validation_zero is None
            else validation_zero.diagnostics["projection_time"],
            "ode_selection": ode_selection,
            "locked_ode_substeps": locked_projection_options.ode_substeps,
            "total_projected_evaluator_calls": int(
                result.diagnostics.get(
                    "total_evaluator_calls",
                    result.diagnostics.get("evaluator_calls", 0),
                )
                + int(ode_selection["evaluations"])
                + 1
                + (2 if cas_reference is not None else 0)
            ),
            "gauge_component_evaluations_per_projected_call": (
                2
                if locked_projection_options.gauge_quadrature
                == "midpoint-richardson"
                else 1
            ),
        }
    )
    result.diagnostics["total_gauge_component_evaluator_calls"] = int(
        result.diagnostics["total_projected_evaluator_calls"]
        * result.diagnostics["gauge_component_evaluations_per_projected_call"]
    )
    if cas_reference is not None:
        assert zero is not None
        _apply_cas_energy_bookkeeping(result, cas_reference, zero.total_energy)
    return result


def solve_projected_lbccsd(
    hamiltonian: QPHamiltonian,
    reference: BogoliubovReference | CASQPReference,
    *,
    projection_options: ProjectionOptions | None = None,
    solver_options: SolverOptions | None = None,
    initial_amplitudes: QPAmplitudes | None = None,
    excitation_space: QPExcitationSpaceLike | None = None,
    iteration_callback: IterationCallback | None = None,
    execution_options: ExecutionOptions | None = None,
    gauge_options: GaugeModeOptions | None = None,
    continuation_options: ProjectionContinuationOptions | None = None,
) -> QPCCSDResult:
    return solve_projected_qpccsd(
        hamiltonian,
        reference,
        projection_options=projection_options,
        solver_options=solver_options,
        initial_amplitudes=initial_amplitudes,
        requested_method="lbccsd",
        excitation_space=excitation_space,
        iteration_callback=iteration_callback,
        execution_options=execution_options,
        gauge_options=gauge_options,
        continuation_options=continuation_options,
    )
