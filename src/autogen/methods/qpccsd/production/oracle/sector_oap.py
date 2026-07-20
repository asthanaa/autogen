from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .fock import build_sparse_fock_hamiltonian
from ..excitation_space import QPExcitationSpace, QPExcitationSpaceLike
from ..models import BogoliubovReference, KernelEvaluation, QPAmplitudes, QPCCSDResult
from ..production import amplitudes_from_vector, amplitudes_to_vector, residual_to_vector


@dataclass(frozen=True)
class _ExcitationAction:
    sources: np.ndarray
    destinations: np.ndarray
    signs: np.ndarray

    @property
    def storage_bytes(self) -> int:
        return int(self.sources.nbytes + self.destinations.nbytes + self.signs.nbytes)


@dataclass(frozen=True)
class _CoordinateAction:
    terms: tuple[tuple[complex, _ExcitationAction], ...]

    @property
    def storage_bytes(self) -> int:
        return int(sum(action.storage_bytes + 16 for _, action in self.terms))


def _local_annihilation(index: int) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.complex128)
    for ket in range(4):
        if not (ket >> index) & 1:
            continue
        parity = (ket & ((1 << index) - 1)).bit_count()
        result[ket & ~(1 << index), ket] = -1.0 if parity % 2 else 1.0
    return result


def _local_qp_to_physical(
    reference: BogoliubovReference,
    pair_index: int,
) -> tuple[np.ndarray, float]:
    first = 2 * pair_index
    indices = (first, first + 1)
    annihilators = (_local_annihilation(0), _local_annihilation(1))
    creators = tuple(operator.conj().T for operator in annihilators)
    beta: list[np.ndarray] = []
    for local_q, q in enumerate(indices):
        operator = np.zeros((4, 4), dtype=np.complex128)
        for local_p, p in enumerate(indices):
            operator += reference.U[p, q].conjugate() * annihilators[local_p]
            operator += reference.V[p, q].conjugate() * creators[local_p]
        beta.append(operator)
    beta_dag = tuple(operator.conj().T for operator in beta)

    vacuum = np.zeros(4, dtype=np.complex128)
    vacuum[0] = reference.u[first]
    vacuum[3] = reference.signs[first] * reference.v[first]
    vacuum /= np.linalg.norm(vacuum)
    columns = (
        vacuum,
        beta_dag[0] @ vacuum,
        beta_dag[1] @ vacuum,
        beta_dag[0] @ beta_dag[1] @ vacuum,
    )
    transformation = np.column_stack(columns)
    unitary_error = float(
        np.max(np.abs(transformation.conj().T @ transformation - np.eye(4)))
    )
    annihilation_error = max(float(np.linalg.norm(operator @ vacuum)) for operator in beta)
    if unitary_error > 1.0e-10 or annihilation_error > 1.0e-10:
        raise ValueError(
            "failed to construct a local Bogoliubov basis transformation: "
            f"unitary_error={unitary_error:.3e}, "
            f"annihilation_error={annihilation_error:.3e}"
        )
    return transformation, annihilation_error


def _apply_local_basis_transforms(
    state: np.ndarray,
    transformations: tuple[np.ndarray, ...],
    *,
    adjoint: bool,
) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape(
        (4,) * len(transformations), order="F"
    )
    for axis, transformation in enumerate(transformations):
        operator = transformation.conj().T if adjoint else transformation
        tensor = np.tensordot(operator, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
    return np.asarray(tensor).reshape(-1, order="F")


def _excitation_action(
    key: tuple[int, ...],
    basis: np.ndarray,
    parity: np.ndarray,
) -> _ExcitationAction:
    mask = sum(1 << index for index in key)
    sources = basis[(basis & mask) == 0]
    signs = np.ones(sources.size, dtype=np.int8)
    for index in key:
        odd = parity[sources & ((1 << index) - 1)]
        signs *= np.where(odd, -1, 1).astype(np.int8)
    destinations = sources | mask
    index_dtype = np.uint32 if basis.size <= np.iinfo(np.uint32).max else np.uint64
    return _ExcitationAction(
        sources=np.asarray(sources, dtype=index_dtype),
        destinations=np.asarray(destinations, dtype=index_dtype),
        signs=signs,
    )


class ExactSectorOAPEvaluator:
    """Exact small-space OAP oracle using QP bitstrings and sector slicing.

    Production code never imports this module.  The oracle stores ``2**M``
    vectors, but avoids the former collection of one sparse Fock-space matrix
    per pair and quadruple excitation.
    """

    def __init__(
        self,
        h1: np.ndarray,
        g2: np.ndarray,
        constant: float,
        reference: BogoliubovReference,
        target_number: int,
        *,
        max_spin_orbitals: int = 16,
        excitation_space: QPExcitationSpaceLike | None = None,
    ) -> None:
        started = time.perf_counter()
        self.reference = reference
        self.nspin = reference.nspin
        if self.nspin > max_spin_orbitals:
            raise ValueError(
                f"exact-sector OAP is guarded at {max_spin_orbitals} spin orbitals"
            )
        if np.asarray(h1).shape != (self.nspin, self.nspin):
            raise ValueError("one-body integral shape does not match the reference")
        if np.asarray(g2).shape != (self.nspin,) * 4:
            raise ValueError("two-body integral shape does not match the reference")
        self.target_number = int(target_number)
        self.dimension = 1 << self.nspin
        self.excitation_space = (
            QPExcitationSpace.full(self.nspin)
            if excitation_space is None
            else excitation_space
        )
        if self.excitation_space.nspin != self.nspin:
            raise ValueError("excitation space and exact oracle dimensions differ")
        self.pairs = self.excitation_space.pair_indices
        self.quads = self.excitation_space.quadruple_indices
        self.hamiltonian = build_sparse_fock_hamiltonian(
            h1,
            g2,
            constant,
            particle_parity=target_number % 2,
        )

        local_data = tuple(
            _local_qp_to_physical(reference, pair_index)
            for pair_index in range(self.nspin // 2)
        )
        self._local_transformations = tuple(value[0] for value in local_data)
        self.vacuum_annihilation_error = max(value[1] for value in local_data)
        qp_vacuum = np.zeros(self.dimension, dtype=np.complex128)
        qp_vacuum[0] = 1.0
        self.vacuum = self._qp_to_physical(qp_vacuum)

        particle_numbers = np.fromiter(
            (determinant.bit_count() for determinant in range(self.dimension)),
            dtype=np.int16,
            count=self.dimension,
        )
        self._number_mask = particle_numbers == self.target_number

        basis = np.arange(self.dimension, dtype=np.uint64)
        parity = np.fromiter(
            (determinant.bit_count() % 2 for determinant in range(self.dimension)),
            dtype=np.int8,
            count=self.dimension,
        )
        pair_expansions = getattr(
            self.excitation_space,
            "pair_expansions",
            tuple(((key, 1.0),) for key in self.pairs),
        )
        quadruple_expansions = getattr(
            self.excitation_space,
            "quadruple_expansions",
            tuple(((key, 1.0),) for key in self.quads),
        )
        self._actions = tuple(
            _CoordinateAction(
                tuple(
                    (complex(coefficient), _excitation_action(key, basis, parity))
                    for key, coefficient in expansion
                )
            )
            for expansion in pair_expansions + quadruple_expansions
        )
        self.action_storage_bytes = sum(action.storage_bytes for action in self._actions)
        self.initialization_time = time.perf_counter() - started

    def _qp_to_physical(self, state: np.ndarray) -> np.ndarray:
        return _apply_local_basis_transforms(
            state,
            self._local_transformations,
            adjoint=False,
        )

    def _physical_to_qp(self, state: np.ndarray) -> np.ndarray:
        return _apply_local_basis_transforms(
            state,
            self._local_transformations,
            adjoint=True,
        )

    def _apply_cluster(self, state: np.ndarray, vector: np.ndarray) -> np.ndarray:
        result = np.zeros(self.dimension, dtype=np.complex128)
        for amplitude, coordinate in zip(vector, self._actions):
            if amplitude == 0.0:
                continue
            for coefficient, action in coordinate.terms:
                result[action.destinations] += (
                    amplitude * coefficient * action.signs * state[action.sources]
                )
        return result

    def _probe_coordinates(self, state_qp: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                sum(
                    np.conj(coefficient)
                    * action.signs[0]
                    * state_qp[action.destinations[0]]
                    for coefficient, action in coordinate.terms
                )
                for coordinate in self._actions
            ],
            dtype=np.complex128,
        )

    def projected_metric_action(self, vector: np.ndarray) -> np.ndarray:
        """Apply the exact small-space matrix ``<Phi|B P_N B^dagger|Phi>``."""

        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.excitation_space.coordinate_count,):
            raise ValueError("projected metric vector has an incompatible shape")
        vacuum_qp = np.zeros(self.dimension, dtype=np.complex128)
        vacuum_qp[0] = 1.0
        excited_qp = self._apply_cluster(vacuum_qp, values)
        excited = self._qp_to_physical(excited_qp)
        projected = np.where(self._number_mask, excited, 0.0)
        return self._probe_coordinates(self._physical_to_qp(projected))

    def _cluster_state_qp(self, amplitudes: QPAmplitudes) -> np.ndarray:
        vector = amplitudes_to_vector(amplitudes, self.excitation_space)
        result = np.zeros(self.dimension, dtype=np.complex128)
        result[0] = 1.0
        term = result.copy()
        for order in range(1, self.nspin // 2 + 1):
            term = self._apply_cluster(term, vector) / float(order)
            result += term
            if np.linalg.norm(term) < 1.0e-15 * max(1.0, np.linalg.norm(result)):
                break
        return result

    def cluster_state(self, amplitudes: QPAmplitudes) -> np.ndarray:
        return self._qp_to_physical(self._cluster_state_qp(amplitudes))

    def evaluate(self, amplitudes: QPAmplitudes) -> KernelEvaluation:
        started = time.perf_counter()
        amplitudes = self.excitation_space.enforce(amplitudes)
        state_qp = self._cluster_state_qp(amplitudes)
        state = self._qp_to_physical(state_qp)
        projected_state = np.where(self._number_mask, state, 0.0)
        denominator = complex(np.vdot(self.vacuum, projected_state))
        if abs(denominator) < 1.0e-12:
            raise ValueError("exact-sector OAP projected norm is numerically zero")
        hstate = self.hamiltonian @ state
        projected_hstate = np.where(self._number_mask, hstate, 0.0)
        energy = complex(np.vdot(self.vacuum, projected_hstate) / denominator)
        sigma_qp = self._physical_to_qp(
            projected_hstate - energy * projected_state
        )
        residual = self._probe_coordinates(sigma_qp) / denominator
        tensors = amplitudes_from_vector(
            residual,
            self.nspin,
            self.excitation_space,
        )
        elapsed = time.perf_counter() - started
        return KernelEvaluation(
            total_energy=np.real_if_close(energy),
            correlation_energy=np.real_if_close(energy),
            r1=np.real_if_close(tensors.t1),
            r2=np.real_if_close(tensors.t2),
            residual_norm=float(np.max(np.abs(residual))) if residual.size else 0.0,
            projected_norm=denominator,
            elapsed=elapsed,
            diagnostics={
                "oracle": "exact-sector auxiliary-kernel OAP",
                "projection_equation_schema": "pn-oap-bpn-v2",
                "projection_solver_schema": "exact-sector-oap-least-squares-v1",
                "projector_ordering": "B_mu P_N (H-E) exp(T)",
                "fock_dimension": self.dimension,
                "fock_backend": "QP bitstrings plus sparse physical Hamiltonian",
                "hamiltonian_nnz": int(self.hamiltonian.nnz),
                "vacuum_annihilation_error": self.vacuum_annihilation_error,
                "allowed_pair_count": self.excitation_space.pair_count,
                "allowed_quadruple_count": self.excitation_space.quadruple_count,
                "forbidden_amplitude_norm": self.excitation_space.forbidden_amplitude_norm(
                    amplitudes
                ),
                "action_storage_bytes": self.action_storage_bytes,
                "oracle_initialization_time": self.initialization_time,
                "oracle_evaluation_time": elapsed,
            },
        )

    __call__ = evaluate


def solve_exact_sector_oap(
    evaluator: ExactSectorOAPEvaluator,
    *,
    initial_amplitudes: QPAmplitudes | None = None,
    tolerance: float = 1.0e-9,
    max_nfev: int = 300,
) -> QPCCSDResult:
    """Solve the exact-sector oracle equations with nonlinear least squares."""

    from scipy.optimize import least_squares

    nspin = evaluator.nspin
    if initial_amplitudes is None:
        initial = np.zeros(evaluator.excitation_space.coordinate_count)
    else:
        initial = np.asarray(
            np.real(
                amplitudes_to_vector(
                    initial_amplitudes,
                    evaluator.excitation_space,
                )
            ),
            dtype=float,
        )

    initial_tensors = amplitudes_from_vector(
        initial,
        nspin,
        evaluator.excitation_space,
    )
    initial_evaluation = evaluator(initial_tensors)
    if initial_evaluation.residual_norm < tolerance:
        return QPCCSDResult(
            converged=True,
            total_energy=initial_evaluation.total_energy,
            correlation_energy=initial_evaluation.correlation_energy,
            amplitudes=initial_tensors,
            residual_norm=initial_evaluation.residual_norm,
            iterations=0,
            requested_method="exact-sector-oap-oracle",
            canonical_method="qpccsd",
            projected=True,
            raw_projected_energy=initial_evaluation.total_energy,
            projection_equation_schema=initial_evaluation.diagnostics[
                "projection_equation_schema"
            ],
            projection_solver_schema=initial_evaluation.diagnostics[
                "projection_solver_schema"
            ],
            projector_ordering=initial_evaluation.diagnostics["projector_ordering"],
            allowed_pair_count=evaluator.excitation_space.pair_count,
            allowed_quadruple_count=evaluator.excitation_space.quadruple_count,
            forbidden_amplitude_norm=0.0,
            diagnostics={
                **initial_evaluation.diagnostics,
                "least_squares_status": 0,
                "least_squares_message": "initial amplitudes satisfy the oracle equations",
            },
        )

    def objective(vector: np.ndarray) -> np.ndarray:
        evaluation = evaluator(
            amplitudes_from_vector(vector, nspin, evaluator.excitation_space)
        )
        residual = residual_to_vector(
            evaluation.r1,
            evaluation.r2,
            evaluator.excitation_space,
        )
        return np.concatenate((np.real(residual), np.imag(residual)))

    solver_tolerance = max(min(0.1 * tolerance, 1.0e-10), 1.0e-13)
    solution = least_squares(
        objective,
        initial,
        xtol=solver_tolerance,
        ftol=solver_tolerance,
        gtol=solver_tolerance,
        max_nfev=int(max_nfev),
        x_scale="jac",
    )
    amplitudes = amplitudes_from_vector(
        solution.x,
        nspin,
        evaluator.excitation_space,
    )
    final = evaluator(amplitudes)
    return QPCCSDResult(
        converged=bool(solution.success and final.residual_norm < tolerance),
        total_energy=final.total_energy,
        correlation_energy=final.correlation_energy,
        amplitudes=amplitudes,
        residual_norm=final.residual_norm,
        iterations=int(solution.nfev),
        requested_method="exact-sector-oap-oracle",
        canonical_method="qpccsd",
        projected=True,
        raw_projected_energy=final.total_energy,
        projection_equation_schema=final.diagnostics[
            "projection_equation_schema"
        ],
        projection_solver_schema=final.diagnostics["projection_solver_schema"],
        projector_ordering=final.diagnostics["projector_ordering"],
        allowed_pair_count=evaluator.excitation_space.pair_count,
        allowed_quadruple_count=evaluator.excitation_space.quadruple_count,
        forbidden_amplitude_norm=evaluator.excitation_space.forbidden_amplitude_norm(
            amplitudes
        ),
        diagnostics={
            **final.diagnostics,
            "least_squares_cost": float(solution.cost),
            "least_squares_optimality": float(solution.optimality),
            "least_squares_status": int(solution.status),
            "least_squares_message": str(solution.message),
        },
    )
