from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ProjectedMetricProjector:
    """Physical-range projector for a Hermitian projected excitation metric.

    Small metrics are diagonalized explicitly, which gives an exact spectral
    cutoff and a direct certification target.  Large metrics discover only
    the low-dimensional null space with matrix-free Hermitian eigensolves and
    never allocate a coordinate-squared array.
    """

    def __init__(
        self,
        matvec: Callable[[np.ndarray], np.ndarray],
        coordinate_count: int,
        *,
        explicit_matrix_builder: Callable[[], np.ndarray] | None = None,
        explicit_max_coordinates: int,
        relative_tolerance: float,
        absolute_tolerance: float,
        max_iterations: int,
        maximum_nullity: int,
    ) -> None:
        self._matvec = matvec
        self._explicit_matrix_builder = explicit_matrix_builder
        self.coordinate_count = int(coordinate_count)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_iterations = int(max_iterations)
        self.maximum_nullity = int(maximum_nullity)
        self.source = "projected-excitation-metric"
        self.gauge_vectors = np.empty(
            (self.coordinate_count, 0), dtype=np.complex128
        )
        self.singular_values = np.empty(0, dtype=float)
        self.metric_eigenvalues = np.empty(0, dtype=float)
        self.metric_rank_estimate: int | None = None
        self.hermiticity_defect = 0.0
        self.positivity_defect = 0.0
        self.identity_defect = 0.0
        self.last_iterative_info = 0
        self.last_iterative_residual = 0.0
        self.matrix_free_solver = "none"
        self._explicit_projector: np.ndarray | None = None
        self._spectral_cutoff = self.absolute_tolerance
        self._largest_eigenvalue = 0.0

        if self.coordinate_count == 0:
            self.metric_rank_estimate = 0
            return
        if self.coordinate_count <= int(explicit_max_coordinates):
            self._build_explicit()
        else:
            self._largest_eigenvalue = self._estimate_largest_eigenvalue()
            self._spectral_cutoff = max(
                self.absolute_tolerance,
                self.relative_tolerance * self._largest_eigenvalue,
            )
            if self._largest_eigenvalue <= self.absolute_tolerance:
                self.metric_rank_estimate = 0
                self.matrix_free_solver = "zero-metric"
            else:
                self._build_matrix_free_nullspace()

    @property
    def active(self) -> bool:
        return self.coordinate_count > 0

    @property
    def gauge_rank(self) -> int:
        if self.metric_rank_estimate is None:
            return 0
        return self.coordinate_count - self.metric_rank_estimate

    @property
    def spectral_cutoff(self) -> float:
        return float(self._spectral_cutoff)

    @property
    def largest_eigenvalue(self) -> float:
        return float(self._largest_eigenvalue)

    @property
    def explicit(self) -> bool:
        return self._explicit_projector is not None

    def metric_action(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.coordinate_count,):
            raise ValueError("metric vector has an incompatible coordinate dimension")
        result = np.asarray(self._matvec(values), dtype=np.complex128)
        if result.shape != values.shape:
            raise ValueError("projected metric action returned an incompatible shape")
        return result

    def _build_explicit(self) -> None:
        if self._explicit_matrix_builder is None:
            eye = np.eye(self.coordinate_count, dtype=np.complex128)
            raw = np.column_stack(
                [
                    self.metric_action(eye[:, column])
                    for column in range(self.coordinate_count)
                ]
            )
        else:
            raw = np.asarray(self._explicit_matrix_builder(), dtype=np.complex128)
            if raw.shape != (self.coordinate_count, self.coordinate_count):
                raise ValueError("explicit projected metric has an incompatible shape")
        scale = max(1.0, float(np.max(np.abs(raw))))
        self.hermiticity_defect = float(np.max(np.abs(raw - raw.conj().T))) / scale
        metric = 0.5 * (raw + raw.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        self.positivity_defect = max(0.0, -float(eigenvalues[0]))
        largest = max(0.0, float(eigenvalues[-1]))
        cutoff = max(self.absolute_tolerance, self.relative_tolerance * largest)
        physical = eigenvalues > cutoff
        physical_vectors = eigenvectors[:, physical]
        gauge_vectors = np.asarray(eigenvectors[:, ~physical], dtype=np.complex128).copy()
        for column in range(gauge_vectors.shape[1]):
            pivot = int(np.argmax(np.abs(gauge_vectors[:, column])))
            phase = gauge_vectors[pivot, column]
            if abs(phase):
                gauge_vectors[:, column] *= np.conj(phase) / abs(phase)
        projector = physical_vectors @ physical_vectors.conj().T
        self._explicit_projector = projector
        self.gauge_vectors = gauge_vectors
        self.metric_eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.singular_values = np.asarray(eigenvalues[~physical], dtype=float)
        self.metric_rank_estimate = int(np.count_nonzero(physical))
        self._largest_eigenvalue = largest
        self._spectral_cutoff = cutoff
        self.identity_defect = float(np.max(np.abs(projector @ projector - projector)))

    def _estimate_largest_eigenvalue(self) -> float:
        # A fixed pseudo-random start avoids exact orthogonality to structured
        # metric ranges while retaining reproducible setup and checkpoints.
        random = np.random.default_rng(1729 + self.coordinate_count)
        vector = random.normal(size=self.coordinate_count) + 1.0j * random.normal(
            size=self.coordinate_count
        )
        vector /= np.linalg.norm(vector)
        estimate = 0.0
        for _ in range(min(self.max_iterations, 20)):
            product = self.metric_action(vector)
            norm = float(np.linalg.norm(product))
            if norm <= self.absolute_tolerance:
                return 0.0
            vector = product / norm
            updated = float(np.real(np.vdot(vector, self.metric_action(vector))))
            if abs(updated - estimate) <= self.relative_tolerance * max(1.0, abs(updated)):
                estimate = updated
                break
            estimate = updated
        return max(0.0, estimate)

    def _build_matrix_free_nullspace(self) -> None:
        if self.coordinate_count <= 1:
            value = self.metric_action(np.ones(self.coordinate_count, dtype=np.complex128))
            null = float(np.linalg.norm(value)) <= self._spectral_cutoff
            self.gauge_vectors = (
                np.ones((self.coordinate_count, 1), dtype=np.complex128)
                if null
                else np.empty((self.coordinate_count, 0), dtype=np.complex128)
            )
            self.metric_rank_estimate = 0 if null else self.coordinate_count
            self.matrix_free_solver = "direct-one-dimensional"
            return

        from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence, eigsh

        operator = LinearOperator(
            (self.coordinate_count, self.coordinate_count),
            matvec=self.metric_action,
            rmatvec=self.metric_action,
            dtype=np.complex128,
        )
        reflection_shift = max(
            self.absolute_tolerance,
            1.05 * self._largest_eigenvalue,
        )

        def reflected_action(vector: np.ndarray) -> np.ndarray:
            return reflection_shift * vector - operator.matvec(vector)

        reflected = LinearOperator(
            (self.coordinate_count, self.coordinate_count),
            matvec=reflected_action,
            rmatvec=reflected_action,
            dtype=np.complex128,
        )
        probe_limit = min(
            max(1, self.maximum_nullity + 1),
            self.coordinate_count - 1,
        )
        trial_count = min(4, probe_limit)
        indices = np.arange(self.coordinate_count, dtype=float)
        initial = np.exp(2.0j * np.pi * (indices + 0.5) / self.coordinate_count)
        initial /= np.linalg.norm(initial)
        while True:
            try:
                reflected_eigenvalues, eigenvectors = eigsh(
                    reflected,
                    k=trial_count,
                    which="LA",
                    v0=initial,
                    tol=self.relative_tolerance,
                    maxiter=max(20, self.max_iterations * self.coordinate_count),
                )
            except ArpackNoConvergence as error:
                if error.eigenvalues is None or error.eigenvectors is None:
                    raise RuntimeError(
                        "projected metric null-space eigensolver did not converge"
                    ) from error
                reflected_eigenvalues = error.eigenvalues
                eigenvectors = error.eigenvectors
            eigenvalues = reflection_shift - np.real(
                np.asarray(reflected_eigenvalues)
            )
            order = np.argsort(np.real(eigenvalues))
            eigenvalues = np.asarray(eigenvalues)[order]
            eigenvectors = np.asarray(eigenvectors)[:, order]
            null = eigenvalues <= self._spectral_cutoff
            if np.count_nonzero(null) < trial_count:
                break
            if trial_count >= probe_limit:
                if probe_limit < self.coordinate_count - 1:
                    raise RuntimeError(
                        "projected metric nullity exceeds metric_maximum_nullity; "
                        "increase the limit before solving"
                    )
                break
            trial_count = min(probe_limit, 2 * trial_count)
        gauge_vectors = np.asarray(eigenvectors[:, null], dtype=np.complex128).copy()
        if gauge_vectors.size:
            gauge_vectors, _ = np.linalg.qr(gauge_vectors, mode="reduced")
        for column in range(gauge_vectors.shape[1]):
            pivot = int(np.argmax(np.abs(gauge_vectors[:, column])))
            phase = gauge_vectors[pivot, column]
            if abs(phase):
                gauge_vectors[:, column] *= np.conj(phase) / abs(phase)
        self.gauge_vectors = gauge_vectors
        self.metric_eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.singular_values = np.asarray(eigenvalues[null], dtype=float)
        self.metric_rank_estimate = self.coordinate_count - gauge_vectors.shape[1]
        self.positivity_defect = max(0.0, -float(eigenvalues[0]))
        left = initial
        right = np.roll(initial, 1) * np.exp(0.37j)
        left_action = self.metric_action(left)
        right_action = self.metric_action(right)
        lhs = np.vdot(left, right_action)
        rhs = np.vdot(left_action, right)
        self.hermiticity_defect = float(abs(lhs - rhs)) / max(
            1.0,
            float(abs(lhs)),
            float(abs(rhs)),
        )
        self.identity_defect = float(
            np.max(
                np.abs(
                    gauge_vectors.conj().T @ gauge_vectors
                    - np.eye(gauge_vectors.shape[1])
                )
            )
        ) if gauge_vectors.shape[1] else 0.0
        self.matrix_free_solver = "arpack-reflected-largest-eigenpairs"

    def project_physical(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.coordinate_count,):
            raise ValueError("residual vector has an incompatible coordinate dimension")
        if not self.coordinate_count:
            return values.copy()
        if self._explicit_projector is not None:
            return self._explicit_projector @ values
        if self._largest_eigenvalue <= self.absolute_tolerance:
            return np.zeros_like(values)
        if self.metric_rank_estimate is not None:
            if not self.gauge_vectors.shape[1]:
                return values.copy()
            return values - self.gauge_vectors @ (
                self.gauge_vectors.conj().T @ values
            )
        from scipy.sparse.linalg import LinearOperator, cg

        cutoff = self._spectral_cutoff

        def shifted_action(candidate: np.ndarray) -> np.ndarray:
            return self.metric_action(candidate) + cutoff * candidate

        operator = LinearOperator(
            (self.coordinate_count, self.coordinate_count),
            matvec=shifted_action,
            rmatvec=shifted_action,
            dtype=np.complex128,
        )
        solution, info = cg(
            operator,
            values,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            maxiter=self.max_iterations,
        )
        self.last_iterative_info = int(info)
        filtered = self.metric_action(solution)
        self.last_iterative_residual = float(
            np.linalg.norm(shifted_action(solution) - values)
        )
        if info < 0 or not np.all(np.isfinite(filtered)):
            raise RuntimeError("matrix-free projected metric solve failed")
        return filtered

    def project_gauge(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        return values - self.project_physical(values)

    def diagnostics(self) -> dict[str, object]:
        return {
            "source": self.source,
            "explicit": self.explicit,
            "coordinate_count": self.coordinate_count,
            "metric_rank_estimate": self.metric_rank_estimate,
            "gauge_rank": self.gauge_rank,
            "spectral_cutoff": self.spectral_cutoff,
            "largest_eigenvalue": self.largest_eigenvalue,
            "hermiticity_defect": self.hermiticity_defect,
            "positivity_defect": self.positivity_defect,
            "projector_identity_defect": self.identity_defect,
            "last_iterative_info": self.last_iterative_info,
            "last_iterative_residual": self.last_iterative_residual,
            "matrix_free_solver": self.matrix_free_solver,
        }


__all__ = ["ProjectedMetricProjector"]
