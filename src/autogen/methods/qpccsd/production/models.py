from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np


BLOCK_NAMES = ("h11", "h20", "h02", "h22", "h31", "h13", "h40", "h04")


@dataclass
class BogoliubovReference:
    """Polynomial-runtime description of a paired Bogoliubov vacuum."""

    U: np.ndarray
    V: np.ndarray
    u: np.ndarray
    v: np.ndarray
    partner: np.ndarray
    signs: np.ndarray
    target_number: int
    quasiparticle_energies: np.ndarray
    chemical_potential: float = 0.0
    hfb_energy: float = 0.0
    converged: bool = True
    iterations: int = 0
    residual_norm: float = 0.0
    pairing_collapsed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U, dtype=np.complex128)
        self.V = np.asarray(self.V, dtype=np.complex128)
        self.u = np.asarray(self.u, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        self.partner = np.asarray(self.partner, dtype=np.int64)
        self.signs = np.asarray(self.signs, dtype=float)
        self.quasiparticle_energies = np.asarray(self.quasiparticle_energies, dtype=float)
        nspin = self.U.shape[0]
        if self.U.shape != (nspin, nspin) or self.V.shape != (nspin, nspin):
            raise ValueError("U and V must be square matrices with the same shape")
        for name, value in (
            ("u", self.u),
            ("v", self.v),
            ("partner", self.partner),
            ("signs", self.signs),
            ("quasiparticle_energies", self.quasiparticle_energies),
        ):
            if value.shape != (nspin,):
                raise ValueError(f"{name} must have shape {(nspin,)}")
        if nspin % 2:
            raise ValueError("paired HFB requires an even number of spin orbitals")
        if np.any(self.partner[self.partner] != np.arange(nspin)):
            raise ValueError("partner must be an involution")

    @property
    def nspin(self) -> int:
        return int(self.U.shape[0])

    @property
    def occupations(self) -> np.ndarray:
        return np.real(np.diag(self.V @ self.V.conj().T))

    @property
    def pairing_tensor(self) -> np.ndarray:
        return self.V @ self.U.T

    def canonical_errors(self) -> tuple[float, float]:
        eye = np.eye(self.nspin)
        normal = self.U.conj().T @ self.U + self.V.conj().T @ self.V - eye
        anomalous = self.U.T @ self.V + self.V.T @ self.U
        return float(np.max(np.abs(normal))), float(np.max(np.abs(anomalous)))


@dataclass
class CASQPReference:
    """CAS-RDM input and projected-AGP quasiparticle reference.

    The production transformation is built in the active natural-orbital
    basis.  Natural occupations determine its magnitudes, while a signed
    projected-AGP least-squares fit to the pair-transfer block of the active
    two-body RDM determines the relative geminal signs and ratios.  This does
    not claim to reproduce the complete CAS two-body RDM.
    """

    bogoliubov: BogoliubovReference
    casscf_energy: float
    mo_coeff: np.ndarray
    spatial_occupations: np.ndarray
    active_rdm1: np.ndarray
    active_rdm2: np.ndarray
    active_spatial_indices: tuple[int, ...]
    inactive_spatial_indices: tuple[int, ...]
    external_spatial_indices: tuple[int, ...]
    frozen_spatial_indices: tuple[int, ...] = ()
    physical_target_number: int | None = None
    rdm_cumulant_norm: float = 0.0
    orbital_labels: tuple[str, ...] = ()
    reference_mode: str = "projected_agp_2rdm"
    signed_geminals: np.ndarray | None = None
    reconstruction_metrics: dict[str, float | bool | str] = field(default_factory=dict)
    source_rdm_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.casscf_energy = float(self.casscf_energy)
        self.mo_coeff = np.asarray(self.mo_coeff, dtype=float)
        self.spatial_occupations = np.asarray(self.spatial_occupations, dtype=float)
        self.active_rdm1 = np.asarray(self.active_rdm1, dtype=float)
        self.active_rdm2 = np.asarray(self.active_rdm2, dtype=float)
        nspatial = self.bogoliubov.nspin // 2
        if self.mo_coeff.ndim != 2 or self.mo_coeff.shape[1] != nspatial:
            raise ValueError("mo_coeff must have one column per correlated spatial orbital")
        if self.spatial_occupations.shape != (nspatial,):
            raise ValueError("spatial_occupations has an incompatible shape")
        if np.any(self.spatial_occupations < -1.0e-10) or np.any(
            self.spatial_occupations > 2.0 + 1.0e-10
        ):
            raise ValueError("spatial occupations must lie in [0, 2]")
        active_count = len(self.active_spatial_indices)
        if self.active_rdm1.shape != (active_count, active_count):
            raise ValueError("active_rdm1 has an incompatible shape")
        if self.active_rdm2.shape != (active_count,) * 4:
            raise ValueError("active_rdm2 has an incompatible shape")
        if self.reference_mode not in {
            "sokolov_linear_1rdm",
            "projected_agp_2rdm",
        }:
            raise ValueError(f"unsupported CAS-QP reference mode {self.reference_mode!r}")
        if self.signed_geminals is not None:
            self.signed_geminals = np.asarray(self.signed_geminals, dtype=float)
            if self.signed_geminals.shape != (active_count,):
                raise ValueError("signed_geminals must have one entry per active orbital")
        partitions = (
            self.inactive_spatial_indices,
            self.active_spatial_indices,
            self.external_spatial_indices,
        )
        flattened = tuple(index for partition in partitions for index in partition)
        if sorted(flattened) != list(range(nspatial)) or len(set(flattened)) != nspatial:
            raise ValueError("inactive, active, and external partitions must cover the space")
        if self.orbital_labels and len(self.orbital_labels) != nspatial:
            raise ValueError("orbital_labels must have one entry per correlated orbital")
        if self.physical_target_number is None:
            self.physical_target_number = int(
                self.bogoliubov.target_number + 2 * len(self.frozen_spatial_indices)
            )

    @property
    def nspin(self) -> int:
        return self.bogoliubov.nspin

    @property
    def nspatial(self) -> int:
        return self.nspin // 2

    @property
    def target_number(self) -> int:
        return self.bogoliubov.target_number

    @property
    def active_spin_indices(self) -> tuple[int, ...]:
        return tuple(
            spin_index
            for spatial_index in self.active_spatial_indices
            for spin_index in (2 * spatial_index, 2 * spatial_index + 1)
        )


@dataclass
class ActiveSpaceState:
    """Exact fixed-active-space state in a declared orbital basis."""

    ci_coefficients: np.ndarray
    electron_count: tuple[int, int]
    rdm1: np.ndarray
    rdm2: np.ndarray
    orbital_basis: str = "cas_natural_orbitals"
    rdm3: np.ndarray | None = None
    rdm4: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ci_coefficients = np.asarray(self.ci_coefficients)
        self.electron_count = tuple(int(value) for value in self.electron_count)
        if len(self.electron_count) != 2 or min(self.electron_count) < 0:
            raise ValueError("electron_count must contain non-negative alpha/beta counts")
        self.rdm1 = np.asarray(self.rdm1)
        self.rdm2 = np.asarray(self.rdm2)
        nactive = int(self.rdm1.shape[0])
        if self.rdm1.shape != (nactive, nactive):
            raise ValueError("active-state rdm1 must be square")
        if self.rdm2.shape != (nactive,) * 4:
            raise ValueError("active-state rdm2 has an incompatible shape")
        expected = (
            comb(nactive, self.electron_count[0]),
            comb(nactive, self.electron_count[1]),
        )
        if self.ci_coefficients.shape != expected:
            raise ValueError(
                f"active CI coefficients must have shape {expected}, "
                f"not {self.ci_coefficients.shape}"
            )
        for name, rank in (("rdm3", 6), ("rdm4", 8)):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value)
            if value.shape != (nactive,) * rank:
                raise ValueError(f"active-state {name} has an incompatible shape")
            setattr(self, name, value)
        norm = float(np.vdot(self.ci_coefficients, self.ci_coefficients).real)
        if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-8:
            raise ValueError(f"active CI vector is not normalized: norm={norm:.12g}")

    @property
    def nactive(self) -> int:
        return int(self.rdm1.shape[0])

    @property
    def electron_number(self) -> int:
        return int(sum(self.electron_count))


@dataclass
class CASContractedReference:
    """Number-conserving Hamiltonian plus an exact active-space reference."""

    active_state: ActiveSpaceState
    casscf_energy: float
    h1_spatial: np.ndarray
    eri_spatial: np.ndarray
    constant_energy: float
    inactive_spatial_indices: tuple[int, ...]
    active_spatial_indices: tuple[int, ...]
    external_spatial_indices: tuple[int, ...]
    target_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.casscf_energy = float(self.casscf_energy)
        self.h1_spatial = np.asarray(self.h1_spatial)
        self.eri_spatial = np.asarray(self.eri_spatial)
        self.constant_energy = float(self.constant_energy)
        self.inactive_spatial_indices = tuple(
            int(value) for value in self.inactive_spatial_indices
        )
        self.active_spatial_indices = tuple(
            int(value) for value in self.active_spatial_indices
        )
        self.external_spatial_indices = tuple(
            int(value) for value in self.external_spatial_indices
        )
        nspatial = int(self.h1_spatial.shape[0])
        if self.h1_spatial.shape != (nspatial, nspatial):
            raise ValueError("contracted-CAS h1_spatial must be square")
        if self.eri_spatial.shape != (nspatial,) * 4:
            raise ValueError("contracted-CAS eri_spatial has an incompatible shape")
        partitions = (
            self.inactive_spatial_indices,
            self.active_spatial_indices,
            self.external_spatial_indices,
        )
        flattened = tuple(index for values in partitions for index in values)
        if sorted(flattened) != list(range(nspatial)) or len(set(flattened)) != nspatial:
            raise ValueError("contracted-CAS orbital partitions must cover the space")
        if len(self.active_spatial_indices) != self.active_state.nactive:
            raise ValueError("active CI and active orbital dimensions differ")
        expected_number = 2 * len(self.inactive_spatial_indices) + self.active_state.electron_number
        if int(self.target_number) != expected_number:
            raise ValueError(
                "target number is inconsistent with inactive occupancy and active CI"
            )


@dataclass
class CASContractedCCSDResult:
    """Result type kept distinct from projected quasiparticle CCSD."""

    converged: bool
    total_energy: float
    dynamic_correlation_energy: float
    residual_norm: float
    iterations: int
    canonical_method: str = "cas-contracted-ccsd"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QPHamiltonian:
    """Normal-ordered quasiparticle Hamiltonian through rank four."""

    constant: complex
    h11: np.ndarray
    h20: np.ndarray
    h02: np.ndarray
    h22: np.ndarray
    h31: np.ndarray
    h13: np.ndarray
    h40: np.ndarray
    h04: np.ndarray

    def __post_init__(self) -> None:
        self.constant = complex(self.constant)
        for name in BLOCK_NAMES:
            setattr(self, name, np.asarray(getattr(self, name)))
        nspin = int(self.h11.shape[0])
        expected = {
            "h11": (nspin, nspin),
            "h20": (nspin, nspin),
            "h02": (nspin, nspin),
            "h22": (nspin,) * 4,
            "h31": (nspin,) * 4,
            "h13": (nspin,) * 4,
            "h40": (nspin,) * 4,
            "h04": (nspin,) * 4,
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

    @property
    def nspin(self) -> int:
        return int(self.h11.shape[0])

    @property
    def h00(self) -> complex:
        return self.constant

    @property
    def E0(self) -> complex:
        return self.constant

    def as_dict(self) -> dict[str, np.ndarray | complex]:
        return {"constant": self.constant, **{name: getattr(self, name) for name in BLOCK_NAMES}}


@dataclass
class QPAmplitudes:
    """Fully antisymmetric two- and four-quasiparticle amplitudes."""

    t1: np.ndarray
    t2: np.ndarray

    def __post_init__(self) -> None:
        self.t1 = np.asarray(self.t1)
        self.t2 = np.asarray(self.t2)
        nspin = int(self.t1.shape[0])
        if self.t1.shape != (nspin, nspin):
            raise ValueError("t1 must be square")
        if self.t2.shape != (nspin,) * 4:
            raise ValueError(f"t2 must have shape {(nspin,) * 4}")

    @classmethod
    def zeros(cls, nspin: int, dtype: Any = float) -> "QPAmplitudes":
        return cls(np.zeros((nspin, nspin), dtype=dtype), np.zeros((nspin,) * 4, dtype=dtype))


@dataclass(frozen=True)
class ExecutionOptions:
    """Numerically neutral controls for contraction and parallel execution."""

    integral_backend: str = "exact"
    parallel_mode: str = "auto"
    workers: int | None = None
    blas_threads: int | None = None
    max_workspace_bytes: int | None = None
    deterministic: bool = True
    pipeline_depth: int = 4
    cholesky_tolerances: tuple[float, ...] = (1.0e-8, 1.0e-10, 1.0e-12)
    factorization_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.integral_backend not in {"exact", "cholesky", "auto-validated"}:
            raise ValueError(
                "integral_backend must be 'exact', 'cholesky', or 'auto-validated'"
            )
        if self.parallel_mode not in {"auto", "serial", "blas", "pipeline"}:
            raise ValueError(
                "parallel_mode must be 'auto', 'serial', 'blas', or 'pipeline'"
            )
        for name, value in (("workers", self.workers), ("blas_threads", self.blas_threads)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")
        if self.max_workspace_bytes is not None and self.max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive when specified")
        if self.pipeline_depth <= 0:
            raise ValueError("pipeline_depth must be positive")
        if not self.cholesky_tolerances or any(
            tolerance <= 0.0 for tolerance in self.cholesky_tolerances
        ):
            raise ValueError("cholesky_tolerances must contain positive values")
        if self.factorization_tolerance <= 0.0:
            raise ValueError("factorization_tolerance must be positive")


@dataclass(frozen=True)
class GaugeModeOptions:
    """Controls removal of redundant particle-number-projected equations.

    ``projected-metric`` is the production algorithm.  It identifies the
    physical residual range from the Hamiltonian-independent projected
    excitation metric.  ``sampled-w2-orbit`` is retained only to reproduce
    historical diagnostics; it is not a certified solver projector.
    """

    enabled: bool = True
    apply_to_solver: bool = True
    method: str = "projected-metric"
    explicit_metric_max_coordinates: int = 256
    metric_relative_tolerance: float = 1.0e-10
    metric_absolute_tolerance: float = 1.0e-12
    metric_max_iterations: int = 64
    metric_maximum_nullity: int = 32
    # Fix amplitudes to a constant section transverse to the redundant
    # projected equations. This is a gauge condition, not a claim that the
    # metric-null vectors are nonlinear symmetry tangents at finite T.
    project_solver_steps: bool = True
    local_tangent_gauge: bool = True
    sample_count: int = 16
    maximum_rank: int = 16
    rank_relative_tolerance: float = 1.0e-10
    rank_absolute_tolerance: float = 1.0e-12
    gauge_weight: float = 0.0
    coarse_newton: bool = False
    weak_subspace_overlap_threshold: float = 0.99
    identity_tolerance: float = 1.0e-8
    certified_weak_subspace_overlap: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"projected-metric", "sampled-w2-orbit"}:
            raise ValueError(
                "gauge method must be 'projected-metric' or 'sampled-w2-orbit'"
            )
        if self.explicit_metric_max_coordinates < 0:
            raise ValueError("explicit metric limit must be non-negative")
        if (
            self.metric_relative_tolerance <= 0.0
            or self.metric_absolute_tolerance <= 0.0
            or self.metric_max_iterations <= 0
            or self.metric_maximum_nullity <= 0
        ):
            raise ValueError("projected metric tolerances and iteration limit must be positive")
        if self.sample_count <= 0 or self.maximum_rank <= 0:
            raise ValueError("gauge sample count and maximum rank must be positive")
        if (
            self.rank_relative_tolerance <= 0.0
            or self.rank_absolute_tolerance <= 0.0
        ):
            raise ValueError("gauge rank tolerances must be positive")
        if not 0.0 <= self.gauge_weight <= 1.0:
            raise ValueError("gauge_weight must lie in [0, 1]")
        if not 0.0 <= self.weak_subspace_overlap_threshold <= 1.0:
            raise ValueError("weak_subspace_overlap_threshold must lie in [0, 1]")
        if self.identity_tolerance <= 0.0:
            raise ValueError("identity_tolerance must be positive")
        if self.certified_weak_subspace_overlap is not None and not (
            0.0 <= self.certified_weak_subspace_overlap <= 1.0
        ):
            raise ValueError("certified weak-subspace overlap must lie in [0, 1]")
        if (
            self.enabled
            and self.apply_to_solver
            and self.method == "sampled-w2-orbit"
            and (
                self.certified_weak_subspace_overlap is None
                or self.certified_weak_subspace_overlap
                < self.weak_subspace_overlap_threshold
            )
        ):
            raise ValueError(
                "gauge removal requires an independently certified weak-subspace overlap"
            )


@dataclass(frozen=True)
class ProjectedResidualBasis:
    """Implicit physical/gauge decomposition of projected CC residuals."""

    gauge_vectors: np.ndarray
    singular_values: np.ndarray
    coordinate_count: int
    source: str = "analytic-w2-number-orbit"
    identity_defect: float = 0.0

    def __post_init__(self) -> None:
        vectors = np.asarray(self.gauge_vectors, dtype=np.complex128)
        singular_values = np.asarray(self.singular_values, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != self.coordinate_count:
            raise ValueError("gauge_vectors have an incompatible coordinate dimension")
        if singular_values.shape != (vectors.shape[1],):
            raise ValueError("one singular value is required per gauge vector")
        if vectors.shape[1]:
            error = np.max(
                np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1]))
            )
            if error > 1.0e-10:
                raise ValueError(f"gauge vectors are not orthonormal: {error:.3e}")
        vectors.setflags(write=False)
        singular_values.setflags(write=False)
        object.__setattr__(self, "gauge_vectors", vectors)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "coordinate_count", int(self.coordinate_count))
        object.__setattr__(self, "identity_defect", float(self.identity_defect))

    @classmethod
    def empty(cls, coordinate_count: int) -> "ProjectedResidualBasis":
        return cls(
            gauge_vectors=np.empty((int(coordinate_count), 0), dtype=np.complex128),
            singular_values=np.empty(0, dtype=float),
            coordinate_count=int(coordinate_count),
            source="disabled-or-zero-orbit",
        )

    @property
    def gauge_rank(self) -> int:
        return int(self.gauge_vectors.shape[1])

    @property
    def active(self) -> bool:
        return self.gauge_rank > 0

    def project_gauge(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.coordinate_count,):
            raise ValueError("residual vector has an incompatible coordinate dimension")
        if not self.gauge_rank:
            return np.zeros_like(values)
        return self.gauge_vectors @ (self.gauge_vectors.conj().T @ values)

    def project_physical(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.complex128)
        return values - self.project_gauge(values)


@dataclass(frozen=True)
class ProjectionOptions:
    target_number: int
    grid_size: int = 9
    parity: str = "even"
    gauge_quadrature: str = "midpoint"
    ode_substeps: int = 1
    grid_shift: float = 0.5
    contour_radius: float | None = None
    overlap_tolerance: float = 1.0e-10
    validation_tolerance: float = 1.0e-8
    validation_residual_tolerance: float = 1.0e-8
    cache_bytes: int = 2 * 1024 * 1024 * 1024
    auto_select_ode: bool = False
    ode_substep_candidates: tuple[int, ...] = (8, 12, 16, 24)
    ode_selection_energy_tolerance: float = 1.0e-10
    ode_selection_residual_tolerance: float = 1.0e-8
    max_grid_refinements: int = 0
    disentanglement_backend: str = "ser2"
    backend_validation_tolerance: float = 1.0e-10
    contour_safety_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.disentanglement_backend not in {"ode2", "ser2", "ser3"}:
            raise ValueError(
                "disentanglement_backend must be 'ode2', 'ser2', or 'ser3'"
            )
        if self.grid_size <= 0 or self.ode_substeps <= 0:
            raise ValueError("grid_size and ode_substeps must be positive")
        if self.parity not in {"even", "full"}:
            raise ValueError("parity must be 'even' or 'full'")
        if self.gauge_quadrature not in {"midpoint", "midpoint-richardson"}:
            raise ValueError(
                "gauge_quadrature must be 'midpoint' or 'midpoint-richardson'"
            )
        if self.parity == "even" and self.target_number % 2:
            raise ValueError("the even-parity grid cannot project an odd particle number")
        if not 0.0 <= self.grid_shift < 1.0:
            raise ValueError("grid_shift must lie in [0, 1)")
        if self.contour_radius is not None and self.contour_radius <= 0.0:
            raise ValueError("contour_radius must be positive")
        if not 0.0 < self.contour_safety_threshold <= 1.0:
            raise ValueError("contour_safety_threshold must lie in (0, 1]")
        if self.contour_safety_threshold <= self.overlap_tolerance:
            raise ValueError(
                "contour_safety_threshold must exceed overlap_tolerance"
            )
        if self.validation_tolerance <= 0.0 or self.validation_residual_tolerance <= 0.0:
            raise ValueError("projection validation tolerances must be positive")
        if self.cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        if not self.ode_substep_candidates or any(
            value <= 0 for value in self.ode_substep_candidates
        ):
            raise ValueError("ode_substep_candidates must contain positive values")
        if tuple(sorted(set(self.ode_substep_candidates))) != self.ode_substep_candidates:
            raise ValueError("ode_substep_candidates must be strictly increasing")
        if (
            self.ode_selection_energy_tolerance <= 0.0
            or self.ode_selection_residual_tolerance <= 0.0
        ):
            raise ValueError("ODE selection tolerances must be positive")
        if self.max_grid_refinements < 0:
            raise ValueError("max_grid_refinements must be non-negative")
        if self.backend_validation_tolerance <= 0.0:
            raise ValueError("backend_validation_tolerance must be positive")


@dataclass(frozen=True)
class ProjectionContinuationOptions:
    """Residual-homotopy stages used to reach the full OAP equations."""

    strengths: tuple[float, ...] = (0.25, 0.50, 0.75, 0.875, 0.9375, 1.0)
    evaluator_calls: tuple[int, ...] = (8, 8, 8, 8, 8, 20)
    residual_tolerances: tuple[float, ...] = (
        1.0e-4,
        1.0e-5,
        1.0e-5,
        1.0e-5,
        1.0e-5,
        1.0e-8,
    )
    predictor_maximum: float = 0.05

    def __post_init__(self) -> None:
        if not (
            len(self.strengths)
            == len(self.evaluator_calls)
            == len(self.residual_tolerances)
        ) or not self.strengths:
            raise ValueError("projection continuation stages have inconsistent lengths")
        if any(not 0.0 < value <= 1.0 for value in self.strengths):
            raise ValueError("projection continuation strengths must lie in (0, 1]")
        if tuple(sorted(self.strengths)) != self.strengths or self.strengths[-1] != 1.0:
            raise ValueError("projection continuation strengths must increase and end at one")
        if any(value <= 0 for value in self.evaluator_calls):
            raise ValueError("projection continuation evaluator budgets must be positive")
        if any(value <= 0.0 for value in self.residual_tolerances):
            raise ValueError("projection continuation tolerances must be positive")
        if self.predictor_maximum <= 0.0:
            raise ValueError("projection continuation predictor maximum must be positive")


@dataclass(frozen=True)
class SolverOptions:
    max_iterations: int = 80
    residual_tolerance: float = 1.0e-8
    energy_tolerance: float = 1.0e-10
    step_max: float = 0.20
    max_backtracks: int = 1
    newton_max_backtracks: int = 6
    backtrack_shrink: float = 0.5
    broyden_history: int = 12
    denominator_floor: float = 1.0e-8
    quasiparticle_level_shift: bool = False
    quasiparticle_level_shift_value: float | None = None
    adaptive_block_preconditioner: bool = True
    block_preconditioner_minimum_scale: float = 0.25
    block_preconditioner_maximum_scale: float = 4.0
    spectral_fallback: bool = False
    spectral_max_evaluations: int = 120
    complex_amplitudes: bool | None = None
    max_evaluator_calls: int = 60
    nonmonotone_window: int = 5
    nonmonotone_factor: float = 1.0
    filter_maximum_growth: float = 5.0e-2
    filter_l2_relative_decrease: float = 1.0e-3
    trust_expand: float = 1.25
    minimum_step_max: float = 1.0e-12
    stagnation_iterations: int = 4
    stagnation_relative_improvement: float = 5.0e-2
    newton_krylov: bool = True
    newton_initial: bool = False
    newton_globalization: str = "hookstep"
    gmres_restart: int = 16
    gmres_max_iterations: int = 1
    max_jvp_calls: int = 48
    newton_after_rejections: int = 3
    newton_cooldown_iterations: int = 4
    newton_step_max: float = 0.05
    newton_level_shift: float | None = None
    newton_level_shift_growth: float = 4.0
    newton_level_shift_max: float = 0.25
    newton_hookstep_min_radius: float = 1.0e-6
    newton_hookstep_maximum_growth: float = 0.25
    newton_hookstep_acceptance_ratio: float = 1.0e-4
    dense_newton_min_coordinates: int = 16
    dense_newton_max_coordinates: int = 128
    dense_newton_workers: int | None = 1
    dense_newton_max_builds: int = 0
    dense_newton_step_max: float = 2.0e-1
    dense_newton_derivative: str = "analytic-jvp"
    dense_newton_finite_difference_step: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if (
            self.max_backtracks < 0
            or self.newton_max_backtracks < 0
            or self.broyden_history < 0
        ):
            raise ValueError("optimizer history and backtrack limits must be non-negative")
        if self.residual_tolerance <= 0.0 or self.energy_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be positive")
        if (
            self.quasiparticle_level_shift_value is not None
            and (
                not np.isfinite(self.quasiparticle_level_shift_value)
                or self.quasiparticle_level_shift_value < 0.0
            )
        ):
            raise ValueError(
                "quasiparticle_level_shift_value must be finite and non-negative "
                "when specified"
            )
        if (
            not self.quasiparticle_level_shift
            and self.quasiparticle_level_shift_value is not None
        ):
            raise ValueError(
                "quasiparticle_level_shift_value requires quasiparticle_level_shift=True"
            )
        if (
            self.block_preconditioner_minimum_scale <= 0.0
            or self.block_preconditioner_maximum_scale
            < self.block_preconditioner_minimum_scale
        ):
            raise ValueError("block preconditioner scale bounds are inconsistent")
        if self.step_max <= 0.0:
            raise ValueError("step_max must be positive")
        if not 0.0 < self.backtrack_shrink < 1.0:
            raise ValueError("backtrack_shrink must lie in (0, 1)")
        if self.spectral_max_evaluations <= 0:
            raise ValueError("spectral_max_evaluations must be positive")
        if self.max_evaluator_calls <= 0:
            raise ValueError("max_evaluator_calls must be positive")
        if (
            self.nonmonotone_window <= 0
            or self.stagnation_iterations <= 0
            or self.newton_after_rejections <= 0
        ):
            raise ValueError("optimizer history windows must be positive")
        if self.newton_cooldown_iterations < 0:
            raise ValueError("newton_cooldown_iterations must be non-negative")
        if not 0.0 < self.stagnation_relative_improvement < 1.0:
            raise ValueError("stagnation_relative_improvement must lie in (0, 1)")
        if self.nonmonotone_factor < 1.0:
            raise ValueError("nonmonotone_factor must be at least one")
        if self.filter_maximum_growth < 0.0:
            raise ValueError("filter_maximum_growth must be non-negative")
        if not 0.0 <= self.filter_l2_relative_decrease < 1.0:
            raise ValueError(
                "filter_l2_relative_decrease must lie in [0, 1)"
            )
        if self.trust_expand < 1.0:
            raise ValueError("trust_expand must be at least one")
        if self.minimum_step_max <= 0.0 or self.minimum_step_max > self.step_max:
            raise ValueError("minimum_step_max must lie in (0, step_max]")
        if (
            self.gmres_restart <= 0
            or self.gmres_max_iterations <= 0
            or self.max_jvp_calls <= 0
        ):
            raise ValueError("GMRES limits must be positive")
        if self.newton_globalization not in {"hookstep", "line-search"}:
            raise ValueError(
                "newton_globalization must be 'hookstep' or 'line-search'"
            )
        if self.newton_step_max <= 0.0 or self.newton_step_max > self.step_max:
            raise ValueError("newton_step_max must lie in (0, step_max]")
        if self.newton_level_shift is not None and self.newton_level_shift < 0.0:
            raise ValueError("newton_level_shift must be non-negative when specified")
        if self.newton_level_shift_growth <= 1.0:
            raise ValueError("newton_level_shift_growth must exceed one")
        if self.newton_level_shift_max <= 0.0:
            raise ValueError("newton_level_shift_max must be positive")
        if (
            self.newton_hookstep_min_radius <= 0.0
            or self.newton_hookstep_min_radius > self.newton_step_max
        ):
            raise ValueError(
                "newton_hookstep_min_radius must lie in (0, newton_step_max]"
            )
        if self.newton_hookstep_maximum_growth < 0.0:
            raise ValueError("newton_hookstep_maximum_growth must be non-negative")
        if not 0.0 <= self.newton_hookstep_acceptance_ratio < 1.0:
            raise ValueError(
                "newton_hookstep_acceptance_ratio must lie in [0, 1)"
            )
        if (
            self.newton_level_shift is not None
            and self.newton_level_shift > self.newton_level_shift_max
        ):
            raise ValueError(
                "newton_level_shift must not exceed newton_level_shift_max"
            )
        if (
            self.dense_newton_min_coordinates < 0
            or self.dense_newton_max_coordinates < 0
            or self.dense_newton_min_coordinates > self.dense_newton_max_coordinates
        ):
            raise ValueError("dense Newton coordinate bounds are inconsistent")
        if self.dense_newton_workers is not None and self.dense_newton_workers <= 0:
            raise ValueError("dense_newton_workers must be positive when specified")
        if self.dense_newton_derivative not in {"finite-difference", "analytic-jvp"}:
            raise ValueError(
                "dense_newton_derivative must be 'finite-difference' or 'analytic-jvp'"
            )
        if self.dense_newton_finite_difference_step <= 0.0:
            raise ValueError("dense_newton_finite_difference_step must be positive")
        if self.dense_newton_max_builds < 0:
            raise ValueError("dense_newton_max_builds must be non-negative")
        if self.dense_newton_step_max <= 0.0:
            raise ValueError("dense_newton_step_max must be positive")
        if (
            self.dense_newton_max_builds > 0
            and self.dense_newton_step_max > self.step_max
        ):
            raise ValueError(
                "dense_newton_step_max must not exceed step_max when dense Newton is enabled"
            )


@dataclass
class KernelEvaluation:
    total_energy: complex
    correlation_energy: complex
    r1: np.ndarray
    r2: np.ndarray
    residual_norm: float
    projected_norm: complex = 1.0 + 0.0j
    elapsed: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_moments: Any | None = None


@dataclass
class KernelDirectionalDerivative:
    total_energy: complex
    correlation_energy: complex
    r1: np.ndarray
    r2: np.ndarray
    elapsed: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_moments: Any | None = None


@dataclass
class QPCCSDResult:
    converged: bool
    total_energy: complex
    correlation_energy: complex
    amplitudes: QPAmplitudes
    residual_norm: float
    iterations: int
    requested_method: str
    canonical_method: str = "qpccsd"
    projected: bool = False
    projection_time: float = 0.0
    grid_error: float | None = None
    hfb_pairing_collapsed: bool | None = None
    reference_energy: float | None = None
    zero_amplitude_energy: complex | None = None
    dynamic_correlation_energy: complex | None = None
    internal_residual_norm: float | None = None
    rdm_cumulant_norm: float | None = None
    allowed_pair_count: int | None = None
    allowed_quadruple_count: int | None = None
    forbidden_amplitude_norm: float | None = None
    raw_projected_energy: complex | None = None
    casscf_plus_dynamic_delta: complex | None = None
    baseline_gap: complex | None = None
    projection_equation_schema: str | None = None
    projection_solver_schema: str | None = None
    projector_ordering: str | None = None
    w2_exact_energy_difference: float | None = None
    w2_exact_residual_difference: float | None = None
    allowed_residual: np.ndarray | None = None
    physical_residual_norm: float | None = None
    gauge_residual_norm: float | None = None
    full_residual_norm: float | None = None
    gauge_rank: int = 0
    gauge_identity_defect: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QPPAVEvaluation:
    """Fixed-amplitude particle-number projection after variation."""

    total_energy: complex
    raw_projected_energy: complex
    validation_energy: complex
    amplitudes: QPAmplitudes
    projected_residual_norm: float
    validation_residual_norm: float
    grid_error: float
    residual_grid_error: float
    validation_passed: bool
    projection_time: float
    zero_amplitude_energy: complex | None = None
    dynamic_correlation_energy: complex | None = None
    baseline_gap: complex | None = None
    validation_raw_projected_energy: complex | None = None
    raw_grid_error: float | None = None
    raw_imaginary_energy_error: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QPPAVResult:
    """Unprojected QPCCSD solve followed by fixed-amplitude projection."""

    converged: bool
    qpccsd: QPCCSDResult
    projection: QPPAVEvaluation
    requested_method: str = "pav-qpccsd"
    canonical_method: str = "pav-qpccsd"

    @property
    def total_energy(self) -> complex:
        return self.projection.total_energy

    @property
    def amplitudes(self) -> QPAmplitudes:
        return self.qpccsd.amplitudes
