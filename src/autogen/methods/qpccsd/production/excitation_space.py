from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement, permutations
from typing import Iterable, Union

import numpy as np

from .models import BogoliubovReference, CASQPReference, QPAmplitudes


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


_QUAD_PERMUTATIONS = tuple(
    (permutation, _permutation_sign(permutation))
    for permutation in permutations(range(4))
)


@dataclass(frozen=True)
class QPExcitationSpace:
    """Independent QPCCSD coordinates with pure-active amplitudes removed."""

    nspin: int
    active_spin_indices: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    quadruple_indices: tuple[tuple[int, int, int, int], ...]
    internal_pair_indices: tuple[tuple[int, int], ...]
    internal_quadruple_indices: tuple[tuple[int, int, int, int], ...]
    _pair_array: np.ndarray = field(init=False, repr=False, compare=False)
    _quadruple_array: np.ndarray = field(init=False, repr=False, compare=False)
    _internal_pair_array: np.ndarray = field(init=False, repr=False, compare=False)
    _internal_quadruple_array: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.nspin <= 0 or self.nspin % 2:
            raise ValueError("nspin must be a positive even integer")
        active = tuple(sorted(set(int(index) for index in self.active_spin_indices)))
        if active != self.active_spin_indices:
            raise ValueError("active_spin_indices must be sorted and unique")
        if any(index < 0 or index >= self.nspin for index in active):
            raise ValueError("active spin-orbital index is outside the QP space")
        object.__setattr__(self, "_pair_array", self._as_index_array(self.pair_indices, 2))
        object.__setattr__(
            self, "_quadruple_array", self._as_index_array(self.quadruple_indices, 4)
        )
        object.__setattr__(
            self, "_internal_pair_array", self._as_index_array(self.internal_pair_indices, 2)
        )
        object.__setattr__(
            self,
            "_internal_quadruple_array",
            self._as_index_array(self.internal_quadruple_indices, 4),
        )

    @staticmethod
    def _as_index_array(keys: tuple[tuple[int, ...], ...], rank: int) -> np.ndarray:
        if not keys:
            return np.empty((0, rank), dtype=np.int64)
        result = np.asarray(keys, dtype=np.int64)
        if result.shape != (len(keys), rank):
            raise ValueError("excitation keys have an incompatible rank")
        return result

    @classmethod
    def full(cls, nspin: int) -> "QPExcitationSpace":
        return cls.from_active_spin_indices(nspin, ())

    @classmethod
    def from_active_spin_indices(
        cls,
        nspin: int,
        active_spin_indices: Iterable[int],
    ) -> "QPExcitationSpace":
        active = tuple(sorted(set(int(index) for index in active_spin_indices)))
        active_set = frozenset(active)
        all_pairs = tuple(combinations(range(int(nspin)), 2))
        all_quads = tuple(combinations(range(int(nspin)), 4))
        internal_pairs = tuple(key for key in all_pairs if set(key) <= active_set)
        internal_quads = tuple(key for key in all_quads if set(key) <= active_set)
        pair_indices = tuple(key for key in all_pairs if not set(key) <= active_set)
        quadruple_indices = tuple(key for key in all_quads if not set(key) <= active_set)
        return cls(
            nspin=int(nspin),
            active_spin_indices=active,
            pair_indices=pair_indices,
            quadruple_indices=quadruple_indices,
            internal_pair_indices=internal_pairs,
            internal_quadruple_indices=internal_quads,
        )

    @property
    def pair_count(self) -> int:
        return len(self.pair_indices)

    @property
    def quadruple_count(self) -> int:
        return len(self.quadruple_indices)

    @property
    def coordinate_count(self) -> int:
        return self.pair_count + self.quadruple_count

    @property
    def internal_coordinate_count(self) -> int:
        return len(self.internal_pair_indices) + len(self.internal_quadruple_indices)

    def _validate_amplitudes(self, amplitudes: QPAmplitudes) -> None:
        if amplitudes.t1.shape != (self.nspin, self.nspin):
            raise ValueError("amplitudes and excitation space have different dimensions")

    def unpack(self, vector: np.ndarray) -> QPAmplitudes:
        vector = np.asarray(vector)
        if vector.shape != (self.coordinate_count,):
            raise ValueError("independent amplitude vector has an incompatible shape")
        t1 = np.zeros((self.nspin, self.nspin), dtype=vector.dtype)
        t2 = np.zeros((self.nspin,) * 4, dtype=vector.dtype)
        if self.pair_count:
            p, q = self._pair_array.T
            values = vector[: self.pair_count]
            t1[p, q] = values
            t1[q, p] = -values
        if self.quadruple_count:
            values = vector[self.pair_count :]
            for permutation, sign in _QUAD_PERMUTATIONS:
                indices = self._quadruple_array[:, permutation]
                t2[tuple(indices.T)] = sign * values
        return QPAmplitudes(t1=t1, t2=t2)

    def pack(self, amplitudes: QPAmplitudes) -> np.ndarray:
        self._validate_amplitudes(amplitudes)
        pair_values = (
            amplitudes.t1[tuple(self._pair_array.T)]
            if self.pair_count
            else np.empty(0, dtype=amplitudes.t1.dtype)
        )
        quad_values = (
            amplitudes.t2[tuple(self._quadruple_array.T)]
            if self.quadruple_count
            else np.empty(0, dtype=amplitudes.t2.dtype)
        )
        return np.concatenate((pair_values, quad_values))

    def residual_vector(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return self.pack(QPAmplitudes(r1, r2))

    def enforce(self, amplitudes: QPAmplitudes) -> QPAmplitudes:
        return self.unpack(self.pack(amplitudes))

    def internal_residual_norm(self, r1: np.ndarray, r2: np.ndarray) -> float:
        values: list[np.ndarray] = []
        if len(self.internal_pair_indices):
            values.append(np.asarray(r1)[tuple(self._internal_pair_array.T)])
        if len(self.internal_quadruple_indices):
            values.append(np.asarray(r2)[tuple(self._internal_quadruple_array.T)])
        if not values:
            return 0.0
        joined = np.concatenate(values)
        return float(np.max(np.abs(joined))) if joined.size else 0.0

    def forbidden_amplitude_norm(self, amplitudes: QPAmplitudes) -> float:
        return self.internal_residual_norm(amplitudes.t1, amplitudes.t2)

    def diagnostics(self) -> dict[str, object]:
        return {
            "excitation_space_kind": "spin-orbital-all-index",
            "coordinate_count": self.coordinate_count,
            "pair_block_counts": {},
            "quadruple_block_counts": {},
        }


def _fermion_one_body_action(
    key: tuple[int, ...],
    creator: int,
    annihilator: int,
) -> tuple[complex, tuple[int, ...]] | None:
    state = sum(1 << index for index in key)
    if not (state >> annihilator) & 1:
        return None
    sign = -1.0 if (state & ((1 << annihilator) - 1)).bit_count() % 2 else 1.0
    state &= ~(1 << annihilator)
    if (state >> creator) & 1:
        return None
    sign *= -1.0 if (state & ((1 << creator) - 1)).bit_count() % 2 else 1.0
    state |= 1 << creator
    result = tuple(index for index in range(max(key + (creator,)) + 1) if (state >> index) & 1)
    return sign, result


def _deterministic_range(projector: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in range(projector.shape[0]):
        vector = np.asarray(projector[:, index], dtype=np.complex128).copy()
        for column in columns:
            vector -= column * np.vdot(column, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        # Several spin-adapted vectors have symmetry-related coefficients with
        # exactly equal magnitudes.  Choosing the numerically largest entry
        # lets sub-ulp BLAS differences select a different phase convention on
        # another machine.  The first significant determinant is an invariant
        # lexicographic anchor and makes checkpoints portable.
        significant = np.flatnonzero(np.abs(vector) > tolerance)
        if not significant.size:
            continue
        pivot = int(significant[0])
        phase = vector[pivot]
        if abs(phase):
            vector *= np.conj(phase) / abs(phase)
        columns.append(vector)
    if not columns:
        return np.empty((projector.shape[0], 0), dtype=np.complex128)
    return np.column_stack(columns)


def _singlet_spin_expansions(
    spatial_key: tuple[int, ...],
) -> tuple[tuple[tuple[tuple[int, ...], complex], ...], ...]:
    rank = len(spatial_key)
    unique_spatial = tuple(sorted(set(spatial_key)))
    spin_orbitals = tuple(
        spin
        for spatial in unique_spatial
        for spin in (2 * spatial, 2 * spatial + 1)
    )

    def keys_with_alpha_count(alpha_count: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            key
            for key in combinations(spin_orbitals, rank)
            if tuple(sorted(index // 2 for index in key)) == spatial_key
            and sum(index % 2 == 0 for index in key) == alpha_count
        )

    m_zero = keys_with_alpha_count(rank // 2)
    m_plus = keys_with_alpha_count(rank // 2 + 1)
    plus_lookup = {key: index for index, key in enumerate(m_plus)}
    s_plus = np.zeros((len(m_plus), len(m_zero)), dtype=np.complex128)
    for column, key in enumerate(m_zero):
        for spatial in unique_spatial:
            action = _fermion_one_body_action(
                key,
                2 * spatial,
                2 * spatial + 1,
            )
            if action is None:
                continue
            coefficient, raised = action
            row = plus_lookup.get(raised)
            if row is not None:
                s_plus[row, column] += coefficient
    s_squared = s_plus.conj().T @ s_plus
    eigenvalues, eigenvectors = np.linalg.eigh(s_squared)
    null = eigenvectors[:, eigenvalues < 1.0e-10]
    singlets = _deterministic_range(null @ null.conj().T)
    return tuple(
        tuple(
            (key, np.real_if_close(vector[row]).item())
            for row, key in enumerate(m_zero)
            if abs(vector[row]) > 1.0e-14
        )
        for vector in singlets.T
    )


@dataclass(frozen=True)
class BlockQPExcitationSpace:
    """Spatial total-singlet QP coordinates in molecular orbital blocks."""

    nspin: int
    active_spin_indices: tuple[int, ...]
    pair_expansions: tuple[tuple[tuple[tuple[int, int], complex], ...], ...]
    quadruple_expansions: tuple[
        tuple[tuple[tuple[int, int, int, int], complex], ...], ...
    ]
    pair_blocks: tuple[str, ...]
    quadruple_blocks: tuple[str, ...]
    internal_pair_indices: tuple[tuple[int, int], ...]
    internal_quadruple_indices: tuple[tuple[int, int, int, int], ...]
    include_active_t1_t2: bool = False

    def __post_init__(self) -> None:
        if self.nspin <= 0 or self.nspin % 2:
            raise ValueError("nspin must be a positive even integer")
        if len(self.pair_expansions) != len(self.pair_blocks):
            raise ValueError("pair block labels and expansions differ")
        if len(self.quadruple_expansions) != len(self.quadruple_blocks):
            raise ValueError("quadruple block labels and expansions differ")
        for name, expansions in (
            ("pair", self.pair_expansions),
            ("quadruple", self.quadruple_expansions),
        ):
            for expansion in expansions:
                norm = sum(abs(coefficient) ** 2 for _, coefficient in expansion)
                if not np.isclose(norm, 1.0, atol=1.0e-12, rtol=0.0):
                    raise ValueError(f"{name} spin expansion is not normalized")

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(expansion[0][0] for expansion in self.pair_expansions)

    @property
    def quadruple_indices(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(expansion[0][0] for expansion in self.quadruple_expansions)

    @property
    def pair_count(self) -> int:
        return len(self.pair_expansions)

    @property
    def quadruple_count(self) -> int:
        return len(self.quadruple_expansions)

    @property
    def coordinate_count(self) -> int:
        return self.pair_count + self.quadruple_count

    @property
    def internal_coordinate_count(self) -> int:
        return len(self.internal_pair_indices) + len(self.internal_quadruple_indices)

    def unpack(self, vector: np.ndarray) -> QPAmplitudes:
        values = np.asarray(vector)
        if values.shape != (self.coordinate_count,):
            raise ValueError("independent amplitude vector has an incompatible shape")
        coefficients = tuple(
            coefficient
            for expansion in self.pair_expansions + self.quadruple_expansions
            for _, coefficient in expansion
        )
        dtype = np.result_type(values.dtype, *(type(value) for value in coefficients))
        t1 = np.zeros((self.nspin, self.nspin), dtype=dtype)
        t2 = np.zeros((self.nspin,) * 4, dtype=dtype)
        for value, expansion in zip(values[: self.pair_count], self.pair_expansions):
            for (p, q), coefficient in expansion:
                component = coefficient * value
                t1[p, q] += component
                t1[q, p] -= component
        for value, expansion in zip(values[self.pair_count :], self.quadruple_expansions):
            for key, coefficient in expansion:
                component = coefficient * value
                for permutation, sign in _QUAD_PERMUTATIONS:
                    indices = tuple(key[index] for index in permutation)
                    t2[indices] += sign * component
        return QPAmplitudes(t1=t1, t2=t2)

    def pack(self, amplitudes: QPAmplitudes) -> np.ndarray:
        if amplitudes.t1.shape != (self.nspin, self.nspin):
            raise ValueError("amplitudes and excitation space have different dimensions")
        coefficient_types = tuple(
            type(coefficient)
            for expansion in self.pair_expansions + self.quadruple_expansions
            for _, coefficient in expansion
        )
        coefficient_dtype = (
            np.result_type(*coefficient_types)
            if coefficient_types
            else np.dtype(float)
        )
        dtype = np.result_type(
            amplitudes.t1.dtype,
            amplitudes.t2.dtype,
            coefficient_dtype,
        )
        pairs = np.asarray(
            [
                sum(np.conj(coefficient) * amplitudes.t1[key] for key, coefficient in expansion)
                for expansion in self.pair_expansions
            ],
            dtype=dtype,
        )
        quadruples = np.asarray(
            [
                sum(np.conj(coefficient) * amplitudes.t2[key] for key, coefficient in expansion)
                for expansion in self.quadruple_expansions
            ],
            dtype=dtype,
        )
        return np.concatenate((pairs, quadruples))

    def residual_vector(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return self.pack(QPAmplitudes(r1, r2))

    def enforce(self, amplitudes: QPAmplitudes) -> QPAmplitudes:
        return self.unpack(self.pack(amplitudes))

    def internal_residual_norm(self, r1: np.ndarray, r2: np.ndarray) -> float:
        values = [np.asarray(r1)[key] for key in self.internal_pair_indices]
        values.extend(np.asarray(r2)[key] for key in self.internal_quadruple_indices)
        return float(np.max(np.abs(values))) if values else 0.0

    def forbidden_amplitude_norm(self, amplitudes: QPAmplitudes) -> float:
        enforced = self.enforce(amplitudes)
        defect = max(
            float(np.max(np.abs(amplitudes.t1 - enforced.t1))),
            float(np.max(np.abs(amplitudes.t2 - enforced.t2))),
        )
        return 0.0 if defect < 1.0e-14 else defect

    def inverse_denominators(self, energies: np.ndarray, floor: float) -> np.ndarray:
        values: list[float] = []
        for expansion in self.pair_expansions:
            values.append(
                sum(
                    abs(coefficient) ** 2 * (energies[key[0]] + energies[key[1]])
                    for key, coefficient in expansion
                )
            )
        for expansion in self.quadruple_expansions:
            values.append(
                sum(
                    abs(coefficient) ** 2 * sum(energies[index] for index in key)
                    for key, coefficient in expansion
                )
            )
        return 1.0 / np.maximum(np.asarray(values, dtype=float), floor)

    def diagnostics(self) -> dict[str, object]:
        from collections import Counter

        return {
            "excitation_space_kind": "spatial-total-singlet-block",
            "coordinate_count": self.coordinate_count,
            "include_active_t1_t2": self.include_active_t1_t2,
            "pair_block_counts": dict(sorted(Counter(self.pair_blocks).items())),
            "quadruple_block_counts": dict(
                sorted(Counter(self.quadruple_blocks).items())
            ),
        }


QPExcitationSpaceLike = Union[QPExcitationSpace, BlockQPExcitationSpace]


_PAIR_BLOCKS = {
    (1, 0, 1): "ia",
    (1, 1, 0): "ix",
    (0, 1, 1): "xa",
}
_QUADRUPLE_BLOCKS = {
    (2, 0, 2): "ijab",
    (2, 2, 0): "ijxy",
    (0, 2, 2): "xyab",
    (2, 1, 1): "ijxa",
    (1, 1, 2): "ixab",
    (1, 2, 1): "ixya",
    (1, 3, 0): "ixyz",
    (0, 3, 1): "xyza",
}


def build_block_spin_adapted_qp_space(
    reference: CASQPReference,
    *,
    include_active_t1_t2: bool = True,
) -> BlockQPExcitationSpace:
    """Build the total-singlet, totally symmetric QPCCSD manifold.

    The production default includes pure-active pair (``xy``) and quadruple
    (``xyzw``) coordinates and uses the direct molecular-Hamiltonian energy.
    Setting ``include_active_t1_t2=False`` reproduces the historical masked
    test manifold and is an explicit experimental choice.
    """

    nspatial = reference.nspatial
    inactive = frozenset(reference.inactive_spatial_indices)
    active = frozenset(reference.active_spatial_indices)
    external = frozenset(reference.external_spatial_indices)
    spatial_irreps = np.asarray(
        reference.metadata.get("orbital_irrep_ids", np.zeros(nspatial, dtype=int)),
        dtype=np.int64,
    )
    if spatial_irreps.shape != (nspatial,):
        raise ValueError("orbital irrep identifiers have an incompatible shape")

    def counts(key: tuple[int, ...]) -> tuple[int, int, int]:
        return (
            sum(index in inactive for index in key),
            sum(index in active for index in key),
            sum(index in external for index in key),
        )

    def totally_symmetric(key: tuple[int, ...]) -> bool:
        irrep = 0
        for index in key:
            irrep ^= int(spatial_irreps[index])
        return irrep == 0

    pair_blocks = dict(_PAIR_BLOCKS)
    quadruple_blocks = dict(_QUADRUPLE_BLOCKS)
    if include_active_t1_t2:
        pair_blocks[(0, 2, 0)] = "xy"
        quadruple_blocks[(0, 4, 0)] = "xyzw"

    pair_records: list[
        tuple[str, tuple[int, ...], tuple[tuple[tuple[int, int], complex], ...]]
    ] = []
    for spatial_key in combinations_with_replacement(range(nspatial), 2):
        block = pair_blocks.get(counts(spatial_key))
        if block is None or not totally_symmetric(spatial_key):
            continue
        for expansion in _singlet_spin_expansions(spatial_key):
            pair_records.append((block, spatial_key, expansion))

    quad_records: list[
        tuple[
            str,
            tuple[int, ...],
            tuple[tuple[tuple[int, int, int, int], complex], ...],
        ]
    ] = []
    for spatial_key in combinations_with_replacement(range(nspatial), 4):
        if max(spatial_key.count(index) for index in set(spatial_key)) > 2:
            continue
        block = quadruple_blocks.get(counts(spatial_key))
        if block is None or not totally_symmetric(spatial_key):
            continue
        for expansion in _singlet_spin_expansions(spatial_key):
            quad_records.append((block, spatial_key, expansion))

    pair_order = {name: index for index, name in enumerate(pair_blocks.values())}
    quad_order = {
        name: index for index, name in enumerate(quadruple_blocks.values())
    }
    pair_records.sort(key=lambda item: (pair_order[item[0]], item[1]))
    quad_records.sort(key=lambda item: (quad_order[item[0]], item[1]))
    internal = QPExcitationSpace.from_active_spin_indices(
        reference.nspin, reference.active_spin_indices
    )
    internal_pairs = () if include_active_t1_t2 else internal.internal_pair_indices
    internal_quadruples = (
        () if include_active_t1_t2 else internal.internal_quadruple_indices
    )
    return BlockQPExcitationSpace(
        nspin=reference.nspin,
        active_spin_indices=reference.active_spin_indices,
        pair_expansions=tuple(record[2] for record in pair_records),
        quadruple_expansions=tuple(record[2] for record in quad_records),
        pair_blocks=tuple(record[0] for record in pair_records),
        quadruple_blocks=tuple(record[0] for record in quad_records),
        internal_pair_indices=internal_pairs,
        internal_quadruple_indices=internal_quadruples,
        include_active_t1_t2=include_active_t1_t2,
    )


def build_external_qp_space(
    reference_or_nspin: CASQPReference | BogoliubovReference | int,
    active_spin_indices: Iterable[int] | None = None,
) -> QPExcitationSpace:
    """Build the dynamic QP space, excluding only pure-active coordinates."""

    if isinstance(reference_or_nspin, CASQPReference):
        if active_spin_indices is not None:
            raise ValueError("CASQPReference already defines the active spin orbitals")
        return QPExcitationSpace.from_active_spin_indices(
            reference_or_nspin.nspin,
            reference_or_nspin.active_spin_indices,
        )
    if isinstance(reference_or_nspin, BogoliubovReference):
        if active_spin_indices is None:
            raise ValueError("active_spin_indices are required for a BogoliubovReference")
        nspin = reference_or_nspin.nspin
    else:
        nspin = int(reference_or_nspin)
    if active_spin_indices is None:
        raise ValueError("active_spin_indices are required")
    return QPExcitationSpace.from_active_spin_indices(nspin, active_spin_indices)


def build_ms0_qp_space(reference: CASQPReference) -> QPExcitationSpace:
    """Legacy spin-orbital filter that enforces only M_S=0 and spatial symmetry."""

    base = build_external_qp_space(reference)
    spatial_irreps = reference.metadata.get("orbital_irrep_ids")
    if spatial_irreps is None:
        raise ValueError("CAS reference does not provide orbital irrep identifiers")
    spatial_irreps = np.asarray(spatial_irreps, dtype=np.int64)
    if spatial_irreps.shape != (reference.nspatial,):
        raise ValueError("orbital irrep identifiers have an incompatible shape")
    spin_irreps = np.repeat(spatial_irreps, 2)

    def spin_scalar(key: tuple[int, ...]) -> bool:
        return sum(index % 2 == 0 for index in key) == len(key) // 2

    def totally_symmetric(key: tuple[int, ...]) -> bool:
        value = 0
        for index in key:
            value ^= int(spin_irreps[index])
        return value == 0

    pairs = tuple(
        key
        for key in base.pair_indices
        if spin_scalar(key) and totally_symmetric(key)
    )
    quadruples = tuple(
        key
        for key in base.quadruple_indices
        if spin_scalar(key) and totally_symmetric(key)
    )
    return QPExcitationSpace(
        nspin=base.nspin,
        active_spin_indices=base.active_spin_indices,
        pair_indices=pairs,
        quadruple_indices=quadruples,
        internal_pair_indices=base.internal_pair_indices,
        internal_quadruple_indices=base.internal_quadruple_indices,
    )


def build_symmetry_adapted_qp_space(
    reference: CASQPReference,
    *,
    include_active_t1_t2: bool = True,
) -> BlockQPExcitationSpace:
    """Build the production full-active symmetry-adapted CCSD manifold."""

    return build_block_spin_adapted_qp_space(
        reference,
        include_active_t1_t2=include_active_t1_t2,
    )


__all__ = [
    "BlockQPExcitationSpace",
    "QPExcitationSpace",
    "QPExcitationSpaceLike",
    "build_block_spin_adapted_qp_space",
    "build_external_qp_space",
    "build_ms0_qp_space",
    "build_symmetry_adapted_qp_space",
]
