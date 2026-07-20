from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .excitation_space import QPExcitationSpace, QPExcitationSpaceLike
from .models import BogoliubovReference, QPAmplitudes


_BINARY_EINSUM_PATH = ("einsum_path", (0, 1))


@dataclass(frozen=True)
class QPAmplitudeTransport:
    """Creation-sector warm start in a new quasiparticle basis."""

    amplitudes: QPAmplitudes
    creation_map: np.ndarray
    annihilation_map: np.ndarray
    spatial_singular_values: np.ndarray
    diagnostics: dict[str, float | str]


def _closest_unitary(overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, singular_values, right_adjoint = np.linalg.svd(overlap, full_matrices=False)
    return left @ right_adjoint, singular_values


def _spin_overlap(spatial_overlap: np.ndarray) -> np.ndarray:
    return np.kron(spatial_overlap, np.eye(2, dtype=spatial_overlap.dtype))


def _transform_rank_four(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = np.einsum(
        "ia,abcd->ibcd",
        matrix,
        tensor,
        optimize=_BINARY_EINSUM_PATH,
    )
    transformed = np.einsum(
        "jb,ibcd->ijcd",
        matrix,
        transformed,
        optimize=_BINARY_EINSUM_PATH,
    )
    transformed = np.einsum(
        "kc,ijcd->ijkd",
        matrix,
        transformed,
        optimize=_BINARY_EINSUM_PATH,
    )
    return np.einsum(
        "ld,ijkd->ijkl",
        matrix,
        transformed,
        optimize=_BINARY_EINSUM_PATH,
    )


def transport_qp_amplitudes(
    amplitudes: QPAmplitudes,
    old_reference: BogoliubovReference,
    new_reference: BogoliubovReference,
    spatial_orbital_overlap: np.ndarray,
    *,
    excitation_space: QPExcitationSpaceLike | None = None,
    orthogonalize_orbital_map: bool = True,
) -> QPAmplitudeTransport:
    """Transport a QPCCSD warm start between tracked molecular geometries.

    The orbital overlap has old orbitals on its rows and new orbitals on its
    columns.  The exact operator map generally mixes new quasiparticle creation
    and annihilation operators because the two Bogoliubov vacua differ.  Only
    the all-creation component is retained as a CC warm start; the discarded
    mixing is reported rather than hidden.
    """

    overlap = np.asarray(spatial_orbital_overlap, dtype=np.complex128)
    nspin = old_reference.nspin
    nspatial = nspin // 2
    if new_reference.nspin != nspin:
        raise ValueError("old and new quasiparticle references have different sizes")
    if amplitudes.t1.shape != (nspin, nspin):
        raise ValueError("amplitudes and quasiparticle references have different sizes")
    if overlap.shape != (nspatial, nspatial):
        raise ValueError(
            "spatial_orbital_overlap must have one old and one new orbital index"
        )

    if orthogonalize_orbital_map:
        orbital_map, singular_values = _closest_unitary(overlap)
        map_kind = "polar-orthogonalized-cross-overlap"
    else:
        orbital_map = overlap
        singular_values = np.linalg.svd(overlap, compute_uv=False)
        map_kind = "raw-cross-overlap"
    spin_map = _spin_overlap(orbital_map)

    physical_creation = spin_map.T @ old_reference.U
    physical_annihilation = spin_map.conj().T @ old_reference.V
    creation_map = (
        new_reference.U.conj().T @ physical_creation
        + new_reference.V.conj().T @ physical_annihilation
    )
    annihilation_map = (
        new_reference.V.T @ physical_creation
        + new_reference.U.T @ physical_annihilation
    )

    transported_t1 = creation_map @ amplitudes.t1 @ creation_map.T
    transported_t2 = _transform_rank_four(amplitudes.t2, creation_map)
    raw = QPAmplitudes(transported_t1, transported_t2)
    space = excitation_space or QPExcitationSpace.full(nspin)
    transported = space.enforce(raw)

    identity = np.eye(nspin, dtype=np.complex128)
    normal_defect = (
        creation_map.conj().T @ creation_map
        + annihilation_map.conj().T @ annihilation_map
        - identity
    )
    anomalous_defect = (
        creation_map.T @ annihilation_map
        + annihilation_map.T @ creation_map
    )
    discarded = max(
        float(np.max(np.abs(raw.t1 - transported.t1), initial=0.0)),
        float(np.max(np.abs(raw.t2 - transported.t2), initial=0.0)),
    )
    diagnostics: dict[str, float | str] = {
        "orbital_map": map_kind,
        "minimum_spatial_overlap_singular_value": float(np.min(singular_values)),
        "maximum_spatial_overlap_singular_value": float(np.max(singular_values)),
        "annihilation_leakage_maximum": float(np.max(np.abs(annihilation_map))),
        "annihilation_leakage_frobenius": float(np.linalg.norm(annihilation_map)),
        "canonical_normal_defect": float(np.max(np.abs(normal_defect))),
        "canonical_anomalous_defect": float(np.max(np.abs(anomalous_defect))),
        "mask_discarded_maximum": discarded,
    }
    return QPAmplitudeTransport(
        amplitudes=transported,
        creation_map=creation_map,
        annihilation_map=annihilation_map,
        spatial_singular_values=singular_values,
        diagnostics=diagnostics,
    )


__all__ = ["QPAmplitudeTransport", "transport_qp_amplitudes"]
