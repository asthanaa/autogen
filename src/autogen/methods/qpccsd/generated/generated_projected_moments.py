"""Generated factorized left moments for PN-OAP-QPCCSD.

The external probe labels remain open throughout these contractions.  This
avoids constructing pair/quadruple probe tensors with separate internal and
external index sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROJECTION_EQUATION_SCHEMA = "pn-oap-bpn-v2"
PROJECTOR_ORDERING = "B_mu P_N (H-E) exp(T)"
_BINARY_EINSUM_PATH = ("einsum_path", (0, 1))


@dataclass(frozen=True)
class ProjectedMoments:
    n0: complex
    n20: np.ndarray
    n40: np.ndarray
    h0: complex
    h20: np.ndarray
    h40: np.ndarray


@dataclass(frozen=True)
class ProjectedNormMoments:
    n20: np.ndarray
    n40: np.ndarray


def _pair_wedge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,kl->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("jk,il->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("jl,ik->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("kl,ij->ijkl", left, right, optimize=_BINARY_EINSUM_PATH)
    )


def _pair_square(pair: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,kl->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
        - np.einsum("ik,jl->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
        + np.einsum("il,jk->ijkl", pair, pair, optimize=_BINARY_EINSUM_PATH)
    )


def _transform_pair(probe_annihilation: np.ndarray, pair: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(probe_annihilation)
    if np.array_equal(rows, columns):
        diagonal = np.diag(probe_annihilation)
        return diagonal[:, None] * pair * diagonal[None, :]
    return probe_annihilation @ pair @ probe_annihilation.T


def _mode_product(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
    """Apply a one-body transformation to one tensor index using BLAS."""

    transformed = np.tensordot(matrix, tensor, axes=(1, axis))
    return np.moveaxis(transformed, 0, axis)


def _transform_quad(probe_annihilation: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(probe_annihilation)
    if np.array_equal(rows, columns):
        diagonal = np.diag(probe_annihilation)
        return (
            np.asarray(quad)
            * diagonal[:, None, None, None]
            * diagonal[None, :, None, None]
            * diagonal[None, None, :, None]
            * diagonal[None, None, None, :]
        )
    transformed = np.asarray(quad)
    for axis in range(4):
        transformed = _mode_product(probe_annihilation, transformed, axis)
    return transformed


def projected_left_moments(
    probe_annihilation: np.ndarray,
    probe_pairing: np.ndarray,
    *,
    n0: complex,
    n20: np.ndarray,
    n40: np.ndarray,
    h0: complex,
    h20: np.ndarray,
    h40: np.ndarray,
) -> ProjectedMoments:
    """Contract transformed pair/quad probes with right vectors through 4qp."""

    a = np.asarray(probe_annihilation)
    k = np.asarray(probe_pairing)
    n20_right = _transform_pair(a, np.asarray(n20))
    h20_right = _transform_pair(a, np.asarray(h20))
    k_square = _pair_square(k)
    return ProjectedMoments(
        n0=complex(n0),
        n20=k * n0 + n20_right,
        n40=(
            k_square * n0
            + _pair_wedge(k, n20_right)
            + _transform_quad(a, np.asarray(n40))
        ),
        h0=complex(h0),
        h20=k * h0 + h20_right,
        h40=(
            k_square * h0
            + _pair_wedge(k, h20_right)
            + _transform_quad(a, np.asarray(h40))
        ),
    )


def projected_left_norm_moments(
    probe_annihilation: np.ndarray,
    probe_pairing: np.ndarray,
    *,
    n0: complex = 0.0,
    n20: np.ndarray,
    n40: np.ndarray,
) -> ProjectedNormMoments:
    """Apply projected pair/quad probes to a linear excitation ket.

    This is the Hamiltonian-independent action required by the projected
    excitation metric.  The scalar right component is zero, so disconnected
    reference terms do not enter.
    """

    a = np.asarray(probe_annihilation)
    k = np.asarray(probe_pairing)
    n20_right = _transform_pair(a, np.asarray(n20))
    transformed_quad = _transform_quad(a, np.asarray(n40))
    if n0 == 0.0:
        return ProjectedNormMoments(
            n20=n20_right,
            n40=_pair_wedge(k, n20_right) + transformed_quad,
        )
    k_square = _pair_square(k)
    return ProjectedNormMoments(
        n20=k * n0 + n20_right,
        n40=(
            k_square * n0
            + _pair_wedge(k, n20_right)
            + transformed_quad
        ),
    )


def projected_left_vacuum_norm_moments(
    probe_pairing: np.ndarray,
) -> ProjectedNormMoments:
    """Return the projected probe moments of the quasiparticle vacuum."""

    pairing = np.asarray(probe_pairing)
    return ProjectedNormMoments(n20=pairing, n40=_pair_square(pairing))


__all__ = [
    "PROJECTOR_ORDERING",
    "PROJECTION_EQUATION_SCHEMA",
    "ProjectedMoments",
    "ProjectedNormMoments",
    "projected_left_moments",
    "projected_left_norm_moments",
    "projected_left_vacuum_norm_moments",
]
