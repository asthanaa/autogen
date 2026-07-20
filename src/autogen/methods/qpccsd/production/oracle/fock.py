from __future__ import annotations

import numpy as np
from scipy import sparse


def _annihilate(state: int, orbital: int) -> tuple[float, int] | None:
    if not (state >> orbital) & 1:
        return None
    sign = -1.0 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return sign, state & ~(1 << orbital)


def _create(state: int, orbital: int) -> tuple[float, int] | None:
    if (state >> orbital) & 1:
        return None
    sign = -1.0 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return sign, state | (1 << orbital)


def build_sparse_fock_hamiltonian(
    h1: np.ndarray,
    g2: np.ndarray,
    constant: float,
    *,
    particle_parity: int | None = None,
) -> sparse.csr_matrix:
    """Build the exact oracle Hamiltonian without a dense ``2**M`` matrix."""

    h1 = np.asarray(h1)
    g2 = np.asarray(g2)
    nspin = int(h1.shape[0])
    if h1.shape != (nspin, nspin):
        raise ValueError("h1 must be square")
    if g2.shape != (nspin,) * 4:
        raise ValueError("g2 shape is incompatible with h1")
    if particle_parity not in {None, 0, 1}:
        raise ValueError("particle_parity must be None, 0, or 1")

    one_terms = tuple(
        tuple((p, h1[p, q]) for p in range(nspin) if h1[p, q] != 0.0)
        for q in range(nspin)
    )
    two_terms = tuple(
        tuple(
            tuple(
                (p, q, 0.25 * g2[p, q, r, s])
                for q in range(nspin)
                for p in range(nspin)
                if g2[p, q, r, s] != 0.0
            )
            for s in range(nspin)
        )
        for r in range(nspin)
    )

    dimension = 1 << nspin
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for ket in range(dimension):
        if particle_parity is not None and ket.bit_count() % 2 != particle_parity:
            continue
        column: dict[int, complex] = {ket: complex(constant)}
        for q in range(nspin):
            right = _annihilate(ket, q)
            if right is None:
                continue
            sign_q, state_q = right
            for p, coefficient in one_terms[q]:
                left = _create(state_q, p)
                if left is None:
                    continue
                sign_p, bra = left
                column[bra] = column.get(bra, 0.0j) + coefficient * sign_p * sign_q

        for r in range(nspin):
            right_r = _annihilate(ket, r)
            if right_r is None:
                continue
            sign_r, state_r = right_r
            for s in range(nspin):
                right_s = _annihilate(state_r, s)
                if right_s is None:
                    continue
                sign_s, state_rs = right_s
                for p, q, coefficient in two_terms[r][s]:
                    left_q = _create(state_rs, q)
                    if left_q is None:
                        continue
                    sign_q, state_rsq = left_q
                    left_p = _create(state_rsq, p)
                    if left_p is None:
                        continue
                    sign_p, bra = left_p
                    contribution = (
                        coefficient * sign_r * sign_s * sign_q * sign_p
                    )
                    column[bra] = column.get(bra, 0.0j) + contribution

        for bra, value in column.items():
            if value != 0.0:
                rows.append(bra)
                columns.append(ket)
                values.append(value)

    result = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return (0.5 * (result + result.getH())).tocsr()


def build_full_fock_hamiltonian(
    h1: np.ndarray,
    g2: np.ndarray,
    constant: float,
) -> np.ndarray:
    return build_sparse_fock_hamiltonian(h1, g2, constant).toarray()
