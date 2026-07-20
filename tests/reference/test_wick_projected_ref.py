from __future__ import annotations

from itertools import permutations

import numpy as np

from autogen.reference.wick_ref import projected_bch_ref_blocks


def _anti2(x: np.ndarray) -> np.ndarray:
    return x - x.T


def _anti4_pair(x: np.ndarray) -> np.ndarray:
    return 0.25 * (x - x.swapaxes(0, 1) - x.swapaxes(2, 3) + x.transpose(1, 0, 3, 2))


def _pair_perm_sign(perm: tuple[int, ...], base: tuple[int, ...]) -> int:
    idx = [base.index(x) for x in perm]
    inv = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if idx[i] > idx[j]:
                inv += 1
    return -1 if (inv % 2) else 1


def _fermion_ops(n: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    dim = 1 << n

    def ann(p: int) -> np.ndarray:
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            if (state >> p) & 1:
                new = state ^ (1 << p)
                sign = -1 if ((state & ((1 << p) - 1)).bit_count() % 2) else 1
                mat[new, state] = sign
        return mat

    annihilators = [ann(p) for p in range(n)]
    creators = [op.T.conj() for op in annihilators]
    vac = np.zeros(dim, dtype=np.complex128)
    vac[0] = 1.0
    return annihilators, creators, vac


def _pairs(n: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(n) for q in range(p + 1, n)]


def _quads(n: int) -> list[tuple[int, int, int, int]]:
    return [
        (p, q, r, s)
        for p in range(n)
        for q in range(p + 1, n)
        for r in range(q + 1, n)
        for s in range(r + 1, n)
    ]


def _build_full_t1_t2(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = _pairs(n)
    quads = _quads(n)
    t1 = np.zeros((n, n))
    t2 = np.zeros((n, n, n, n))
    k = 0
    for p, q in pairs:
        val = theta[k]
        t1[p, q] = val
        t1[q, p] = -val
        k += 1
    for p, q, r, s in quads:
        val = theta[k]
        base = (p, q, r, s)
        for perm in permutations(base):
            t2[perm] = _pair_perm_sign(perm, base) * val
        k += 1
    return t1, t2


def _op_from_h22(h22: np.ndarray, creators: list[np.ndarray], annihilators: list[np.ndarray]) -> np.ndarray:
    dim = creators[0].shape[0]
    out = np.zeros((dim, dim), dtype=np.complex128)
    for p, q in _pairs(h22.shape[0]):
        left = creators[p] @ creators[q]
        for r, s in _pairs(h22.shape[0]):
            coef = h22[p, q, r, s]
            if abs(coef) > 1e-12:
                out += coef * (left @ (annihilators[s] @ annihilators[r]))
    return out


def _op_from_t1(t1: np.ndarray, creators: list[np.ndarray]) -> np.ndarray:
    dim = creators[0].shape[0]
    out = np.zeros((dim, dim), dtype=np.complex128)
    for p, q in _pairs(t1.shape[0]):
        coef = t1[p, q]
        if abs(coef) > 1e-12:
            out += coef * (creators[p] @ creators[q])
    return out


def _op_from_t2(t2: np.ndarray, creators: list[np.ndarray]) -> np.ndarray:
    dim = creators[0].shape[0]
    out = np.zeros((dim, dim), dtype=np.complex128)
    for p, q, r, s in _quads(t2.shape[0]):
        coef = t2[p, q, r, s]
        if abs(coef) > 1e-12:
            out += coef * (creators[p] @ creators[q] @ creators[r] @ creators[s])
    return out


def _pair_state(z: np.ndarray, creators: list[np.ndarray], vac: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    dim = vac.shape[0]
    pair_gen = np.zeros((dim, dim), dtype=np.complex128)
    for p, q in _pairs(n):
        coef = z[p, q]
        if abs(coef) > 1e-12:
            # This sign reproduces <0| a_p a_q |phi> = Z_pq.
            pair_gen += -coef * (creators[p] @ creators[q])

    state = vac.copy()
    term = vac.copy()
    for k in range(1, n // 2 + 1):
        term = (pair_gen @ term) / k
        state += term
    return state


def _exact_projected(
    op: np.ndarray,
    phi: np.ndarray,
    annihilators: list[np.ndarray],
    vac: np.ndarray,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    n = len(annihilators)
    pairs = _pairs(n)
    quads = _quads(n)
    denom = np.vdot(vac, phi)

    energy = np.vdot(vac, op @ phi) / denom
    r1 = np.zeros((n, n), dtype=np.complex128)
    r2 = np.zeros((n, n, n, n), dtype=np.complex128)
    for p, q in pairs:
        val = np.vdot(vac, annihilators[q] @ (annihilators[p] @ (op @ phi))) / denom
        r1[p, q] = val
        r1[q, p] = -val
    for p, q, r, s in quads:
        val = np.vdot(
            vac,
            annihilators[s] @ (annihilators[r] @ (annihilators[q] @ (annihilators[p] @ (op @ phi)))),
        ) / denom
        base = (p, q, r, s)
        for perm in permutations(base):
            r2[perm] = _pair_perm_sign(perm, base) * val
    return energy, r1, r2


def _zero_ham(n: int) -> dict[str, np.ndarray]:
    zeros2 = np.zeros((n, n))
    zeros4 = np.zeros((n, n, n, n))
    return {
        "h11": zeros2.copy(),
        "h20": zeros2.copy(),
        "h02": zeros2.copy(),
        "h22": zeros4.copy(),
        "h31": zeros4.copy(),
        "h13": zeros4.copy(),
        "h40": zeros4.copy(),
        "h04": zeros4.copy(),
    }


def test_projected_h22_order0_matches_exact_gaussian_state():
    rng = np.random.default_rng(1)
    n = 4
    annihilators, creators, vac = _fermion_ops(n)

    z = 0.1 * _anti2(rng.normal(size=(n, n)))
    h22 = _anti4_pair(rng.normal(size=(n, n, n, n)))
    phi = _pair_state(z, creators, vac)

    ham = _zero_ham(n)
    ham["h22"] = h22

    blocks, _ = projected_bch_ref_blocks(
        ham,
        np.zeros((n, n)),
        np.zeros((n, n, n, n)),
        z,
        targets=(0, 2, 4),
        max_order=0,
    )
    exact = _exact_projected(_op_from_h22(h22, creators, annihilators), phi, annihilators, vac)

    assert np.allclose(blocks[0], exact[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(blocks[2], exact[1], atol=1e-10, rtol=1e-10)
    assert np.allclose(blocks[4], exact[2], atol=1e-10, rtol=1e-10)


def test_projected_h22_order1_matches_exact_gaussian_state():
    rng = np.random.default_rng(2)
    n = 4
    annihilators, creators, vac = _fermion_ops(n)

    z = 0.1 * _anti2(rng.normal(size=(n, n)))
    h22 = _anti4_pair(rng.normal(size=(n, n, n, n)))
    theta = 0.05 * rng.normal(size=len(_pairs(n)) + len(_quads(n)))
    t1, t2 = _build_full_t1_t2(theta, n)
    phi = _pair_state(z, creators, vac)

    ham = _zero_ham(n)
    ham["h22"] = h22

    h22_op = _op_from_h22(h22, creators, annihilators)
    t_op = _op_from_t1(t1, creators) + _op_from_t2(t2, creators)
    first_order = h22_op + h22_op @ t_op - t_op @ h22_op

    blocks, _ = projected_bch_ref_blocks(ham, t1, t2, z, targets=(0, 2, 4), max_order=1)
    exact = _exact_projected(first_order, phi, annihilators, vac)

    assert np.allclose(blocks[0], exact[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(blocks[2], exact[1], atol=1e-10, rtol=1e-10)
    assert np.allclose(blocks[4], exact[2], atol=1e-10, rtol=1e-10)
