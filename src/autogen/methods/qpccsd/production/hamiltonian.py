from __future__ import annotations

from itertools import permutations
import math

import numpy as np

from .models import BogoliubovReference, QPHamiltonian


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _antisymmetrize_groups(tensor: np.ndarray, groups: tuple[tuple[int, ...], ...]) -> np.ndarray:
    result = np.asarray(tensor)
    for group in groups:
        if len(group) < 2:
            continue
        source = result
        result = np.zeros_like(source)
        for permutation in permutations(range(len(group))):
            axes = list(range(source.ndim))
            for destination, selected in zip(group, permutation):
                axes[destination] = group[selected]
            result += _permutation_sign(permutation) * source.transpose(axes)
        result /= math.factorial(len(group))
    return result


def _normal_order(
    operators: tuple[tuple[str, int], ...],
    sign: int = 1,
    contractions: tuple[tuple[int, int], ...] = (),
) -> list[tuple[int, tuple[tuple[str, int], ...], tuple[tuple[int, int], ...]]]:
    for index in range(len(operators) - 1):
        if operators[index][0] != "A" or operators[index + 1][0] != "C":
            continue
        swapped = list(operators)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        terms = _normal_order(tuple(swapped), -sign, contractions)
        left_label = operators[index][1]
        right_label = operators[index + 1][1]
        contracted = operators[:index] + operators[index + 2 :]
        terms.extend(_normal_order(contracted, sign, contractions + ((left_label, right_label),)))
        return terms
    return [(sign, operators, contractions)]


def _contract_coefficient(
    coefficient: np.ndarray,
    contractions: tuple[tuple[int, int], ...],
    output_labels: tuple[int, ...],
) -> np.ndarray:
    labels = list("abcdefgh")[: coefficient.ndim]
    for left, right in contractions:
        labels[right] = labels[left]
    output = "".join(labels[index] for index in output_labels)
    return np.einsum(f"{''.join(labels)}->{output}", coefficient, optimize=True)


def _accumulate_normal_ordered(
    blocks: dict[str, np.ndarray],
    coefficient: np.ndarray,
    operators: tuple[tuple[str, int], ...],
) -> None:
    for sign, ordered, contractions in _normal_order(operators):
        name = "".join(kind for kind, _ in ordered)
        labels = tuple(label for _, label in ordered)
        value = sign * _contract_coefficient(coefficient, contractions, labels)
        if name in blocks:
            blocks[name] += value
        else:
            blocks[name] = np.asarray(value)


def _canonical_blocks(
    h1: np.ndarray,
    g2: np.ndarray,
    reference: BogoliubovReference,
) -> dict[str, np.ndarray]:
    """Return coefficients in canonical creator-then-annihilator order."""

    U = reference.U
    V = reference.V
    dtype = np.result_type(h1, g2, U, V, np.complex128)
    blocks: dict[str, np.ndarray] = {"": np.array(0.0, dtype=dtype)}

    one_body_choices = (
        (("C", U.conj()), ("A", V)),
        (("C", V.conj()), ("A", U)),
    )
    for first in one_body_choices[0]:
        for second in one_body_choices[1]:
            coefficient = np.einsum(
                "pq,pa,qb->ab", h1, first[1], second[1], optimize=True
            )
            _accumulate_normal_ordered(
                blocks,
                coefficient,
                ((first[0], 0), (second[0], 1)),
            )

    two_body_choices = (
        (("C", U.conj()), ("A", V)),
        (("C", U.conj()), ("A", V)),
        (("C", V.conj()), ("A", U)),
        (("C", V.conj()), ("A", U)),
    )
    for mask in range(16):
        selected = tuple(two_body_choices[index][(mask >> index) & 1] for index in range(4))
        coefficient = 0.25 * np.einsum(
            "pqrs,pa,qb,sc,rd->abcd",
            g2,
            selected[0][1],
            selected[1][1],
            selected[2][1],
            selected[3][1],
            optimize=True,
        )
        _accumulate_normal_ordered(
            blocks,
            coefficient,
            tuple((choice[0], index) for index, choice in enumerate(selected)),
        )
    return blocks


def build_qp_hamiltonian(
    h1: np.ndarray,
    g2: np.ndarray,
    reference: BogoliubovReference,
    *,
    constant: float | complex = 0.0,
) -> QPHamiltonian:
    """Normal order a two-body Hamiltonian with respect to a Bogoliubov vacuum.

    ``g2[p,q,r,s]`` multiplies ``c_p^dag c_q^dag c_s c_r / 4``.
    The implementation performs only tensor contractions and stores no Fock-space
    operators or determinant sectors.
    """

    h1 = np.asarray(h1)
    g2 = np.asarray(g2)
    nspin = reference.nspin
    if h1.shape != (nspin, nspin) or g2.shape != (nspin,) * 4:
        raise ValueError("integral dimensions do not match the Bogoliubov reference")
    canonical = _canonical_blocks(h1, g2, reference)
    dtype = np.result_type(h1, g2, reference.U, reference.V, np.complex128)
    zero2 = np.zeros((nspin, nspin), dtype=dtype)
    zero4 = np.zeros((nspin,) * 4, dtype=dtype)

    c11 = np.asarray(canonical.get("CA", zero2))
    c20 = _antisymmetrize_groups(np.asarray(canonical.get("CC", zero2)), ((0, 1),))
    c02 = _antisymmetrize_groups(np.asarray(canonical.get("AA", zero2)), ((0, 1),))
    c22 = _antisymmetrize_groups(
        np.asarray(canonical.get("CCAA", zero4)), ((0, 1), (2, 3))
    )
    c31 = _antisymmetrize_groups(
        np.asarray(canonical.get("CCCA", zero4)), ((0, 1, 2),)
    )
    c13 = _antisymmetrize_groups(
        np.asarray(canonical.get("CAAA", zero4)), ((1, 2, 3),)
    )
    c40 = _antisymmetrize_groups(
        np.asarray(canonical.get("CCCC", zero4)), ((0, 1, 2, 3),)
    )
    c04 = _antisymmetrize_groups(
        np.asarray(canonical.get("AAAA", zero4)), ((0, 1, 2, 3),)
    )

    return QPHamiltonian(
        constant=complex(constant) + complex(canonical.get("", 0.0)),
        h11=c11,
        h20=2.0 * c20,
        h02=2.0 * c02.T,
        h22=4.0 * c22.swapaxes(2, 3),
        h31=6.0 * c31,
        h13=6.0 * c13,
        h40=24.0 * c40,
        h04=24.0 * c04.transpose(3, 2, 1, 0),
    )
