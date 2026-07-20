from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import math
import string
from typing import Iterable

import numpy as np

from .models import QPHamiltonian


@dataclass(frozen=True)
class _Term:
    tensor: np.ndarray
    labels: tuple[str, ...]
    creators: tuple[str, ...]
    annihilators: tuple[str, ...]

    def renamed(self, prefix: str) -> "_Term":
        mapping = {label: f"{prefix}{index}" for index, label in enumerate(self.labels)}
        return _Term(
            self.tensor,
            tuple(mapping[label] for label in self.labels),
            tuple(mapping[label] for label in self.creators),
            tuple(mapping[label] for label in self.annihilators),
        )


@dataclass(frozen=True)
class _Token:
    kind: str
    label: str
    index: int


_BLOCK_META = {
    "h11": (("p", "q"), ("p",), ("q",), 1.0),
    "h20": (("p", "q"), ("p", "q"), (), 0.5),
    "h02": (("p", "q"), (), ("q", "p"), 0.5),
    "h22": (("p", "q", "r", "s"), ("p", "q"), ("s", "r"), 0.25),
    "h31": (("p", "q", "r", "s"), ("p", "q", "r"), ("s",), 1.0 / 6.0),
    "h13": (("p", "q", "r", "s"), ("p",), ("q", "r", "s"), 1.0 / 6.0),
    "h40": (("p", "q", "r", "s"), ("p", "q", "r", "s"), (), 1.0 / 24.0),
    "h04": (("p", "q", "r", "s"), (), ("p", "q", "r", "s"), 1.0 / 24.0),
}
_RANK_TO_NAME = {
    (1, 1): "h11",
    (2, 0): "h20",
    (0, 2): "h02",
    (2, 2): "h22",
    (3, 1): "h31",
    (1, 3): "h13",
    (4, 0): "h40",
    (0, 4): "h04",
}


def _hamiltonian_terms(hamiltonian: QPHamiltonian) -> list[_Term]:
    terms = []
    for name, (labels, creators, annihilators, prefactor) in _BLOCK_META.items():
        terms.append(
            _Term(
                prefactor * np.asarray(getattr(hamiltonian, name)),
                labels,
                creators,
                annihilators,
            )
        )
    return terms


def _z_term(z: np.ndarray) -> _Term:
    return _Term(0.5 * np.asarray(z), ("p", "q"), (), ("q", "p"))


def _count_inversions(values: Iterable[int]) -> int:
    values = tuple(values)
    return sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values)))


def _contraction_sign(ops_in: list[_Token], pairs: list[tuple[_Token, _Token]]) -> int:
    ops = list(ops_in)
    sign = 1
    for left, right in sorted(pairs, key=lambda pair: min(ops_in.index(pair[0]), ops_in.index(pair[1]))):
        pos_left = ops.index(left)
        pos_right = ops.index(right)
        if pos_left > pos_right:
            pos_left, pos_right = pos_right, pos_left
        if (pos_right - pos_left - 1) % 2:
            sign *= -1
        ops.pop(pos_right)
        ops.pop(pos_left)
    seen_annihilators = 0
    inversions = 0
    for token in ops:
        if token.kind == "a":
            seen_annihilators += 1
        else:
            inversions += seen_annihilators
    return -sign if inversions % 2 else sign


def _einsum_merge(
    left: np.ndarray,
    left_labels: list[str],
    right: np.ndarray,
    right_labels: list[str],
    output_labels: list[str],
) -> np.ndarray:
    labels = []
    for label in left_labels + right_labels:
        if label not in labels:
            labels.append(label)
    symbols = string.ascii_lowercase + string.ascii_uppercase
    if len(labels) > len(symbols):
        raise ValueError("too many Wick labels for NumPy einsum")
    mapping = dict(zip(labels, symbols))
    subs_left = "".join(mapping[label] for label in left_labels)
    subs_right = "".join(mapping[label] for label in right_labels)
    subs_output = "".join(mapping[label] for label in output_labels)
    return np.einsum(f"{subs_left},{subs_right}->{subs_output}", left, right, optimize=True)


def _contract(left: _Term, right: _Term, pairs: list[tuple[str, str]], sign: int) -> _Term:
    contracted_left = {label for label, _ in pairs}
    contracted_right = {label for _, label in pairs}
    right_to_left = {right_label: left_label for left_label, right_label in pairs}
    right_labels = [right_to_left.get(label, label) for label in right.labels]
    output_labels = [
        label
        for label in list(left.labels) + right_labels
        if label not in contracted_left
    ]
    tensor = sign * _einsum_merge(
        left.tensor,
        list(left.labels),
        right.tensor,
        right_labels,
        output_labels,
    )
    creators = tuple(label for label in left.creators if label not in contracted_left) + tuple(
        right_to_left.get(label, label)
        for label in right.creators
        if label not in contracted_right
    )
    annihilators = tuple(label for label in left.annihilators if label not in contracted_left) + tuple(
        right_to_left.get(label, label)
        for label in right.annihilators
        if label not in contracted_right
    )
    return _Term(np.asarray(tensor), tuple(output_labels), creators, annihilators)


def _connected_left_product(left_in: _Term, right_in: _Term) -> list[_Term]:
    """Return connected contractions in [left, right] for pure-annihilation left."""

    left = left_in.renamed("L")
    right = right_in.renamed("R")
    tokens = []
    index = 0
    left_ann = {}
    for label in left.annihilators:
        token = _Token("a", label, index)
        tokens.append(token)
        left_ann[label] = token
        index += 1
    right_cre = {}
    for label in right.creators:
        token = _Token("c", label, index)
        tokens.append(token)
        right_cre[label] = token
        index += 1
    for label in right.annihilators:
        tokens.append(_Token("a", label, index))
        index += 1

    results = []
    max_contractions = min(len(left.annihilators), len(right.creators))
    for count in range(1, max_contractions + 1):
        for left_selection in combinations(left.annihilators, count):
            for right_selection in combinations(right.creators, count):
                for order in permutations(range(count)):
                    pairs = [
                        (left_selection[pos], right_selection[order[pos]])
                        for pos in range(count)
                    ]
                    token_pairs = [(left_ann[a], right_cre[c]) for a, c in pairs]
                    results.append(_contract(left, right, pairs, _contraction_sign(tokens, token_pairs)))
    return results


def _permutation_sign(source: tuple[str, ...], target: tuple[str, ...]) -> int:
    positions = [source.index(label) for label in target]
    return -1 if _count_inversions(positions) % 2 else 1


def _canonical_tensor(term: _Term) -> np.ndarray:
    if term.tensor.ndim == 0:
        return np.asarray(term.tensor)
    creators = tuple(sorted(term.creators))
    annihilators = tuple(sorted(term.annihilators))
    order = creators + annihilators
    axes = [term.labels.index(label) for label in order]
    sign = _permutation_sign(term.creators, creators) * _permutation_sign(term.annihilators, annihilators)
    return sign * term.tensor.transpose(axes)


def _merge_ranks(terms: Iterable[_Term], nspin: int, dtype: np.dtype) -> dict[tuple[int, int], np.ndarray]:
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for term in terms:
        rank = (len(term.creators), len(term.annihilators))
        if sum(rank) > 4:
            continue
        if rank not in blocks:
            shape = () if rank == (0, 0) else (nspin,) * sum(rank)
            blocks[rank] = np.zeros(shape, dtype=dtype)
        blocks[rank] += _canonical_tensor(term)
    return blocks


def _canonical_term(rank: tuple[int, int], tensor: np.ndarray) -> _Term:
    ncre, nann = rank
    creators = tuple(f"c{index}" for index in range(ncre))
    annihilators = tuple(f"a{index}" for index in range(nann))
    return _Term(tensor, creators + annihilators, creators, annihilators)


def _from_canonical_block(name: str, canonical: np.ndarray) -> np.ndarray:
    labels, creators, annihilators, prefactor = _BLOCK_META[name]
    ordered = tuple(sorted(creators)) + tuple(sorted(annihilators))
    axes = [labels.index(label) for label in ordered]
    inverse = np.argsort(axes)
    sign = _permutation_sign(creators, tuple(sorted(creators)))
    sign *= _permutation_sign(annihilators, tuple(sorted(annihilators)))
    return (sign / prefactor) * canonical.transpose(tuple(inverse))


def _antisymmetrize_axes(tensor: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    if len(axes) < 2:
        return np.asarray(tensor)
    result = np.zeros_like(tensor)
    for permutation in permutations(axes):
        order = list(range(tensor.ndim))
        for destination, source in zip(axes, permutation):
            order[destination] = source
        result += _permutation_sign(axes, permutation) * tensor.transpose(order)
    return result / math.factorial(len(axes))


def _canonicalize_block(name: str, tensor: np.ndarray) -> np.ndarray:
    creator_count = len(_BLOCK_META[name][1])
    annihilator_count = len(_BLOCK_META[name][2])
    result = np.asarray(tensor)
    if creator_count > 1:
        result = _antisymmetrize_axes(result, tuple(range(creator_count)))
    if annihilator_count > 1:
        result = _antisymmetrize_axes(
            result,
            tuple(range(creator_count, creator_count + annihilator_count)),
        )
    return result


def similarity_transform_deexcitation(
    hamiltonian: QPHamiltonian,
    z: np.ndarray,
) -> QPHamiltonian:
    """Evaluate H_Z = exp(Z) H exp(-Z) as a terminating Wick polynomial."""

    z = np.asarray(z)
    if z.shape != (hamiltonian.nspin, hamiltonian.nspin):
        raise ValueError("z has an incompatible shape")
    dtype = np.result_type(*hamiltonian.as_dict().values(), z, np.complex128)
    nspin = hamiltonian.nspin
    current = _merge_ranks(_hamiltonian_terms(hamiltonian), nspin, dtype)
    accumulated: dict[tuple[int, int], np.ndarray] = {}

    z_operator = _z_term(z)
    for order in range(5):
        factor = 1.0 / math.factorial(order)
        for rank, tensor in current.items():
            if rank not in accumulated:
                accumulated[rank] = np.zeros_like(tensor)
            accumulated[rank] += factor * tensor
        if order == 4:
            break
        next_terms = []
        for rank, tensor in current.items():
            if np.max(np.abs(tensor)) <= 1.0e-15:
                continue
            next_terms.extend(_connected_left_product(z_operator, _canonical_term(rank, tensor)))
        current = _merge_ranks(next_terms, nspin, dtype)

    zero2 = np.zeros((nspin, nspin), dtype=dtype)
    zero4 = np.zeros((nspin,) * 4, dtype=dtype)
    values = {}
    for rank, name in _RANK_TO_NAME.items():
        canonical = accumulated.get(rank)
        if canonical is None:
            canonical = zero2 if sum(rank) == 2 else zero4
        values[name] = _canonicalize_block(
            name,
            _from_canonical_block(name, canonical),
        )
    scalar = accumulated.get((0, 0), np.array(0.0, dtype=dtype))
    return QPHamiltonian(constant=hamiltonian.constant + scalar, **values)


def _select_active_axes(
    tensor: np.ndarray,
    support: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    """Select Thouless-active axes while preserving views for contiguous CAS blocks."""

    if not axes:
        return np.asarray(tensor)
    if support.size and np.array_equal(
        support,
        np.arange(int(support[0]), int(support[-1]) + 1),
    ):
        selector: slice | np.ndarray = slice(int(support[0]), int(support[-1]) + 1)
        index = tuple(selector if axis in axes else slice(None) for axis in range(tensor.ndim))
        return np.asarray(tensor[index])
    selected = np.asarray(tensor)
    for axis in axes:
        selected = np.take(selected, support, axis=axis)
    return selected


class ScalarSimilarityTransformKernel:
    """Evaluate ``<0| exp(Z) H exp(-Z) exp(W) |0>`` without transformed blocks.

    The generic similarity transformation above is the reference implementation
    needed by projected residuals.  A scalar PAV evaluation only needs its 00,
    02, and 04 pieces.  Wick expanding those pieces and immediately contracting
    them with ``W1`` and ``W2`` leaves twenty scalar contractions.  Restricting
    every index attached to ``Z`` to its exact support is particularly important
    for CAS references: the cost then depends on the active quasiparticle count
    rather than transforming every full-space rank-four Hamiltonian block.
    """

    backend = "direct-support-pruned-scalar-wick-v1"

    def __init__(
        self,
        hamiltonian: QPHamiltonian,
        support: Iterable[int],
    ) -> None:
        self.hamiltonian = hamiltonian
        indices = np.unique(np.asarray(tuple(support), dtype=np.int64))
        if np.any(indices < 0) or np.any(indices >= hamiltonian.nspin):
            raise ValueError("Thouless support contains an invalid orbital index")
        self.support = indices
        self.support_size = int(indices.size)
        self._inactive = np.setdiff1d(
            np.arange(hamiltonian.nspin, dtype=np.int64),
            indices,
            assume_unique=True,
        )

        # These are views for the contiguous active blocks produced by the CAS
        # reference builder and compact copies otherwise.  They are reused for
        # every gauge point and every scalar PAV evaluation.
        self.h11_aq = _select_active_axes(hamiltonian.h11, indices, (0,))
        self.h20_aa = _select_active_axes(hamiltonian.h20, indices, (0, 1))
        self.h22_aars = _select_active_axes(hamiltonian.h22, indices, (0, 1))
        self.h31_aaas = _select_active_axes(hamiltonian.h31, indices, (0, 1, 2))
        self.h13_aqrs = _select_active_axes(hamiltonian.h13, indices, (0,))
        self.h40_aaaa = _select_active_axes(hamiltonian.h40, indices, (0, 1, 2, 3))

        selected_blocks = (
            ("h11", self.h11_aq),
            ("h20", self.h20_aa),
            ("h22", self.h22_aars),
            ("h31", self.h31_aaas),
            ("h13", self.h13_aqrs),
            ("h40", self.h40_aaaa),
        )
        unique_arrays = {
            id(value): value
            for name, value in selected_blocks
            if not np.shares_memory(value, getattr(hamiltonian, name))
        }
        self.precomputed_bytes = int(sum(value.nbytes for value in unique_arrays.values()))

    @classmethod
    def from_deexcitations(
        cls,
        hamiltonian: QPHamiltonian,
        deexcitations: Iterable[np.ndarray],
    ) -> "ScalarSimilarityTransformKernel":
        support: set[int] = set()
        for value in deexcitations:
            z = np.asarray(value)
            if z.shape != (hamiltonian.nspin, hamiltonian.nspin):
                raise ValueError("z has an incompatible shape")
            rows, columns = np.nonzero(z)
            support.update(int(index) for index in rows)
            support.update(int(index) for index in columns)
        return cls(hamiltonian, sorted(support))

    def _active_pair(self, tensor: np.ndarray) -> np.ndarray:
        return _select_active_axes(tensor, self.support, (0, 1))

    def transformed_constant(self, z: np.ndarray) -> complex:
        z = np.asarray(z)
        if z.shape != (self.hamiltonian.nspin, self.hamiltonian.nspin):
            raise ValueError("z has an incompatible shape")
        if self._inactive.size and (
            np.any(z[self._inactive, :]) or np.any(z[:, self._inactive])
        ):
            raise ValueError("z contains entries outside the precomputed support")
        if not self.support_size:
            return complex(self.hamiltonian.constant)
        za = self._active_pair(z)
        constant = complex(self.hamiltonian.constant)
        constant += 0.5 * np.einsum(
            "pq,pq->", self.h20_aa, za, optimize=True
        )
        constant += 0.125 * np.einsum(
            "pqrs,pq,rs->", self.h40_aaaa, za, za, optimize="greedy"
        )
        return complex(constant)

    def evaluate(
        self,
        z: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> tuple[complex, complex]:
        """Return the correlated scalar and its zero-amplitude baseline."""

        nspin = self.hamiltonian.nspin
        z = np.asarray(z)
        w1 = np.asarray(w1)
        w2 = np.asarray(w2)
        if z.shape != (nspin, nspin):
            raise ValueError("z has an incompatible shape")
        if w1.shape != (nspin, nspin) or w2.shape != (nspin,) * 4:
            raise ValueError("W amplitudes have incompatible shapes")

        hamiltonian = self.hamiltonian
        transformed_constant = self.transformed_constant(z)
        energy = transformed_constant
        # Bare 02/04 terms do not carry a Thouless-active index.
        energy += 0.5 * np.einsum(
            "pq,pq->", hamiltonian.h02, w1, optimize=True
        )
        energy += (1.0 / 24.0) * np.einsum(
            "pqrs,srqp->", hamiltonian.h04, w2, optimize=True
        )
        energy += 0.125 * np.einsum(
            "pqrs,sr,qp->", hamiltonian.h04, w1, w1, optimize="greedy"
        )
        if not self.support_size:
            return complex(energy), complex(hamiltonian.constant)

        active = self.support
        za = self._active_pair(z)
        w1_qa = _select_active_axes(w1, active, (1,))
        w1_aa = _select_active_axes(w1, active, (0, 1))
        w2_rsba = _select_active_axes(w2, active, (2, 3))
        w2_scba = _select_active_axes(w2, active, (1, 2, 3))
        w2_srqa = _select_active_axes(w2, active, (3,))
        w2_dcba = _select_active_axes(w2, active, (0, 1, 2, 3))

        energy += np.einsum(
            "pq,pa,qa->", self.h11_aq, za, w1_qa, optimize="greedy"
        )
        energy += 0.5 * np.einsum(
            "pq,pa,qb,ba->", self.h20_aa, za, za, w1_aa, optimize="greedy"
        )

        energy += 0.25 * np.einsum(
            "pqrs,pq,rs->", self.h22_aars, za, w1, optimize="greedy"
        )
        energy += 0.25 * np.einsum(
            "pqrs,pa,qb,rsba->",
            self.h22_aars,
            za,
            za,
            w2_rsba,
            optimize="greedy",
        )
        # H22 is antisymmetric within, but not across, its creator and
        # annihilator pairs.  Its three disconnected W1 products are therefore
        # retained explicitly rather than folded into one multiplicity.
        energy += 0.25 * np.einsum(
            "pqrs,pa,qb,rs,ba->",
            self.h22_aars,
            za,
            za,
            w1,
            w1_aa,
            optimize="greedy",
        )
        energy -= 0.25 * np.einsum(
            "pqrs,pa,qb,rb,sa->",
            self.h22_aars,
            za,
            za,
            w1_qa,
            w1_qa,
            optimize="greedy",
        )
        energy += 0.25 * np.einsum(
            "pqrs,pa,qb,ra,sb->",
            self.h22_aars,
            za,
            za,
            w1_qa,
            w1_qa,
            optimize="greedy",
        )

        energy -= 0.5 * np.einsum(
            "pqrs,pr,qa,sa->",
            self.h31_aaas,
            za,
            za,
            w1_qa,
            optimize="greedy",
        )
        energy += (1.0 / 6.0) * np.einsum(
            "pqrs,pa,qb,rc,scba->",
            self.h31_aaas,
            za,
            za,
            za,
            w2_scba,
            optimize="greedy",
        )
        energy += 0.5 * np.einsum(
            "pqrs,pa,qb,rc,sc,ba->",
            self.h31_aaas,
            za,
            za,
            za,
            w1_qa,
            w1_aa,
            optimize="greedy",
        )

        energy += (1.0 / 6.0) * np.einsum(
            "pqrs,pa,srqa->",
            self.h13_aqrs,
            za,
            w2_srqa,
            optimize="greedy",
        )
        energy += 0.5 * np.einsum(
            "pqrs,pa,sr,qa->",
            self.h13_aqrs,
            za,
            w1,
            w1_qa,
            optimize="greedy",
        )

        energy += 0.25 * np.einsum(
            "pqrs,pq,ra,sb,ba->",
            self.h40_aaaa,
            za,
            za,
            za,
            w1_aa,
            optimize="greedy",
        )
        energy += (1.0 / 24.0) * np.einsum(
            "pqrs,pa,qb,rc,sd,dcba->",
            self.h40_aaaa,
            za,
            za,
            za,
            za,
            w2_dcba,
            optimize="greedy",
        )
        energy += 0.125 * np.einsum(
            "pqrs,pa,qb,rc,sd,dc,ba->",
            self.h40_aaaa,
            za,
            za,
            za,
            za,
            w1_aa,
            w1_aa,
            optimize="greedy",
        )
        return complex(energy), transformed_constant


def evaluate_similarity_transformed_energy(
    hamiltonian: QPHamiltonian,
    z: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> tuple[complex, complex]:
    """One-shot scalar Wick evaluation used by tests and small applications."""

    kernel = ScalarSimilarityTransformKernel.from_deexcitations(hamiltonian, (z,))
    return kernel.evaluate(z, w1, w2)
