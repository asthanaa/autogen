"""Reference Wick-engine evaluator for qp BCH R2 (4,0) block."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import math
import re
import string
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Term:
    tensor: np.ndarray
    labels: Tuple[str, ...]
    creators: Tuple[str, ...]
    annihilators: Tuple[str, ...]
    origin: Tuple[str, ...] = ()

    def renamed(self, prefix: str) -> "Term":
        mapping = {label: f"{prefix}{label}" for label in self.labels}
        labels = tuple(mapping[label] for label in self.labels)
        creators = tuple(mapping[label] for label in self.creators)
        annihilators = tuple(mapping[label] for label in self.annihilators)
        return Term(self.tensor, labels, creators, annihilators, self.origin)

    def scaled(self, factor: float) -> "Term":
        return Term(self.tensor * factor, self.labels, self.creators, self.annihilators, self.origin)

    @property
    def n_creators(self) -> int:
        return len(self.creators)

    @property
    def n_annihilators(self) -> int:
        return len(self.annihilators)


@dataclass(frozen=True)
class OpToken:
    kind: str  # 'c' or 'a'
    label: str
    index: int


_LABEL_SORT_RE = re.compile(r"([A-Za-z]+)(\d*)$")


def _count_inversions(order: Sequence[int]) -> int:
    inversions = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inversions += 1
    return inversions


_AXIS_PERM_SIGNS = {
    n: [(perm, -1 if (_count_inversions(perm) % 2) else 1) for perm in permutations(range(n))]
    for n in range(5)
}


def bch_ref_block(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    n_creators: int,
    max_order: int = 4,
    contraction_provider=None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute the (n_creators,0) block of Hbar via explicit BCH."""
    blocks, parts = bch_ref_blocks(
        ham, t1, t2, targets=(n_creators,), max_order=max_order, contraction_provider=contraction_provider
    )
    return blocks[n_creators], parts[n_creators]


def bch_ref_blocks(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    targets: Iterable[int],
    max_order: int = 4,
    contraction_provider=None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Compute multiple (n_creators,0) blocks of Hbar in one BCH pass."""
    target_list = sorted(set(targets))
    h_terms = _make_h_terms(ham)
    t_terms = _make_t_terms(t1, t2)

    n = _infer_n(t1, t2)
    dtype = np.result_type(t1, t2)
    blocks: Dict[int, np.ndarray] = {}
    parts: Dict[int, Dict[str, np.ndarray]] = {}
    for n_cre in target_list:
        if n_cre == 0:
            blocks[n_cre] = np.array(0.0, dtype=dtype)
        else:
            blocks[n_cre] = np.zeros((n,) * n_cre, dtype=dtype)
        parts[n_cre] = {}

    current = h_terms
    for order in range(0, max_order + 1):
        factor = 1.0 / math.factorial(order)
        for term in current:
            if term.n_annihilators != 0:
                continue
            if term.n_creators not in blocks:
                continue
            tensor = _extract_term_block_tensor(term, term.n_creators, 0)
            key = _origin_key(term)
            block = blocks[term.n_creators]
            if key not in parts[term.n_creators]:
                parts[term.n_creators][key] = np.zeros_like(block)
            parts[term.n_creators][key] += factor * tensor
            blocks[term.n_creators] += factor * tensor
        if order == max_order:
            break
        current = _prune_terms(
            _commutator(current, t_terms, contraction_provider),
            remaining=max_order - order - 1,
        )

    return blocks, parts


def bch_ref_operator_blocks(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    ranks: Iterable[Tuple[int, int]],
    max_order: int = 4,
    contraction_provider=None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], Dict[str, np.ndarray]]]:
    """Compute normal-ordered Hbar blocks for the requested (n_creators, n_annihilators) ranks."""
    rank_list = sorted(set(ranks))
    if not rank_list:
        return {}, {}

    h_terms = _make_h_terms(ham)
    t_terms = _make_t_terms(t1, t2)

    n = _infer_n(t1, t2)
    dtype = np.result_type(t1, t2)
    blocks: Dict[Tuple[int, int], np.ndarray] = {}
    parts: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for rank in rank_list:
        n_cre, n_ann = rank
        if n_cre == 0 and n_ann == 0:
            blocks[rank] = np.array(0.0, dtype=dtype)
        else:
            blocks[rank] = np.zeros((n,) * (n_cre + n_ann), dtype=dtype)
        parts[rank] = {}

    max_cre = max(rank[0] for rank in rank_list)
    max_ann = max(rank[1] for rank in rank_list)

    current = h_terms
    for order in range(0, max_order + 1):
        factor = 1.0 / math.factorial(order)
        for term in current:
            rank = (term.n_creators, term.n_annihilators)
            if rank not in blocks:
                continue
            # For ordinary normal-ordered operator blocks we want the block
            # coefficient tensor itself, not a vacuum matrix element with
            # external projectors. Using the latter is only appropriate for the
            # creator-only (n, 0) blocks.
            tensor = _tensor_for_operator_block(term)
            key = _origin_key(term)
            block = blocks[rank]
            if key not in parts[rank]:
                parts[rank][key] = np.zeros_like(block)
            parts[rank][key] += factor * tensor
            blocks[rank] += factor * tensor
        if order == max_order:
            break
        current = _prune_rank_terms(
            _commutator(current, t_terms, contraction_provider),
            remaining=max_order - order - 1,
            max_cre=max_cre,
            max_ann=max_ann,
        )

    return blocks, parts


def r2_bch_ref(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    max_order: int = 4,
    contraction_provider=None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute R2 = Hbar^{40} via explicit BCH with Wick contractions."""
    return bch_ref_block(
        ham, t1, t2, n_creators=4, max_order=max_order, contraction_provider=contraction_provider
    )


def _make_h_terms(ham: Dict[str, np.ndarray]) -> List[Term]:
    terms = []
    if "h11" in ham:
        terms.append(Term(ham["h11"], ("p", "q"), ("p",), ("q",), ("H11",)))
    if "h20" in ham:
        terms.append(Term(0.5 * ham["h20"], ("p", "q"), ("p", "q"), (), ("H20",)))
    if "h02" in ham:
        terms.append(Term(0.5 * ham["h02"], ("p", "q"), (), ("q", "p"), ("H02",)))
    if "h22" in ham:
        terms.append(Term(0.25 * ham["h22"], ("p", "q", "r", "s"), ("p", "q"), ("s", "r"), ("H22",)))
    if "h31" in ham:
        terms.append(Term((1.0 / 6.0) * ham["h31"], ("p", "q", "r", "s"), ("p", "q", "r"), ("s",), ("H31",)))
    if "h13" in ham:
        # h13 is the Hermitian adjoint of h31 with the tensor convention used in
        # build_bogoliubov_hamiltonian(), so it carries the same +1/6 prefactor.
        terms.append(Term((1.0 / 6.0) * ham["h13"], ("p", "q", "r", "s"), ("p",), ("q", "r", "s"), ("H13",)))
    if "h40" in ham:
        terms.append(Term((1.0 / 24.0) * ham["h40"], ("p", "q", "r", "s"), ("p", "q", "r", "s"), (), ("H40",)))
    if "h04" in ham:
        # h04 follows the direct Hermitian-adjoint convention of h40, so the
        # annihilator ordering matches the tensor-label ordering.
        terms.append(Term((1.0 / 24.0) * ham["h04"], ("p", "q", "r", "s"), (), ("p", "q", "r", "s"), ("H04",)))
    return terms


def _make_t_terms(t1: np.ndarray, t2: np.ndarray) -> List[Term]:
    terms = []
    if t1 is not None:
        terms.append(Term(0.5 * t1, ("p", "q"), ("p", "q"), (), ("T1",)))
    if t2 is not None:
        terms.append(Term((1.0 / 24.0) * t2, ("p", "q", "r", "s"), ("p", "q", "r", "s"), (), ("T2",)))
    return terms


def _commutator(
    terms_a: Sequence[Term], terms_b: Sequence[Term], contraction_provider=None
) -> List[Term]:
    terms = []
    for a in terms_a:
        for b in terms_b:
            terms.extend(_wick_product(a, b, contraction_provider))
            for term in _wick_product(b, a, contraction_provider):
                terms.append(term.scaled(-1.0))
    return terms


def _wick_product(a_in: Term, b_in: Term, contraction_provider=None) -> List[Term]:
    a = a_in.renamed("A")
    b = b_in.renamed("B")

    token_index = 0
    a_tokens = []
    for label in a.creators:
        a_tokens.append(OpToken("c", label, token_index))
        token_index += 1
    for label in a.annihilators:
        a_tokens.append(OpToken("a", label, token_index))
        token_index += 1
    b_tokens = []
    for label in b.creators:
        b_tokens.append(OpToken("c", label, token_index))
        token_index += 1
    for label in b.annihilators:
        b_tokens.append(OpToken("a", label, token_index))
        token_index += 1
    ops = a_tokens + b_tokens

    if contraction_provider is None:
        a_ann = list(a.annihilators)
        b_cre = list(b.creators)
        a_ann_tokens = {
            token.label: token for token in ops if token.kind == "a" and token.label in a_ann
        }
        b_cre_tokens = {
            token.label: token for token in ops if token.kind == "c" and token.label in b_cre
        }
        results = []
        max_k = min(len(a_ann), len(b_cre))
        for k in range(0, max_k + 1):
            for a_sel in combinations(a_ann, k):
                for b_sel in combinations(b_cre, k):
                    for perm in permutations(range(k)):
                        pairs = [(a_sel[i], b_sel[perm[i]]) for i in range(k)]
                        pair_tokens = [(a_ann_tokens[al], b_cre_tokens[bl]) for al, bl in pairs]
                        sign = _contraction_sign(ops, pair_tokens)
                        results.append(_contract_terms(a, b, pairs, sign))
        return results

    results = []
    max_k = min(len(a_tokens), len(b_tokens))
    for k in range(0, max_k + 1):
        for a_sel in combinations(a_tokens, k):
            for b_sel in combinations(b_tokens, k):
                for perm in permutations(range(k)):
                    pair_tokens = [(a_sel[i], b_sel[perm[i]]) for i in range(k)]
                    pair_labels = []
                    contractions = []
                    skip = False
                    for left_tok, right_tok in pair_tokens:
                        if hasattr(contraction_provider, "get"):
                            contr = contraction_provider.get(left_tok.kind, right_tok.kind)
                        else:
                            contr = contraction_provider(left_tok.kind, right_tok.kind)
                        if contr is None:
                            skip = True
                            break
                        pair_labels.append((left_tok.label, right_tok.label))
                        contractions.append(contr)
                    if skip:
                        continue
                    sign = _contraction_sign(ops, pair_tokens)
                    results.append(_contract_terms(a, b, pair_labels, sign, contractions))
    return results


def _contraction_sign(ops_in: Sequence[OpToken], pair_tokens: Sequence[Tuple[OpToken, OpToken]]) -> int:
    ops = list(ops_in)
    sign = 1
    sorted_pairs = sorted(pair_tokens, key=lambda pair: min(ops_in.index(pair[0]), ops_in.index(pair[1])))
    for tok_left, tok_right in sorted_pairs:
        pos_a = ops.index(tok_left)
        pos_c = ops.index(tok_right)
        if pos_a > pos_c:
            pos_a, pos_c = pos_c, pos_a
        distance = pos_c - pos_a - 1
        if distance % 2:
            sign *= -1
        ops.pop(pos_c)
        ops.pop(pos_a)
    inversions = 0
    seen_ann = 0
    for tok in ops:
        if tok.kind == "a":
            seen_ann += 1
        else:
            inversions += seen_ann
    if inversions % 2:
        sign *= -1
    return sign


def _contract_terms(
    a: Term,
    b: Term,
    pairs: Sequence[Tuple[str, str]],
    sign: int,
    contractions: Sequence[np.ndarray] | None = None,
) -> Term:
    contracted_a = {a_label for a_label, _ in pairs}
    contracted_b = {b_label for _, b_label in pairs}

    labels_a = list(a.labels)
    labels_b = list(b.labels)

    if contractions is None:
        shared_map = {b_label: a_label for a_label, b_label in pairs}
        labels_b = [shared_map.get(lbl, lbl) for lbl in labels_b]
        out_labels = [lbl for lbl in labels_a + labels_b if lbl not in contracted_a]
        tensor = _einsum_merge(a.tensor, labels_a, b.tensor, labels_b, out_labels)
        tensor = tensor * sign
        creators = tuple(lbl for lbl in a.creators if lbl not in contracted_a) + tuple(
            shared_map.get(lbl, lbl) for lbl in b.creators if lbl not in contracted_b
        )
        annihilators = tuple(lbl for lbl in a.annihilators if lbl not in contracted_a) + tuple(
            shared_map.get(lbl, lbl) for lbl in b.annihilators if lbl not in contracted_b
        )
        return Term(np.asarray(tensor), tuple(out_labels), creators, annihilators, a.origin + b.origin)

    out_labels = [lbl for lbl in labels_a if lbl not in contracted_a] + [
        lbl for lbl in labels_b if lbl not in contracted_b
    ]

    tensors = [a.tensor, b.tensor]
    labels_list = [labels_a, labels_b]
    for (a_label, b_label), contr in zip(pairs, contractions):
        tensors.append(contr)
        labels_list.append([a_label, b_label])

    tensor = _einsum_merge_many(tensors, labels_list, out_labels)
    tensor = tensor * sign

    creators = tuple(lbl for lbl in a.creators if lbl not in contracted_a) + tuple(
        lbl for lbl in b.creators if lbl not in contracted_b
    )
    annihilators = tuple(lbl for lbl in a.annihilators if lbl not in contracted_a) + tuple(
        lbl for lbl in b.annihilators if lbl not in contracted_b
    )
    return Term(np.asarray(tensor), tuple(out_labels), creators, annihilators, a.origin + b.origin)


def _einsum_merge(
    a: np.ndarray,
    labels_a: Sequence[str],
    b: np.ndarray,
    labels_b: Sequence[str],
    out_labels: Sequence[str],
) -> np.ndarray:
    if not labels_a and not labels_b:
        return a * b
    if not labels_a:
        return b * a
    if not labels_b:
        return a * b

    unique_labels = []
    for label in list(labels_a) + list(labels_b):
        if label not in unique_labels:
            unique_labels.append(label)
    symbols = _label_symbols(len(unique_labels))
    label_map = {label: sym for label, sym in zip(unique_labels, symbols)}

    subs_a = "".join(label_map[label] for label in labels_a)
    subs_b = "".join(label_map[label] for label in labels_b)
    subs_out = "".join(label_map[label] for label in out_labels)
    expr = f"{subs_a},{subs_b}->{subs_out}"
    return np.einsum(expr, a, b, optimize=True)


def _einsum_merge_many(
    tensors: Sequence[np.ndarray],
    labels_list: Sequence[Sequence[str]],
    out_labels: Sequence[str],
) -> np.ndarray:
    if not tensors:
        return np.array(1.0)
    if len(tensors) == 1:
        return tensors[0]

    unique_labels = []
    for labels in labels_list:
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
    symbols = _label_symbols(len(unique_labels))
    label_map = {label: sym for label, sym in zip(unique_labels, symbols)}

    subs = ["".join(label_map[label] for label in labels) for labels in labels_list]
    subs_out = "".join(label_map[label] for label in out_labels)
    expr = f"{','.join(subs)}->{subs_out}"
    return np.einsum(expr, *tensors, optimize=True)


def _antisym_tensor_axes(tensor: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    axes = tuple(axes)
    if len(axes) <= 1:
        return np.asarray(tensor)
    if len(axes) not in _AXIS_PERM_SIGNS:
        raise ValueError(f"Unsupported antisym rank: {len(axes)}")

    out = np.zeros_like(tensor)
    for perm, sign in _AXIS_PERM_SIGNS[len(axes)]:
        perm_axes = list(range(tensor.ndim))
        for src_pos, dst_pos in enumerate(perm):
            perm_axes[axes[src_pos]] = axes[dst_pos]
        out += sign * tensor.transpose(perm_axes)
    return out


def _canonicalize_term(term: Term) -> Term:
    tensor = np.asarray(term.tensor)
    if tensor.ndim == 0:
        return term

    label_to_axis = {label: idx for idx, label in enumerate(term.labels)}
    creator_sources = {label[0] for label in term.creators if label and label[0] in {"A", "B"}}
    if term.n_creators > 1 and len(creator_sources) > 1:
        tensor = _antisym_tensor_axes(tensor, [label_to_axis[label] for label in term.creators])
    return Term(tensor, term.labels, term.creators, term.annihilators, term.origin)


def _canonicalize_terms(terms: Iterable[Term]) -> List[Term]:
    return [_canonicalize_term(term) for term in terms]


def _label_symbols(n: int) -> List[str]:
    symbols = list(string.ascii_lowercase + string.ascii_uppercase)
    if n > len(symbols):
        raise ValueError("Too many labels for einsum symbol set.")
    return symbols[:n]


def _tensor_for_creators(term: Term) -> np.ndarray:
    if term.tensor.ndim == 0:
        return term.tensor
    label_to_axis = {label: idx for idx, label in enumerate(term.labels)}
    axes = [label_to_axis[label] for label in term.creators]
    return term.tensor.transpose(axes)


def _tensor_for_operator_block(term: Term) -> np.ndarray:
    if term.tensor.ndim == 0:
        return term.tensor

    sorted_creators = tuple(sorted(term.creators, key=_label_sort_key))
    sorted_annihilators = tuple(sorted(term.annihilators, key=_label_sort_key))

    label_to_axis = {label: idx for idx, label in enumerate(term.labels)}
    axes = [label_to_axis[label] for label in sorted_creators + sorted_annihilators]
    tensor = term.tensor.transpose(axes)

    sign = _perm_sign_to_sorted(term.creators, sorted_creators)
    sign *= _perm_sign_to_sorted(term.annihilators, sorted_annihilators)
    return sign * tensor


def _extract_term_block_tensor(term: Term, n_left: int, n_right: int) -> np.ndarray:
    left_labels = tuple(f"L{i}" for i in range(n_left))
    right_labels = tuple(f"R{i}" for i in range(n_right))

    token_index = 0
    ops: List[OpToken] = []
    for label in left_labels:
        ops.append(OpToken("a", label, token_index))
        token_index += 1
    for label in term.creators:
        ops.append(OpToken("c", label, token_index))
        token_index += 1
    for label in term.annihilators:
        ops.append(OpToken("a", label, token_index))
        token_index += 1
    for label in right_labels:
        ops.append(OpToken("c", label, token_index))
        token_index += 1

    if len(ops) % 2:
        raise ValueError("Cannot extract odd-rank vacuum block from normal-ordered term.")

    n = term.tensor.shape[0] if term.tensor.ndim else 1
    delta = np.eye(n, dtype=np.result_type(term.tensor))
    dtype = np.result_type(term.tensor)
    if n_left + n_right == 0:
        total = np.array(0.0, dtype=dtype)
    else:
        total = np.zeros((n,) * (n_left + n_right), dtype=dtype)

    used = [False] * len(ops)
    pairs: List[Tuple[OpToken, OpToken]] = []

    def dfs() -> None:
        try:
            left_idx = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            sign = _contraction_sign(ops, pairs)
            tensors = [term.tensor]
            labels_list: List[Sequence[str]] = [term.labels]
            for left, right in pairs:
                tensors.append(delta)
                labels_list.append((left.label, right.label))
            contrib = _einsum_merge_many(tensors, labels_list, left_labels + right_labels)
            total[...] += sign * contrib
            return

        left = ops[left_idx]
        if left.kind != "a":
            return

        used[left_idx] = True
        for right_idx in range(left_idx + 1, len(ops)):
            if used[right_idx]:
                continue
            right = ops[right_idx]
            if right.kind != "c":
                continue
            used[right_idx] = True
            pairs.append((left, right))
            dfs()
            pairs.pop()
            used[right_idx] = False
        used[left_idx] = False

    dfs()
    return total


def _label_sort_key(label: str) -> Tuple[str, int, str]:
    match = _LABEL_SORT_RE.fullmatch(label)
    if match is None:
        return (label, 0, label)
    stem, suffix = match.groups()
    return (stem, int(suffix or 0), label)


def _perm_sign_to_sorted(labels: Sequence[str], sorted_labels: Sequence[str]) -> int:
    if tuple(labels) == tuple(sorted_labels):
        return 1
    remaining = list(labels)
    order = []
    for label in sorted_labels:
        idx = remaining.index(label)
        order.append(idx)
        remaining[idx] = None
    inversions = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inversions += 1
    return -1 if (inversions % 2) else 1


def _origin_key(term: Term) -> str:
    h_tag = next((tag for tag in term.origin if tag.startswith("H")), "H?")
    t1_count = sum(tag == "T1" for tag in term.origin)
    t2_count = sum(tag == "T2" for tag in term.origin)
    parts = []
    if t1_count:
        parts.append("T1" * t1_count)
    if t2_count:
        parts.append("T2" * t2_count)
    if not parts:
        return h_tag
    return f"{h_tag}_{''.join(parts)}"


def _prune_terms(terms: Iterable[Term], remaining: int) -> List[Term]:
    pruned = []
    for term in terms:
        n_c = term.n_creators
        n_a = term.n_annihilators
        if n_c > 4:
            continue
        if n_c + 4 * remaining < 4:
            continue
        if n_a > 4 * remaining:
            continue
        pruned.append(term)
    return pruned


def _infer_n(t1: np.ndarray | None, t2: np.ndarray | None) -> int:
    if t1 is not None:
        return t1.shape[0]
    if t2 is not None:
        return t2.shape[0]
    raise ValueError("Cannot infer dimension from t1/t2.")


def projected_bch_ref_blocks(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    z: np.ndarray,
    targets: Iterable[int] = (0, 2, 4),
    max_order: int = 4,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Compute projected qp BCH kernels from ordinary normal-ordered Hbar blocks.

    Contractions follow the projected qp rules:
      <beta_i beta_j> = Z_ij
      <beta_i beta_j^\u2020> = delta_ij
    """
    target_list = sorted(set(targets))
    projector_labels = {
        0: (),
        2: ("P0", "P1"),
        4: ("P0", "P1", "P2", "P3"),
    }
    needed_ranks: set[Tuple[int, int]] = set()
    if 0 in target_list:
        needed_ranks.update({(0, 0), (0, 2), (0, 4)})
    if 2 in target_list:
        needed_ranks.update({(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (2, 0), (2, 2)})
    if 4 in target_list:
        needed_ranks.update({(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (4, 0)})

    op_blocks, op_parts = bch_ref_operator_blocks(
        ham,
        t1,
        t2,
        ranks=needed_ranks,
        max_order=max_order,
    )

    n = _infer_n(t1, t2)
    dtype = np.result_type(t1, t2, z)
    blocks: Dict[int, np.ndarray] = {}
    parts: Dict[int, Dict[str, np.ndarray]] = {}
    for n_cre in target_list:
        if n_cre == 0:
            blocks[n_cre] = np.array(0.0, dtype=dtype)
        else:
            blocks[n_cre] = np.zeros((n,) * n_cre, dtype=dtype)
        parts[n_cre] = {}

    for rank, tensor in op_blocks.items():
        term = _term_from_operator_block(rank, tensor)
        for n_cre in target_list:
            labels = projector_labels.get(n_cre)
            if labels is None:
                raise ValueError(f"Unsupported projected target rank: {n_cre}")
            contrib = _project_normal_ordered_term(term, z, labels)
            if contrib is None:
                continue
            blocks[n_cre] += contrib

    for rank, rank_parts in op_parts.items():
        for key, tensor in rank_parts.items():
            term = _term_from_operator_block(rank, tensor)
            part_key = f"{rank[0]}{rank[1]}:{key}"
            for n_cre in target_list:
                labels = projector_labels.get(n_cre)
                if labels is None:
                    raise ValueError(f"Unsupported projected target rank: {n_cre}")
                contrib = _project_normal_ordered_term(term, z, labels)
                if contrib is None:
                    continue
                if part_key not in parts[n_cre]:
                    parts[n_cre][part_key] = np.zeros_like(blocks[n_cre])
                parts[n_cre][part_key] += contrib

    return blocks, parts


def projected_bch_ref_block(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    z: np.ndarray,
    n_creators: int,
    max_order: int = 4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    blocks, parts = projected_bch_ref_blocks(
        ham,
        t1,
        t2,
        z,
        targets=(n_creators,),
        max_order=max_order,
    )
    return blocks[n_creators], parts[n_creators]


def _build_projected_raw_terms(
    ham: Dict[str, np.ndarray],
    t1: np.ndarray,
    t2: np.ndarray,
    max_order: int,
) -> List[Term]:
    h_terms = _make_h_terms(ham)
    t_terms = _make_t_terms(t1, t2)

    current = h_terms
    collected: List[Term] = []
    for order in range(0, max_order + 1):
        factor = 1.0 / math.factorial(order)
        collected.extend(term.scaled(factor) for term in current)
        if order == max_order:
            break
        current = _prune_projected_terms(
            _commutator(current, t_terms, contraction_provider=None),
            remaining=max_order - order - 1,
        )
    return collected


def _term_from_operator_block(rank: Tuple[int, int], tensor: np.ndarray) -> Term:
    n_cre, n_ann = rank
    creators = tuple(f"c{i}" for i in range(n_cre))
    annihilators = tuple(f"a{i}" for i in range(n_ann))
    labels = creators + annihilators
    return Term(np.asarray(tensor), labels, creators, annihilators, (f"O{n_cre}{n_ann}",))


def _prune_projected_terms(terms: Iterable[Term], remaining: int) -> List[Term]:
    pruned = []
    for term in terms:
        n_c = term.n_creators
        n_a = term.n_annihilators
        if n_c > 4 + n_a:
            continue
        if n_c > 8:
            continue
        if n_a > 8:
            continue
        if n_c > 4 + n_a + 4 * remaining:
            continue
        pruned.append(term)
    return pruned


def _prune_rank_terms(
    terms: Iterable[Term],
    remaining: int,
    *,
    max_cre: int,
    max_ann: int,
) -> List[Term]:
    """Prune ordinary BCH terms against requested operator-block ranks.

    In Hbar BCH with T containing only creators, the creator count can only stay
    the same or increase, while the annihilator count can only stay the same or
    decrease through contractions. That lets us cut branches much more
    aggressively than the generic projected-term bound.
    """
    pruned = []
    for term in terms:
        n_c = term.n_creators
        n_a = term.n_annihilators
        if n_c > max_cre:
            continue
        if n_a > max_ann + 4 * remaining:
            continue
        pruned.append(term)
    return pruned


def _project_normal_ordered_term(
    term: Term,
    z: np.ndarray,
    projector_labels: Sequence[str],
) -> np.ndarray | None:
    token_index = 0
    ops: List[OpToken] = []
    for label in projector_labels:
        ops.append(OpToken("a", label, token_index))
        token_index += 1
    for label in term.creators:
        ops.append(OpToken("c", label, token_index))
        token_index += 1
    for label in term.annihilators:
        ops.append(OpToken("a", label, token_index))
        token_index += 1

    if len(ops) % 2:
        return None

    n = z.shape[0]
    delta = np.eye(n, dtype=np.result_type(term.tensor, z))
    dtype = np.result_type(term.tensor, z)
    if projector_labels:
        total = np.zeros((n,) * len(projector_labels), dtype=dtype)
    else:
        total = np.array(0.0, dtype=dtype)

    used = [False] * len(ops)
    pairs: List[Tuple[OpToken, OpToken]] = []

    def dfs() -> None:
        try:
            left_idx = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            sign = _contraction_sign(ops, pairs)
            contrib = _project_pairing_contrib(term, z, delta, projector_labels, pairs, sign)
            if contrib is not None:
                total[...] += contrib
            return

        left = ops[left_idx]
        if left.kind != "a":
            return

        used[left_idx] = True
        for right_idx in range(left_idx + 1, len(ops)):
            if used[right_idx]:
                continue
            right = ops[right_idx]
            if right.kind not in {"a", "c"}:
                continue
            used[right_idx] = True
            pairs.append((left, right))
            dfs()
            pairs.pop()
            used[right_idx] = False
        used[left_idx] = False

    dfs()
    if projector_labels:
        total = total.transpose(tuple(reversed(range(len(projector_labels)))))
    return total


def _project_pairing_contrib(
    term: Term,
    z: np.ndarray,
    delta: np.ndarray,
    projector_labels: Sequence[str],
    pairs: Sequence[Tuple[OpToken, OpToken]],
    sign: int,
) -> np.ndarray | None:
    tensors = [term.tensor]
    labels_list: List[Sequence[str]] = [term.labels]
    for left, right in pairs:
        if right.kind == "c":
            tensors.append(delta)
            labels_list.append((left.label, right.label))
        else:
            tensors.append(z)
            labels_list.append((left.label, right.label))
    contrib = _einsum_merge_many(tensors, labels_list, projector_labels)
    return sign * np.asarray(contrib)
