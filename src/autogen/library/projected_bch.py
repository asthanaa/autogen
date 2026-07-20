from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import combinations, permutations, product
import math
import os
from typing import Iterable, Sequence


@dataclass(frozen=True)
class _OpToken:
    kind: str
    label: str
    index: int


@dataclass(frozen=True)
class SymbolicTerm:
    coeff: float
    factors: tuple[tuple[str, tuple[str, ...]], ...]
    creators: tuple[str, ...]
    annihilators: tuple[str, ...]
    origin: tuple[str, ...] = ()

    def renamed(self, suffix: str) -> "SymbolicTerm":
        mapping: dict[str, str] = {}
        suffix_tag = {"A": "0", "B": "1"}.get(suffix, suffix)

        def rename(label: str) -> str:
            if label not in mapping:
                mapping[label] = f"{label}{suffix_tag}"
            return mapping[label]

        factors = tuple(
            (name, tuple(rename(label) for label in labels))
            for name, labels in self.factors
        )
        creators = tuple(rename(label) for label in self.creators)
        annihilators = tuple(rename(label) for label in self.annihilators)
        return SymbolicTerm(
            coeff=self.coeff,
            factors=factors,
            creators=creators,
            annihilators=annihilators,
            origin=self.origin,
        )

    def scaled(self, factor: float) -> "SymbolicTerm":
        return SymbolicTerm(
            coeff=factor * self.coeff,
            factors=self.factors,
            creators=self.creators,
            annihilators=self.annihilators,
            origin=self.origin,
        )

    @property
    def n_creators(self) -> int:
        return len(self.creators)

    @property
    def n_annihilators(self) -> int:
        return len(self.annihilators)


def build_projected_qp_ccsd_terms(
    max_order: int = 4,
) -> dict[str, list[tuple[str, list[tuple[str, str]], float]]]:
    # Projection introduces external indices p,q,r,s. Rename the BCH term's
    # internal dummy labels away from those names first so mixed delta/Z
    # contractions cannot accidentally collapse onto the external projector
    # indices.
    raw_terms = [_rename_internal_labels(term) for term in _build_hbar_terms(max_order=max_order)]
    return {
        "scalar": _project_terms(raw_terms, (), ""),
        "X1": _project_terms(raw_terms, ("q", "p"), "pq"),
        "X2": _project_terms(raw_terms, ("s", "r", "q", "p"), "pqrs"),
    }


def build_projected_qp_ccsd_detailed_terms(
    max_order: int = 4,
) -> dict[str, list[dict]]:
    raw_terms = [_rename_internal_labels(term) for term in _build_hbar_terms(max_order=max_order)]
    return {
        "scalar": _project_terms_impl(raw_terms, (), "", detailed=True),
        "X1": _project_terms_impl(raw_terms, ("q", "p"), "pq", detailed=True),
        "X2": _project_terms_impl(raw_terms, ("s", "r", "q", "p"), "pqrs", detailed=True),
    }


@contextlib.contextmanager
def _suppress_output() -> None:
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


@contextlib.contextmanager
def _bog_qp_context() -> None:
    prev = os.environ.get("AUTOGEN_BOGOLIUBOV_QP")
    os.environ["AUTOGEN_BOGOLIUBOV_QP"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AUTOGEN_BOGOLIUBOV_QP", None)
        else:
            os.environ["AUTOGEN_BOGOLIUBOV_QP"] = prev


def _map_autogen_factor_name(name: str) -> str:
    if name.startswith("T1"):
        return "t1"
    if name.startswith("T2"):
        return "t2"
    if name.startswith("H") and len(name) >= 3 and name[1:].isdigit():
        return f"h{name[1:]}"
    raise ValueError(f"Unsupported autogen factor name: {name}")


def _map_origin_name(name: str) -> str:
    if name.startswith("T1"):
        return "T1"
    if name.startswith("T2"):
        return "T2"
    if name.startswith("H") and len(name) >= 3 and name[1:].isdigit():
        return name
    raise ValueError(f"Unsupported autogen origin name: {name}")


def _label_name(label) -> str:
    return label if isinstance(label, str) else label.name


def _final_op(term):
    if not term.st or not term.st[0]:
        return None
    for op in reversed(term.st[0]):
        if getattr(op, "kind", None) == "op":
            return op
    return None


def _raw_terms_from_op(op_name: str):
    from autogen.library import change_terms, make_op

    dict_ind = {}
    ops, dict_ind = make_op.make_op([op_name], dict_ind)
    if not ops:
        return []
    ops[0].map_org = ops
    terms = change_terms.change_terms1(ops[0].st, ops[0].co, 1.0, dict_ind, ops)
    for term in terms:
        term.compress()
        term.build_map_org()
    return terms


def _autogen_term_to_symbolic(term, scale: float) -> SymbolicTerm:
    op = _final_op(term)
    if op is None:
        creators: tuple[str, ...] = ()
        annihilators: tuple[str, ...] = ()
    else:
        creators = tuple(_label_name(label) for label in op.upper)
        annihilators = tuple(_label_name(label) for label in op.lower)

    factors = []
    origin = []
    for op_obj, coeff in zip(term.large_op_list, term.coeff_list):
        name = _map_autogen_factor_name(op_obj.name)
        labels = tuple(_label_name(label) for label in coeff)
        factors.append((name, labels))
        origin.append(_map_origin_name(op_obj.name))

    return SymbolicTerm(
        coeff=scale * term.fac,
        factors=tuple(factors),
        creators=creators,
        annihilators=annihilators,
        origin=tuple(origin),
    )


def _build_autogen_hbar_terms(max_order: int) -> list[SymbolicTerm]:
    from autogen.main_tools import commutator as comm

    h_ops = ("H11", "H20", "H02", "H22", "H31", "H13", "H40", "H04")
    t1_labels = ("T1qp", "T1qp1", "T1qp2", "T1qp3")
    t2_labels = ("T2qp", "T2qp1")

    built_terms: list[SymbolicTerm] = []
    with _bog_qp_context(), _suppress_output():
        for h_op in h_ops:
            for n in range(max_order + 1):
                factor = 1.0 / math.factorial(n)
                for seq in product(("T1", "T2"), repeat=n):
                    t1_count = seq.count("T1")
                    t2_count = n - t1_count
                    if t1_count > len(t1_labels) or t2_count > len(t2_labels):
                        continue
                    t1_idx = 0
                    t2_idx = 0
                    t_ops = []
                    for kind in seq:
                        if kind == "T1":
                            t_ops.append(t1_labels[t1_idx])
                            t1_idx += 1
                        else:
                            t_ops.append(t2_labels[t2_idx])
                            t2_idx += 1
                    if not t_ops:
                        h_terms = _raw_terms_from_op(h_op)
                    else:
                        h_terms = [h_op]
                    for t_op in t_ops:
                        h_terms = comm.comm(h_terms, [t_op], last=0)
                        if not h_terms:
                            break
                    if not h_terms:
                        continue
                    for term in h_terms:
                        built_terms.append(_autogen_term_to_symbolic(term, factor))
    return built_terms


def _build_hbar_terms(max_order: int) -> list[SymbolicTerm]:
    h_terms = _make_h_terms()
    t_terms = _make_t_terms()
    current = h_terms
    collected: list[SymbolicTerm] = []

    for order in range(0, max_order + 1):
        factor = 1.0 / math.factorial(order)
        collected.extend(term.scaled(factor) for term in current)
        if order == max_order:
            break
        current = _prune_terms(
            _commutator(current, t_terms),
            remaining=max_order - order - 1,
        )

    return collected


def _make_h_terms() -> list[SymbolicTerm]:
    return [
        SymbolicTerm(1.0, (("h11", ("p", "q")),), ("p",), ("q",), ("H11",)),
        SymbolicTerm(0.5, (("h20", ("p", "q")),), ("p", "q"), (), ("H20",)),
        SymbolicTerm(0.5, (("h02", ("p", "q")),), (), ("q", "p"), ("H02",)),
        SymbolicTerm(
            0.25,
            (("h22", ("p", "q", "r", "s")),),
            ("p", "q"),
            ("s", "r"),
            ("H22",),
        ),
        SymbolicTerm(
            1.0 / 6.0,
            (("h31", ("p", "q", "r", "s")),),
            ("p", "q", "r"),
            ("s",),
            ("H31",),
        ),
        SymbolicTerm(
            1.0 / 6.0,
            (("h13", ("p", "q", "r", "s")),),
            ("p",),
            ("q", "r", "s"),
            ("H13",),
        ),
        SymbolicTerm(
            1.0 / 24.0,
            (("h40", ("p", "q", "r", "s")),),
            ("p", "q", "r", "s"),
            (),
            ("H40",),
        ),
        SymbolicTerm(
            1.0 / 24.0,
            (("h04", ("p", "q", "r", "s")),),
            (),
            ("p", "q", "r", "s"),
            ("H04",),
        ),
    ]


def _make_t_terms() -> list[SymbolicTerm]:
    return [
        SymbolicTerm(0.5, (("t1", ("p", "q")),), ("p", "q"), (), ("T1",)),
        SymbolicTerm(
            1.0 / 24.0,
            (("t2", ("p", "q", "r", "s")),),
            ("p", "q", "r", "s"),
            (),
            ("T2",),
        ),
    ]


def _commutator(
    terms_a: Iterable[SymbolicTerm],
    terms_b: Iterable[SymbolicTerm],
) -> list[SymbolicTerm]:
    terms: list[SymbolicTerm] = []
    for a in terms_a:
        for b in terms_b:
            terms.extend(_wick_product(a, b))
            for term in _wick_product(b, a):
                terms.append(term.scaled(-1.0))
    return terms


def _wick_product(a_in: SymbolicTerm, b_in: SymbolicTerm) -> list[SymbolicTerm]:
    # Ordinary BCH contractions only: annihilator from the left factor with
    # creator from the right factor.
    a = a_in.renamed("A")
    b = b_in.renamed("B")

    token_index = 0
    a_tokens = []
    for label in a.creators:
        a_tokens.append(_OpToken("c", label, token_index))
        token_index += 1
    for label in a.annihilators:
        a_tokens.append(_OpToken("a", label, token_index))
        token_index += 1
    b_tokens = []
    for label in b.creators:
        b_tokens.append(_OpToken("c", label, token_index))
        token_index += 1
    for label in b.annihilators:
        b_tokens.append(_OpToken("a", label, token_index))
        token_index += 1
    ops = a_tokens + b_tokens

    a_ann = list(a.annihilators)
    b_cre = list(b.creators)
    a_ann_tokens = {
        token.label: token
        for token in ops
        if token.kind == "a" and token.label in a_ann
    }
    b_cre_tokens = {
        token.label: token
        for token in ops
        if token.kind == "c" and token.label in b_cre
    }

    results: list[SymbolicTerm] = []
    max_k = min(len(a_ann), len(b_cre))
    for k in range(0, max_k + 1):
        for a_sel in combinations(a_ann, k):
            for b_sel in combinations(b_cre, k):
                for perm in permutations(range(k)):
                    pairs = [(a_sel[i], b_sel[perm[i]]) for i in range(k)]
                    pair_tokens = [
                        (a_ann_tokens[a_label], b_cre_tokens[b_label])
                        for a_label, b_label in pairs
                    ]
                    sign = _contraction_sign(ops, pair_tokens)
                    results.append(_contract_terms(a, b, pairs, sign))
    return results


def _contraction_sign(
    ops_in: Sequence[_OpToken],
    pair_tokens: Sequence[tuple[_OpToken, _OpToken]],
) -> int:
    ops = list(ops_in)
    sign = 1
    sorted_pairs = sorted(
        pair_tokens,
        key=lambda pair: min(ops_in.index(pair[0]), ops_in.index(pair[1])),
    )
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


def _apply_label_map(
    factors: tuple[tuple[str, tuple[str, ...]], ...],
    mapping: dict[str, str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not mapping:
        return factors
    return tuple(
        (name, tuple(mapping.get(label, label) for label in labels))
        for name, labels in factors
    )


def _contract_terms(
    a: SymbolicTerm,
    b: SymbolicTerm,
    pairs: Sequence[tuple[str, str]],
    sign: int,
) -> SymbolicTerm:
    contracted_a = {a_label for a_label, _ in pairs}
    contracted_b = {b_label for _, b_label in pairs}
    shared_map = {b_label: a_label for a_label, b_label in pairs}

    factors = list(a.factors)
    factors.extend(_apply_label_map(b.factors, shared_map))

    creators = tuple(label for label in a.creators if label not in contracted_a) + tuple(
        shared_map.get(label, label)
        for label in b.creators
        if label not in contracted_b
    )
    annihilators = tuple(
        label for label in a.annihilators if label not in contracted_a
    ) + tuple(
        shared_map.get(label, label)
        for label in b.annihilators
        if label not in contracted_b
    )

    return SymbolicTerm(
        coeff=sign * a.coeff * b.coeff,
        factors=tuple(factors),
        creators=creators,
        annihilators=annihilators,
        origin=a.origin + b.origin,
    )


def _prune_terms(terms: Iterable[SymbolicTerm], remaining: int) -> list[SymbolicTerm]:
    # T only adds creators. Terms that already have too many creators relative
    # to the available annihilators can never contribute to scalar/R1/R2
    # projected kernels.
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


def _project_terms(
    raw_terms: Iterable[SymbolicTerm],
    projector_annihilators: tuple[str, ...],
    out_labels: str,
) -> list[tuple[str, list[tuple[str, str]], float]]:
    return _project_terms_impl(raw_terms, projector_annihilators, out_labels, detailed=False)


def _project_terms_impl(
    raw_terms: Iterable[SymbolicTerm],
    projector_annihilators: tuple[str, ...],
    out_labels: str,
    *,
    detailed: bool,
):
    structs = []
    for term in raw_terms:
        structs.extend(_project_term(term, projector_annihilators, out_labels, detailed=detailed))
    return structs


def _project_term(
    term: SymbolicTerm,
    projector_annihilators: tuple[str, ...],
    out_labels: str,
    *,
    detailed: bool,
) -> list:
    ops = []
    token_index = 0
    for label in projector_annihilators:
        ops.append(_OpToken("a", label, token_index))
        token_index += 1
    for label in term.creators:
        ops.append(_OpToken("c", label, token_index))
        token_index += 1
    for label in term.annihilators:
        ops.append(_OpToken("a", label, token_index))
        token_index += 1

    if len(ops) % 2:
        return []

    used = [False] * len(ops)
    pair_tokens: list[tuple[_OpToken, _OpToken]] = []
    structs: list[tuple[str, list[tuple[str, str]], float]] = []

    def dfs() -> None:
        try:
            left_idx = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            sign = _contraction_sign(ops, pair_tokens)
            struct = _projected_struct_from_pairs(term, pair_tokens, sign, out_labels, detailed=detailed)
            if struct is not None:
                structs.append(struct)
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
            pair_tokens.append((left, right))
            dfs()
            pair_tokens.pop()
            used[right_idx] = False
        used[left_idx] = False

    dfs()
    return structs


def _rename_internal_labels(term: SymbolicTerm) -> SymbolicTerm:
    ordered_labels: list[str] = []
    for _name, labels in term.factors:
        for label in labels:
            if label not in ordered_labels:
                ordered_labels.append(label)
    for label in term.creators + term.annihilators:
        if label not in ordered_labels:
            ordered_labels.append(label)

    mapping = {label: f"u{idx}" for idx, label in enumerate(ordered_labels)}
    factors = tuple(
        (name, tuple(mapping[label] for label in labels))
        for name, labels in term.factors
    )
    creators = tuple(mapping[label] for label in term.creators)
    annihilators = tuple(mapping[label] for label in term.annihilators)
    return SymbolicTerm(
        coeff=term.coeff,
        factors=factors,
        creators=creators,
        annihilators=annihilators,
        origin=term.origin,
    )


def _projected_struct_from_pairs(
    term: SymbolicTerm,
    pair_tokens: Sequence[tuple[_OpToken, _OpToken]],
    sign: int,
    out_labels: str,
    *,
    detailed: bool,
):
    delta_map: dict[str, str] = {}
    z_factors: list[tuple[str, tuple[str, ...]]] = []

    for left, right in pair_tokens:
        if right.kind == "c":
            delta_map[right.label] = left.label
        else:
            z_factors.append(("z", (left.label, right.label)))

    factors = list(_apply_label_map(term.factors, delta_map))
    factors.extend(z_factors)
    coeff = sign * term.coeff
    if abs(coeff) < 1e-12:
        return None

    struct_factors = [(name, "".join(labels)) for name, labels in factors]
    if not detailed:
        return out_labels, struct_factors, coeff
    return {
        "output_labels": out_labels,
        "tensors": [{"name": name, "labels": labels} for name, labels in struct_factors],
        "coeff": coeff,
        "origin": list(term.origin),
        "origin_key": "_".join(term.origin),
        "hamiltonian_block": next((name.lower() for name in term.origin if name.startswith("H")), "unknown"),
        "bch_order": sum(1 for name in term.origin if name in {"T1", "T2"}),
    }
