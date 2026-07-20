from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[a-z][0-9]*")
EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
OUTPUT_KEY_ALIASES = {
    "scalar": "energy",
    "X1": "r1",
    "X2": "r2",
    "energy": "energy",
    "r1": "r1",
    "r2": "r2",
}
LABEL_DIM_WEIGHTS = {"o": 2, "v": 4, "p": 4}


StructuredTensor = tuple[str, str]
StructuredTerm = tuple[str, list[StructuredTensor], float]


def deserialize_coefficient(term: dict[str, Any]) -> float:
    """Read the exact-rational schema, with legacy float compatibility."""

    encoded = term.get("coefficient")
    if encoded is not None:
        if not isinstance(encoded, dict):
            raise TypeError("coefficient must be a rational JSON object")
        numerator = int(encoded["numerator"])
        denominator = int(encoded["denominator"])
        if denominator <= 0:
            raise ValueError("coefficient denominator must be positive")
        return float(Fraction(numerator, denominator))
    return float(term["coeff"])


def serialize_coefficient(value: float, *, max_denominator: int = 1 << 20) -> dict[str, int]:
    """Encode a generated floating coefficient as a checked exact rational."""

    fraction = Fraction(float(value)).limit_denominator(max_denominator)
    if abs(float(fraction) - float(value)) > 1.0e-12:
        raise ValueError(f"coefficient {value!r} is not rational within generator tolerance")
    return {
        "numerator": int(fraction.numerator),
        "denominator": int(fraction.denominator),
    }


@dataclass(frozen=True)
class PlannedStep:
    name: str
    expr: str
    arg_refs: tuple[str, ...]
    arg_labels: tuple[str, ...]
    out_labels: str
    contracted_labels: tuple[str, ...]
    formal_scaling: int
    flop_cost: int
    result_rank: int
    result_size: int
    signature: str
    reuse_count: int


@dataclass(frozen=True)
class PlannedTerm:
    name: str
    coeff: float
    output_labels: str
    tensor_signature: tuple[str, ...]
    root_ref: str
    root_labels: str
    steps: tuple[PlannedStep, ...]
    formal_scaling: int
    flop_cost: int
    peak_rank: int
    peak_size: int
    contraction_count: int
    signature: str


@dataclass(frozen=True)
class OutputPlan:
    output_labels: str
    raw_count: int
    grouped_count: int
    terms: tuple[PlannedTerm, ...]
    contraction_count: int
    max_formal_scaling: int
    scaling_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PlannedGroup:
    name: str
    output_key: str
    output_labels: str
    signature: tuple[str, ...]
    raw_count: int
    grouped_count: int
    term_names: tuple[str, ...]
    contraction_count: int
    max_formal_scaling: int


@dataclass(frozen=True)
class ProjectedRuntimePlan:
    outputs: dict[str, OutputPlan]
    intermediates: tuple[PlannedStep, ...]
    groups: tuple[PlannedGroup, ...]
    reduction: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class _TreeNode:
    labels: tuple[str, ...]
    signature: tuple[Any, ...]
    signature_key: str
    children: tuple["_TreeNode", ...]
    leaf_name: str | None
    expr: str | None
    arg_labels: tuple[str, ...]
    contracted_labels: tuple[str, ...]
    formal_scaling: int
    flop_cost: int
    peak_rank: int
    peak_size: int
    contraction_count: int
    step_formal_scaling: int
    step_flop_cost: int
    result_rank: int
    result_size: int
    leaves_count: int


def normalize_output_key(name: str) -> str:
    try:
        return OUTPUT_KEY_ALIASES[name]
    except KeyError as exc:
        raise KeyError(f"Unsupported projected output key {name!r}") from exc


def normalize_projected_output_structs(outputs: dict[str, list[StructuredTerm]]) -> dict[str, list[StructuredTerm]]:
    normalized: dict[str, list[StructuredTerm]] = {}
    for key, terms in outputs.items():
        normalized[normalize_output_key(key)] = terms
    return normalized


def deserialize_structs(terms: Iterable[dict]) -> list[StructuredTerm]:
    out: list[StructuredTerm] = []
    for term in terms:
        out.append(
            (
                term["output_labels"],
                [(tensor["name"], tensor["labels"]) for tensor in term["tensors"]],
                deserialize_coefficient(term),
            )
        )
    return out


def serialize_structs(terms: Iterable[StructuredTerm]) -> list[dict]:
    out = []
    for output_labels, tensors, coeff in terms:
        out.append(
            {
                "output_labels": output_labels,
                "tensors": [{"name": name, "labels": labels} for name, labels in tensors],
                "coefficient": serialize_coefficient(coeff),
            }
        )
    return out


@lru_cache(maxsize=None)
def load_projected_codegen_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def raw_projected_outputs_from_plan(path: str | Path) -> dict[str, list[StructuredTerm]]:
    data = load_projected_codegen_plan(str(path))
    return {
        normalize_output_key(key): deserialize_structs(block["raw_terms"])
        for key, block in data["outputs"].items()
    }


def _split_labels(labels: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(labels))


def _sort_labels(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(labels))


def _parity_from_perm(orig: tuple[str, ...], perm: tuple[str, ...]) -> int:
    idx: list[int] = []
    used = [False] * len(orig)
    for tok in perm:
        for pos, val in enumerate(orig):
            if not used[pos] and val == tok:
                idx.append(pos)
                used[pos] = True
                break
    inversions = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if idx[i] > idx[j]:
                inversions += 1
    return -1 if (inversions % 2) else 1


def canonicalize_projected_tensor(name: str, labels: str) -> tuple[str, int]:
    toks = _split_labels(labels)
    if name in {"t1", "z", "h02", "h20"}:
        perm = _sort_labels(toks)
        return "".join(perm), _parity_from_perm(toks, perm)
    if name == "t2":
        # Bogoliubov T2 creates four quasiparticles.  Unlike particle-hole
        # CCSD doubles, it is one fully antisymmetric rank-four tensor, not a
        # tensor with separate occupied/virtual antisymmetry groups.
        perm = _sort_labels(toks)
        return "".join(perm), _parity_from_perm(toks, perm)
    if name == "h22":
        creators = toks[:2]
        annihilators = toks[2:]
        perm_cre = _sort_labels(creators)
        perm_ann = _sort_labels(annihilators)
        sign = _parity_from_perm(creators, perm_cre) * _parity_from_perm(annihilators, perm_ann)
        return "".join(perm_cre + perm_ann), sign
    if name == "h31":
        creators = toks[:3]
        annihilator = toks[3:]
        perm_cre = _sort_labels(creators)
        return "".join(perm_cre + annihilator), _parity_from_perm(creators, perm_cre)
    if name == "h13":
        creator = toks[:1]
        annihilators = toks[1:]
        perm_ann = _sort_labels(annihilators)
        return "".join(creator + perm_ann), _parity_from_perm(annihilators, perm_ann)
    if name in {"h04", "h40"}:
        perm = _sort_labels(toks)
        return "".join(perm), _parity_from_perm(toks, perm)
    return labels, 1


def _term_key(
    output_labels: str,
    tensors: Iterable[StructuredTensor],
    *,
    canonical: bool,
) -> tuple[str, tuple[StructuredTensor, ...], int]:
    sign = 1
    keyed_tensors: list[StructuredTensor] = []
    for name, labels in tensors:
        if canonical:
            labels, tensor_sign = canonicalize_projected_tensor(name, labels)
            sign *= tensor_sign
        keyed_tensors.append((name, labels))
    keyed_tensors.sort()
    return output_labels, tuple(keyed_tensors), sign


def combine_identical_terms(terms: Iterable[StructuredTerm]) -> list[StructuredTerm]:
    combined: dict[tuple[str, tuple[StructuredTensor, ...]], float] = defaultdict(float)
    for output_labels, tensors, coeff in terms:
        combined[(output_labels, tuple(sorted(tensors)))] += float(coeff)
    out: list[StructuredTerm] = []
    for (output_labels, tensors), coeff in combined.items():
        if abs(coeff) <= 1e-12:
            continue
        out.append((output_labels, list(tensors), coeff))
    return out


def merge_projected_terms(terms: Iterable[StructuredTerm]) -> list[StructuredTerm]:
    merged: dict[tuple[str, tuple[StructuredTensor, ...]], float] = defaultdict(float)
    for output_labels, tensors, coeff in terms:
        out_labels, keyed_tensors, sign = _term_key(output_labels, tensors, canonical=True)
        merged[(out_labels, keyed_tensors)] += sign * float(coeff)
    out: list[StructuredTerm] = []
    for (output_labels, tensors), coeff in merged.items():
        if abs(coeff) <= 1e-12:
            continue
        out.append((output_labels, list(tensors), coeff))
    return out


def _safe_einsum_subs(inputs: list[str], output: str) -> str:
    tokens = []
    for labels in inputs + [output]:
        tokens.extend(_split_labels(labels))
    mapping: dict[str, str] = {}
    symbol_iter = iter(EINSUM_SYMBOLS)
    for token in tokens:
        mapping.setdefault(token, next(symbol_iter))

    def remap(labels: str) -> str:
        return "".join(mapping[token] for token in _split_labels(labels))

    return ",".join(remap(labels) for labels in inputs) + "->" + remap(output)


def _label_weight(token: str) -> int:
    return LABEL_DIM_WEIGHTS.get(token[0], 4)


def _labels_size(tokens: Iterable[str]) -> int:
    size = 1
    for token in tokens:
        size *= _label_weight(token)
    return size


def _global_label_order(output_tokens: tuple[str, ...], tensors: list[StructuredTensor]) -> tuple[str, ...]:
    ordered: list[str] = []
    for token in output_tokens:
        if token not in ordered:
            ordered.append(token)
    for _name, labels in tensors:
        for token in _split_labels(labels):
            if token not in ordered:
                ordered.append(token)
    return tuple(ordered)


def _tree_signature_key(signature: tuple[Any, ...]) -> str:
    return repr(signature)


def _compare_cost(node: _TreeNode) -> tuple[Any, ...]:
    return (
        node.formal_scaling,
        node.flop_cost,
        node.peak_rank,
        node.peak_size,
        node.contraction_count,
        node.signature_key,
    )


def _make_leaf_node(name: str, labels: tuple[str, ...]) -> _TreeNode:
    signature = ("leaf", name, "".join(labels))
    return _TreeNode(
        labels=labels,
        signature=signature,
        signature_key=_tree_signature_key(signature),
        children=(),
        leaf_name=name,
        expr=None,
        arg_labels=(),
        contracted_labels=(),
        formal_scaling=0,
        flop_cost=0,
        peak_rank=len(labels),
        peak_size=_labels_size(labels),
        contraction_count=0,
        step_formal_scaling=0,
        step_flop_cost=0,
        result_rank=len(labels),
        result_size=_labels_size(labels),
        leaves_count=1,
    )


def _make_unary_node(child: _TreeNode, target_labels: tuple[str, ...]) -> _TreeNode:
    expr = _safe_einsum_subs(["".join(child.labels)], "".join(target_labels))
    signature = ("perm", child.signature, "".join(target_labels))
    step_formal = len(set(child.labels))
    step_flop = _labels_size(child.labels)
    result_size = _labels_size(target_labels)
    return _TreeNode(
        labels=target_labels,
        signature=signature,
        signature_key=_tree_signature_key(signature),
        children=(child,),
        leaf_name=None,
        expr=expr,
        arg_labels=("".join(child.labels),),
        contracted_labels=(),
        formal_scaling=max(child.formal_scaling, step_formal),
        flop_cost=child.flop_cost + step_flop,
        peak_rank=max(child.peak_rank, len(target_labels)),
        peak_size=max(child.peak_size, result_size),
        contraction_count=child.contraction_count + 1,
        step_formal_scaling=step_formal,
        step_flop_cost=step_flop,
        result_rank=len(target_labels),
        result_size=result_size,
        leaves_count=child.leaves_count,
    )


def _make_binary_node(left: _TreeNode, right: _TreeNode, target_labels: tuple[str, ...]) -> _TreeNode:
    ordered = tuple(sorted((left, right), key=lambda node: node.signature_key))
    ordered_left, ordered_right = ordered
    arg_labels = ("".join(ordered_left.labels), "".join(ordered_right.labels))
    expr = _safe_einsum_subs(list(arg_labels), "".join(target_labels))
    contracted = tuple(sorted((set(ordered_left.labels) & set(ordered_right.labels)) - set(target_labels)))
    step_labels = tuple(sorted(set(ordered_left.labels) | set(ordered_right.labels)))
    step_formal = len(step_labels)
    step_flop = _labels_size(step_labels)
    result_size = _labels_size(target_labels)
    signature = (
        "contract",
        ordered_left.signature,
        ordered_right.signature,
        "".join(target_labels),
        contracted,
    )
    return _TreeNode(
        labels=target_labels,
        signature=signature,
        signature_key=_tree_signature_key(signature),
        children=ordered,
        leaf_name=None,
        expr=expr,
        arg_labels=arg_labels,
        contracted_labels=contracted,
        formal_scaling=max(ordered_left.formal_scaling, ordered_right.formal_scaling, step_formal),
        flop_cost=ordered_left.flop_cost + ordered_right.flop_cost + step_flop,
        peak_rank=max(ordered_left.peak_rank, ordered_right.peak_rank, len(target_labels)),
        peak_size=max(ordered_left.peak_size, ordered_right.peak_size, result_size),
        contraction_count=ordered_left.contraction_count + ordered_right.contraction_count + 1,
        step_formal_scaling=step_formal,
        step_flop_cost=step_flop,
        result_rank=len(target_labels),
        result_size=result_size,
        leaves_count=ordered_left.leaves_count + ordered_right.leaves_count,
    )


def _plan_structured_tree(output_labels: str, tensors: list[StructuredTensor]) -> _TreeNode:
    output_tokens = tuple(_split_labels(output_labels))
    n_terms = len(tensors)
    if n_terms == 0:
        raise ValueError("Projected term has no tensors")

    operand_labels = [tuple(_split_labels(labels)) for _name, labels in tensors]
    operand_presence = [frozenset(labels) for labels in operand_labels]
    global_order = _global_label_order(output_tokens, tensors)
    output_set = set(output_tokens)
    all_mask = (1 << n_terms) - 1
    leaf_nodes = tuple(_make_leaf_node(name, labels) for (name, _raw_labels), labels in zip(tensors, operand_labels))

    @lru_cache(maxsize=None)
    def _subset_presence(mask: int) -> frozenset[str]:
        labels: set[str] = set()
        for idx in range(n_terms):
            if mask & (1 << idx):
                labels.update(operand_presence[idx])
        return frozenset(labels)

    @lru_cache(maxsize=None)
    def _result_labels(mask: int) -> tuple[str, ...]:
        subset = _subset_presence(mask)
        outside = _subset_presence(all_mask ^ mask)
        keep = output_set | set(outside)
        return tuple(label for label in global_order if label in subset and label in keep)

    @lru_cache(maxsize=None)
    def _solve(mask: int) -> _TreeNode:
        if mask & (mask - 1) == 0:
            idx = mask.bit_length() - 1
            leaf = leaf_nodes[idx]
            target_labels = _result_labels(mask)
            if leaf.labels == target_labels:
                return leaf
            if set(leaf.labels) != set(target_labels):
                raise ValueError(
                    "Singleton projected planner would need a reduction step; "
                    f"leaf={leaf.labels} target={target_labels}"
                )
            return _make_unary_node(leaf, target_labels)

        target_labels = _result_labels(mask)
        lowest_bit = mask & -mask
        best: _TreeNode | None = None
        submask = (mask - 1) & mask
        while submask:
            if submask & lowest_bit:
                other = mask ^ submask
                if other:
                    left = _solve(submask)
                    right = _solve(other)
                    candidate = _make_binary_node(left, right, target_labels)
                    if best is None or _compare_cost(candidate) < _compare_cost(best):
                        best = candidate
            submask = (submask - 1) & mask

        if best is None:
            raise RuntimeError(f"Unable to build projected contraction tree for mask={mask}")

        if mask == all_mask and best.labels != output_tokens:
            if set(best.labels) != set(output_tokens):
                raise ValueError(f"Root labels {best.labels} do not match output labels {output_tokens}")
            best = _make_unary_node(best, output_tokens)
        return best

    return _solve(all_mask)


def plan_structured_term(
    output_labels: str,
    tensors: list[StructuredTensor],
    coeff: float = 1.0,
    *,
    name: str = "term",
) -> PlannedTerm:
    root = _plan_structured_tree(output_labels, tensors)
    steps: list[PlannedStep] = []
    counter = 0

    def _emit(node: _TreeNode) -> tuple[str, str]:
        nonlocal counter
        if node.leaf_name is not None:
            return node.leaf_name, "".join(node.labels)
        arg_refs: list[str] = []
        for child in node.children:
            ref, _labels = _emit(child)
            arg_refs.append(ref)
        step_name = f"{name}_S{counter}"
        counter += 1
        steps.append(
            PlannedStep(
                name=step_name,
                expr=node.expr or "",
                arg_refs=tuple(arg_refs),
                arg_labels=node.arg_labels,
                out_labels="".join(node.labels),
                contracted_labels=node.contracted_labels,
                formal_scaling=node.step_formal_scaling,
                flop_cost=node.step_flop_cost,
                result_rank=node.result_rank,
                result_size=node.result_size,
                signature=node.signature_key,
                reuse_count=1,
            )
        )
        return step_name, "".join(node.labels)

    root_ref, root_labels = _emit(root)
    return PlannedTerm(
        name=name,
        coeff=float(coeff),
        output_labels=output_labels,
        tensor_signature=tuple(name for name, _labels in tensors),
        root_ref=root_ref,
        root_labels=root_labels,
        steps=tuple(steps),
        formal_scaling=root.formal_scaling,
        flop_cost=root.flop_cost,
        peak_rank=root.peak_rank,
        peak_size=root.peak_size,
        contraction_count=root.contraction_count,
        signature=root.signature_key,
    )


def _collect_internal_nodes(node: _TreeNode, counter: dict[str, tuple[_TreeNode, int]]) -> None:
    if node.leaf_name is None:
        existing = counter.get(node.signature_key)
        if existing is None:
            counter[node.signature_key] = (node, 1)
        else:
            counter[node.signature_key] = (existing[0], existing[1] + 1)
    for child in node.children:
        _collect_internal_nodes(child, counter)


def _select_shared_subtrees(term_roots: Iterable[_TreeNode]) -> tuple[PlannedStep, ...]:
    if os.getenv("AUTOGEN_PROJECTED_ENABLE_SHARED_SUBTREES", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return ()
    min_reuse = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MIN_REUSE", "2"))
    min_benefit = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MIN_BENEFIT", "32768"))
    min_step_scaling = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MIN_STEP_SCALING", "5"))
    max_rank = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MAX_RANK", "6"))
    max_size = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MAX_SIZE", str(4**6)))
    max_shared_count = int(os.getenv("AUTOGEN_PROJECTED_SHARED_MAX_COUNT", "256"))
    node_counts: dict[str, tuple[_TreeNode, int]] = {}
    for root in term_roots:
        _collect_internal_nodes(root, node_counts)

    selected: dict[str, PlannedStep] = {}
    ordered_candidates = sorted(
        (
            (signature_key, node, reuse_count, node.step_flop_cost * max(reuse_count - 1, 0))
            for signature_key, (node, reuse_count) in node_counts.items()
            if reuse_count >= min_reuse
            and len(node.children) == 2
            and node.step_formal_scaling >= min_step_scaling
        ),
        key=lambda item: (
            -item[3],
            -item[1].step_formal_scaling,
            -item[2],
            item[1].leaves_count,
            item[0],
        ),
    )
    for signature_key, node, reuse_count, benefit in ordered_candidates:
        if len(selected) >= max_shared_count:
            break
        if node.result_rank > max_rank or node.result_size > max_size:
            continue
        if benefit < min_benefit:
            continue
        arg_refs: list[str] = []
        for child in node.children:
            if child.leaf_name is not None:
                arg_refs.append(child.leaf_name)
            else:
                shared_child = selected.get(child.signature_key)
                if shared_child is None:
                    arg_refs = []
                    break
                arg_refs.append(shared_child.name)
        if not arg_refs:
            continue
        step_name = f"W_s{node.step_formal_scaling}_r{node.result_rank}_{len(selected)}"
        selected[signature_key] = PlannedStep(
            name=step_name,
            expr=node.expr or "",
            arg_refs=tuple(arg_refs),
            arg_labels=node.arg_labels,
            out_labels="".join(node.labels),
            contracted_labels=node.contracted_labels,
            formal_scaling=node.step_formal_scaling,
            flop_cost=node.step_flop_cost,
            result_rank=node.result_rank,
            result_size=node.result_size,
            signature=signature_key,
            reuse_count=reuse_count,
        )
    return tuple(selected[key] for key in selected)


def _compile_term_plan(
    *,
    output_key: str,
    term_index: int,
    coeff: float,
    output_labels: str,
    tensor_signature: tuple[str, ...],
    root: _TreeNode,
    shared_map: dict[str, PlannedStep],
) -> PlannedTerm:
    steps: list[PlannedStep] = []
    counter = 0

    def _emit(node: _TreeNode) -> tuple[str, str]:
        nonlocal counter
        if node.leaf_name is not None:
            return node.leaf_name, "".join(node.labels)
        shared = shared_map.get(node.signature_key)
        if shared is not None:
            return shared.name, shared.out_labels
        arg_refs: list[str] = []
        for child in node.children:
            ref, _labels = _emit(child)
            arg_refs.append(ref)
        step_name = f"T_{output_key}_{term_index}_{counter}"
        counter += 1
        steps.append(
            PlannedStep(
                name=step_name,
                expr=node.expr or "",
                arg_refs=tuple(arg_refs),
                arg_labels=node.arg_labels,
                out_labels="".join(node.labels),
                contracted_labels=node.contracted_labels,
                formal_scaling=node.step_formal_scaling,
                flop_cost=node.step_flop_cost,
                result_rank=node.result_rank,
                result_size=node.result_size,
                signature=node.signature_key,
                reuse_count=1,
            )
        )
        return step_name, "".join(node.labels)

    root_ref, root_labels = _emit(root)
    return PlannedTerm(
        name=f"{output_key}_{term_index}",
        coeff=float(coeff),
        output_labels=output_labels,
        tensor_signature=tensor_signature,
        root_ref=root_ref,
        root_labels=root_labels,
        steps=tuple(steps),
        formal_scaling=root.formal_scaling,
        flop_cost=root.flop_cost,
        peak_rank=root.peak_rank,
        peak_size=root.peak_size,
        contraction_count=root.contraction_count,
        signature=root.signature_key,
    )


def _summarize_output(terms: list[PlannedTerm]) -> tuple[int, tuple[tuple[int, int], ...]]:
    if not terms:
        return 0, ()
    histogram = Counter(term.formal_scaling for term in terms)
    return max(histogram), tuple(sorted((int(scale), int(count)) for scale, count in histogram.items()))


def build_runtime_plan(
    raw_outputs: dict[str, list[StructuredTerm]],
    *,
    reduction: str = "canonical",
    use_intermediates: bool = True,
) -> ProjectedRuntimePlan:
    del use_intermediates  # Planned pairwise lowering is always used.
    normalized = normalize_projected_output_structs(raw_outputs)
    reducer = merge_projected_terms if reduction == "canonical" else combine_identical_terms
    grouped_outputs = {key: reducer(terms) for key, terms in normalized.items()}

    term_roots: list[tuple[str, int, StructuredTerm, _TreeNode]] = []
    signature_groups: dict[str, dict[tuple[str, ...], list[int]]] = defaultdict(lambda: defaultdict(list))
    for output_key in normalized:
        for index, term in enumerate(grouped_outputs[output_key]):
            output_labels, tensors, _coeff = term
            root = _plan_structured_tree(output_labels, tensors)
            term_roots.append((output_key, index, term, root))
            signature_groups[output_key][tuple(name for name, _labels in tensors)].append(index)

    shared_steps = _select_shared_subtrees(root for _output_key, _idx, _term, root in term_roots)
    shared_map = {step.signature: step for step in shared_steps}

    compiled_outputs: dict[str, OutputPlan] = {}
    compiled_groups: list[PlannedGroup] = []
    all_terms_by_output: dict[str, list[PlannedTerm]] = {key: [] for key in normalized}
    term_lookup: dict[tuple[str, int], PlannedTerm] = {}
    for output_key, index, term, root in term_roots:
        output_labels, tensors, coeff = term
        planned = _compile_term_plan(
            output_key=output_key,
            term_index=index,
            coeff=coeff,
            output_labels=output_labels,
            tensor_signature=tuple(name for name, _labels in tensors),
            root=root,
            shared_map=shared_map,
        )
        all_terms_by_output[output_key].append(planned)
        term_lookup[(output_key, index)] = planned

    for output_key in normalized:
        output_labels = grouped_outputs[output_key][0][0] if grouped_outputs[output_key] else normalized[output_key][0][0]
        terms = tuple(all_terms_by_output[output_key])
        max_formal, histogram = _summarize_output(list(terms))
        compiled_outputs[output_key] = OutputPlan(
            output_labels=output_labels,
            raw_count=len(normalized[output_key]),
            grouped_count=len(grouped_outputs[output_key]),
            terms=terms,
            contraction_count=sum(len(term.steps) for term in terms),
            max_formal_scaling=max_formal,
            scaling_histogram=histogram,
        )
        for signature, term_indices in sorted(signature_groups[output_key].items(), key=lambda item: item[0]):
            group_terms = [term_lookup[(output_key, idx)] for idx in term_indices]
            compiled_groups.append(
                PlannedGroup(
                    name="G_" + output_key + "_" + "_".join(signature),
                    output_key=output_key,
                    output_labels=output_labels,
                    signature=signature,
                    raw_count=sum(1 for term in normalized[output_key] if tuple(name for name, _labels in term[1]) == signature),
                    grouped_count=len(group_terms),
                    term_names=tuple(term.name for term in group_terms),
                    contraction_count=sum(len(term.steps) for term in group_terms),
                    max_formal_scaling=max((term.formal_scaling for term in group_terms), default=0),
                )
            )

    all_steps = [step for output in compiled_outputs.values() for term in output.terms for step in term.steps]
    scaling_hist = Counter(step.formal_scaling for step in shared_steps)
    scaling_hist.update(step.formal_scaling for step in all_steps)
    stats = {
        "intermediate_count": len(shared_steps),
        "group_count": len(compiled_groups),
        "local_contraction_count": len(all_steps),
        "final_call_count": len(shared_steps) + len(all_steps),
        "max_formal_scaling": max((step.formal_scaling for step in (*shared_steps, *all_steps)), default=0),
        "scaling_histogram": {str(scale): count for scale, count in sorted(scaling_hist.items())},
    }
    return ProjectedRuntimePlan(
        outputs=compiled_outputs,
        intermediates=shared_steps,
        groups=tuple(compiled_groups),
        reduction=reduction,
        stats=stats,
    )


def build_runtime_plan_from_codegen(path: str | Path, *, reduction: str = "canonical", use_intermediates: bool = True) -> ProjectedRuntimePlan:
    return build_runtime_plan(raw_projected_outputs_from_plan(path), reduction=reduction, use_intermediates=use_intermediates)


def evaluate_runtime_plan(
    plan: ProjectedRuntimePlan,
    arrays: dict[str, np.ndarray],
    *,
    targets: Iterable[str] | None = None,
) -> dict[str, np.ndarray | complex]:
    chosen = list(targets) if targets is not None else list(plan.outputs)
    work = dict(arrays)
    for step in plan.intermediates:
        if step.name in work:
            continue
        work[step.name] = np.einsum(step.expr, *(work[name] for name in step.arg_refs), optimize=False)

    outs: dict[str, np.ndarray | complex] = {}
    nspin = int(work["h11"].shape[0])
    for output_key in chosen:
        output_plan = plan.outputs[output_key]
        if output_plan.output_labels:
            rank = len(_split_labels(output_plan.output_labels))
            out: np.ndarray | complex = np.zeros((nspin,) * rank, dtype=np.complex128)
        else:
            out = np.array(0.0 + 0.0j)
        for term in output_plan.terms:
            for step in term.steps:
                if step.name in work:
                    continue
                work[step.name] = np.einsum(step.expr, *(work[name] for name in step.arg_refs), optimize=False)
            out += term.coeff * work[term.root_ref]
        outs[output_key] = np.real_if_close(out)
    return outs


def write_numpy_runtime_module(
    plan: ProjectedRuntimePlan,
    path: str | Path,
    *,
    tensor_names: Iterable[str],
) -> None:
    """Emit a standalone NumPy evaluator for a compiled contraction plan.

    The generated module has no dependency on the symbolic generator.  Local
    contraction slots are reused for every term so Python does not retain all
    rank-four intermediates until the function returns.
    """
    tensor_names = tuple(tensor_names)
    signature = ", ".join(tensor_names)
    lines = [
        '"""Generated connected qp-CCSD Wick contractions; do not edit."""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        f"TENSOR_NAMES = {tensor_names!r}",
        f"MAX_FORMAL_SCALING = {int(plan.stats['max_formal_scaling'])}",
        f"CONTRACTION_COUNT = {int(plan.stats['final_call_count'])}",
        "",
        f"def compute_outputs({signature}):",
    ]
    for name in tensor_names:
        lines.append(f"    {name} = np.asarray({name})")
    dtype_args = ", ".join(tensor_names)
    lines.append(f"    dtype = np.result_type({dtype_args}, np.complex128)")

    for step in plan.intermediates:
        lines.append(
            f"    {step.name} = np.einsum({step.expr!r}, "
            f"{', '.join(step.arg_refs)}, optimize=False)"
        )

    for output_key, output in plan.outputs.items():
        if output.output_labels:
            rank = len(_split_labels(output.output_labels))
            shape = ", ".join(["t1.shape[0]"] * rank)
            lines.append(f"    {output_key} = np.zeros(({shape}), dtype=dtype)")
        else:
            lines.append(f"    {output_key} = np.array(0.0 + 0.0j, dtype=dtype)")

        for term in output.terms:
            step_slots: dict[str, str] = {}
            for slot, step in enumerate(term.steps):
                slot_name = f"_s{slot}"
                step_slots[step.name] = slot_name
                args = [step_slots.get(ref, ref) for ref in step.arg_refs]
                lines.append(
                    f"    {slot_name} = np.einsum({step.expr!r}, {', '.join(args)}, optimize=False)"
                )
            root = step_slots.get(term.root_ref, term.root_ref)
            lines.append(f"    {output_key} += ({term.coeff!r}) * {root}")
            if step_slots:
                lines.append("    del " + ", ".join(step_slots.values()))

    result_items = ", ".join(f"{key!r}: np.real_if_close({key})" for key in plan.outputs)
    lines.append(f"    return {{{result_items}}}")
    lines.extend(
        [
            "",
            f"def compute_energy({signature}):",
            f"    return compute_outputs({signature})['energy']",
            "",
            f"def compute_r1({signature}):",
            f"    return compute_outputs({signature})['r1']",
            "",
            f"def compute_r2({signature}):",
            f"    return compute_outputs({signature})['r2']",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))


def summarize_runtime_plan(plan: ProjectedRuntimePlan) -> dict[str, Any]:
    largest_intermediates = sorted(
        (
            {
                "name": step.name,
                "reuse_count": step.reuse_count,
                "formal_scaling": step.formal_scaling,
                "flop_cost": step.flop_cost,
                "result_rank": step.result_rank,
                "result_size": step.result_size,
                "contracted_labels": list(step.contracted_labels),
                "out_labels": step.out_labels,
            }
            for step in plan.intermediates
        ),
        key=lambda item: (item["result_size"], item["formal_scaling"], item["reuse_count"]),
        reverse=True,
    )
    worst_groups = sorted(
        (
            {
                "name": group.name,
                "output_key": group.output_key,
                "signature": list(group.signature),
                "raw_count": group.raw_count,
                "grouped_count": group.grouped_count,
                "contraction_count": group.contraction_count,
                "max_formal_scaling": group.max_formal_scaling,
            }
            for group in plan.groups
        ),
        key=lambda item: (
            item["max_formal_scaling"],
            item["contraction_count"],
            item["grouped_count"],
            item["name"],
        ),
        reverse=True,
    )
    return {
        "stats": plan.stats,
        "outputs": {
            key: {
                "raw_count": output.raw_count,
                "grouped_count": output.grouped_count,
                "term_count": len(output.terms),
                "contraction_count": output.contraction_count,
                "max_formal_scaling": output.max_formal_scaling,
                "scaling_histogram": {str(scale): count for scale, count in output.scaling_histogram},
            }
            for key, output in plan.outputs.items()
        },
        "worst_groups": worst_groups[:10],
        "largest_intermediates": largest_intermediates[:10],
    }


def write_canonical_term_artifact(raw_outputs: dict[str, list[StructuredTerm]], path: str | Path) -> None:
    normalized = normalize_projected_output_structs(raw_outputs)
    data = {
        "schema_version": 2,
        "coefficient_encoding": "exact-rational",
        "amplitude_convention": "T=t1/2!+t2/4!",
        "bch_order": 4,
        "connected_only": True,
        "outputs": {
        key: {
            "count_raw": len(terms),
            "count_canonical": len(merge_projected_terms(terms)),
            "terms": serialize_structs(merge_projected_terms(terms)),
        }
        for key, terms in normalized.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def write_grouped_term_artifact(raw_outputs: dict[str, list[StructuredTerm]], path: str | Path) -> None:
    plan = build_runtime_plan(raw_outputs, reduction="canonical", use_intermediates=True)
    data = {
        "stats": plan.stats,
        "intermediates": [
            {
                "name": step.name,
                "reuse_count": step.reuse_count,
                "expr": step.expr,
                "arg_refs": list(step.arg_refs),
                "arg_labels": list(step.arg_labels),
                "out_labels": step.out_labels,
                "contracted_labels": list(step.contracted_labels),
                "formal_scaling": step.formal_scaling,
                "flop_cost": step.flop_cost,
                "result_rank": step.result_rank,
                "result_size": step.result_size,
            }
            for step in plan.intermediates
        ],
        "outputs": {
            key: {
                "raw_count": output.raw_count,
                "grouped_count": output.grouped_count,
                "term_count": len(output.terms),
                "contraction_count": output.contraction_count,
                "max_formal_scaling": output.max_formal_scaling,
                "scaling_histogram": {str(scale): count for scale, count in output.scaling_histogram},
                "groups": [
                    {
                        "name": group.name,
                        "signature": list(group.signature),
                        "raw_count": group.raw_count,
                        "grouped_count": group.grouped_count,
                        "term_names": list(group.term_names),
                        "contraction_count": group.contraction_count,
                        "max_formal_scaling": group.max_formal_scaling,
                    }
                    for group in plan.groups
                    if group.output_key == key
                ],
                "terms": [
                    {
                        "name": term.name,
                        "coeff": term.coeff,
                        "tensor_signature": list(term.tensor_signature),
                        "root_ref": term.root_ref,
                        "root_labels": term.root_labels,
                        "formal_scaling": term.formal_scaling,
                        "flop_cost": term.flop_cost,
                        "peak_rank": term.peak_rank,
                        "peak_size": term.peak_size,
                        "contraction_count": term.contraction_count,
                        "steps": [
                            {
                                "name": step.name,
                                "expr": step.expr,
                                "arg_refs": list(step.arg_refs),
                                "arg_labels": list(step.arg_labels),
                                "out_labels": step.out_labels,
                                "contracted_labels": list(step.contracted_labels),
                                "formal_scaling": step.formal_scaling,
                                "flop_cost": step.flop_cost,
                                "result_rank": step.result_rank,
                                "result_size": step.result_size,
                                "signature": step.signature,
                            }
                            for step in term.steps
                        ],
                    }
                    for term in output.terms
                ],
            }
            for key, output in plan.outputs.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
