from __future__ import annotations

# Generate einsum-based evaluators from Autogen contraction output.

import contextlib
import copy
import itertools
import math
import re
from pathlib import Path
import json
import os
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
GENERATED_DIR = ROOT / "generated_code"
METHODS_DIR = GENERATED_DIR / "methods"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from autogen.main_tools import commutator as comm  # noqa: E402
from autogen.main_tools import multi_cont  # noqa: E402
from autogen.main_tools import product as prod  # noqa: E402
from autogen.library import change_terms  # noqa: E402
from autogen.library import compare_utils  # noqa: E402
from autogen.library import full_con  # noqa: E402
from autogen.library import make_op  # noqa: E402
from autogen.library import compare as cpre  # noqa: E402
from autogen.library.projected_bch import build_projected_qp_ccsd_terms  # noqa: E402
from autogen.codegen.canonicalize import canonicalize_g as _canonicalize_g_impl  # noqa: E402
from autogen.codegen.canonicalize import safe_einsum_subs as _safe_einsum_subs_impl  # noqa: E402
from autogen.codegen.canonicalize import tokenize_labels as _tokenize_labels_impl  # noqa: E402
from autogen.codegen.compat import load_spec_namespace  # noqa: E402
from autogen.codegen.projected_terms import write_canonical_term_artifact, write_grouped_term_artifact  # noqa: E402
from autogen.codegen.spec_model import default_output_name as _default_output_name  # noqa: E402
from autogen.codegen.spec_model import parse_legacy_spec_terms  # noqa: E402
from autogen.codegen.spec_model import resolve_output_name as _resolve_output_name  # noqa: E402
from autogen.pkg import fix_uv  # noqa: E402


TENSOR_MAP = {
    "V2": "g",
    "F1": "f",
    "T1": "t1",
    "T2": "t2",
    "R1": "r1",
    "R2": "r2",
    "D1": "d1",
    "D2": "d2",
    "X1": "x1",
    "X2": "x2",
}

OCC_SET = set("ijklmn")
VIRT_SET = set("abcdefgh")
GEN_SET = set("pqrst")


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


@contextlib.contextmanager
def _spin_summed_context(enabled: bool):
    prev_env = os.environ.get("AUTOGEN_SPIN_SUMMED")
    prev_fix = fix_uv.SPIN_SUMMED
    os.environ["AUTOGEN_SPIN_SUMMED"] = "1" if enabled else "0"
    fix_uv.SPIN_SUMMED = enabled
    try:
        yield
    finally:
        if prev_env is None:
            os.environ.pop("AUTOGEN_SPIN_SUMMED", None)
        else:
            os.environ["AUTOGEN_SPIN_SUMMED"] = prev_env
        fix_uv.SPIN_SUMMED = prev_fix


def _root_depth_for(path: Path) -> int:
    return len(path.relative_to(ROOT).parts)


def _ensure_output_package(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel = output_dir.resolve().relative_to(ROOT)
    except ValueError:
        return
    if not rel.parts or rel.parts[0] != "generated_code":
        return
    cur = ROOT
    for part in rel.parts:
        cur = cur / part
        init_path = cur / "__init__.py"
        if not init_path.exists():
            init_path.write_text("")


def canonicalize_g(labels):
    return _canonicalize_g_impl(labels)


def _bog_qp_enabled():
    return os.getenv("AUTOGEN_BOGOLIUBOV_QP") == "1"


def _op_counts(name: str):
    if not name:
        return None
    if _bog_qp_enabled():
        if name[0] == "T" and len(name) > 1 and name[1].isdigit():
            if name[1] == "1":
                return 2, 0
            if name[1] == "2":
                return 4, 0
        if name[0] == "X" and len(name) > 1 and name[1].isdigit():
            if name[1] == "1":
                return 0, 2
            if name[1] == "2":
                return 0, 4
    if name[0] in {"T", "D", "R", "X"} and len(name) > 1 and name[1].isdigit():
        n = int(name[1])
        return n, n
    if name.startswith("F1") or name.startswith("H11"):
        return 1, 1
    if name.startswith("V2") or name.startswith("H22"):
        return 2, 2
    if name.startswith("H") and len(name) >= 3 and name[1].isdigit() and name[2].isdigit():
        return int(name[1]), int(name[2])
    return None


def _can_fully_contract(ops):
    total_up = 0
    total_low = 0
    for op in ops:
        counts = _op_counts(op)
        if counts is None:
            return True
        total_up += counts[0]
        total_low += counts[1]
    return total_up == total_low


def _label_type(label):
    if not label:
        return "p"
    base = label[0]
    if base in VIRT_SET:
        return "v"
    if base in OCC_SET:
        return "o"
    return "p"


def _sort_labels(labels):
    order = {"v": 0, "o": 1, "p": 2}
    return sorted(labels, key=lambda x: (order[_label_type(x)], x))


_LABEL_RE = re.compile(r"[a-z][0-9]*")
_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_QP_ANTISYM_TENSOR_GROUPS = {
    "t1": ((0, 1),),
    "t2": ((0, 1, 2, 3),),
    "h02": ((0, 1),),
    "h20": ((0, 1),),
    "h22": ((0, 1), (2, 3)),
    "h31": ((0, 1, 2),),
    "h13": ((1, 2, 3),),
    "h04": ((0, 1, 2, 3),),
    "h40": ((0, 1, 2, 3),),
    "z": ((0, 1),),
}


def _tokenize_labels(labels: str):
    return _tokenize_labels_impl(labels)


def _safe_einsum_subs(subs: str) -> str:
    return _safe_einsum_subs_impl(subs)


def _extract_z_tensors(term):
    z_tensors = []
    if not getattr(term, "st", None):
        return z_tensors
    for block in term.st:
        for op in block:
            if getattr(op, "kind", None) != "z":
                continue
            labels = "".join(item.name for item in op.upper)
            if labels:
                z_tensors.append(("z", labels))
    return z_tensors


def _perm_sign_from_order(order):
    inv = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inv += 1
    return -1 if (inv % 2) else 1


def _canonicalize_antisym_tensor(name: str, labels: str):
    rules = _QP_ANTISYM_TENSOR_GROUPS.get(name)
    tokens = _tokenize_labels(labels)
    if not rules or len(tokens) <= 1:
        return labels, 1.0, False

    tokens = list(tokens)
    sign = 1.0
    for group in rules:
        group_tokens = [tokens[idx] for idx in group]
        if len(set(group_tokens)) < len(group_tokens):
            return labels, 0.0, True
        sorted_tokens = sorted(group_tokens)
        if group_tokens == sorted_tokens:
            continue
        order = [sorted_tokens.index(tok) for tok in group_tokens]
        sign *= _perm_sign_from_order(order)
        for pos, tok in zip(group, sorted_tokens):
            tokens[pos] = tok
    return "".join(tokens), sign, False


def _canonicalize_antisym_output(output_labels: str):
    if not _bog_qp_enabled():
        return output_labels, 1.0
    tokens = _tokenize_labels(output_labels)
    if len(tokens) not in (2, 4):
        return output_labels, 1.0
    if len(set(tokens)) < len(tokens):
        return output_labels, 0.0
    sorted_tokens = sorted(tokens)
    if tokens == sorted_tokens:
        return output_labels, 1.0
    order = [sorted_tokens.index(tok) for tok in tokens]
    return "".join(sorted_tokens), float(_perm_sign_from_order(order))


def _canonicalize_qp_antisymmetry(output_labels, tensors, coeff):
    if coeff == 0.0:
        return output_labels, tensors, 0.0

    output_labels, out_sign = _canonicalize_antisym_output(output_labels)
    coeff *= out_sign
    if coeff == 0.0:
        return output_labels, tensors, 0.0

    canon_tensors = []
    for name, labels in tensors:
        new_labels, sign, is_zero = _canonicalize_antisym_tensor(name, labels)
        if is_zero:
            return output_labels, tensors, 0.0
        coeff *= sign
        if coeff == 0.0:
            return output_labels, tensors, 0.0
        canon_tensors.append((name, new_labels))

    # Tensor products commute; sorting merges equivalent terms emitted in different orders.
    canon_tensors.sort()
    return output_labels, canon_tensors, coeff


def _canonicalize_term_labels(output_labels, tensors):
    mapping = {}
    def label_iter(letters):
        idx = 0
        size = len(letters)
        while True:
            base = letters[idx % size]
            suffix = idx // size
            yield f"{base}{suffix}" if suffix else base
            idx += 1

    occ_iter = label_iter("ijklmn")
    virt_iter = label_iter("abcdefgh")
    gen_iter = label_iter("pqrst")

    def map_label(label):
        if label in mapping:
            return mapping[label]
        base = label[0] if label else ""
        if base in VIRT_SET:
            new = next(virt_iter)
        elif base in OCC_SET:
            new = next(occ_iter)
        else:
            new = next(gen_iter)
        mapping[label] = new
        return new

    output_tokens = _tokenize_labels(output_labels)
    output_can = "".join(map_label(label) for label in output_tokens)
    tensors_can = []
    for name, labels in tensors:
        label_tokens = _tokenize_labels(labels)
        tensors_can.append((name, "".join(map_label(label) for label in label_tokens)))
    return output_can, tensors_can


def _pair_key(op1, op2):
    name1, labels1 = op1
    name2, labels2 = op2
    labels1_tokens = _tokenize_labels(labels1)
    labels2_tokens = _tokenize_labels(labels2)
    common = set(labels1_tokens) & set(labels2_tokens)
    out_labels = []
    for label in labels1_tokens:
        if label not in common and label not in out_labels:
            out_labels.append(label)
    for label in labels2_tokens:
        if label not in common and label not in out_labels:
            out_labels.append(label)
    out_labels = _sort_labels(out_labels)
    ops = tuple(sorted(((name1, labels1), (name2, labels2))))
    return ops, tuple(sorted(common)), tuple(out_labels)


def _term_to_residual_struct(term, tensor_map=None, require_output=True):
    tensors = []
    output_labels = None
    mapping = tensor_map or TENSOR_MAP
    z_tensors = _extract_z_tensors(term)
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        if op.name.startswith("X"):
            if output_labels is not None:
                raise ValueError("Multiple X projectors found in term.")
            output_labels = output_labels_from_xop(coeff)
            continue
        tensor_name = mapping.get(op.name)
        if not tensor_name and op.name.startswith("T") and len(op.name) > 1:
            tensor_name = "t1" if op.name[1] == "1" else "t2"
        if (
            not tensor_name
            and op.name.startswith("H")
            and len(op.name) >= 3
            and op.name[1].isdigit()
            and op.name[2].isdigit()
        ):
            tensor_name = f"h{op.name[1:]}"
        if not tensor_name:
            raise ValueError(f"Unsupported operator {op.name}")
        tensors.append((tensor_name, "".join(coeff)))
    if z_tensors:
        tensors.extend(z_tensors)
    if output_labels is None:
        if require_output:
            raise ValueError("No X projector found in amplitude term.")
        output_labels = ""
    return output_labels, tensors, term.fac


def _spin_adapt_output_spins(output_labels):
    if not output_labels:
        return {}
    tokens = _tokenize_labels(output_labels)
    if len(tokens) == 2:
        return {tokens[0]: 0, tokens[1]: 0}
    if len(tokens) == 4:
        return {
            tokens[0]: 0,
            tokens[2]: 0,
            tokens[1]: 1,
            tokens[3]: 1,
        }
    return {label: 0 for label in tokens}


def _spin_adapt_expand_g(tensors):
    expanded = [(list(tensors), 1.0)]
    for idx, (name, labels) in enumerate(tensors):
        label_tokens = _tokenize_labels(labels)
        if name != "g" or len(label_tokens) != 4:
            continue
        next_expanded = []
        for base, sign in expanded:
            direct = list(base)
            direct[idx] = (name, labels)
            next_expanded.append((direct, sign))
            exch_labels = "".join(label_tokens[:2] + [label_tokens[3], label_tokens[2]])
            exchange = list(base)
            exchange[idx] = (name, exch_labels)
            next_expanded.append((exchange, -sign))
        expanded = next_expanded
    return expanded


def _spin_adapt_spin_factor(tensors, fixed_spins):
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _name, labels in tensors:
        for label in _tokenize_labels(labels):
            find(label)
    for label in fixed_spins:
        find(label)

    for name, labels in tensors:
        label_tokens = _tokenize_labels(labels)
        if name in {"f", "t1", "r1", "d1"} and len(label_tokens) >= 2:
            union(label_tokens[0], label_tokens[1])
        elif name in {"t2", "r2", "d2"} and len(label_tokens) >= 4:
            union(label_tokens[0], label_tokens[2])
            union(label_tokens[1], label_tokens[3])
        elif name == "g" and len(label_tokens) >= 4:
            union(label_tokens[0], label_tokens[2])
            union(label_tokens[1], label_tokens[3])

    groups = {}
    for label in parent:
        root = find(label)
        groups.setdefault(root, []).append(label)

    fixed_by_root = {}
    for root, labels in groups.items():
        spins = {fixed_spins[label] for label in labels if label in fixed_spins}
        if len(spins) > 1:
            return 0
        if spins:
            fixed_by_root[root] = next(iter(spins))

    free = sum(1 for root in groups if root not in fixed_by_root)
    return 2 ** free


def _canonicalize_spin_summed_tensor(name, labels):
    label_tokens = _tokenize_labels(labels)
    if name in {"g", "t2"} and len(label_tokens) == 4:
        left = sorted(label_tokens[:2])
        right = sorted(label_tokens[2:])
        return name, "".join(left + right)
    return name, labels


def _spin_adapt_structs(output_labels, tensors, coeff):
    fixed_spins = _spin_adapt_output_spins(output_labels)
    expanded = _spin_adapt_expand_g(tensors)
    combined = {}
    for tensors_exp, sign in expanded:
        spin_factor = _spin_adapt_spin_factor(tensors_exp, fixed_spins)
        if spin_factor == 0:
            continue
        tensors_can = tuple(
            _canonicalize_spin_summed_tensor(name, labels)
            for name, labels in tensors_exp
        )
        key = (output_labels, tensors_can)
        combined[key] = combined.get(key, 0.0) + coeff * sign * spin_factor
    structs = []
    for (out_labels, tensors_can), value in combined.items():
        if abs(value) < 1e-12:
            continue
        structs.append((out_labels, list(tensors_can), value))
    return structs


def _spin_output_fixed_spins(output_tokens):
    if len(output_tokens) == 4:
        v1, v2, o1, o2 = output_tokens
        return {v1: 0, o1: 0, v2: 1, o2: 1}
    return {}


def _spin_output_weight(output_tokens, spins):
    if len(output_tokens) == 2:
        v1, o1 = output_tokens
        if spins[v1] != spins[o1]:
            return 0.0
        return 1.0 / math.sqrt(2.0)
    if len(output_tokens) == 4:
        v1, v2, o1, o2 = output_tokens
        if spins[v1] != 0 or spins[o1] != 0 or spins[v2] != 1 or spins[o2] != 1:
            return 0.0
        return math.sqrt(2.0)
    return 1.0


def _spin_tensor_contribs(name, labels, spins, tokens=None):
    if tokens is None:
        tokens = _tokenize_labels(labels)
    if name in {"f", "t1", "d1"}:
        if len(tokens) != 2:
            return []
        if spins[tokens[0]] != spins[tokens[1]]:
            return []
        return [(name, labels, 1.0)]
    if name == "r1":
        if len(tokens) != 2:
            return []
        if spins[tokens[0]] != spins[tokens[1]]:
            return []
        return [(name, labels, 1.0 / math.sqrt(2.0))]
    if name in {"t2", "d2", "r2"}:
        if len(tokens) != 4:
            return []
        v1, v2, o1, o2 = tokens
        s_v1 = spins[v1]
        s_v2 = spins[v2]
        s_o1 = spins[o1]
        s_o2 = spins[o2]
        base = 1.0 / math.sqrt(2.0) if name == "r2" else 1.0
        contribs = []
        if s_v1 == s_v2 == s_o1 == s_o2:
            contribs.append((name, labels, base))
            swapped = "".join([v1, v2, o2, o1])
            contribs.append((name, swapped, -base))
            return contribs
        if s_v1 == s_v2 or s_o1 == s_o2:
            return contribs
        if s_v1 == 0 and s_v2 == 1:
            v_alpha, v_beta = v1, v2
            swap_v = False
        elif s_v1 == 1 and s_v2 == 0:
            v_alpha, v_beta = v2, v1
            swap_v = True
        else:
            return contribs
        if s_o1 == 0 and s_o2 == 1:
            o_alpha, o_beta = o1, o2
            swap_o = False
        elif s_o1 == 1 and s_o2 == 0:
            o_alpha, o_beta = o2, o1
            swap_o = True
        else:
            return contribs
        sign = base if swap_v == swap_o else -base
        reordered = "".join([v_alpha, v_beta, o_alpha, o_beta])
        contribs.append((name, reordered, sign))
        return contribs
    if name == "g":
        if len(tokens) != 4:
            return []
        p, q, r, s = tokens
        s_p = spins[p]
        s_q = spins[q]
        s_r = spins[r]
        s_s = spins[s]
        contribs = []
        if s_p == s_r and s_q == s_s:
            contribs.append((name, labels, 1.0))
        if s_p == s_s and s_q == s_r:
            swapped = "".join([p, q, s, r])
            contribs.append((name, swapped, -1.0))
        return contribs
    return [(name, labels, 1.0)]


def _spin_sum_eom_structs(output_labels, tensors, coeff):
    output_tokens = _tokenize_labels(output_labels)
    if not output_tokens:
        return [(output_labels, tensors, coeff)]

    labels = []
    for label in output_tokens:
        if label not in labels:
            labels.append(label)
    tensor_tokens = []
    for name, tensor_labels in tensors:
        tokens = _tokenize_labels(tensor_labels)
        tensor_tokens.append((name, tensor_labels, tokens))
        for label in tokens:
            if label not in labels:
                labels.append(label)

    fixed_spins = _spin_output_fixed_spins(output_tokens)
    remaining = [label for label in labels if label not in fixed_spins]
    combined = {}

    for bits in range(1 << len(remaining)):
        spins = dict(fixed_spins)
        for idx, label in enumerate(remaining):
            spins[label] = (bits >> idx) & 1

        weight = _spin_output_weight(output_tokens, spins)
        if weight == 0.0:
            continue

        contrib_lists = []
        for name, tensor_labels, tokens in tensor_tokens:
            contribs = _spin_tensor_contribs(name, tensor_labels, spins, tokens)
            if not contribs:
                break
            contrib_lists.append(contribs)
        else:
            for combo in itertools.product(*contrib_lists):
                factor = coeff * weight
                out_tensors = []
                for name, tensor_labels, fac in combo:
                    factor *= fac
                    out_tensors.append((name, tensor_labels))
                if abs(factor) < 1e-12:
                    continue
                key = (output_labels, tuple(out_tensors))
                combined[key] = combined.get(key, 0.0) + factor

    structs = []
    for (out_labels, tensors_key), value in combined.items():
        if abs(value) < 1e-12:
            continue
        structs.append((out_labels, list(tensors_key), value))
    return structs


def _select_intermediates(terms):
    min_count = int(os.getenv("AUTOGEN_INTERMEDIATE_MIN", "3"))
    max_intermediates = int(os.getenv("AUTOGEN_INTERMEDIATE_MAX", "80"))

    def dim_cost(labels):
        weights = {"v": 4, "o": 2, "p": 4}
        cost = 1
        for label in labels:
            cost *= weights[_label_type(label)]
        return cost

    counts = {}
    costs = {}
    for output_labels, tensors, _ in terms:
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                key = _pair_key(tensors[i], tensors[j])
                common = key[1]
                if not common:
                    continue
                counts[key] = counts.get(key, 0) + 1
                cost = dim_cost(tensors[i][1]) * dim_cost(tensors[j][1])
                costs[key] = costs.get(key, 0) + cost
    candidates = []
    for key, count in counts.items():
        if count >= min_count:
            candidates.append((key, count, costs.get(key, 0)))
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if max_intermediates > 0:
        candidates = candidates[:max_intermediates]
    inter_map = {}
    for idx, (key, count, _cost) in enumerate(candidates):
        ops, _common, out_labels = key
        name = f"I{idx}"
        inter_map[key] = {
            "name": name,
            "ops": ops,
            "out_labels": "".join(out_labels),
            "count": count,
        }
    return inter_map


def _apply_intermediates(tensors, inter_map):
    tensors = list(tensors)
    used = True
    while used:
        used = False
        best = None
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                key = _pair_key(tensors[i], tensors[j])
                info = inter_map.get(key)
                if info is None:
                    continue
                if best is None or info["count"] > best[0]["count"]:
                    best = (info, i, j)
        if best is not None:
            info, i, j = best
            new_tensors = []
            for idx, op in enumerate(tensors):
                if idx in (i, j):
                    continue
                new_tensors.append(op)
            new_tensors.append((info["name"], info["out_labels"]))
            tensors = new_tensors
            used = True
    return tensors


def _group_terms_by_subs(terms):
    grouped = {}
    for output_labels, tensors, coeff in terms:
        subs_in = ",".join(labels for _, labels in tensors)
        subs = f"{subs_in}->{output_labels}"
        key = (subs, tuple(tensors))
        grouped.setdefault(key, 0.0)
        grouped[key] += coeff
    return grouped


def build_terms(list_char_op, merge: bool = True, quiet: bool = False):
    # Mirror driv3-style contraction flow to get fully contracted terms.
    if not _can_fully_contract(list_char_op):
        return []
    with suppress_output(quiet):
        dict_ind = {}
        lou, dict_ind = make_op.make_op(list_char_op, dict_ind)
        st, co = lou[0].st, lou[0].co
        for i in range(1, len(lou)):
            # Contract one operator at a time to keep the contraction logic identical to driv3.
            st, co = multi_cont.multi_cont(st, lou[i].st, co, lou[i].co)

        st, co = full_con.full_con(st, co)
        terms = change_terms.change_terms1(st, co, 1.0, dict_ind, lou)
        for term in terms:
            term.compress()
            term.build_map_org()

        if merge:
            def merge_terms(rep, term, flo):
                rep.fac = rep.fac + term.fac * flo
                term.fac = 0.0

            compare_utils.reduce_terms_two_stage(
                terms,
                compare_utils.fast_compare,
                merge_terms,
                key_func=compare_utils.coarse_key,
                secondary_key_func=compare_utils.matrix_key,
            )

        return [term for term in terms if term.fac != 0.0]


def build_eom_bch_terms(
    max_order: int = 4,
    t1_labels: tuple[str, ...] = ("T1", "T11", "T12", "T13"),
    t2_labels: tuple[str, ...] = ("T2", "T21"),
    h_ops: tuple[str, ...] = ("F1", "V2"),
    outputs: tuple[str, ...] = ("X1", "X2"),
    quiet: bool = False,
):
    from math import factorial
    from itertools import product

    built_terms = []

    with suppress_output(quiet):
        for h_op in h_ops:
            for n in range(max_order + 1):
                fac = 1.0 / factorial(n)
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
                    h_terms = [h_op]
                    for t_op in t_ops:
                        h_terms = comm.comm(h_terms, [t_op], last=0)
                        if not h_terms:
                            break
                    if not h_terms:
                        continue
                    for r_op in ("R1", "R2"):
                        hr_terms = comm.comm(h_terms, [r_op], last=0)
                        if not hr_terms:
                            continue
                        for output_key in outputs:
                            xr_terms = prod.prod([output_key], hr_terms, fac)
                            for term in xr_terms:
                                built_terms.append((output_key, term))

    return built_terms


def build_bch_terms(
    max_order: int = 4,
    t1_labels: tuple[str, ...] = ("T1", "T11", "T12", "T13"),
    t2_labels: tuple[str, ...] = ("T2", "T21"),
    h_ops: tuple[str, ...] = ("F1", "V2"),
    outputs: tuple[str, ...] = ("X1", "X2", "scalar"),
    quiet: bool = False,
):
    def _raw_terms_from_op(op_name: str):
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

    def term_op_rank(term):
        if not term.st or not term.st[0]:
            return 0, 0
        for op in reversed(term.st[0]):
            if op.kind == "op":
                return len(op.upper), len(op.lower)
        return 0, 0

    built_terms = []

    with suppress_output(quiet):
        for h_op in h_ops:
            for n in range(max_order + 1):
                fac = 1.0 / math.factorial(n)
                for seq in itertools.product(("T1", "T2"), repeat=n):
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
                    for output_key in outputs:
                        if output_key == "scalar":
                            for term in h_terms:
                                if term_op_rank(term) != (0, 0):
                                    continue
                                scaled = copy.deepcopy(term)
                                scaled.fac *= fac
                                built_terms.append((output_key, scaled))
                        else:
                            xr_terms = prod.prod([output_key], h_terms, fac)
                            for term in xr_terms:
                                built_terms.append((output_key, term))

    return built_terms


def term_to_einsum(term):
    # Map each operator to a tensor name and build an einsum signature.
    tensors = []
    prefactor = 1.0
    z_tensors = _extract_z_tensors(term)
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        tensor_name = TENSOR_MAP.get(op.name)
        if not tensor_name and op.name.startswith("T") and len(op.name) > 1:
            tensor_name = "t1" if op.name[1] == "1" else "t2"
        if (
            not tensor_name
            and op.name.startswith("H")
            and len(op.name) >= 3
            and op.name[1].isdigit()
            and op.name[2].isdigit()
        ):
            tensor_name = f"h{op.name[1:]}"
        if not tensor_name:
            raise ValueError(f"Unsupported operator {op.name}")
        labels = "".join(coeff)
        if tensor_name == "g":
            labels, sign = canonicalize_g(labels)
            prefactor *= sign
        tensors.append((tensor_name, labels))
    if z_tensors:
        tensors.extend(z_tensors)

    all_inds = []
    for _, labels in tensors:
        for label in labels:
            if label not in all_inds:
                all_inds.append(label)
    sum_set = set(term.sum_list)
    out_labels = "".join([label for label in all_inds if label not in sum_set])

    subs_in = ",".join(labels for _, labels in tensors)
    subs = f"{subs_in}->{out_labels}"

    args = []
    for tensor_name, labels in tensors:
        if tensor_name in {"g", "f"}:
            args.append(f"view_tensor({tensor_name}, '{labels}', o, v)")
        else:
            args.append(tensor_name)

    return subs, args, prefactor


def output_labels_from_xop(coeff):
    virt = [label for label in coeff if label and label[0] in VIRT_SET]
    occ = [label for label in coeff if label and label[0] in OCC_SET]
    if virt or occ:
        return "".join(virt + occ)
    if _bog_qp_enabled():
        return "".join(coeff)
    return "".join(virt + occ)


def term_to_residual_einsum(term):
    tensors = []
    output_labels = None
    z_tensors = _extract_z_tensors(term)
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        if op.name.startswith("X"):
            output_labels = output_labels_from_xop(coeff)
            continue
        tensor_name = TENSOR_MAP.get(op.name)
        if not tensor_name and op.name.startswith("T") and len(op.name) > 1:
            tensor_name = "t1" if op.name[1] == "1" else "t2"
        if not tensor_name:
            raise ValueError(f"Unsupported operator {op.name}")
        labels = "".join(coeff)
        tensors.append((tensor_name, labels))
    if z_tensors:
        tensors.extend(z_tensors)
    if output_labels is None:
        raise ValueError("No X projector found in amplitude term.")

    subs_in = ",".join(labels for _, labels in tensors)
    subs = f"{subs_in}->{output_labels}"

    args = []
    for tensor_name, labels in tensors:
        if tensor_name in {"g", "f"}:
            args.append(f"view_tensor({tensor_name}, '{labels}', o, v)")
        else:
            args.append(tensor_name)
    return subs, args


def build_ccsd_amplitude_terms(quiet: bool = False, subset: str = "both"):
    x1_specs = [
        (1.0, ["X1", "F1"]),
        (1.0, ["X1", "F1", "T1"]),
        (1.0, ["X1", "F1", "T2"]),
        (0.5, ["X1", "F1", "T1", "T11"]),
        (0.5, ["X1", "F1", "T2", "T21"]),
        (1.0, ["X1", "F1", "T1", "T2"]),
        (1.0, ["X1", "V2"]),
        (1.0, ["X1", "V2", "T1"]),
        (1.0, ["X1", "V2", "T2"]),
        (0.5, ["X1", "V2", "T1", "T11"]),
        (0.5, ["X1", "V2", "T2", "T21"]),
        (1.0, ["X1", "V2", "T1", "T2"]),
        (1.0 / 6.0, ["X1", "V2", "T1", "T11", "T12"]),
    ]
    x2_specs = [
        (1.0, ["X2", "F1"]),
        (1.0, ["X2", "F1", "T1"]),
        (0.5, ["X2", "F1", "T1", "T11"]),
        (1.0, ["X2", "F1", "T2"]),
        (0.5, ["X2", "F1", "T2", "T21"]),
        (1.0, ["X2", "F1", "T1", "T2"]),
        (1.0, ["X2", "V2", "T1"]),
        (0.5, ["X2", "V2", "T1", "T11"]),
        (1.0 / 6.0, ["X2", "V2", "T1", "T11", "T12"]),
        (1.0, ["X2", "V2", "T2"]),
        (0.5, ["X2", "V2", "T2", "T21"]),
        (1.0, ["X2", "V2", "T1", "T2"]),
        (0.5, ["X2", "V2", "T1", "T11", "T2"]),
        (1.0 / 24.0, ["X2", "V2", "T1", "T11", "T12", "T13"]),
    ]

    x1_terms = []
    x2_terms = []
    subset = subset.lower()
    if subset not in {"both", "x1", "x2"}:
        raise ValueError(f"Unsupported CCSD amplitude subset: {subset}")
    if subset in {"both", "x1"}:
        for fac, ops in x1_specs:
            terms = build_terms(ops, merge=False, quiet=quiet)
            for term in terms:
                term.fac *= fac
            x1_terms.extend(terms)
    if subset in {"both", "x2"}:
        for fac, ops in x2_specs:
            terms = build_terms(ops, merge=False, quiet=quiet)
            for term in terms:
                term.fac *= fac
            x2_terms.extend(terms)

    return x1_terms, x2_terms


def _infer_output_key(ops):
    x_ops = [name for name in ops if name.startswith("X")]
    if len(x_ops) > 1:
        raise ValueError("Multiple X projectors found in operator list.")
    if x_ops:
        return x_ops[0]
    return "scalar"


def default_output_name(output_key):
    return _default_output_name(output_key)


def resolve_output_name(output_key, output_names):
    return _resolve_output_name(output_key, output_names)


def load_spec(spec_path):
    return load_spec_namespace(spec_path)


def parse_spec_terms(spec):
    return parse_legacy_spec_terms(spec)


def _ordered_tensor_names(names):
    tensor_order = ["f", "g", "t1", "t2", "r1", "r2", "d1", "d2", "x1", "x2"]
    ordered = [name for name in tensor_order if name in names]
    extras = sorted(name for name in names if name not in tensor_order)
    return ordered + extras


def emit_spec_residuals(
    output_dir,
    spec_terms,
    output_names,
    tensor_map,
    view_tensors,
    mode: str = "full",
    quiet: bool = False,
    spin_adapted: bool = False,
    filename: str = "residuals.py",
    spin_summed_override: bool | None = None,
    _allow_spin_adapted: bool = True,
    prebuilt_terms=None,
    eom_mode: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = mode.lower()
    if mode not in {"full", "intermediates"}:
        raise ValueError(f"Unsupported spec mode: {mode}")
    spin_summed = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
    if spin_summed_override is not None:
        spin_summed = spin_summed_override
    mode_flag = None
    if spin_summed and (spin_adapted or eom_mode):
        mode_flag = os.getenv("AUTOGEN_SPIN_SUMMED_MODE")
        if mode_flag is None:
            mode_flag = "direct"
        else:
            mode_flag = mode_flag.lower().strip()
        if spin_adapted and not eom_mode and mode_flag == "spinorb":
            allow_spinorb = os.getenv("AUTOGEN_CCSD_ALLOW_SPINORB", "0") == "1"
            if not allow_spinorb:
                mode_flag = "direct"
        if eom_mode and mode_flag == "direct":
            allow_direct = os.getenv("AUTOGEN_EOM_ALLOW_DIRECT", "1") == "1"
            if not allow_direct:
                mode_flag = "spinorb"
    if spin_summed and (spin_adapted or eom_mode) and _allow_spin_adapted:
        if mode_flag == "spinorb":
            emit_spec_residuals(
                output_dir,
                spec_terms,
                output_names,
                tensor_map,
                view_tensors,
                mode=mode,
                quiet=quiet,
                spin_adapted=False,
                filename="residuals_spinorb.py",
                spin_summed_override=False,
                _allow_spin_adapted=False,
                eom_mode=eom_mode,
                prebuilt_terms=prebuilt_terms,
            )
            emit_spin_adapted_wrapper(
                output_dir,
                output_names,
                mode=mode,
                eom_mode=eom_mode,
            )
            return

    built_terms = []
    output_order = []
    adapt_direct = (
        spin_summed
        and spin_adapted
        and eom_mode
        and mode_flag == "direct"
    )
    build_spin_summed = spin_summed and not adapt_direct
    with _spin_summed_context(build_spin_summed):
        if prebuilt_terms is not None:
            for output_key, term in prebuilt_terms:
                if output_key not in output_order:
                    output_order.append(output_key)
                built_terms.append((output_key, term))
        else:
            for term_spec in spec_terms:
                output_key = term_spec["output_key"]
                if output_key not in output_order:
                    output_order.append(output_key)
                terms = build_terms(term_spec["ops"], merge=False, quiet=quiet)
                for term in terms:
                    term.fac *= term_spec["fac"]
                    built_terms.append((output_key, term))

    if not built_terms:
        raise ValueError("No terms produced from spec.")

    output_structs = {key: [] for key in output_order}
    output_labels_map = {}
    all_structs = []
    for output_key, term in built_terms:
        output_labels, tensors, coeff = _term_to_residual_struct(
            term, tensor_map=tensor_map, require_output=False
        )
        if output_key != "scalar" and not output_labels:
            raise ValueError(f"Output '{output_key}' requires an X projector.")
        if adapt_direct:
            structs = _spin_sum_eom_structs(output_labels, tensors, coeff)
        else:
            structs = [(output_labels, tensors, coeff)]
        for out_labels, tensors, coeff in structs:
            out_labels, tensors, coeff = _canonicalize_qp_antisymmetry(
                out_labels, tensors, coeff
            )
            if abs(coeff) < 1e-12:
                continue
            out_labels, tensors = _canonicalize_term_labels(out_labels, tensors)
            if output_key in output_labels_map:
                if output_labels_map[output_key] != out_labels:
                    raise ValueError(
                        f"Output '{output_key}' has inconsistent labels: "
                        f"{output_labels_map[output_key]} vs {out_labels}"
                    )
            else:
                output_labels_map[output_key] = out_labels
            struct = (out_labels, tensors, coeff)
            output_structs[output_key].append(struct)
            all_structs.append(struct)

    base_tensor_names = []
    base_tensor_set = set()
    for _out, tensors, _coeff in all_structs:
        for name, _labels in tensors:
            if name.startswith("I"):
                continue
            if name not in base_tensor_set:
                base_tensor_set.add(name)
                base_tensor_names.append(name)

    base_tensor_names = _ordered_tensor_names(base_tensor_names)
    view_tensor_names = [name for name in base_tensor_names if name in view_tensors]
    non_view_tensor_names = [name for name in base_tensor_names if name not in view_tensors]

    if mode == "intermediates":
        inter_map = _select_intermediates(all_structs)
        output_structs = {
            key: [
                (output_labels, _apply_intermediates(tensors, inter_map), coeff)
                for output_labels, tensors, coeff in terms
            ]
            for key, terms in output_structs.items()
        }
        used_intermediates = set()
        for terms in output_structs.values():
            for _out, tensors, _coeff in terms:
                for name, _labels in tensors:
                    if name.startswith("I"):
                        used_intermediates.add(name)
        inter_defs = [
            info for info in inter_map.values() if info["name"] in used_intermediates
        ]
        inter_defs.sort(key=lambda info: int(info["name"][1:]))
    else:
        inter_defs = []

    lines = []
    lines.append("import numpy as np")
    lines.append("import os")
    lines.append("import re")
    lines.append("")
    mode_tag = mode_flag or ("direct" if spin_summed else "spinorb")
    bog_qp = _bog_qp_enabled()
    lines.append(f"AUTOGEN_SPIN_SUMMED = {spin_summed}")
    lines.append(f"AUTOGEN_SPIN_SUMMED_MODE = {mode_tag!r}")
    lines.append(f"AUTOGEN_BOGOLIUBOV_QP = {bog_qp}")
    lines.append(
        "AUTOGEN_QP_T1_SIGN = float(os.getenv("
        "'AUTOGEN_QP_T1_SIGN', '-1.0' if AUTOGEN_BOGOLIUBOV_QP else '1.0'))"
    )
    lines.append(f"AUTOGEN_INTERMEDIATES = {mode == 'intermediates'}")
    lines.append(f"VIEW_TENSORS = {tuple(view_tensor_names)}")
    lines.append("OCC = set('ijklmn')")
    lines.append("VIRT = set('abcdefgh')")
    lines.append("_LABEL_RE = re.compile(r\"[a-z][0-9]*\")")
    lines.append("")
    lines.append("def _apply_t1_sign(t1):")
    lines.append("    if AUTOGEN_QP_T1_SIGN == 1.0:")
    lines.append("        return t1")
    lines.append("    return AUTOGEN_QP_T1_SIGN * t1")
    lines.append("")
    lines.append("def _iter_labels(labels):")
    lines.append("    if not labels:")
    lines.append("        return []")
    lines.append("    return _LABEL_RE.findall(labels)")
    lines.append("")
    lines.append("def view_tensor(tensor, labels, o, v):")
    lines.append("    label_tokens = _iter_labels(labels)")
    lines.append("    idx = []")
    lines.append("    list_axes = []")
    lines.append("    for axis, label in enumerate(label_tokens):")
    lines.append("        if label and label[0] in OCC:")
    lines.append("            idx.append(o)")
    lines.append("            list_axes.append(axis)")
    lines.append("        elif label and label[0] in VIRT:")
    lines.append("            idx.append(v)")
    lines.append("            list_axes.append(axis)")
    lines.append("        else:")
    lines.append("            idx.append(slice(None))")
    lines.append("    if not list_axes:")
    lines.append("        return tensor[tuple(idx)]")
    lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
    lines.append("    ix_iter = iter(ix)")
    lines.append("    full_idx = []")
    lines.append("    for axis in range(len(idx)):")
    lines.append("        if axis in list_axes:")
    lines.append("            full_idx.append(next(ix_iter))")
    lines.append("        else:")
    lines.append("            full_idx.append(idx[axis])")
    lines.append("    return tensor[tuple(full_idx)]")
    lines.append("")
    lines.append("def zeros_for_output(labels, o, v, dtype=None):")
    lines.append("    label_tokens = _iter_labels(labels)")
    lines.append("    if not label_tokens:")
    lines.append("        return 0.0")
    lines.append("    shape = []")
    lines.append("    for label in label_tokens:")
    lines.append("        if label and label[0] in OCC:")
    lines.append("            shape.append(len(o))")
    lines.append("        elif label and label[0] in VIRT:")
    lines.append("            shape.append(len(v))")
    lines.append("        else:")
    lines.append("            shape.append(len(o) + len(v))")
    lines.append("    if dtype is None:")
    lines.append("        return np.zeros(tuple(shape))")
    lines.append("    return np.zeros(tuple(shape), dtype=dtype)")
    lines.append("")
    lines.append("def _get_viewer(tensor_map, o, v):")
    lines.append("    views = {}")
    lines.append("    def get_view(name, labels):")
    lines.append("        key = (name, labels)")
    lines.append("        if key in views:")
    lines.append("            return views[key]")
    lines.append("        tensor = tensor_map[name]")
    lines.append("        views[key] = view_tensor(tensor, labels, o, v)")
    lines.append("        return views[key]")
    lines.append("    return get_view")
    lines.append("")

    tensor_args = ", ".join(base_tensor_names + ["o", "v"])
    if mode == "intermediates":
        non_view_args = ", ".join(["get_view"] + non_view_tensor_names)
        lines.append(f"def compute_intermediates({non_view_args}):")
        if inter_defs:
            for info in inter_defs:
                name = info["name"]
                labels1 = info["ops"][0][1]
                labels2 = info["ops"][1][1]
                out_labels = info["out_labels"]
                subs = _safe_einsum_subs(f"{labels1},{labels2}->{out_labels}")
                args = []
                for op_name, op_labels in info["ops"]:
                    if op_name in view_tensor_names:
                        args.append(f"get_view('{op_name}', '{op_labels}')")
                    else:
                        args.append(op_name)
                lines.append(
                    f"    {name} = np.einsum('{subs}', {', '.join(args)}, optimize=True)"
                )
            lines.append("    return {")
            for info in inter_defs:
                name = info["name"]
                lines.append(f"        '{name}': {name},")
            lines.append("    }")
        else:
            lines.append("    return {}")
        lines.append("")
        lines.append(f"def compute_outputs({tensor_args}):")
        lines.append("    tensor_map = {")
        for name in view_tensor_names:
            lines.append(f"        '{name}': {name},")
        lines.append("    }")
        lines.append("    get_view = _get_viewer(tensor_map, o, v)")
        dtype_args = ", ".join(base_tensor_names)
        if dtype_args:
            lines.append(f"    dtype = np.result_type({dtype_args})")
        else:
            lines.append("    dtype = None")
        if "t1" in base_tensor_names:
            lines.append("    t1 = _apply_t1_sign(t1)")
        lines.append(f"    inter = compute_intermediates({non_view_args})")
        for info in inter_defs:
            name = info["name"]
            lines.append(f"    {name} = inter['{name}']")
        lines.append("    outputs = {}")
        for output_key in output_order:
            output_name = resolve_output_name(output_key, output_names)
            output_labels = output_labels_map[output_key]
            lines.append(
                f"    {output_name} = zeros_for_output('{output_labels}', o, v, dtype=dtype)"
            )
            grouped = _group_terms_by_subs(output_structs[output_key])
            for (subs, tensors), coeff in sorted(grouped.items()):
                if coeff == 0.0:
                    continue
                safe_subs = _safe_einsum_subs(subs)
                args = []
                for name, labels in tensors:
                    if name in used_intermediates:
                        args.append(name)
                    elif name in view_tensor_names:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    {output_name} += ({coeff}) * np.einsum('{safe_subs}', {', '.join(args)}, optimize=True)"
                )
            lines.append(f"    outputs['{output_name}'] = {output_name}")
        lines.append("    return outputs")
        lines.append("")
        for output_key in output_order:
            output_name = resolve_output_name(output_key, output_names)
            lines.append(f"def compute_{output_name}({tensor_args}):")
            lines.append(f"    return compute_outputs({tensor_args})['{output_name}']")
            lines.append("")
    else:
        for output_key in output_order:
            output_name = resolve_output_name(output_key, output_names)
            output_labels = output_labels_map[output_key]
            lines.append(f"def compute_{output_name}({tensor_args}):")
            lines.append("    tensor_map = {")
            for name in view_tensor_names:
                lines.append(f"        '{name}': {name},")
            lines.append("    }")
            lines.append("    get_view = _get_viewer(tensor_map, o, v)")
            dtype_args = ", ".join(base_tensor_names)
            if dtype_args:
                lines.append(f"    dtype = np.result_type({dtype_args})")
            else:
                lines.append("    dtype = None")
            if "t1" in base_tensor_names:
                lines.append("    t1 = _apply_t1_sign(t1)")
            lines.append(
                f"    out = zeros_for_output('{output_labels}', o, v, dtype=dtype)"
            )
            grouped = _group_terms_by_subs(output_structs[output_key])
            for (subs, tensors), coeff in sorted(grouped.items()):
                if coeff == 0.0:
                    continue
                safe_subs = _safe_einsum_subs(subs)
                args = []
                for name, labels in tensors:
                    if name in view_tensor_names:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    out += ({coeff}) * np.einsum('{safe_subs}', {', '.join(args)}, optimize=True)"
                )
            lines.append("    return out")
            lines.append("")

    (output_dir / filename).write_text("\n".join(lines) + "\n")


def _build_structured_codegen_plan(
    output_structs,
    output_names,
    view_tensors,
    *,
    use_intermediates: bool = False,
    canonicalize_qp_antisymmetry: bool = True,
):
    output_order = [key for key in ("scalar", "X1", "X2") if key in output_structs]
    normalized_structs = {key: [] for key in output_order}
    output_labels_map = {}
    all_structs = []
    for output_key in output_order:
        for out_labels, tensors, coeff in output_structs[output_key]:
            if canonicalize_qp_antisymmetry:
                out_labels, tensors, coeff = _canonicalize_qp_antisymmetry(
                    out_labels, list(tensors), coeff
                )
                if abs(coeff) < 1e-12:
                    continue
            out_labels, tensors = _canonicalize_term_labels(out_labels, tensors)
            if output_key in output_labels_map:
                if output_labels_map[output_key] != out_labels:
                    raise ValueError(
                        f"Output '{output_key}' has inconsistent labels: "
                        f"{output_labels_map[output_key]} vs {out_labels}"
                    )
            else:
                output_labels_map[output_key] = out_labels
            struct = (out_labels, tensors, coeff)
            normalized_structs[output_key].append(struct)
            all_structs.append(struct)

    base_tensor_names = []
    base_tensor_set = set()
    for _out_labels, tensors, _coeff in all_structs:
        for name, _labels in tensors:
            if name not in base_tensor_set:
                base_tensor_set.add(name)
                base_tensor_names.append(name)
    base_tensor_names = _ordered_tensor_names(base_tensor_names)
    view_tensor_names = [name for name in base_tensor_names if name in view_tensors]
    non_view_tensor_names = [name for name in base_tensor_names if name not in view_tensors]

    inter_map = {}
    inter_defs = []
    emitted_structs = normalized_structs
    if use_intermediates:
        inter_map = _select_intermediates(all_structs)
        emitted_structs = {
            key: [
                (out_labels, _apply_intermediates(tensors, inter_map), coeff)
                for out_labels, tensors, coeff in normalized_structs[key]
            ]
            for key in output_order
        }
        used_intermediates = set()
        for terms in emitted_structs.values():
            for _out_labels, tensors, _coeff in terms:
                for name, _labels in tensors:
                    if name.startswith("I"):
                        used_intermediates.add(name)
        inter_defs = [
            info for info in inter_map.values() if info["name"] in used_intermediates
        ]
        inter_defs.sort(key=lambda info: int(info["name"][1:]))
        inter_defs, emitted_structs = _dedupe_intermediate_defs(inter_defs, emitted_structs)

    return {
        "output_order": output_order,
        "normalized_structs": normalized_structs,
        "emitted_structs": emitted_structs,
        "output_labels_map": output_labels_map,
        "base_tensor_names": base_tensor_names,
        "view_tensor_names": view_tensor_names,
        "non_view_tensor_names": non_view_tensor_names,
        "inter_defs": inter_defs,
    }


def _serialize_structs(terms):
    from autogen.codegen.projected_terms import serialize_coefficient

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


def _write_codegen_metadata(path: Path, plan: dict, output_names) -> None:
    data = {
        "outputs": {
            key: {
                "name": resolve_output_name(key, output_names),
                "labels": plan["output_labels_map"][key],
                "raw_terms": _serialize_structs(plan["normalized_structs"][key]),
                "emitted_terms": _serialize_structs(plan["emitted_structs"][key]),
            }
            for key in plan["output_order"]
        },
        "intermediates": [
            {
                "name": info["name"],
                "ops": [{"name": name, "labels": labels} for name, labels in info["ops"]],
                "out_labels": info["out_labels"],
                "count": info["count"],
            }
            for info in plan["inter_defs"]
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def _dedupe_intermediate_defs(inter_defs, emitted_structs):
    if not inter_defs:
        return inter_defs, emitted_structs

    dedup = {}
    alias = {}
    unique_defs = []
    for info in inter_defs:
        labels1 = info["ops"][0][1]
        labels2 = info["ops"][1][1]
        out_labels = info["out_labels"]
        subs = _safe_einsum_subs(f"{labels1},{labels2}->{out_labels}")
        key = (subs, tuple(op_name for op_name, _ in info["ops"]))
        keep = dedup.get(key)
        if keep is None:
            dedup[key] = info["name"]
            unique_defs.append(info)
        else:
            alias[info["name"]] = keep

    if not alias:
        return inter_defs, emitted_structs

    deduped_structs = {}
    for output_key, terms in emitted_structs.items():
        deduped_terms = []
        for out_labels, tensors, coeff in terms:
            deduped_tensors = [(alias.get(name, name), labels) for name, labels in tensors]
            deduped_terms.append((out_labels, deduped_tensors, coeff))
        deduped_structs[output_key] = deduped_terms
    return unique_defs, deduped_structs


def emit_structured_residuals(
    output_dir,
    output_structs,
    output_names,
    view_tensors,
    filename: str = "residuals.py",
    bog_qp: bool = False,
    use_intermediates: bool = False,
    metadata_filename: str | None = None,
    canonicalize_qp_antisymmetry: bool = True,
):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    plan = _build_structured_codegen_plan(
        output_structs,
        output_names,
        view_tensors,
        use_intermediates=use_intermediates,
        canonicalize_qp_antisymmetry=canonicalize_qp_antisymmetry,
    )
    output_order = plan["output_order"]
    emitted_structs = plan["emitted_structs"]
    output_labels_map = plan["output_labels_map"]
    base_tensor_names = plan["base_tensor_names"]
    view_tensor_names = plan["view_tensor_names"]
    non_view_tensor_names = plan["non_view_tensor_names"]
    inter_defs = plan["inter_defs"]

    lines = []
    lines.append("import numpy as np")
    lines.append("import os")
    lines.append("import re")
    lines.append("")
    lines.append("AUTOGEN_SPIN_SUMMED = False")
    lines.append("AUTOGEN_SPIN_SUMMED_MODE = 'spinorb'")
    lines.append(f"AUTOGEN_BOGOLIUBOV_QP = {bog_qp}")
    lines.append(
        "AUTOGEN_QP_T1_SIGN = float(os.getenv("
        "'AUTOGEN_QP_T1_SIGN', '-1.0' if AUTOGEN_BOGOLIUBOV_QP else '1.0'))"
    )
    lines.append(f"AUTOGEN_INTERMEDIATES = {bool(inter_defs)}")
    lines.append(f"VIEW_TENSORS = {tuple(view_tensor_names)}")
    lines.append("OCC = set('ijklmn')")
    lines.append("VIRT = set('abcdefgh')")
    lines.append("_LABEL_RE = re.compile(r\"[a-z][0-9]*\")")
    lines.append("")
    lines.append("def _apply_t1_sign(t1):")
    lines.append("    if AUTOGEN_QP_T1_SIGN == 1.0:")
    lines.append("        return t1")
    lines.append("    return AUTOGEN_QP_T1_SIGN * t1")
    lines.append("")
    lines.append("def _iter_labels(labels):")
    lines.append("    if not labels:")
    lines.append("        return []")
    lines.append("    return _LABEL_RE.findall(labels)")
    lines.append("")
    lines.append("def view_tensor(tensor, labels, o, v):")
    lines.append("    label_tokens = _iter_labels(labels)")
    lines.append("    idx = []")
    lines.append("    list_axes = []")
    lines.append("    for axis, label in enumerate(label_tokens):")
    lines.append("        if label and label[0] in OCC:")
    lines.append("            idx.append(o)")
    lines.append("            list_axes.append(axis)")
    lines.append("        elif label and label[0] in VIRT:")
    lines.append("            idx.append(v)")
    lines.append("            list_axes.append(axis)")
    lines.append("        else:")
    lines.append("            idx.append(slice(None))")
    lines.append("    if not list_axes:")
    lines.append("        return tensor[tuple(idx)]")
    lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
    lines.append("    ix_iter = iter(ix)")
    lines.append("    full_idx = []")
    lines.append("    for axis in range(len(idx)):")
    lines.append("        if axis in list_axes:")
    lines.append("            full_idx.append(next(ix_iter))")
    lines.append("        else:")
    lines.append("            full_idx.append(idx[axis])")
    lines.append("    return tensor[tuple(full_idx)]")
    lines.append("")
    lines.append("def zeros_for_output(labels, o, v, dtype=None):")
    lines.append("    label_tokens = _iter_labels(labels)")
    lines.append("    if not label_tokens:")
    lines.append("        return np.array(0.0, dtype=dtype) if dtype is not None else 0.0")
    lines.append("    shape = []")
    lines.append("    for label in label_tokens:")
    lines.append("        if label and label[0] in OCC:")
    lines.append("            shape.append(len(o))")
    lines.append("        elif label and label[0] in VIRT:")
    lines.append("            shape.append(len(v))")
    lines.append("        else:")
    lines.append("            shape.append(len(o) + len(v))")
    lines.append("    if dtype is None:")
    lines.append("        return np.zeros(tuple(shape))")
    lines.append("    return np.zeros(tuple(shape), dtype=dtype)")
    lines.append("")
    lines.append("def _get_viewer(tensor_map, o, v):")
    lines.append("    views = {}")
    lines.append("    def get_view(name, labels):")
    lines.append("        key = (name, labels)")
    lines.append("        if key in views:")
    lines.append("            return views[key]")
    lines.append("        tensor = tensor_map[name]")
    lines.append("        views[key] = view_tensor(tensor, labels, o, v)")
    lines.append("        return views[key]")
    lines.append("    return get_view")
    lines.append("")
    tensor_args = ", ".join(base_tensor_names + ["o", "v"])
    dtype_args = ", ".join(base_tensor_names)
    if inter_defs:
        non_view_args = ", ".join(["get_view"] + non_view_tensor_names)
        lines.append(f"def compute_intermediates({non_view_args}):")
        for info in inter_defs:
            name = info["name"]
            labels1 = info["ops"][0][1]
            labels2 = info["ops"][1][1]
            out_labels = info["out_labels"]
            subs = _safe_einsum_subs(f"{labels1},{labels2}->{out_labels}")
            args = []
            for op_name, op_labels in info["ops"]:
                if op_name in view_tensor_names:
                    args.append(f"get_view('{op_name}', '{op_labels}')")
                else:
                    args.append(op_name)
            lines.append(
                f"    {name} = np.einsum('{subs}', {', '.join(args)}, optimize=True)"
            )
        lines.append("    return {")
        for info in inter_defs:
            name = info["name"]
            lines.append(f"        '{name}': {name},")
        lines.append("    }")
        lines.append("")

    lines.append(f"def compute_outputs({tensor_args}):")
    lines.append("    tensor_map = {")
    for name in view_tensor_names:
        lines.append(f"        '{name}': {name},")
    lines.append("    }")
    lines.append("    get_view = _get_viewer(tensor_map, o, v)")
    if dtype_args:
        lines.append(f"    dtype = np.result_type({dtype_args})")
    else:
        lines.append("    dtype = None")
    if "t1" in base_tensor_names:
        lines.append("    t1 = _apply_t1_sign(t1)")
    if inter_defs:
        lines.append(f"    inter = compute_intermediates({non_view_args})")
        for info in inter_defs:
            name = info["name"]
            lines.append(f"    {name} = inter['{name}']")
    lines.append("    outputs = {}")
    used_intermediates = {info["name"] for info in inter_defs}
    for output_key in output_order:
        output_name = resolve_output_name(output_key, output_names)
        output_labels = output_labels_map[output_key]
        lines.append(
            f"    {output_name} = zeros_for_output('{output_labels}', o, v, dtype=dtype)"
        )
        grouped = _group_terms_by_subs(emitted_structs[output_key])
        for (subs, tensors), coeff in sorted(grouped.items()):
            if abs(coeff) < 1e-12:
                continue
            safe_subs = _safe_einsum_subs(subs)
            args = []
            for name, labels in tensors:
                if name in used_intermediates:
                    args.append(name)
                elif name in view_tensor_names:
                    args.append(f"get_view('{name}', '{labels}')")
                else:
                    args.append(name)
            lines.append(
                f"    {output_name} += ({coeff}) * np.einsum('{safe_subs}', {', '.join(args)}, optimize=True)"
            )
        lines.append(f"    outputs['{output_name}'] = {output_name}")
    lines.append("    return outputs")
    lines.append("")
    for output_key in output_order:
        output_name = resolve_output_name(output_key, output_names)
        lines.append(f"def compute_{output_name}({tensor_args}):")
        lines.append(f"    return compute_outputs({tensor_args})['{output_name}']")
        lines.append("")
    (output_dir / filename).write_text("\n".join(lines) + "\n")
    if metadata_filename is not None:
        _write_codegen_metadata(output_dir / metadata_filename, plan, output_names)


def emit_spin_adapted_wrapper(
    output_dir,
    output_names,
    mode: str = "full",
    eom_mode: bool = False,
):
    output_dir = Path(output_dir)
    r1_name = resolve_output_name("X1", output_names)
    r2_name = resolve_output_name("X2", output_names)
    inter_flag = mode == "intermediates"

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    if eom_mode:
        lines.append("from pyscf.cc import addons, eom_rccsd, eom_uccsd")
    else:
        lines.append("from pyscf.cc import addons")
    lines.append("")
    lines.append("from . import residuals_spinorb as spinorb")
    lines.append("")
    lines.append("AUTOGEN_SPIN_SUMMED = True")
    lines.append("AUTOGEN_SPIN_SUMMED_MODE = 'spinorb'")
    lines.append(f"AUTOGEN_INTERMEDIATES = {inter_flag}")
    lines.append("VIEW_TENSORS = ('f', 'g')")
    lines.append("")
    lines.append("def _orbspin(nocc, nmo):")
    lines.append("    orbspin = np.zeros(2 * nmo, dtype=int)")
    lines.append("    orbspin[1::2] = 1")
    lines.append("    return orbspin")
    lines.append("")
    lines.append("def _build_spin_orbital_fock(f, nocc):")
    lines.append("    nmo = f.shape[0]")
    lines.append("    nocc_so = 2 * nocc")
    lines.append("    nvir_so = 2 * (nmo - nocc)")
    lines.append("    f_so = np.zeros((nocc_so + nvir_so, nocc_so + nvir_so))")
    lines.append("    for p in range(nmo):")
    lines.append("        for q in range(nmo):")
    lines.append("            for spin in (0, 1):")
    lines.append("                if p < nocc:")
    lines.append("                    p_so = 2 * p + spin")
    lines.append("                else:")
    lines.append("                    p_so = nocc_so + 2 * (p - nocc) + spin")
    lines.append("                if q < nocc:")
    lines.append("                    q_so = 2 * q + spin")
    lines.append("                else:")
    lines.append("                    q_so = nocc_so + 2 * (q - nocc) + spin")
    lines.append("                f_so[p_so, q_so] = f[p, q]")
    lines.append("    return f_so")
    lines.append("")
    lines.append("def _build_spin_orbital_g(g_raw, nocc):")
    lines.append("    nmo = g_raw.shape[0]")
    lines.append("    nocc_so = 2 * nocc")
    lines.append("    nvir_so = 2 * (nmo - nocc)")
    lines.append("    nso = nocc_so + nvir_so")
    lines.append("    g_phys = g_raw.transpose(0, 2, 1, 3)")
    lines.append("    g_so = np.zeros((nso, nso, nso, nso))")
    lines.append("    for p in range(nmo):")
    lines.append("        for q in range(nmo):")
    lines.append("            for r in range(nmo):")
    lines.append("                for s in range(nmo):")
    lines.append("                    val = g_phys[p, q, r, s]")
    lines.append("                    for sp in (0, 1):")
    lines.append("                        for sq in (0, 1):")
    lines.append("                            for sr in (0, 1):")
    lines.append("                                for ss in (0, 1):")
    lines.append("                                    if sp != sr or sq != ss:")
    lines.append("                                        continue")
    lines.append("                                    if p < nocc:")
    lines.append("                                        p_so = 2 * p + sp")
    lines.append("                                    else:")
    lines.append("                                        p_so = nocc_so + 2 * (p - nocc) + sp")
    lines.append("                                    if q < nocc:")
    lines.append("                                        q_so = 2 * q + sq")
    lines.append("                                    else:")
    lines.append("                                        q_so = nocc_so + 2 * (q - nocc) + sq")
    lines.append("                                    if r < nocc:")
    lines.append("                                        r_so = 2 * r + sr")
    lines.append("                                    else:")
    lines.append("                                        r_so = nocc_so + 2 * (r - nocc) + sr")
    lines.append("                                    if s < nocc:")
    lines.append("                                        s_so = 2 * s + ss")
    lines.append("                                    else:")
    lines.append("                                        s_so = nocc_so + 2 * (s - nocc) + ss")
    lines.append("                                    g_so[p_so, q_so, r_so, s_so] = val")
    lines.append("    g_so_as = g_so - g_so.transpose(0, 1, 3, 2)")
    lines.append("    return g_so_as")
    lines.append("")

    if eom_mode:
        lines.append("def compute_outputs(f, g, t1, t2, r1, r2, o, v):")
        lines.append("    nocc = len(o)")
        lines.append("    nmo = f.shape[0]")
        lines.append("    orbspin = _orbspin(nocc, nmo)")
        lines.append("    f_so = _build_spin_orbital_fock(f, nocc)")
        lines.append("    g_so = _build_spin_orbital_g(g, nocc)")
        lines.append("    t1_so = addons.spatial2spin(t1.T, orbspin).T")
        lines.append("    t2_so = addons.spatial2spin(")
        lines.append("        t2.transpose(2, 3, 0, 1), orbspin")
        lines.append("    ).transpose(2, 3, 0, 1)")
        lines.append("    r1_so = eom_rccsd.spatial2spin_singlet(r1.T, orbspin).T")
        lines.append("    r2_so = eom_rccsd.spatial2spin_singlet(")
        lines.append("        r2.transpose(2, 3, 0, 1), orbspin")
        lines.append("    ).transpose(2, 3, 0, 1)")
        lines.append("    o_so = list(range(2 * nocc))")
        lines.append("    v_so = list(range(2 * nocc, 2 * nmo))")
        lines.append("    outs = spinorb.compute_outputs(f_so, g_so, t1_so, t2_so, r1_so, r2_so, o_so, v_so)")
        lines.append(f"    s1_so = outs['{r1_name}'].T")
        lines.append(f"    s2_so = outs['{r2_name}'].transpose(2, 3, 0, 1)")
        lines.append("    s1a, s1b = eom_uccsd.spin2spatial_eomee(s1_so, orbspin)")
        lines.append("    s2aa, s2ab, s2bb = eom_uccsd.spin2spatial_eomee(s2_so, orbspin)")
        lines.append("    sqrt2 = 2 ** 0.5")
        lines.append("    s1 = (0.5 * (s1a + s1b) * sqrt2).T")
        lines.append("    s2 = s2ab.transpose(2, 3, 0, 1) * sqrt2")
        lines.append(f"    return {{{r1_name!r}: s1, {r2_name!r}: s2}}")
        lines.append("")
        lines.append(f"def compute_{r1_name}(f, g, t1, t2, r1, r2, o, v):")
        lines.append(
            f"    return compute_outputs(f, g, t1, t2, r1, r2, o, v)[{r1_name!r}]"
        )
        lines.append("")
        lines.append(f"def compute_{r2_name}(f, g, t1, t2, r1, r2, o, v):")
        lines.append(
            f"    return compute_outputs(f, g, t1, t2, r1, r2, o, v)[{r2_name!r}]"
        )
    else:
        lines.append("def compute_outputs(f, g, t1, t2, o, v):")
        lines.append("    nocc = len(o)")
        lines.append("    nmo = f.shape[0]")
        lines.append("    orbspin = _orbspin(nocc, nmo)")
        lines.append("    f_so = _build_spin_orbital_fock(f, nocc)")
        lines.append("    g_so = _build_spin_orbital_g(g, nocc)")
        lines.append("    t1_so = addons.spatial2spin(t1.T, orbspin).T")
        lines.append("    t2_so = addons.spatial2spin(")
        lines.append("        t2.transpose(2, 3, 0, 1), orbspin")
        lines.append("    ).transpose(2, 3, 0, 1)")
        lines.append("    o_so = list(range(2 * nocc))")
        lines.append("    v_so = list(range(2 * nocc, 2 * nmo))")
        lines.append("    outs = spinorb.compute_outputs(f_so, g_so, t1_so, t2_so, o_so, v_so)")
        lines.append(f"    r1_so = outs['{r1_name}'].T")
        lines.append(f"    r2_so = outs['{r2_name}'].transpose(2, 3, 0, 1)")
        lines.append("    r1a, r1b = addons.spin2spatial(r1_so, orbspin)")
        lines.append("    r2aa, r2ab, r2bb = addons.spin2spatial(r2_so, orbspin)")
        lines.append("    r1 = r1a.T")
        lines.append("    r2 = r2ab.transpose(2, 3, 0, 1)")
        lines.append(f"    return {{{r1_name!r}: r1, {r2_name!r}: r2}}")
        lines.append("")
        lines.append(f"def compute_{r1_name}(f, g, t1, t2, o, v):")
        lines.append(f"    return compute_outputs(f, g, t1, t2, o, v)[{r1_name!r}]")
        lines.append("")
        lines.append(f"def compute_{r2_name}(f, g, t1, t2, o, v):")
        lines.append(f"    return compute_outputs(f, g, t1, t2, o, v)[{r2_name!r}]")

    lines.append("")
    (output_dir / "residuals.py").write_text("\n".join(lines) + "\n")


def _apply_qpccsd_r2_prefactor_fix(path: Path, *, quiet: bool = False) -> bool:
    """Apply qpCCSD residual fixes (R1/R2 scaling + missing h11 terms)."""
    if not path.exists():
        return False
    text = path.read_text()
    if "AUTOGEN_BOGOLIUBOV_QP" not in text:
        return False

    def _patch_block(block: str, kind: str) -> str:
        if kind == "r1":
            get_view_name = "get_view_r1"
            insert = (
                "    def get_view_r1(name, labels):\n"
                "        view = get_view(name, labels)\n"
                "        if name == 'h22':\n"
                "            return -0.5 * view\n"
                "        if name == 'h02':\n"
                "            return 0.5 * view\n"
                "        if name == 'h04':\n"
                "            return -0.25 * view\n"
                "        if name == 'h13':\n"
                "            return -0.5 * view\n"
                "        return view\n"
            )
            replace_target = "get_view_r1("
            # insert missing h11 commutator terms
            marker = "    out = zeros_for_output('pq', o, v, dtype=dtype)\n"
            h11_lines = (
                "    out += (-1.0) * np.einsum('pr,rq->pq', get_view_r1('h11', 'pq'), t1, optimize=True)\n"
                "    out += (-1.0) * np.einsum('pr,qr->pq', t1, get_view_r1('h11', 'pq'), optimize=True)\n"
            )
        else:
            get_view_name = "get_view_r2"
            insert = (
                "    def get_view_r2(name, labels):\n"
                "        view = get_view(name, labels)\n"
                "        if name in ('h11', 'h22'):\n"
                "            return 0.5 * view\n"
                "        if name == 'h04':\n"
                "            return 0.125 * view\n"
                "        if name == 'h31':\n"
                "            return -0.5 * view\n"
                "        if name == 'h13':\n"
                "            return 0.25 * view\n"
                "        if name == 'h02':\n"
                "            return 0.25 * view\n"
                "        return view\n"
            )
            replace_target = "get_view_r2("
            marker = None
            h11_lines = None

        # remove any existing get_view_r{kind}
        import re

        block = re.sub(r"\n\s*def get_view_r[12]\(.*?\n\s*return view\n", "\n", block, flags=re.DOTALL)
        target = "    get_view = _get_viewer(tensor_map, o, v)\n"
        if target in block:
            block = block.replace(target, target + insert, 1)

        # route all views through the new helper
        block = block.replace("get_view(", replace_target)
        block = block.replace(f"view = {get_view_name}(name, labels)", "view = get_view(name, labels)")

        if marker and marker in block and h11_lines:
            if "get_view_r1('h11'" not in block:
                block = block.replace(marker, marker + h11_lines, 1)
        return block

    # patch compute_r1
    start = text.find("def compute_r1")
    if start != -1:
        end = text.find("\ndef ", start + 1)
        if end == -1:
            end = len(text)
        block = text[start:end]
        block = _patch_block(block, "r1")
        text = text[:start] + block + text[end:]

    # patch compute_r2
    start = text.find("def compute_r2")
    if start != -1:
        end = text.find("\ndef ", start + 1)
        if end == -1:
            end = len(text)
        block = text[start:end]
        block = _patch_block(block, "r2")
        text = text[:start] + block + text[end:]

    path.write_text(text)
    if not quiet:
        print(f"Applied qpCCSD residual fixes to {path}")
    return True


def _module_path_from_dir(output_dir: Path):
    rel = output_dir.relative_to(ROOT)
    return ".".join(rel.parts)


def _validate_ccsd_outputs(spec_terms, output_names):
    output_keys = {term["output_key"] for term in spec_terms}
    if not output_keys and output_names:
        output_keys = set(output_names.keys())
    if "X1" not in output_keys or "X2" not in output_keys:
        raise ValueError("CCSD solver requires X1 and X2 outputs in the spec.")
    r1_name = resolve_output_name("X1", output_names)
    r2_name = resolve_output_name("X2", output_names)
    return r1_name, r2_name


def _load_qp_ccsd_projected_oracle_sections():
    path = ROOT / "scripts" / "qp_ccsd_projected_terms.txt"
    if not path.exists():
        raise FileNotFoundError(f"Projected qpCCSD oracle not found: {path}")

    sections = {"energy": [], "r1": [], "r2": []}
    current = None
    heading_map = {
        "compute_energy": "energy",
        "compute_r1": "r1",
        "compute_r2": "r2",
    }

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("== ") and stripped.endswith(" =="):
            current = None
            for marker, key in heading_map.items():
                if marker in stripped:
                    current = key
                    break
            continue
        if current and stripped.startswith("out +="):
            sections[current].append(stripped)

    missing = [name for name, lines in sections.items() if not lines]
    if missing:
        raise ValueError(
            "Projected qpCCSD oracle is incomplete: missing "
            + ", ".join(sorted(missing))
        )
    return sections


def emit_qp_ccsd_projected_residuals(output_dir, filename: str = "residuals_pnp.py"):
    projected_structs = build_projected_qp_ccsd_terms(max_order=4)
    use_intermediates = os.getenv("AUTOGEN_PROJECTED_INTERMEDIATES", "1") != "0"
    emit_structured_residuals(
        output_dir,
        projected_structs,
        {"scalar": "energy", "X1": "r1", "X2": "r2"},
        ["h02", "h04", "h11", "h13", "h20", "h22", "h31", "h40"],
        filename=filename,
        bog_qp=True,
        use_intermediates=use_intermediates,
        metadata_filename="projected_codegen_plan.json",
        canonicalize_qp_antisymmetry=False,
    )
    write_canonical_term_artifact(projected_structs, Path(output_dir) / "projected_canonical_terms.json")
    write_grouped_term_artifact(projected_structs, Path(output_dir) / "projected_grouped_terms.json")


def emit_ccsd_solver(output_dir, spec_terms, output_names, spin_orbital=False):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    module_path = _module_path_from_dir(output_dir)
    root_depth = _root_depth_for(output_dir)
    r1_name, r2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import os")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_integrals, run_scf")
    if spin_orbital:
        lines.append("from generated_code.pyscf_integrals import build_spin_orbital_integrals, spin_orbital_slices")
    lines.append(f"from {module_path} import residuals")
    lines.append("")
    lines.append("def mp2_init(f, g, o, v):")
    lines.append("    eps = np.diag(f)")
    lines.append("    eps_occ = eps[o]")
    lines.append("    eps_virt = eps[v]")
    lines.append("    denom_ai = eps_occ[None, :] - eps_virt[:, None]")
    lines.append("    denom_abij = (")
    lines.append("        eps_occ[None, None, :, None]")
    lines.append("        + eps_occ[None, None, None, :]")
    lines.append("        - eps_virt[:, None, None, None]")
    lines.append("        - eps_virt[None, :, None, None]")
    lines.append("    )")
    lines.append("    denom_ai = np.where(abs(denom_ai) < 1e-12, 1e-12, denom_ai)")
    lines.append("    denom_abij = np.where(abs(denom_abij) < 1e-12, 1e-12, denom_abij)")
    lines.append("    g_ijab = g[np.ix_(o, o, v, v)]")
    lines.append("    t1 = f[np.ix_(o, v)].T / denom_ai")
    lines.append("    t2 = g_ijab.transpose(2, 3, 0, 1) / denom_abij")
    lines.append("    return t1, t2, denom_ai, denom_abij")
    lines.append("")
    lines.append("def compute_energy(f, g_raw, t1, t2, o, v, spin_orbital=False):")
    lines.append("    if spin_orbital:")
    lines.append("        f_ov = f[np.ix_(o, v)]")
    lines.append("        g_ijab = g_raw[np.ix_(o, o, v, v)]")
    lines.append("        t1_ia = t1.T")
    lines.append("        t2_ijab = t2.transpose(2, 3, 0, 1)")
    lines.append("        e = np.einsum('ia,ia->', f_ov, t1_ia)")
    lines.append("        e += 0.25 * np.einsum('ijab,abij->', g_ijab, t2)")
    lines.append("        e += 0.5 * np.einsum('ijab,ai,bj->', g_ijab, t1, t1)")
    lines.append("        return e")
    lines.append("    f_ov = f[np.ix_(o, v)]")
    lines.append("    t1_ia = t1.T")
    lines.append("    t2_ijab = t2.transpose(2, 3, 0, 1)")
    lines.append("    tau = t2_ijab + np.einsum('ia,jb->ijab', t1_ia, t1_ia)")
    lines.append("    eris_ovvo = g_raw[np.ix_(o, v, v, o)]")
    lines.append("    e = 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)")
    lines.append("    e += 2.0 * np.einsum('ijab,iabj->', tau, eris_ovvo)")
    lines.append("    e -= np.einsum('jiab,iabj->', tau, eris_ovvo)")
    lines.append("    return e")
    lines.append("")
    lines.append("def solve_ccsd(mol=None, max_iter=50, tol=1e-8, damping=0.0, diis_start=2, max_diis=6):")
    lines.append("    if mol is None:")
    lines.append("        mol = build_h2o_631g()")
    lines.append("    mf = run_scf(mol)")
    lines.append("    ints = compute_integrals(mol, mf=mf)")
    if spin_orbital:
        lines.append("    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is True:")
        lines.append("        raise ValueError('Spin-orbital solver requires AUTOGEN_SPIN_SUMMED=0 residuals.')")
        lines.append("    f, g = build_spin_orbital_integrals(ints['f'], ints['g_raw'])")
        lines.append("    g_raw = g")
        lines.append("    o, v = spin_orbital_slices(ints['nocc'], ints['nmo'])")
    else:
        lines.append("    f = ints['f']")
        lines.append("    g_raw = ints['g_raw']")
        lines.append("    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is True:")
        lines.append("        g = g_raw")
        lines.append("    else:")
        lines.append("        g = ints['g']")
        lines.append("    nocc = ints['nocc']")
        lines.append("    nmo = ints['nmo']")
        lines.append("    o = list(range(nocc))")
        lines.append("    v = list(range(nocc, nmo))")
    lines.append("")
    lines.append("    t1, t2, denom_ai, denom_abij = mp2_init(f, g, o, v)")
    lines.append(f"    energy = compute_energy(f, g_raw, t1, t2, o, v, spin_orbital={spin_orbital})")
    lines.append("")
    lines.append("    t1_list = []")
    lines.append("    t2_list = []")
    lines.append("    err_list = []")
    lines.append("    tensor_args = {'f': f, 'g': g, 't1': t1, 't2': t2, 'o': o, 'v': v}")
    lines.append("")
    lines.append("    def diis_extrapolate(t1_list, t2_list, err_list):")
    lines.append("        n = len(err_list)")
    lines.append("        b = np.empty((n + 1, n + 1))")
    lines.append("        b[-1, :] = -1.0")
    lines.append("        b[:, -1] = -1.0")
    lines.append("        b[-1, -1] = 0.0")
    lines.append("        for i in range(n):")
    lines.append("            for j in range(n):")
    lines.append("                b[i, j] = np.vdot(err_list[i], err_list[j]).real")
    lines.append("        rhs = np.zeros(n + 1)")
    lines.append("        rhs[-1] = -1.0")
    lines.append("        try:")
    lines.append("            coeff = np.linalg.solve(b, rhs)[:-1]")
    lines.append("        except np.linalg.LinAlgError:")
    lines.append("            return t1_list[-1], t2_list[-1]")
    lines.append("        t1_new = sum(c * t for c, t in zip(coeff, t1_list))")
    lines.append("        t2_new = sum(c * t for c, t in zip(coeff, t2_list))")
    lines.append("        return t1_new, t2_new")
    lines.append("")
    lines.append("    for it in range(1, max_iter + 1):")
    lines.append("        tensor_args['t1'] = t1")
    lines.append("        tensor_args['t2'] = t2")
    lines.append("        if hasattr(residuals, 'compute_outputs'):")
    lines.append(f"            outs = residuals.compute_outputs(f, g, t1, t2, o, v)")
    lines.append(f"            r1 = outs['{r1_name}']")
    lines.append(f"            r2 = outs['{r2_name}']")
    lines.append("        else:")
    lines.append(f"            r1 = residuals.compute_{r1_name}(f, g, t1, t2, o, v)")
    lines.append(f"            r2 = residuals.compute_{r2_name}(f, g, t1, t2, o, v)")
    lines.append("        t1_new = t1 + r1 / denom_ai")
    lines.append("        t2_new = t2 + r2 / denom_abij")
    lines.append("        if damping > 0.0:")
    lines.append("            t1_new = (1.0 - damping) * t1_new + damping * t1")
    lines.append("            t2_new = (1.0 - damping) * t2_new + damping * t2")
    lines.append("        err = np.concatenate([r1.ravel(), r2.ravel()])")
    lines.append("        t1_list.append(t1_new.copy())")
    lines.append("        t2_list.append(t2_new.copy())")
    lines.append("        err_list.append(err)")
    lines.append("        if len(err_list) > max_diis:")
    lines.append("            t1_list.pop(0)")
    lines.append("            t2_list.pop(0)")
    lines.append("            err_list.pop(0)")
    lines.append("        if it >= diis_start and len(err_list) >= 2:")
    lines.append("            t1_new, t2_new = diis_extrapolate(t1_list, t2_list, err_list)")
    lines.append(f"        new_energy = compute_energy(f, g_raw, t1_new, t2_new, o, v, spin_orbital={spin_orbital})")
    lines.append("        r_norm = max(np.max(np.abs(r1)), np.max(np.abs(r2)))")
    lines.append("        e_diff = abs(new_energy - energy)")
    lines.append("        print(f'iter {it:3d}  energy {new_energy: .10f}  |R| {r_norm:.3e}  dE {e_diff:.3e}')")
    lines.append("        t1, t2, energy = t1_new, t2_new, new_energy")
    lines.append("        if r_norm < tol and e_diff < tol:")
    lines.append("            break")
    lines.append("    return energy, t1, t2")
    lines.append("")
    lines.append("def main():")
    lines.append("    energy, _t1, _t2 = solve_ccsd()")
    lines.append("    print('CCSD correlation energy (iterative):', energy)")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    (output_dir / "solver.py").write_text("\n".join(lines) + "\n")


def emit_qp_ccsd_solver(output_dir, spec_terms, output_names):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    module_path = _module_path_from_dir(output_dir)
    root_depth = _root_depth_for(output_dir)
    r1_name, r2_name = _validate_ccsd_outputs(spec_terms, output_names)
    energy_name = resolve_output_name("scalar", output_names)
    expect_bog_qp = _bog_qp_enabled()

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from generated_code.pyscf_integrals import (")
    lines.append("    build_bogoliubov_base,")
    lines.append("    build_bogoliubov_hamiltonian,")
    lines.append("    build_h2o_631g,")
    lines.append("    run_scf,")
    lines.append("    spin_orbital_slices,")
    lines.append(")")
    lines.append(f"from {module_path} import residuals")
    lines.append("try:")
    lines.append(f"    from {module_path} import residuals_pnp")
    lines.append("except Exception:")
    lines.append("    residuals_pnp = None")
    lines.append(f"from {module_path} import pnp_u1_qp")
    lines.append("")
    lines.append("def mp2_init(h11, h22, o, v):")
    lines.append("    eps = np.diag(h11)")
    lines.append("    eps_occ = eps[o]")
    lines.append("    eps_virt = eps[v]")
    lines.append("    denom_ai = eps_occ[None, :] - eps_virt[:, None]")
    lines.append("    denom_abij = (")
    lines.append("        eps_occ[None, None, :, None]")
    lines.append("        + eps_occ[None, None, None, :]")
    lines.append("        - eps_virt[:, None, None, None]")
    lines.append("        - eps_virt[None, :, None, None]")
    lines.append("    )")
    lines.append("    denom_ai = np.where(abs(denom_ai) < 1e-12, 1e-12, denom_ai)")
    lines.append("    denom_abij = np.where(abs(denom_abij) < 1e-12, 1e-12, denom_abij)")
    lines.append("    h22_ijab = h22[np.ix_(o, o, v, v)]")
    lines.append("    t1 = h11[np.ix_(o, v)].T / denom_ai")
    lines.append("    t2 = h22_ijab.transpose(2, 3, 0, 1) / denom_abij")
    lines.append("    return t1, t2, denom_ai, denom_abij")
    lines.append("")
    lines.append("def solve_qp_ccsd(mol=None, mf=None, dm1=None, max_iter=50, tol=1e-8, damping=0.0, diis_start=2, max_diis=6, use_diis=True, line_search=True, ls_max_steps=6, ls_shrink=0.5, step_cap=0.2, step_min=1e-4, pnp_u1=False, pnp_opts=None):")
    lines.append("    if mol is None:")
    lines.append("        mol = build_h2o_631g()")
    lines.append("    if mf is None:")
    lines.append("        mf = run_scf(mol)")
    lines.append("    e_nuc = mol.energy_nuc()")
    lines.append("    base = build_bogoliubov_base(mol, mf=mf, dm1=dm1)")
    lines.append("    ints = build_bogoliubov_hamiltonian(mol, mf=mf, dm1=dm1, base=base)")
    lines.append("    ref_energy = ints['E0']")
    lines.append("    residuals_use = residuals")
    lines.append("    if pnp_u1:")
    lines.append("        if residuals_pnp is None:")
    lines.append("            raise ValueError(")
    lines.append("                'Projected qpCCSD requires residuals_pnp. '")
    lines.append("                'Regenerate with AUTOGEN_QP_Z_CONTRACTION=1 and '")
    lines.append("                'AUTOGEN_RESIDUALS_BASENAME=residuals_pnp.py.'")
    lines.append("            )")
    lines.append("        residuals_use = residuals_pnp")
    lines.append("    if getattr(residuals_use, 'AUTOGEN_SPIN_SUMMED', None) is True:")
    lines.append("        raise ValueError('qp-CCSD solver requires AUTOGEN_SPIN_SUMMED=0 residuals.')")
    lines.append("    h11 = ints['h11']")
    lines.append("    h20 = ints['h20']")
    lines.append("    h02 = ints['h02']")
    lines.append("    h22 = ints['h22']")
    lines.append("    h31 = ints['h31']")
    lines.append("    h13 = ints['h13']")
    lines.append("    h40 = ints['h40']")
    lines.append("    h04 = ints['h04']")
    lines.append("    bog_qp = getattr(residuals_use, 'AUTOGEN_BOGOLIUBOV_QP', False)")
    if expect_bog_qp:
        lines.append("    if not bog_qp:")
        lines.append("        raise ValueError(")
        lines.append("            'qp-CCSD solver expects Bogoliubov-QP residuals. '")
        lines.append("            'Regenerate with AUTOGEN_BOGOLIUBOV_QP=1.'")
        lines.append("        )")
        lines.append("    required = {'h11', 'h20', 'h22', 'h31', 'h40'}")
    else:
        lines.append("    required = {'h11', 'h22'}")
    lines.append("    view = set(getattr(residuals_use, 'VIEW_TENSORS', ()))")
    lines.append("    if not required.issubset(view):")
    lines.append("        raise ValueError(")
    lines.append("            'qp-CCSD residuals are missing required tensors. '")
    lines.append("            f'Missing: {required - view}'")
    lines.append("        )")
    lines.append("    o, v = spin_orbital_slices(ints['nocc'], ints['nmo'])")
    lines.append("    pnp_cache = None")
    lines.append("    if pnp_u1:")
    lines.append("        pnp_opts = pnp_opts or {}")
    lines.append("        pnp_cache = pnp_u1_qp.build_pnp_cache(base, mol.nelectron, **pnp_opts)")
    lines.append("")
    lines.append("    if bog_qp:")
    lines.append("        eps = np.diag(h11)")
    lines.append("        denom_ai = eps[:, None] + eps[None, :]")
    lines.append("        denom_abij = (")
    lines.append("            eps[:, None, None, None]")
    lines.append("            + eps[None, :, None, None]")
    lines.append("            + eps[None, None, :, None]")
    lines.append("            + eps[None, None, None, :]")
    lines.append("        )")
    lines.append("        denom_ai = np.where(abs(denom_ai) < 1e-12, 1e-12, denom_ai)")
    lines.append("        denom_abij = np.where(abs(denom_abij) < 1e-12, 1e-12, denom_abij)")
    lines.append("        t1 = np.zeros_like(h11)")
    lines.append("        t2 = np.zeros_like(h40)")
    lines.append("    else:")
    lines.append("        t1, t2, denom_ai, denom_abij = mp2_init(h11, h22, o, v)")
    lines.append("    z = np.zeros_like(h11, dtype=np.result_type(h11, 1.0j))")
    lines.append("    energy = ref_energy")
    lines.append("")
    lines.append("    tensor_args = {")
    lines.append("        't1': t1,")
    lines.append("        't2': t2,")
    lines.append("        'h11': h11,")
    lines.append("        'h20': h20,")
    lines.append("        'h02': h02,")
    lines.append("        'h22': h22,")
    lines.append("        'h31': h31,")
    lines.append("        'h13': h13,")
    lines.append("        'h40': h40,")
    lines.append("        'h04': h04,")
    lines.append("        'z': z,")
    lines.append("        'o': o,")
    lines.append("        'v': v,")
    lines.append("    }")
    lines.append("    outputs_params = None")
    lines.append("    r1_params = None")
    lines.append("    r2_params = None")
    lines.append("    energy_params = None")
    lines.append("")
    lines.append("    def _param_list(func):")
    lines.append("        return func.__code__.co_varnames[:func.__code__.co_argcount]")
    lines.append("")
    lines.append("    def _as_real_if_close(val, tol=1e-10):")
    lines.append("        arr = np.asarray(val)")
    lines.append("        if np.iscomplexobj(arr) and np.max(np.abs(arr.imag)) < tol:")
    lines.append("            return arr.real")
    lines.append("        return val")
    lines.append("")
    lines.append("    if pnp_cache is not None:")
    lines.append("        if hasattr(residuals_use, 'compute_outputs'):")
    lines.append("            params = _param_list(residuals_use.compute_outputs)")
    lines.append("        else:")
    lines.append("            params = _param_list(residuals_use.compute_r1)")
    lines.append("        if 'z' not in params:")
    lines.append("            raise ValueError(")
    lines.append("                'Projected qpCCSD requires Z contractions. '")
    lines.append("                'Regenerate with AUTOGEN_QP_Z_CONTRACTION=1.'")
    lines.append("            )")
    lines.append("")
    lines.append("    t1_list = []")
    lines.append("    t2_list = []")
    lines.append("    err_list = []")
    lines.append("")
    lines.append("    def diis_extrapolate(t1_list, t2_list, err_list):")
    lines.append("        n = len(err_list)")
    lines.append("        b = np.empty((n + 1, n + 1))")
    lines.append("        b[-1, :] = -1.0")
    lines.append("        b[:, -1] = -1.0")
    lines.append("        b[-1, -1] = 0.0")
    lines.append("        for i in range(n):")
    lines.append("            for j in range(n):")
    lines.append("                b[i, j] = np.vdot(err_list[i], err_list[j]).real")
    lines.append("        rhs = np.zeros(n + 1)")
    lines.append("        rhs[-1] = -1.0")
    lines.append("        try:")
    lines.append("            coeff = np.linalg.solve(b, rhs)[:-1]")
    lines.append("        except np.linalg.LinAlgError:")
    lines.append("            return t1_list[-1], t2_list[-1]")
    lines.append("        t1_new = sum(c * t for c, t in zip(coeff, t1_list))")
    lines.append("        t2_new = sum(c * t for c, t in zip(coeff, t2_list))")
    lines.append("        return t1_new, t2_new")
    lines.append("")
    lines.append("    def evaluate(t1_eval, t2_eval):")
    lines.append("        nonlocal outputs_params, r1_params, r2_params, energy_params")
    lines.append("        tensor_args['t1'] = t1_eval")
    lines.append("        tensor_args['t2'] = t2_eval")
    lines.append("        if pnp_cache is not None:")
    lines.append("            r1_acc = None")
    lines.append("            r2_acc = None")
    lines.append("            e_acc = 0.0 + 0.0j")
    lines.append("            for cache in pnp_cache:")
    lines.append("                tensor_args['z'] = cache.Z")
    lines.append("                if hasattr(residuals_use, 'compute_outputs'):")
    lines.append("                    if outputs_params is None:")
    lines.append("                        outputs_params = residuals_use.compute_outputs.__code__.co_varnames[:residuals_use.compute_outputs.__code__.co_argcount]")
    lines.append("                    call_args = {name: tensor_args[name] for name in outputs_params}")
    lines.append("                    outs = residuals_use.compute_outputs(**call_args)")
    lines.append(f"                    r1_k = outs['{r1_name}']")
    lines.append(f"                    r2_k = outs['{r2_name}']")
    lines.append(f"                    e_k = ref_energy + outs.get('{energy_name}', 0.0)")
    lines.append("                else:")
    lines.append(f"                    if r1_params is None:")
    lines.append(f"                        r1_params = residuals_use.compute_{r1_name}.__code__.co_varnames[:residuals_use.compute_{r1_name}.__code__.co_argcount]")
    lines.append(f"                    if r2_params is None:")
    lines.append(f"                        r2_params = residuals_use.compute_{r2_name}.__code__.co_varnames[:residuals_use.compute_{r2_name}.__code__.co_argcount]")
    lines.append(f"                    r1_k = residuals_use.compute_{r1_name}(**{{name: tensor_args[name] for name in r1_params}})")
    lines.append(f"                    r2_k = residuals_use.compute_{r2_name}(**{{name: tensor_args[name] for name in r2_params}})")
    lines.append(f"                    if hasattr(residuals_use, 'compute_{energy_name}'):")
    lines.append(f"                        if energy_params is None:")
    lines.append(f"                            energy_params = residuals_use.compute_{energy_name}.__code__.co_varnames[:residuals_use.compute_{energy_name}.__code__.co_argcount]")
    lines.append(f"                        e_k = ref_energy + residuals_use.compute_{energy_name}(**{{name: tensor_args[name] for name in energy_params}})")
    lines.append("                    else:")
    lines.append("                        e_k = energy")
    lines.append("                if r1_acc is None:")
    lines.append("                    r1_acc = np.zeros_like(r1_k, dtype=complex)")
    lines.append("                    r2_acc = np.zeros_like(r2_k, dtype=complex)")
    lines.append("                r1_acc += cache.y * r1_k")
    lines.append("                r2_acc += cache.y * r2_k")
    lines.append("                e_acc += cache.y * e_k")
    lines.append("            r1_eval = _as_real_if_close(r1_acc)")
    lines.append("            r2_eval = _as_real_if_close(r2_acc)")
    lines.append("            e_eval = _as_real_if_close(e_acc)")
    lines.append("        else:")
    lines.append("            if hasattr(residuals_use, 'compute_outputs'):")
    lines.append("                if outputs_params is None:")
    lines.append("                    outputs_params = residuals_use.compute_outputs.__code__.co_varnames[:residuals_use.compute_outputs.__code__.co_argcount]")
    lines.append("                call_args = {name: tensor_args[name] for name in outputs_params}")
    lines.append("                outs = residuals_use.compute_outputs(**call_args)")
    lines.append(f"                r1_eval = outs['{r1_name}']")
    lines.append(f"                r2_eval = outs['{r2_name}']")
    lines.append(f"                e_eval = ref_energy + outs.get('{energy_name}', 0.0)")
    lines.append("            else:")
    lines.append(f"                if r1_params is None:")
    lines.append(f"                    r1_params = residuals_use.compute_{r1_name}.__code__.co_varnames[:residuals_use.compute_{r1_name}.__code__.co_argcount]")
    lines.append(f"                if r2_params is None:")
    lines.append(f"                    r2_params = residuals_use.compute_{r2_name}.__code__.co_varnames[:residuals_use.compute_{r2_name}.__code__.co_argcount]")
    lines.append(f"                r1_eval = residuals_use.compute_{r1_name}(**{{name: tensor_args[name] for name in r1_params}})")
    lines.append(f"                r2_eval = residuals_use.compute_{r2_name}(**{{name: tensor_args[name] for name in r2_params}})")
    lines.append(f"                if hasattr(residuals_use, 'compute_{energy_name}'):")
    lines.append(f"                    if energy_params is None:")
    lines.append(f"                        energy_params = residuals_use.compute_{energy_name}.__code__.co_varnames[:residuals_use.compute_{energy_name}.__code__.co_argcount]")
    lines.append(f"                    e_eval = ref_energy + residuals_use.compute_{energy_name}(**{{name: tensor_args[name] for name in energy_params}})")
    lines.append("                else:")
    lines.append("                    e_eval = energy")
    lines.append("        return r1_eval, r2_eval, e_eval")
    lines.append("")
    lines.append("    for it in range(1, max_iter + 1):")
    lines.append("        r1, r2, new_energy = evaluate(t1, t2)")
    lines.append("        r_norm = max(np.max(np.abs(r1)), np.max(np.abs(r2)))")
    lines.append("        e_diff = abs(new_energy - energy)")
    lines.append("")
    lines.append("        dt1 = r1 / denom_ai")
    lines.append("        dt2 = r2 / denom_abij")
    lines.append("        t1_prop = t1 + dt1")
    lines.append("        t2_prop = t2 + dt2")
    lines.append("        if damping > 0.0:")
    lines.append("            t1_prop = (1.0 - damping) * t1_prop + damping * t1")
    lines.append("            t2_prop = (1.0 - damping) * t2_prop + damping * t2")
    lines.append("")
    lines.append("        err = np.concatenate([r1.ravel(), r2.ravel()])")
    lines.append("        if use_diis:")
    lines.append("            t1_list.append(t1_prop.copy())")
    lines.append("            t2_list.append(t2_prop.copy())")
    lines.append("            err_list.append(err)")
    lines.append("            if len(err_list) > max_diis:")
    lines.append("                t1_list.pop(0)")
    lines.append("                t2_list.pop(0)")
    lines.append("                err_list.pop(0)")
    lines.append("            if it >= diis_start and len(err_list) >= 2:")
    lines.append("                t1_prop, t2_prop = diis_extrapolate(t1_list, t2_list, err_list)")
    lines.append("")
    lines.append("        dt1_step = t1_prop - t1")
    lines.append("        dt2_step = t2_prop - t2")
    lines.append("        scale = 1.0")
    lines.append("        if step_cap is not None and step_cap > 0.0:")
    lines.append("            max_step = max(np.max(np.abs(dt1_step)), np.max(np.abs(dt2_step)))")
    lines.append("            if max_step > step_cap:")
    lines.append("                scale = step_cap / max_step")
    lines.append("")
    lines.append("        t1_new = t1 + scale * dt1_step")
    lines.append("        t2_new = t2 + scale * dt2_step")
    lines.append("        energy_next = new_energy")
    lines.append("        r_norm_report = r_norm")
    lines.append("        if line_search:")
    lines.append("            best_scale = scale")
    lines.append("            best_r = None")
    lines.append("            best_energy = None")
    lines.append("            best_t1 = None")
    lines.append("            best_t2 = None")
    lines.append("            for _ in range(ls_max_steps + 1):")
    lines.append("                t1_trial = t1 + best_scale * dt1_step")
    lines.append("                t2_trial = t2 + best_scale * dt2_step")
    lines.append("                r1_trial, r2_trial, e_trial = evaluate(t1_trial, t2_trial)")
    lines.append("                r_trial = max(np.max(np.abs(r1_trial)), np.max(np.abs(r2_trial)))")
    lines.append("                if best_r is None or r_trial < best_r:")
    lines.append("                    best_r = r_trial")
    lines.append("                    best_energy = e_trial")
    lines.append("                    best_t1 = t1_trial")
    lines.append("                    best_t2 = t2_trial")
    lines.append("                if r_trial <= r_norm:")
    lines.append("                    break")
    lines.append("                best_scale *= ls_shrink")
    lines.append("                if best_scale < step_min:")
    lines.append("                    break")
    lines.append("            if best_t1 is not None:")
    lines.append("                t1_new = best_t1")
    lines.append("                t2_new = best_t2")
    lines.append("                energy_next = best_energy")
    lines.append("                r_norm_report = best_r")
    lines.append("")
    lines.append("        e_diff = abs(energy_next - energy)")
    lines.append("        print(f'iter {it:3d}  energy {energy_next: .10f}  |R| {r_norm_report:.3e}  dE {e_diff:.3e}')")
    lines.append("        t1, t2, energy = t1_new, t2_new, energy_next")
    lines.append("        if r_norm_report < tol and e_diff < tol:")
    lines.append("            break")
    lines.append("    return energy, t1, t2")
    lines.append("")
    lines.append("def main():")
    lines.append("    energy, _t1, _t2 = solve_qp_ccsd()")
    lines.append("    print('qp-CCSD total energy (iterative):', energy)")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    (output_dir / "solver.py").write_text("\n".join(lines) + "\n")


def _is_qp_ccsd_target(output_dir: Path, spec_path: str | os.PathLike[str]) -> bool:
    stem = Path(spec_path).stem.lower()
    name = output_dir.name.lower()
    return name == "qp_ccsd" or stem == "qp_ccsd_spec"


def emit_ccsd_pyscf_test(output_dir, spec_terms, output_names, pyscf_mol=None, spin_orbital=False):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    module_path = _module_path_from_dir(output_dir)
    root_depth = _root_depth_for(output_dir)
    _r1_name, _r2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import os")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from pyscf import scf, cc, gto")
    lines.append(f"from {module_path}.solver import solve_ccsd")
    lines.append("")
    atom = "H 0 0 0; H 0 0 0.74"
    basis = "sto-3g"
    unit = "Angstrom"
    charge = 0
    spin = 0
    if pyscf_mol:
        atom = pyscf_mol.get("atom", atom)
        basis = pyscf_mol.get("basis", basis)
        unit = pyscf_mol.get("unit", unit)
        charge = pyscf_mol.get("charge", charge)
        spin = pyscf_mol.get("spin", spin)
    lines.append("def build_molecule():")
    lines.append("    return gto.M(")
    lines.append(f"        atom={atom!r},")
    lines.append(f"        basis={basis!r},")
    lines.append(f"        unit={unit!r},")
    lines.append(f"        charge={charge},")
    lines.append(f"        spin={spin},")
    lines.append("    )")
    lines.append("")
    lines.append("def main():")
    lines.append("    mol = build_molecule()")
    lines.append("    mf = scf.RHF(mol).run()")
    lines.append("    mycc = cc.CCSD(mf).run()")
    lines.append("")
    if spin_orbital:
        lines.append("    if getattr(solve_ccsd.__globals__['residuals'], 'AUTOGEN_SPIN_SUMMED', None) is True:")
        lines.append("        raise ValueError('Spin-orbital test requires AUTOGEN_SPIN_SUMMED=0 residuals.')")
    lines.append("    e_autogen, _t1, _t2 = solve_ccsd(mol=mol)")
    lines.append("    diff = abs(e_autogen - mycc.e_corr)")
    lines.append("    print('PySCF CCSD corr energy:', mycc.e_corr)")
    lines.append("    print('Autogen CCSD corr energy:', e_autogen)")
    lines.append("    print('abs diff:', diff)")
    lines.append("    assert diff < 1e-7")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    (output_dir / "pyscf_test.py").write_text("\n".join(lines) + "\n")


def emit_eom_solver(output_dir, spec_terms, output_names, spin_orbital=False):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    module_path = _module_path_from_dir(output_dir)
    root_depth = _root_depth_for(output_dir)
    s1_name, s2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import os")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from pyscf import cc, lib")
    lines.append("from pyscf.cc import eom_rccsd")
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_integrals, run_scf")
    lines.append(f"from {module_path} import residuals")
    lines.append("")
    lines.append(f"SPIN_ORBITAL = {spin_orbital}")
    lines.append("")
    lines.append("def _abij_to_iajb(r2):")
    lines.append("    return r2.transpose(2, 0, 3, 1)")
    lines.append("")
    lines.append("def _iajb_to_abij(r2):")
    lines.append("    return r2.transpose(1, 3, 0, 2)")
    lines.append("")
    lines.append("def pack(r1, r2, nocc, nvirt):")
    lines.append("    r1_ia = r1.T")
    lines.append("    r2_iajb = _abij_to_iajb(r2)")
    lines.append("    nov = nocc * nvirt")
    lines.append("    mat = r2_iajb.reshape(nov, nov)")
    lines.append("    tril = np.tril_indices(nov)")
    lines.append("    return np.concatenate([r1_ia.ravel(), mat[tril]])")
    lines.append("")
    lines.append("def unpack(vec, nocc, nvirt):")
    lines.append("    nov = nocc * nvirt")
    lines.append("    n1 = nov")
    lines.append("    r1_ia = vec[:n1].reshape(nocc, nvirt)")
    lines.append("    mat = lib.unpack_tril(vec[n1:], filltriu=lib.SYMMETRIC)")
    lines.append("    r2_iajb = mat.reshape(nocc, nvirt, nocc, nvirt)")
    lines.append("    r2 = _iajb_to_abij(r2_iajb)")
    lines.append("    return r1_ia.T, r2")
    lines.append("")
    lines.append("def build_reference(mol, mf=None, t1=None, t2=None):")
    lines.append("    if mf is None:")
    lines.append("        mf = run_scf(mol)")
    lines.append("    if t1 is None or t2 is None:")
    lines.append("        mycc = cc.CCSD(mf).run()")
    lines.append("        t1 = mycc.t1.T")
    lines.append("        t2 = mycc.t2.transpose(2, 3, 0, 1)")
    lines.append("    ints = compute_integrals(mol, mf=mf)")
    lines.append("    f = ints['f']")
    lines.append("    g_raw = ints['g_raw']")
    lines.append("    mode = getattr(residuals, 'AUTOGEN_SPIN_SUMMED_MODE', 'direct')")
    lines.append("    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is True:")
    lines.append("        if mode == 'spinorb':")
    lines.append("            g = g_raw")
    lines.append("        else:")
    lines.append("            g = g_raw.transpose(0, 2, 1, 3)")
    lines.append("    else:")
    lines.append("        g = ints['g']")
    lines.append("    nocc = ints['nocc']")
    lines.append("    nmo = ints['nmo']")
    lines.append("    o = list(range(nocc))")
    lines.append("    v = list(range(nocc, nmo))")
    lines.append("    return f, g, t1, t2, o, v")
    lines.append("")
    lines.append("def sigma_vector(vec, f, g, t1, t2, o, v):")
    lines.append("    if isinstance(vec, (list, tuple)):")
    lines.append("        return np.vstack([sigma_vector(vec_row, f, g, t1, t2, o, v) for vec_row in vec])")
    lines.append("    if vec.ndim == 2:")
    lines.append("        return np.vstack([sigma_vector(vec_row, f, g, t1, t2, o, v) for vec_row in vec])")
    lines.append("    nocc = len(o)")
    lines.append("    nvirt = len(v)")
    lines.append("    r1, r2 = unpack(vec, nocc, nvirt)")
    lines.append(f"    s1 = residuals.compute_{s1_name}(f, g, t1, t2, r1, r2, o, v)")
    lines.append(f"    s2 = residuals.compute_{s2_name}(f, g, t1, t2, r1, r2, o, v)")
    lines.append("    return pack(s1, s2, nocc, nvirt)")
    lines.append("")
    lines.append("def solve_eom_ccsd(mol=None, mf=None, t1=None, t2=None, nroots=3, max_iter=50, tol=1e-8, max_space=20):")
    lines.append("    if mol is None:")
    lines.append("        mol = build_h2o_631g()")
    lines.append("    if SPIN_ORBITAL:")
    lines.append("        raise ValueError('EE-EOM-CCSD solver is spin-summed RHF only.')")
    lines.append("    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is not True:")
    lines.append("        raise ValueError('EE-EOM-CCSD solver requires spin-summed residuals.')")
    lines.append("    f, g, t1, t2, o, v = build_reference(mol, mf=mf, t1=t1, t2=t2)")
    lines.append("    use_pyscf_diag = os.getenv('AUTOGEN_EOM_USE_PYSCF_DIAG', '1') != '0'")
    lines.append("    diag = None")
    lines.append("    guess = None")
    lines.append("    max_memory = 4000")
    lines.append("    eff_tol = tol")
    lines.append("    if use_pyscf_diag and mf is not None:")
    lines.append("        try:")
    lines.append("            mycc = cc.CCSD(mf)")
    lines.append("            mycc.t1 = t1.T")
    lines.append("            mycc.t2 = t2.transpose(2, 3, 0, 1)")
    lines.append("            eom = eom_rccsd.EOMEESinglet(mycc)")
    lines.append("            imds = eom.make_imds()")
    lines.append("            diag = eom.get_diag(imds)")
    lines.append("            guess = eom.get_init_guess(nroots, koopmans=False, diag=diag)")
    lines.append("            max_memory = max(0, eom.max_memory - lib.current_memory()[0])")
    lines.append("            eff_tol = eom.conv_tol")
    lines.append("        except Exception:")
    lines.append("            diag = None")
    lines.append("            guess = None")
    lines.append("    if diag is None:")
    lines.append("        eps = np.diag(f)")
    lines.append("        eps_occ = eps[o]")
    lines.append("        eps_virt = eps[v]")
    lines.append("        denom_ai = eps_virt[:, None] - eps_occ[None, :]")
    lines.append("        denom_abij = (")
    lines.append("            eps_virt[:, None, None, None]")
    lines.append("            + eps_virt[None, :, None, None]")
    lines.append("            - eps_occ[None, None, :, None]")
    lines.append("            - eps_occ[None, None, None, :]")
    lines.append("        )")
    lines.append("        denom_iajb = denom_abij.transpose(2, 0, 3, 1)")
    lines.append("        nov = len(o) * len(v)")
    lines.append("        denom_mat = denom_iajb.reshape(nov, nov)")
    lines.append("        tril = np.tril_indices(nov)")
    lines.append("        diag = np.concatenate([denom_ai.T.ravel(), denom_mat[tril]])")
    lines.append("        guess_idx = np.argsort(diag)[:nroots]")
    lines.append("        guess = []")
    lines.append("        for idx in guess_idx:")
    lines.append("            v0 = np.zeros(diag.size)")
    lines.append("            v0[idx] = 1.0")
    lines.append("            guess.append(v0)")
    lines.append("    matvec = lambda vec: sigma_vector(vec, f, g, t1, t2, o, v)")
    lines.append("    def precond(r, e0, x0):")
    lines.append("        return r / (e0 - diag + 1e-12)")
    lines.append("    real_system = np.isrealobj(f)")
    lines.append("    def pickeig(w, v, nroots, envs):")
    lines.append("        real_idx = np.where(abs(w.imag) < 1e-3)[0]")
    lines.append("        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)")
    lines.append("    conv, es, _vecs = lib.davidson_nosym1(")
    lines.append("        matvec, guess, precond, pick=pickeig,")
    lines.append("        tol=eff_tol, max_cycle=max_iter, max_space=max_space, max_memory=max_memory, nroots=nroots,")
    lines.append("    )")
    lines.append("    return np.real_if_close(es)")
    lines.append("")
    lines.append("def main():")
    lines.append("    eigvals = solve_eom_ccsd()")
    lines.append("    print('EE-EOM-CCSD excitation energies:', eigvals)")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    (output_dir / "eom_solver.py").write_text("\n".join(lines) + "\n")


def emit_eom_pyscf_test(output_dir, spec_terms, output_names, pyscf_mol=None, spin_orbital=False):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    module_path = _module_path_from_dir(output_dir)
    root_depth = _root_depth_for(output_dir)
    _s1_name, _s2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from pyscf import scf, cc, gto")
    lines.append("from pyscf.cc import eom_rccsd")
    lines.append(f"from {module_path}.eom_solver import solve_eom_ccsd")
    lines.append(f"from {module_path} import residuals")
    lines.append("")
    atom = "H 0 0 0; H 0 0 0.74"
    basis = "sto-3g"
    unit = "Angstrom"
    charge = 0
    spin = 0
    if pyscf_mol:
        atom = pyscf_mol.get("atom", atom)
        basis = pyscf_mol.get("basis", basis)
        unit = pyscf_mol.get("unit", unit)
        charge = pyscf_mol.get("charge", charge)
        spin = pyscf_mol.get("spin", spin)
    lines.append("def build_molecule():")
    lines.append("    return gto.M(")
    lines.append(f"        atom={atom!r},")
    lines.append(f"        basis={basis!r},")
    lines.append(f"        unit={unit!r},")
    lines.append(f"        charge={charge},")
    lines.append(f"        spin={spin},")
    lines.append("    )")
    lines.append("")
    lines.append("def main():")
    lines.append("    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is not True:")
    lines.append("        raise ValueError('EE-EOM-CCSD test requires spin-summed residuals.')")
    lines.append("    mol = build_molecule()")
    lines.append("    mf = scf.RHF(mol).run()")
    lines.append("    mycc = cc.CCSD(mf).run()")
    lines.append("    e_pyscf, _vecs = eom_rccsd.EOMEESinglet(mycc).kernel(nroots=3)")
    lines.append("    e_autogen = solve_eom_ccsd(mol=mol, mf=mf, t1=mycc.t1.T, t2=mycc.t2.transpose(2, 3, 0, 1), nroots=3)")
    lines.append("    e_pyscf = np.array(e_pyscf)")
    lines.append("    e_autogen = np.array(e_autogen)")
    lines.append("    e_pyscf = np.sort(e_pyscf)")
    lines.append("    e_autogen = np.sort(e_autogen)")
    lines.append("    diff = np.max(np.abs(e_pyscf[:len(e_autogen)] - e_autogen))")
    lines.append("    print('PySCF EE-EOM-CCSD:', e_pyscf)")
    lines.append("    print('Autogen EE-EOM-CCSD:', e_autogen)")
    lines.append("    print('abs diff:', diff)")
    lines.append("    assert diff < 1e-6")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    (output_dir / "eom_pyscf_test.py").write_text("\n".join(lines) + "\n")

def make_name(list_char_op):
    return "".join(name.lower() for name in list_char_op)


def collect_tensor_labels(terms):
    labels_map = {}
    for term in terms:
        for op, coeff in zip(term.large_op_list, term.coeff_list):
            tensor_name = TENSOR_MAP.get(op.name)
            if tensor_name and tensor_name not in labels_map:
                labels_map[tensor_name] = "".join(coeff)
    return labels_map


def symmetry_factor(list_char_op):
    counts = {}
    for name in list_char_op:
        counts[name] = counts.get(name, 0) + 1
    if counts.get("V2") == 1 and counts.get("T1") == 2 and len(list_char_op) == 3:
        return 0.5
    return 1.0


def emit_einsum_code(list_char_op, terms, output_path):
    # Emit a runnable Python script for a specific operator list.
    expr_name = make_name(list_char_op)
    func_name = f"compute_{expr_name}"
    root_depth = _root_depth_for(output_path.parent)
    _ensure_output_package(output_path.parent)
    labels_map = collect_tensor_labels(terms)
    tensor_order = ["g", "f", "t1", "t2", "d1", "d2", "x1", "x2"]
    needed = [name for name in tensor_order if name in labels_map]
    sym_factor = symmetry_factor(list_char_op)
    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    special_f1t1 = list_char_op == ["F1", "T1"]
    special_v2t1t1 = list_char_op == ["V2", "T1", "T1"]
    special_v2t2 = list_char_op == ["V2", "T2"]
    use_ccsd_ai = any(name in {"t1", "t2"} for name in needed) and not (special_f1t1 or special_v2t1t1 or special_v2t2)
    use_ccsd_ijab = special_f1t1 or special_v2t1t1 or special_v2t2
    imports = ["build_h2o_631g", "compute_integrals", "run_scf"]
    if use_ccsd_ai:
        imports.append("compute_ccsd_amplitudes")
    if use_ccsd_ijab:
        imports.append("compute_ccsd_amplitudes_ijab")
    import_line = "from generated_code.pyscf_integrals import " + ", ".join(imports)
    lines.append(import_line)
    lines.append("")
    lines.append("OCC = set('ijklmn')")
    lines.append("VIRT = set('abcdefgh')")
    lines.append("")
    lines.append("def view_tensor(tensor, labels, o, v):")
    lines.append("    idx = []")
    lines.append("    list_axes = []")
    lines.append("    for axis, label in enumerate(labels):")
    lines.append("        if label in OCC:")
    lines.append("            idx.append(o)")
    lines.append("            list_axes.append(axis)")
    lines.append("        elif label in VIRT:")
    lines.append("            idx.append(v)")
    lines.append("            list_axes.append(axis)")
    lines.append("        else:")
    lines.append("            idx.append(slice(None))")
    lines.append("    if not list_axes:")
    lines.append("        return tensor[tuple(idx)]")
    lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
    lines.append("    ix_iter = iter(ix)")
    lines.append("    full_idx = []")
    lines.append("    for axis in range(len(idx)):")
    lines.append("        if axis in list_axes:")
    lines.append("            full_idx.append(next(ix_iter))")
    lines.append("        else:")
    lines.append("            full_idx.append(idx[axis])")
    lines.append("    return tensor[tuple(full_idx)]")
    lines.append("")
    lines.append("def zeros_for_labels(labels, nocc, nvirt, nmo):")
    lines.append("    shape = []")
    lines.append("    for label in labels:")
    lines.append("        if label in OCC:")
    lines.append("            shape.append(nocc)")
    lines.append("        elif label in VIRT:")
    lines.append("            shape.append(nvirt)")
    lines.append("        else:")
    lines.append("            shape.append(nmo)")
    lines.append("    return np.zeros(tuple(shape))")
    lines.append("")
    tensor_args = ", ".join(needed + ["o", "v"])
    if special_f1t1:
        # Use PySCF's CCSD energy contraction for spatial orbitals.
        lines.append("def compute_f1t1(f, t1_ia, o, v):")
        lines.append("    f_ov = f[np.ix_(o, v)]")
        lines.append("    return 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)")
    elif special_v2t1t1:
        # Match PySCF energy formula via eris.ovvo slices.
        lines.append("def compute_v2t1t1(g_raw, t1_ia, o, v):")
        lines.append("    eris_ovvo = g_raw[np.ix_(o, v, v, o)]")
        lines.append("    tau = np.einsum('ia,jb->ijab', t1_ia, t1_ia)")
        lines.append("    e = 2.0 * np.einsum('ijab,iabj->', tau, eris_ovvo)")
        lines.append("    e -= np.einsum('jiab,iabj->', tau, eris_ovvo)")
        lines.append("    return e")
    elif special_v2t2:
        # Match PySCF energy formula via eris.ovvo slices.
        lines.append("def compute_v2t2(g_raw, t2_ijab, o, v):")
        lines.append("    eris_ovvo = g_raw[np.ix_(o, v, v, o)]")
        lines.append("    e = 2.0 * np.einsum('ijab,iabj->', t2_ijab, eris_ovvo)")
        lines.append("    e -= np.einsum('jiab,iabj->', t2_ijab, eris_ovvo)")
        lines.append("    return e")
    else:
        lines.append(f"def {func_name}({tensor_args}):")
        lines.append("    total = 0.0")
        for i, term in enumerate(terms):
            subs, args, prefactor = term_to_einsum(term)
            arg_list = ", ".join(args)
            lines.append(
                f"    term_{i} = ({term.fac * prefactor * sym_factor}) * np.einsum('{subs}', {arg_list})"
            )
            lines.append(f"    total += term_{i}")
        lines.append("    return total")
    lines.append("")
    lines.append("def main():")
    lines.append("    mol = build_h2o_631g()")
    lines.append("    mf = run_scf(mol)")
    lines.append("    ints = compute_integrals(mol, mf=mf)")
    if "g" in needed:
        lines.append("    g = ints['g']")
        lines.append("    g_raw = ints['g_raw']")
    if "f" in needed:
        lines.append("    f = ints['f']")
    lines.append("    nocc = ints['nocc']")
    lines.append("    nmo = ints['nmo']")
    lines.append("    nvirt = nmo - nocc")
    lines.append("    o = list(range(nocc))")
    lines.append("    v = list(range(nocc, nmo))")
    lines.append("")
    if use_ccsd_ai:
        lines.append("    t1_ai, t2_abij = compute_ccsd_amplitudes(mf)")
    if use_ccsd_ijab:
        lines.append("    t1_ia, t2_ijab = compute_ccsd_amplitudes_ijab(mf)")
    for name in needed:
        if name in {"g", "f"}:
            continue
        if name == "t1":
            lines.append("    t1 = t1_ai")
            continue
        if name == "t2":
            lines.append("    t2 = t2_abij")
            continue
        labels = labels_map[name]
        lines.append(f"    {name} = zeros_for_labels('{labels}', nocc, nvirt, nmo)")
    if special_f1t1:
        lines.append("    value = compute_f1t1(f, t1_ia, o, v)")
    elif special_v2t1t1:
        lines.append("    value = compute_v2t1t1(g_raw, t1_ia, o, v)")
    elif special_v2t2:
        lines.append("    value = compute_v2t2(g_raw, t2_ijab, o, v)")
    else:
        args_call = ", ".join(needed + ["o", "v"])
        lines.append(f"    value = {func_name}({args_call})")
    lines.append(f"    print('{expr_name} value:', value)")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")

    output_path.write_text("\n".join(lines) + "\n")


def emit_ccsd_energy(output_path):
    # Emit a convenience script that sums the CCSD energy pieces.
    _ensure_output_package(output_path.parent)
    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    root_depth = _root_depth_for(output_path.parent)
    lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("# CCSD energy driver that matches PySCF's spatial-orbital energy formula.")
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_ccsd_amplitudes_ijab, compute_integrals, run_scf")
    lines.append("from generated_code.methods.ccsd.f1t1_einsum import compute_f1t1")
    lines.append("from generated_code.methods.ccsd.v2t1t1_einsum import compute_v2t1t1")
    lines.append("from generated_code.methods.ccsd.v2t2_einsum import compute_v2t2")
    lines.append("")
    lines.append("")
    lines.append("def main():")
    lines.append("    # Build molecule + SCF, then reuse the SCF object for integrals and CCSD.")
    lines.append("    mol = build_h2o_631g()")
    lines.append("    mf = run_scf(mol)")
    lines.append("    ints = compute_integrals(mol, mf=mf)")
    lines.append("    f = ints['f']")
    lines.append("    g_raw = ints['g_raw']")
    lines.append("    nocc = ints['nocc']")
    lines.append("    nmo = ints['nmo']")
    lines.append("    o = list(range(nocc))")
    lines.append("    v = list(range(nocc, nmo))")
    lines.append("")
    lines.append("    # CCSD amplitudes in PySCF-native ia/ijab layout.")
    lines.append("    t1_ia, t2_ijab = compute_ccsd_amplitudes_ijab(mf)")
    lines.append("")
    lines.append("    # Energy pieces (F1T1, V2T1T1, V2T2) in PySCF convention.")
    lines.append("    e_f1t1 = compute_f1t1(f, t1_ia, o, v)")
    lines.append("    e_v2t1t1 = compute_v2t1t1(g_raw, t1_ia, o, v)")
    lines.append("    e_v2t2 = compute_v2t2(g_raw, t2_ijab, o, v)")
    lines.append("")
    lines.append("    e_ccsd = e_f1t1 + e_v2t1t1 + e_v2t2")
    lines.append("    print('CCSD correlation energy pieces:')")
    lines.append("    print('  F1T1 :', e_f1t1)")
    lines.append("    print('  V2T1T1 :', e_v2t1t1)")
    lines.append("    print('  V2T2 :', e_v2t2)")
    lines.append("    print('  total :', e_ccsd)")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    output_path.write_text("\n".join(lines) + "\n")


def emit_ccsd_amplitude(output_dir, mode: str = "runtime", subset: str = "both", quiet: bool = False):
    output_dir = Path(output_dir)
    _ensure_output_package(output_dir)
    root_depth = _root_depth_for(output_dir)
    subset = subset.lower()
    if subset not in {"both", "x1", "x2"}:
        raise ValueError(f"Unsupported CCSD amplitude subset: {subset}")
    mode = mode.lower()
    if mode not in {"full", "runtime", "intermediates"}:
        raise ValueError(f"Unsupported CCSD amplitude mode: {mode}")

    spin_summed = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
    if mode == "runtime":
        lines = []
        lines.append("import numpy as np")
        lines.append("")
        lines.append("from pathlib import Path")
        lines.append("import contextlib")
        lines.append("import os")
        lines.append("import sys")
        lines.append("")
        lines.append(f"QUIET = {quiet}")
        lines.append(f"SUBSET = '{subset}'")
        lines.append("")
        lines.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
        lines.append("SRC = ROOT / 'src'")
        lines.append("sys.path.insert(0, str(SRC))")
        lines.append("")
        lines.append("# Residual builder using Autogen-generated terms.")
        lines.append("from autogen.main_tools import multi_cont")
        lines.append("from autogen.library import change_terms, full_con, make_op, compare as cpre")
        lines.append("")
        lines.append(f"AUTOGEN_SPIN_SUMMED = {spin_summed}")
        lines.append("AUTOGEN_INTERMEDIATES = False")
        lines.append("OCC = set('ijklmn')")
        lines.append("VIRT = set('abcdefgh')")
        lines.append("")
        lines.append("@contextlib.contextmanager")
        lines.append("def suppress_output(enabled):")
        lines.append("    if not enabled:")
        lines.append("        yield")
        lines.append("        return")
        lines.append("    with open(os.devnull, 'w') as devnull:")
        lines.append("        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):")
        lines.append("            yield")
        lines.append("")
        lines.append("def view_tensor(tensor, labels, o, v):")
        lines.append("    idx = []")
        lines.append("    list_axes = []")
        lines.append("    for axis, label in enumerate(labels):")
        lines.append("        if label in OCC:")
        lines.append("            idx.append(o)")
        lines.append("            list_axes.append(axis)")
        lines.append("        elif label in VIRT:")
        lines.append("            idx.append(v)")
        lines.append("            list_axes.append(axis)")
        lines.append("        else:")
        lines.append("            idx.append(slice(None))")
        lines.append("    if not list_axes:")
        lines.append("        return tensor[tuple(idx)]")
        lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
        lines.append("    ix_iter = iter(ix)")
        lines.append("    full_idx = []")
        lines.append("    for axis in range(len(idx)):")
        lines.append("        if axis in list_axes:")
        lines.append("            full_idx.append(next(ix_iter))")
        lines.append("        else:")
        lines.append("            full_idx.append(idx[axis])")
        lines.append("    return tensor[tuple(full_idx)]")
        lines.append("")
        lines.append("def build_terms(list_char_op):")
        lines.append("    # Mirror driv3-style contraction flow to get fully contracted terms.")
        lines.append("    with suppress_output(QUIET):")
        lines.append("        dict_ind = {}")
        lines.append("        lou, dict_ind = make_op.make_op(list_char_op, dict_ind)")
        lines.append("        st, co = lou[0].st, lou[0].co")
        lines.append("        for i in range(1, len(lou)):")
        lines.append("            st, co = multi_cont.multi_cont(st, lou[i].st, co, lou[i].co)")
        lines.append("        st, co = full_con.full_con(st, co)")
        lines.append("        terms = change_terms.change_terms1(st, co, 1.0, dict_ind, lou)")
        lines.append("        for term in terms:")
        lines.append("            term.compress()")
        lines.append("            term.build_map_org()")
        lines.append("        for i in range(len(terms)):")
        lines.append("            for j in range(i + 1, len(terms)):")
        lines.append("                if terms[i].fac != 0.0 and terms[j].fac != 0.0:")
        lines.append("                    flo = cpre.compare(terms[i], terms[j])")
        lines.append("                    if flo != 0:")
        lines.append("                        terms[i].fac = terms[i].fac + terms[j].fac * flo")
        lines.append("                        terms[j].fac = 0.0")
        lines.append("        return [term for term in terms if term.fac != 0.0]")
        lines.append("")
        lines.append("def output_labels_from_xop(coeff):")
        lines.append("    virt = [label for label in coeff if label in VIRT]")
        lines.append("    occ = [label for label in coeff if label in OCC]")
        lines.append("    return ''.join(virt + occ)")
        lines.append("")
        lines.append("def term_to_residual_einsum(term):")
        lines.append("    tensors = []")
        lines.append("    output_labels = None")
        lines.append("    for op, coeff in zip(term.large_op_list, term.coeff_list):")
        lines.append("        if op.name in {'X1', 'X2'}:")
        lines.append("            output_labels = output_labels_from_xop(coeff)")
        lines.append("            continue")
        lines.append("        if op.name == 'F1':")
        lines.append("            tensor_name = 'f'")
        lines.append("        elif op.name == 'V2':")
        lines.append("            tensor_name = 'g'")
        lines.append("        elif op.name.startswith('T') and len(op.name) > 1:")
        lines.append("            tensor_name = 't1' if op.name[1] == '1' else 't2'")
        lines.append("        else:")
        lines.append("            raise ValueError(f'Unsupported operator {op.name}')")
        lines.append("        tensors.append((tensor_name, ''.join(coeff)))")
        lines.append("    if output_labels is None:")
        lines.append("        raise ValueError('No X1/X2 projector found in amplitude term.')")
        lines.append("    subs_in = ','.join(labels for _, labels in tensors)")
        lines.append("    subs = f\"{subs_in}->{output_labels}\"")
        lines.append("    return subs, tensors")
        lines.append("")
        lines.append("def build_ccsd_amplitude_terms():")
        lines.append("    # Term lists from tests/ccsd_amplitude.py.")
        lines.append("    x1_specs = [")
        lines.append("        (1.0, ['X1', 'F1']),")
        lines.append("        (1.0, ['X1', 'F1', 'T1']),")
        lines.append("        (1.0, ['X1', 'F1', 'T2']),")
        lines.append("        (0.5, ['X1', 'F1', 'T1', 'T11']),")
        lines.append("        (0.5, ['X1', 'F1', 'T2', 'T21']),")
        lines.append("        (1.0, ['X1', 'F1', 'T1', 'T2']),")
        lines.append("        (1.0, ['X1', 'V2']),")
        lines.append("        (1.0, ['X1', 'V2', 'T1']),")
        lines.append("        (1.0, ['X1', 'V2', 'T2']),")
        lines.append("        (0.5, ['X1', 'V2', 'T1', 'T11']),")
        lines.append("        (0.5, ['X1', 'V2', 'T2', 'T21']),")
        lines.append("        (1.0, ['X1', 'V2', 'T1', 'T2']),")
        lines.append("        (1.0/6.0, ['X1', 'V2', 'T1', 'T11', 'T12']),")
        lines.append("    ]")
        lines.append("    x2_specs = [")
        lines.append("        (1.0, ['X2', 'F1']),")
        lines.append("        (1.0, ['X2', 'F1', 'T1']),")
        lines.append("        (0.5, ['X2', 'F1', 'T1', 'T11']),")
        lines.append("        (1.0, ['X2', 'F1', 'T2']),")
        lines.append("        (0.5, ['X2', 'F1', 'T2', 'T21']),")
        lines.append("        (1.0, ['X2', 'F1', 'T1', 'T2']),")
        lines.append("        (1.0, ['X2', 'V2', 'T1']),")
        lines.append("        (0.5, ['X2', 'V2', 'T1', 'T11']),")
        lines.append("        (1.0/6.0, ['X2', 'V2', 'T1', 'T11', 'T12']),")
        lines.append("        (1.0, ['X2', 'V2', 'T2']),")
        lines.append("        (0.5, ['X2', 'V2', 'T2', 'T21']),")
        lines.append("        (1.0, ['X2', 'V2', 'T1', 'T2']),")
        lines.append("        (0.5, ['X2', 'V2', 'T1', 'T11', 'T2']),")
        lines.append("        (1.0/24.0, ['X2', 'V2', 'T1', 'T11', 'T12', 'T13']),")
        lines.append("    ]")
        lines.append("    x1_terms = []")
        lines.append("    x2_terms = []")
        lines.append("    if SUBSET in ('both', 'x1'):")
        lines.append("        for fac, ops in x1_specs:")
        lines.append("            terms = build_terms(ops)")
        lines.append("            for term in terms:")
        lines.append("                term.fac *= fac")
        lines.append("            x1_terms.extend(terms)")
        lines.append("    if SUBSET in ('both', 'x2'):")
        lines.append("        for fac, ops in x2_specs:")
        lines.append("            terms = build_terms(ops)")
        lines.append("            for term in terms:")
        lines.append("                term.fac *= fac")
        lines.append("            x2_terms.extend(terms)")
        lines.append("    return x1_terms, x2_terms")
        lines.append("")
        lines.append("_TERMS_CACHE = None")
        lines.append("# Cache terms to avoid regenerating them each iteration.")
        lines.append("def _get_terms():")
        lines.append("    global _TERMS_CACHE")
        lines.append("    if _TERMS_CACHE is None:")
        lines.append("        _TERMS_CACHE = build_ccsd_amplitude_terms()")
        lines.append("    return _TERMS_CACHE")
        lines.append("")
        lines.append("def compute_r1(f, g, t1, t2, o, v):")
        lines.append("    if SUBSET == 'x2':")
        lines.append("        return np.zeros((len(v), len(o)))")
        lines.append("    x1_terms, _ = _get_terms()")
        lines.append("    r1 = np.zeros((len(v), len(o)))")
        lines.append("    for term in x1_terms:")
        lines.append("        subs, tensors = term_to_residual_einsum(term)")
        lines.append("        args = []")
        lines.append("        for name, labels in tensors:")
        lines.append("            if name == 'g':")
        lines.append("                args.append(view_tensor(g, labels, o, v))")
        lines.append("            elif name == 'f':")
        lines.append("                args.append(view_tensor(f, labels, o, v))")
        lines.append("            elif name == 't1':")
        lines.append("                args.append(t1)")
        lines.append("            elif name == 't2':")
        lines.append("                args.append(t2)")
        lines.append("        r1 += term.fac * np.einsum(subs, *args)")
        lines.append("    return r1")
        lines.append("")
        lines.append("def compute_r2(f, g, t1, t2, o, v):")
        lines.append("    if SUBSET == 'x1':")
        lines.append("        return np.zeros((len(v), len(v), len(o), len(o)))")
        lines.append("    _, x2_terms = _get_terms()")
        lines.append("    r2 = np.zeros((len(v), len(v), len(o), len(o)))")
        lines.append("    for term in x2_terms:")
        lines.append("        subs, tensors = term_to_residual_einsum(term)")
        lines.append("        args = []")
        lines.append("        for name, labels in tensors:")
        lines.append("            if name == 'g':")
        lines.append("                args.append(view_tensor(g, labels, o, v))")
        lines.append("            elif name == 'f':")
        lines.append("                args.append(view_tensor(f, labels, o, v))")
        lines.append("            elif name == 't1':")
        lines.append("                args.append(t1)")
        lines.append("            elif name == 't2':")
        lines.append("                args.append(t2)")
        lines.append("        r2 += term.fac * np.einsum(subs, *args)")
        lines.append("    return r2")
        (output_dir / "residuals.py").write_text("\n".join(lines) + "\n")
    elif mode == "intermediates":
        x1_terms, x2_terms = build_ccsd_amplitude_terms(quiet=quiet, subset=subset)
        x1_struct = []
        x2_struct = []
        for term in x1_terms:
            output_labels, tensors, coeff = _term_to_residual_struct(term)
            output_labels, tensors, coeff = _canonicalize_qp_antisymmetry(
                output_labels, tensors, coeff
            )
            if abs(coeff) < 1e-12:
                continue
            output_labels, tensors = _canonicalize_term_labels(output_labels, tensors)
            x1_struct.append((output_labels, tensors, coeff))
        for term in x2_terms:
            output_labels, tensors, coeff = _term_to_residual_struct(term)
            output_labels, tensors, coeff = _canonicalize_qp_antisymmetry(
                output_labels, tensors, coeff
            )
            if abs(coeff) < 1e-12:
                continue
            output_labels, tensors = _canonicalize_term_labels(output_labels, tensors)
            x2_struct.append((output_labels, tensors, coeff))

        all_struct = x1_struct + x2_struct
        inter_map = _select_intermediates(all_struct)
        x1_struct = [
            (output_labels, _apply_intermediates(tensors, inter_map), coeff)
            for output_labels, tensors, coeff in x1_struct
        ]
        x2_struct = [
            (output_labels, _apply_intermediates(tensors, inter_map), coeff)
            for output_labels, tensors, coeff in x2_struct
        ]

        used_intermediates = set()
        for _out, tensors, _coeff in x1_struct + x2_struct:
            for name, _labels in tensors:
                if name.startswith("I"):
                    used_intermediates.add(name)

        inter_defs = [
            info for info in inter_map.values() if info["name"] in used_intermediates
        ]
        inter_defs.sort(key=lambda info: int(info["name"][1:]))

        grouped_x1 = _group_terms_by_subs(x1_struct)
        grouped_x2 = _group_terms_by_subs(x2_struct)

        lines = []
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"AUTOGEN_SPIN_SUMMED = {spin_summed}")
        lines.append("AUTOGEN_INTERMEDIATES = True")
        lines.append("OCC = set('ijklmn')")
        lines.append("VIRT = set('abcdefgh')")
        lines.append("")
        lines.append("def view_tensor(tensor, labels, o, v):")
        lines.append("    idx = []")
        lines.append("    list_axes = []")
        lines.append("    for axis, label in enumerate(labels):")
        lines.append("        if label in OCC:")
        lines.append("            idx.append(o)")
        lines.append("            list_axes.append(axis)")
        lines.append("        elif label in VIRT:")
        lines.append("            idx.append(v)")
        lines.append("            list_axes.append(axis)")
        lines.append("        else:")
        lines.append("            idx.append(slice(None))")
        lines.append("    if not list_axes:")
        lines.append("        return tensor[tuple(idx)]")
        lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
        lines.append("    ix_iter = iter(ix)")
        lines.append("    full_idx = []")
        lines.append("    for axis in range(len(idx)):")
        lines.append("        if axis in list_axes:")
        lines.append("            full_idx.append(next(ix_iter))")
        lines.append("        else:")
        lines.append("            full_idx.append(idx[axis])")
        lines.append("    return tensor[tuple(full_idx)]")
        lines.append("")
        lines.append("def _get_viewer(f, g, o, v):")
        lines.append("    views = {}")
        lines.append("    def get_view(name, labels):")
        lines.append("        key = (name, labels)")
        lines.append("        if key in views:")
        lines.append("            return views[key]")
        lines.append("        tensor = f if name == 'f' else g")
        lines.append("        views[key] = view_tensor(tensor, labels, o, v)")
        lines.append("        return views[key]")
        lines.append("    return get_view")
        lines.append("")
        lines.append("def compute_intermediates(t1, t2, get_view):")
        if inter_defs:
            for info in inter_defs:
                name = info["name"]
                op1, op2 = info["ops"]
                labels1 = op1[1]
                labels2 = op2[1]
                out_labels = info["out_labels"]
                subs = f"{labels1},{labels2}->{out_labels}"
                args = []
                for op_name, op_labels in info["ops"]:
                    if op_name in {"f", "g"}:
                        args.append(f"get_view('{op_name}', '{op_labels}')")
                    else:
                        args.append(op_name)
                lines.append(
                    f"    {name} = np.einsum('{subs}', {', '.join(args)}, optimize=True)"
                )
            lines.append("    return {")
            for info in inter_defs:
                name = info["name"]
                lines.append(f"        '{name}': {name},")
            lines.append("    }")
        else:
            lines.append("    return {}")
        lines.append("")
        lines.append("def compute_r1_r2(f, g, t1, t2, o, v):")
        lines.append("    get_view = _get_viewer(f, g, o, v)")
        if inter_defs:
            lines.append("    inter = compute_intermediates(t1, t2, get_view)")
            for info in inter_defs:
                name = info["name"]
                lines.append(f"    {name} = inter['{name}']")
        else:
            lines.append("    inter = {}")
        if not x1_struct:
            lines.append("    r1 = np.zeros((len(v), len(o)))")
        else:
            lines.append("    r1 = np.zeros((len(v), len(o)))")
            for (subs, tensors), coeff in sorted(grouped_x1.items()):
                if coeff == 0.0:
                    continue
                args = []
                for name, labels in tensors:
                    if name in used_intermediates:
                        args.append(name)
                    elif name in {"f", "g"}:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    r1 += ({coeff}) * np.einsum('{subs}', {', '.join(args)}, optimize=True)"
                )
        if not x2_struct:
            lines.append("    r2 = np.zeros((len(v), len(v), len(o), len(o)))")
        else:
            lines.append("    r2 = np.zeros((len(v), len(v), len(o), len(o)))")
            for (subs, tensors), coeff in sorted(grouped_x2.items()):
                if coeff == 0.0:
                    continue
                args = []
                for name, labels in tensors:
                    if name in used_intermediates:
                        args.append(name)
                    elif name in {"f", "g"}:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    r2 += ({coeff}) * np.einsum('{subs}', {', '.join(args)}, optimize=True)"
                )
        lines.append("    return r1, r2")
        lines.append("")
        lines.append("def compute_r1(f, g, t1, t2, o, v):")
        lines.append("    r1, _ = compute_r1_r2(f, g, t1, t2, o, v)")
        lines.append("    return r1")
        lines.append("")
        lines.append("def compute_r2(f, g, t1, t2, o, v):")
        lines.append("    _, r2 = compute_r1_r2(f, g, t1, t2, o, v)")
        lines.append("    return r2")
        (output_dir / "residuals.py").write_text("\n".join(lines) + "\n")
    else:
        x1_terms, x2_terms = build_ccsd_amplitude_terms(quiet=quiet, subset=subset)
        lines = []
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"AUTOGEN_SPIN_SUMMED = {spin_summed}")
        lines.append("AUTOGEN_INTERMEDIATES = False")
        lines.append("OCC = set('ijklmn')")
        lines.append("VIRT = set('abcdefgh')")
        lines.append("")
        lines.append("def view_tensor(tensor, labels, o, v):")
        lines.append("    idx = []")
        lines.append("    list_axes = []")
        lines.append("    for axis, label in enumerate(labels):")
        lines.append("        if label in OCC:")
        lines.append("            idx.append(o)")
        lines.append("            list_axes.append(axis)")
        lines.append("        elif label in VIRT:")
        lines.append("            idx.append(v)")
        lines.append("            list_axes.append(axis)")
        lines.append("        else:")
        lines.append("            idx.append(slice(None))")
        lines.append("    if not list_axes:")
        lines.append("        return tensor[tuple(idx)]")
        lines.append("    ix = np.ix_(*[idx[a] for a in list_axes])")
        lines.append("    ix_iter = iter(ix)")
        lines.append("    full_idx = []")
        lines.append("    for axis in range(len(idx)):")
        lines.append("        if axis in list_axes:")
        lines.append("            full_idx.append(next(ix_iter))")
        lines.append("        else:")
        lines.append("            full_idx.append(idx[axis])")
        lines.append("    return tensor[tuple(full_idx)]")
        lines.append("")
        lines.append("def compute_r1(f, g, t1, t2, o, v):")
        if not x1_terms:
            lines.append("    return np.zeros((len(v), len(o)))")
        else:
            lines.append("    r1 = np.zeros((len(v), len(o)))")
            for i, term in enumerate(x1_terms):
                subs, args = term_to_residual_einsum(term)
                arg_list = ", ".join(args)
                lines.append(
                    f"    term_{i} = ({term.fac}) * np.einsum('{subs}', {arg_list})"
                )
                lines.append(f"    r1 += term_{i}")
            lines.append("    return r1")
        lines.append("")
        lines.append("def compute_r2(f, g, t1, t2, o, v):")
        if not x2_terms:
            lines.append("    return np.zeros((len(v), len(v), len(o), len(o)))")
        else:
            lines.append("    r2 = np.zeros((len(v), len(v), len(o), len(o)))")
            for i, term in enumerate(x2_terms):
                subs, args = term_to_residual_einsum(term)
                arg_list = ", ".join(args)
                lines.append(
                    f"    term_{i} = ({term.fac}) * np.einsum('{subs}', {arg_list})"
                )
                lines.append(f"    r2 += term_{i}")
            lines.append("    return r2")
        (output_dir / "residuals.py").write_text("\n".join(lines) + "\n")

    solver = []
    solver.append("import numpy as np")
    solver.append("")
    solver.append("from pathlib import Path")
    solver.append("import sys")
    solver.append("")
    solver.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    solver.append("sys.path.insert(0, str(ROOT))")
    solver.append("")
    solver.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_integrals, run_scf")
    solver.append("from generated_code.methods.ccsd.ccsd_amplitude import residuals as residuals")
    solver.append("")
    solver.append("def mp2_init(f, g, o, v):")
    solver.append("    # MP2-like starting amplitudes in spatial-orbital form.")
    solver.append("    eps = np.diag(f)")
    solver.append("    eps_occ = eps[o]")
    solver.append("    eps_virt = eps[v]")
    solver.append("    denom_ai = eps_occ[None, :] - eps_virt[:, None]")
    solver.append("    denom_abij = (")
    solver.append("        eps_occ[None, None, :, None]")
    solver.append("        + eps_occ[None, None, None, :]")
    solver.append("        - eps_virt[:, None, None, None]")
    solver.append("        - eps_virt[None, :, None, None]")
    solver.append("    )")
    solver.append("    denom_ai = np.where(abs(denom_ai) < 1e-12, 1e-12, denom_ai)")
    solver.append("    denom_abij = np.where(abs(denom_abij) < 1e-12, 1e-12, denom_abij)")
    solver.append("    g_ijab = g[np.ix_(o, o, v, v)]")
    solver.append("    t1 = f[np.ix_(o, v)].T / denom_ai")
    solver.append("    t2 = g_ijab.transpose(2, 3, 0, 1) / denom_abij")
    solver.append("    return t1, t2, denom_ai, denom_abij")
    solver.append("")
    solver.append("def compute_energy(f, g_raw, t1, t2, o, v):")
    solver.append("    # Match PySCF RHF CCSD energy formula (spatial orbitals).")
    solver.append("    f_ov = f[np.ix_(o, v)]")
    solver.append("    t1_ia = t1.T")
    solver.append("    t2_ijab = t2.transpose(2, 3, 0, 1)")
    solver.append("    tau = t2_ijab + np.einsum('ia,jb->ijab', t1_ia, t1_ia)")
    solver.append("    eris_ovvo = g_raw[np.ix_(o, v, v, o)]")
    solver.append("    e = 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)")
    solver.append("    e += 2.0 * np.einsum('ijab,iabj->', tau, eris_ovvo)")
    solver.append("    e -= np.einsum('jiab,iabj->', tau, eris_ovvo)")
    solver.append("    return e")
    solver.append("")
    solver.append("def solve_ccsd(mol=None, max_iter=50, tol=1e-8, damping=0.0, diis_start=2, max_diis=6):")
    solver.append("    # Jacobi iteration with optional DIIS acceleration.")
    solver.append("    if mol is None:")
    solver.append("        mol = build_h2o_631g()")
    solver.append("    mf = run_scf(mol)")
    solver.append("    ints = compute_integrals(mol, mf=mf)")
    solver.append("    f = ints['f']")
    solver.append("    g = ints['g']")
    solver.append("    g_raw = ints['g_raw']")
    solver.append("    nocc = ints['nocc']")
    solver.append("    nmo = ints['nmo']")
    solver.append("    o = list(range(nocc))")
    solver.append("    v = list(range(nocc, nmo))")
    solver.append("")
    solver.append("    t1, t2, denom_ai, denom_abij = mp2_init(f, g, o, v)")
    solver.append("    energy = compute_energy(f, g_raw, t1, t2, o, v)")
    solver.append("")
    solver.append("    t1_list = []")
    solver.append("    t2_list = []")
    solver.append("    err_list = []")
    solver.append("")
    solver.append("    def diis_extrapolate(t1_list, t2_list, err_list):")
    solver.append("        n = len(err_list)")
    solver.append("        b = np.empty((n + 1, n + 1))")
    solver.append("        b[-1, :] = -1.0")
    solver.append("        b[:, -1] = -1.0")
    solver.append("        b[-1, -1] = 0.0")
    solver.append("        for i in range(n):")
    solver.append("            for j in range(n):")
    solver.append("                b[i, j] = np.dot(err_list[i], err_list[j])")
    solver.append("        rhs = np.zeros(n + 1)")
    solver.append("        rhs[-1] = -1.0")
    solver.append("        coeff = np.linalg.solve(b, rhs)[:-1]")
    solver.append("        t1_new = sum(c * t for c, t in zip(coeff, t1_list))")
    solver.append("        t2_new = sum(c * t for c, t in zip(coeff, t2_list))")
    solver.append("        return t1_new, t2_new")
    solver.append("")
    solver.append("    for it in range(1, max_iter + 1):")
    solver.append("        if hasattr(residuals, 'compute_r1_r2'):")
    solver.append("            r1, r2 = residuals.compute_r1_r2(f, g, t1, t2, o, v)")
    solver.append("        else:")
    solver.append("            r1 = residuals.compute_r1(f, g, t1, t2, o, v)")
    solver.append("            r2 = residuals.compute_r2(f, g, t1, t2, o, v)")
    solver.append("        t1_new = t1 + r1 / denom_ai")
    solver.append("        t2_new = t2 + r2 / denom_abij")
    solver.append("        if damping > 0.0:")
    solver.append("            t1_new = (1.0 - damping) * t1_new + damping * t1")
    solver.append("            t2_new = (1.0 - damping) * t2_new + damping * t2")
    solver.append("        err = np.concatenate([r1.ravel(), r2.ravel()])")
    solver.append("        t1_list.append(t1_new.copy())")
    solver.append("        t2_list.append(t2_new.copy())")
    solver.append("        err_list.append(err)")
    solver.append("        if len(err_list) > max_diis:")
    solver.append("            t1_list.pop(0)")
    solver.append("            t2_list.pop(0)")
    solver.append("            err_list.pop(0)")
    solver.append("        if it >= diis_start and len(err_list) >= 2:")
    solver.append("            t1_new, t2_new = diis_extrapolate(t1_list, t2_list, err_list)")
    solver.append("        new_energy = compute_energy(f, g_raw, t1_new, t2_new, o, v)")
    solver.append("        r_norm = max(np.max(np.abs(r1)), np.max(np.abs(r2)))")
    solver.append("        e_diff = abs(new_energy - energy)")
    solver.append("        print(f'iter {it:3d}  energy {new_energy: .10f}  |R| {r_norm:.3e}  dE {e_diff:.3e}')")
    solver.append("        t1, t2, energy = t1_new, t2_new, new_energy")
    solver.append("        if r_norm < tol and e_diff < tol:")
    solver.append("            break")
    solver.append("    return energy, t1, t2")
    (output_dir / "solver.py").write_text("\n".join(solver) + "\n")

    energy = []
    energy.append("from pathlib import Path")
    energy.append("import sys")
    energy.append("")
    energy.append(f"ROOT = Path(__file__).resolve().parents[{root_depth}]")
    energy.append("sys.path.insert(0, str(ROOT))")
    energy.append("")
    energy.append("from generated_code.methods.ccsd.ccsd_amplitude.solver import solve_ccsd")
    energy.append("# Driver that prints the converged CCSD correlation energy.")
    energy.append("")
    energy.append("def main():")
    energy.append("    energy, t1, t2 = solve_ccsd()")
    energy.append("    print('CCSD correlation energy (iterative):', energy)")
    energy.append("")
    energy.append("if __name__ == '__main__':")
    energy.append("    main()")
    (output_dir / "ccsd_energy.py").write_text("\n".join(energy) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/gen_einsum.py V2 T1 T1")
        print("   or: python scripts/gen_einsum.py CCSD_ENERGY")
        print(
            "   or: python scripts/gen_einsum.py CCSD_AMPLITUDE "
            "[--runtime|--full|--intermediates] [--x1-only|--x2-only] [--quiet]"
        )
        print(
            "   or: python scripts/gen_einsum.py --spec path/to/spec.py "
            "[--full|--intermediates] [--out OUTPUT_DIR] [--tasks solver,pyscf_test] "
            "[--projection-only|--full-gen] [--quiet]"
        )
        sys.exit(1)
    args = list(sys.argv[1:])
    quiet = False
    mode = None
    subset = "both"
    spec_path = None
    output_override = None
    projection_only = None
    if "--quiet" in args:
        quiet = True
        args = [arg for arg in args if arg != "--quiet"]
    if "--projection-only" in args:
        projection_only = True
        args = [arg for arg in args if arg != "--projection-only"]
    if "--full-gen" in args:
        projection_only = False
        args = [arg for arg in args if arg != "--full-gen"]
    if "--spec" in args:
        spec_idx = args.index("--spec")
        if spec_idx + 1 >= len(args):
            raise ValueError("--spec requires a path.")
        spec_path = args[spec_idx + 1]
        args = args[:spec_idx] + args[spec_idx + 2 :]
    tasks_override = None
    if "--tasks" in args:
        tasks_idx = args.index("--tasks")
        if tasks_idx + 1 >= len(args):
            raise ValueError("--tasks requires a comma-separated list.")
        tasks_override = args[tasks_idx + 1]
        args = args[:tasks_idx] + args[tasks_idx + 2 :]
    if "--out" in args:
        out_idx = args.index("--out")
        if out_idx + 1 >= len(args):
            raise ValueError("--out requires a path.")
        output_override = args[out_idx + 1]
        args = args[:out_idx] + args[out_idx + 2 :]
    if "--x1-only" in args and "--x2-only" in args:
        print("Choose only one of --x1-only or --x2-only.")
        sys.exit(1)
    if "--x1-only" in args:
        subset = "x1"
        args = [arg for arg in args if arg != "--x1-only"]
    if "--x2-only" in args:
        subset = "x2"
        args = [arg for arg in args if arg != "--x2-only"]
    if "--runtime" in args or "--fast" in args:
        mode = "runtime"
        args = [arg for arg in args if arg not in {"--runtime", "--fast"}]
    if "--full" in args:
        mode = "full"
        args = [arg for arg in args if arg != "--full"]
    if "--intermediates" in args:
        mode = "intermediates"
        args = [arg for arg in args if arg != "--intermediates"]
    list_char_op = args
    if spec_path is not None:
        if subset != "both":
            raise ValueError("--x1-only/--x2-only are not valid with --spec.")
        if list_char_op:
            raise ValueError("--spec cannot be combined with operator arguments.")
        if mode is None:
            mode = "full"
        if mode not in {"full", "intermediates"}:
            raise ValueError("--spec supports only --full or --intermediates.")
        spec = load_spec(spec_path)
        (
            terms,
            output_names,
            tensor_map,
            view_tensors,
            output_dir,
            tasks,
            pyscf_mol,
            spin_orbital,
            spin_adapted,
        ) = parse_spec_terms(spec)
        prebuilt_terms = None
        if spec.get("EOM_BCH"):
            output_keys = tuple(output_names.keys()) if output_names else ("X1", "X2")
            t1_labels = tuple(spec.get("EOM_T1_LABELS", ("T1", "T11", "T12", "T13")))
            t2_labels = tuple(spec.get("EOM_T2_LABELS", ("T2", "T21")))
            h_ops = tuple(spec.get("EOM_H_OPS", ("F1", "V2")))
            max_order = int(spec.get("EOM_MAX_ORDER", 4))
            spin_summed_env = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
            build_spin_summed = spin_summed_env
            if spin_summed_env and spin_adapted:
                build_spin_summed = False
            with _spin_summed_context(build_spin_summed):
                prebuilt_terms = build_eom_bch_terms(
                    max_order=max_order,
                    t1_labels=t1_labels,
                    t2_labels=t2_labels,
                    h_ops=h_ops,
                    outputs=output_keys,
                    quiet=quiet,
                )
        elif spec.get("BCH"):
            output_keys = tuple(output_names.keys()) if output_names else ("X1", "X2", "scalar")
            t1_labels = tuple(spec.get("BCH_T1_LABELS", spec.get("T1_LABELS", ("T1", "T11", "T12", "T13"))))
            t2_labels = tuple(spec.get("BCH_T2_LABELS", spec.get("T2_LABELS", ("T2", "T21"))))
            h_ops = tuple(spec.get("BCH_H_OPS", spec.get("H_OPS", ("F1", "V2"))))
            max_order = int(spec.get("BCH_MAX_ORDER", spec.get("MAX_ORDER", 4)))
            spin_summed_env = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
            build_spin_summed = spin_summed_env
            if spin_summed_env and spin_adapted:
                build_spin_summed = False
            with _spin_summed_context(build_spin_summed):
                prebuilt_terms = build_bch_terms(
                    max_order=max_order,
                    t1_labels=t1_labels,
                    t2_labels=t2_labels,
                    h_ops=h_ops,
                    outputs=output_keys,
                    quiet=quiet,
                )
        if tasks_override is not None:
            tasks = [task.strip() for task in tasks_override.split(",") if task.strip()]
        if output_override is not None:
            output_dir = output_override
        if output_dir is None:
            output_dir = METHODS_DIR / Path(spec_path).stem
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
        # Default to projection-only for qp-CCSD unless explicitly overridden.
        if projection_only is None and output_dir.name == "qp_ccsd":
            projection_only = True
        residuals_basename = os.getenv("AUTOGEN_RESIDUALS_BASENAME", "residuals.py")
        if projection_only:
            if os.getenv("AUTOGEN_QP_Z_CONTRACTION") != "1":
                raise ValueError(
                    "--projection-only requires AUTOGEN_QP_Z_CONTRACTION=1 "
                    "to generate Z contractions."
                )
            residuals_basename = "residuals_pnp.py"
            if _is_qp_ccsd_target(output_dir, spec_path):
                emit_qp_ccsd_projected_residuals(
                    output_dir,
                    filename=residuals_basename,
                )
            else:
                emit_spec_residuals(
                    output_dir,
                    terms,
                    output_names,
                    tensor_map,
                    view_tensors,
                    mode=mode,
                    quiet=quiet,
                    spin_adapted=spin_adapted,
                    filename=residuals_basename,
                    prebuilt_terms=prebuilt_terms,
                    eom_mode=bool(spec.get("EOM_BCH")),
                )
            print(f"Wrote {output_dir} (projection-only)")
            return
        emit_spec_residuals(
            output_dir,
            terms,
            output_names,
            tensor_map,
            view_tensors,
            mode=mode,
            quiet=quiet,
            spin_adapted=spin_adapted,
            filename=residuals_basename,
            prebuilt_terms=prebuilt_terms,
            eom_mode=bool(spec.get("EOM_BCH")),
        )
        if output_dir.name == "qp_ccsd" and os.getenv("AUTOGEN_QP_R2_PREFAC_FIX", "1") != "0":
            _apply_qpccsd_r2_prefactor_fix(output_dir / residuals_basename, quiet=quiet)
            _apply_qpccsd_r2_prefactor_fix(output_dir / "residuals_pnp.py", quiet=quiet)
        tasks = [task.lower() for task in tasks]
        if "all" in tasks:
            tasks = ["solver", "pyscf_test"]
        if "solver" in tasks:
            emit_ccsd_solver(
                output_dir,
                terms,
                output_names,
                spin_orbital=spin_orbital,
            )
        if "qp_ccsd_solver" in tasks:
            emit_qp_ccsd_solver(
                output_dir,
                terms,
                output_names,
            )
        if "pyscf_test" in tasks:
            emit_ccsd_pyscf_test(
                output_dir,
                terms,
                output_names,
                pyscf_mol=pyscf_mol,
                spin_orbital=spin_orbital,
            )
        if "eom_solver" in tasks:
            emit_eom_solver(
                output_dir,
                terms,
                output_names,
                spin_orbital=spin_orbital,
            )
        if "eom_pyscf_test" in tasks:
            emit_eom_pyscf_test(
                output_dir,
                terms,
                output_names,
                pyscf_mol=pyscf_mol,
                spin_orbital=spin_orbital,
            )
        print(f"Wrote {output_dir}")
        return
    if list_char_op == ["CCSD_ENERGY"]:
        # Ensure component scripts exist before generating the CCSD driver.
        ccsd_dir = METHODS_DIR / "ccsd"
        _ensure_output_package(ccsd_dir)
        for ops in (["F1", "T1"], ["V2", "T1", "T1"], ["V2", "T2"]):
            terms = build_terms(ops, quiet=quiet)
            expr_name = make_name(ops)
            comp_path = ccsd_dir / f"{expr_name}_einsum.py"
            emit_einsum_code(ops, terms, comp_path)
        output_path = ccsd_dir / "ccsd_energy.py"
        emit_ccsd_energy(output_path)
        print(f"Wrote {output_path}")
        return
    if list_char_op == ["CCSD_AMPLITUDE"]:
        if mode is None:
            mode = "runtime"
        ccsd_amp_dir = METHODS_DIR / "ccsd" / "ccsd_amplitude"
        emit_ccsd_amplitude(
            ccsd_amp_dir,
            mode=mode,
            subset=subset,
            quiet=quiet,
        )
        print(f"Wrote {ccsd_amp_dir}")
        return
    terms = build_terms(list_char_op, quiet=quiet)
    expr_name = make_name(list_char_op)
    output_path = GENERATED_DIR / f"{expr_name}_einsum.py"
    emit_einsum_code(list_char_op, terms, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
