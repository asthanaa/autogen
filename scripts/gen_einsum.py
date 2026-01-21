from __future__ import annotations

# Generate einsum-based evaluators from Autogen contraction output.

import contextlib
from pathlib import Path
import os
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from autogen.main_tools import multi_cont  # noqa: E402
from autogen.library import change_terms  # noqa: E402
from autogen.library import full_con  # noqa: E402
from autogen.library import make_op  # noqa: E402
from autogen.library import compare as cpre  # noqa: E402
from autogen.pkg import fix_uv  # noqa: E402


TENSOR_MAP = {
    "V2": "g",
    "F1": "f",
    "T1": "t1",
    "T2": "t2",
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


def canonicalize_g(labels):
    # Normalize antisymmetrized two-electron labels to ijab ordering.
    if len(labels) != 4:
        return labels, 1.0
    occ_idx = [i for i, label in enumerate(labels) if label in OCC_SET]
    virt_idx = [i for i, label in enumerate(labels) if label in VIRT_SET]
    if occ_idx != [0, 1] or virt_idx != [2, 3]:
        return labels, 1.0
    occ_labels = [labels[i] for i in occ_idx]
    virt_labels = [labels[i] for i in virt_idx]
    occ_sorted = sorted(occ_labels)
    virt_sorted = sorted(virt_labels)
    sign = 1.0
    if occ_labels != occ_sorted:
        sign *= -1.0
    if virt_labels != virt_sorted:
        sign *= -1.0
    return "".join(occ_sorted + virt_sorted), sign


def _label_type(label):
    if label in VIRT_SET:
        return "v"
    if label in OCC_SET:
        return "o"
    return "p"


def _sort_labels(labels):
    order = {"v": 0, "o": 1, "p": 2}
    return sorted(labels, key=lambda x: (order[_label_type(x)], x))


def _canonicalize_term_labels(output_labels, tensors):
    mapping = {}
    occ_iter = iter("ijklmn")
    virt_iter = iter("abcdefg")
    gen_iter = iter("pqrst")

    def map_label(label):
        if label in mapping:
            return mapping[label]
        if label in VIRT_SET:
            new = next(virt_iter)
        elif label in OCC_SET:
            new = next(occ_iter)
        else:
            new = next(gen_iter)
        mapping[label] = new
        return new

    output_can = "".join(map_label(label) for label in output_labels)
    tensors_can = []
    for name, labels in tensors:
        tensors_can.append((name, "".join(map_label(label) for label in labels)))
    return output_can, tensors_can


def _pair_key(op1, op2):
    name1, labels1 = op1
    name2, labels2 = op2
    common = set(labels1) & set(labels2)
    out_labels = []
    for label in labels1:
        if label not in common and label not in out_labels:
            out_labels.append(label)
    for label in labels2:
        if label not in common and label not in out_labels:
            out_labels.append(label)
    out_labels = _sort_labels(out_labels)
    ops = tuple(sorted(((name1, labels1), (name2, labels2))))
    return ops, tuple(sorted(common)), tuple(out_labels)


def _term_to_residual_struct(term, tensor_map=None, require_output=True):
    tensors = []
    output_labels = None
    mapping = tensor_map or TENSOR_MAP
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        if op.name.startswith("X"):
            if output_labels is not None:
                raise ValueError("Multiple X projectors found in term.")
            output_labels = output_labels_from_xop(coeff)
            continue
        tensor_name = mapping.get(op.name)
        if not tensor_name and op.name.startswith("T") and len(op.name) > 1:
            tensor_name = "t1" if op.name[1] == "1" else "t2"
        if not tensor_name:
            raise ValueError(f"Unsupported operator {op.name}")
        tensors.append((tensor_name, "".join(coeff)))
    if output_labels is None:
        if require_output:
            raise ValueError("No X projector found in amplitude term.")
        output_labels = ""
    return output_labels, tensors, term.fac


def _spin_adapt_output_spins(output_labels):
    if not output_labels:
        return {}
    if len(output_labels) == 2:
        return {output_labels[0]: 0, output_labels[1]: 0}
    if len(output_labels) == 4:
        return {
            output_labels[0]: 0,
            output_labels[2]: 0,
            output_labels[1]: 1,
            output_labels[3]: 1,
        }
    return {label: 0 for label in output_labels}


def _spin_adapt_expand_g(tensors):
    expanded = [(list(tensors), 1.0)]
    for idx, (name, labels) in enumerate(tensors):
        if name != "g" or len(labels) != 4:
            continue
        next_expanded = []
        for base, sign in expanded:
            direct = list(base)
            direct[idx] = (name, labels)
            next_expanded.append((direct, sign))
            exch_labels = labels[:2] + labels[3] + labels[2]
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
        for label in labels:
            find(label)
    for label in fixed_spins:
        find(label)

    for name, labels in tensors:
        if name in {"f", "t1"}:
            union(labels[0], labels[1])
        elif name == "t2":
            union(labels[0], labels[2])
            union(labels[1], labels[3])
        elif name == "g":
            union(labels[0], labels[2])
            union(labels[1], labels[3])

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
    if name in {"g", "t2"} and len(labels) == 4:
        left = sorted(labels[:2])
        right = sorted(labels[2:])
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
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    if terms[i].fac != 0.0 and terms[j].fac != 0.0:
                        flo = cpre.compare(terms[i], terms[j])
                        if flo != 0:
                            terms[i].fac = terms[i].fac + terms[j].fac * flo
                            terms[j].fac = 0.0

        return [term for term in terms if term.fac != 0.0]


def term_to_einsum(term):
    # Map each operator to a tensor name and build an einsum signature.
    tensors = []
    prefactor = 1.0
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        tensor_name = TENSOR_MAP.get(op.name)
        if not tensor_name and op.name.startswith("T") and len(op.name) > 1:
            tensor_name = "t1" if op.name[1] == "1" else "t2"
        if not tensor_name:
            raise ValueError(f"Unsupported operator {op.name}")
        labels = "".join(coeff)
        if tensor_name == "g":
            labels, sign = canonicalize_g(labels)
            prefactor *= sign
        tensors.append((tensor_name, labels))

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
    virt = [label for label in coeff if label in VIRT_SET]
    occ = [label for label in coeff if label in OCC_SET]
    return "".join(virt + occ)


def term_to_residual_einsum(term):
    tensors = []
    output_labels = None
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
    if output_key.startswith("X") and output_key[1:].isdigit():
        return f"r{output_key[1:]}"
    if output_key == "scalar":
        return "scalar"
    return output_key.lower()


def resolve_output_name(output_key, output_names):
    name = output_names.get(output_key)
    if name is None:
        name = default_output_name(output_key)
    if not name.isidentifier():
        raise ValueError(f"Invalid output name '{name}' for '{output_key}'.")
    return name


def load_spec(spec_path):
    return runpy.run_path(str(spec_path))


def parse_spec_terms(spec):
    if "TERMS" not in spec:
        raise ValueError("Spec file must define TERMS.")
    raw_terms = spec["TERMS"]
    if not isinstance(raw_terms, (list, tuple)):
        raise ValueError("TERMS must be a list or tuple.")

    tensor_map = dict(TENSOR_MAP)
    overrides = spec.get("TENSOR_MAP")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("TENSOR_MAP must be a dict.")
        tensor_map.update({str(k): str(v) for k, v in overrides.items()})

    output_names = spec.get("OUTPUTS", {})
    if output_names is None:
        output_names = {}
    if not isinstance(output_names, dict):
        raise ValueError("OUTPUTS must be a dict.")
    output_names = {str(k): str(v) for k, v in output_names.items()}

    view_tensors = spec.get("VIEW_TENSORS", ("g", "f"))
    if isinstance(view_tensors, str):
        view_tensors = [view_tensors]
    if not isinstance(view_tensors, (list, tuple)):
        raise ValueError("VIEW_TENSORS must be a list or tuple.")
    view_tensors = [str(name) for name in view_tensors]

    output_dir = spec.get("OUTPUT_DIR")
    if output_dir is not None:
        output_dir = str(output_dir)

    tasks = spec.get("TASKS", [])
    if tasks is None:
        tasks = []
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("TASKS must be a list or tuple.")
    tasks = [str(task) for task in tasks]

    pyscf_mol = spec.get("PYSCF_MOL")
    if pyscf_mol is not None:
        if not isinstance(pyscf_mol, dict):
            raise ValueError("PYSCF_MOL must be a dict.")
        pyscf_mol = {str(k): v for k, v in pyscf_mol.items()}
    spin_orbital = bool(spec.get("SPIN_ORBITAL", False))
    spin_adapted = bool(spec.get("SPIN_ADAPTED", False))

    parsed = []
    for item in raw_terms:
        output_key = None
        if isinstance(item, dict):
            ops = item.get("ops")
            fac = item.get("fac", 1.0)
            output_key = item.get("output")
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                fac, ops = item
            elif len(item) == 3:
                output_key, fac, ops = item
            else:
                raise ValueError("Term tuples must have 2 or 3 entries.")
        else:
            raise ValueError("Each term must be a dict or tuple.")

        if ops is None:
            raise ValueError("Each term must define ops.")
        if not isinstance(ops, (list, tuple)):
            raise ValueError("ops must be a list or tuple.")
        ops = [str(op) for op in ops]

        if output_key is None:
            output_key = _infer_output_key(ops)
        else:
            output_key = str(output_key)

        parsed.append({"output_key": output_key, "fac": float(fac), "ops": ops})

    return (
        parsed,
        output_names,
        tensor_map,
        view_tensors,
        output_dir,
        tasks,
        pyscf_mol,
        spin_orbital,
        spin_adapted,
    )


def _ordered_tensor_names(names):
    tensor_order = ["f", "g", "t1", "t2", "d1", "d2", "x1", "x2"]
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
):
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = mode.lower()
    if mode not in {"full", "intermediates"}:
        raise ValueError(f"Unsupported spec mode: {mode}")
    spin_summed = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
    if spin_summed_override is not None:
        spin_summed = spin_summed_override
    if spin_summed and spin_adapted and _allow_spin_adapted:
        mode_flag = os.getenv("AUTOGEN_SPIN_SUMMED_MODE", "direct").lower().strip()
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
            )
            emit_spin_adapted_wrapper(output_dir, output_names, mode=mode)
            return

    built_terms = []
    output_order = []
    with _spin_summed_context(spin_summed):
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
        structs = [(output_labels, tensors, coeff)]
        for out_labels, tensors, coeff in structs:
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
    lines.append("")
    lines.append(f"AUTOGEN_SPIN_SUMMED = {spin_summed}")
    lines.append(f"AUTOGEN_INTERMEDIATES = {mode == 'intermediates'}")
    lines.append(f"VIEW_TENSORS = {tuple(view_tensor_names)}")
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
    lines.append("def zeros_for_output(labels, o, v):")
    lines.append("    if not labels:")
    lines.append("        return 0.0")
    lines.append("    shape = []")
    lines.append("    for label in labels:")
    lines.append("        if label in OCC:")
    lines.append("            shape.append(len(o))")
    lines.append("        elif label in VIRT:")
    lines.append("            shape.append(len(v))")
    lines.append("        else:")
    lines.append("            shape.append(len(o) + len(v))")
    lines.append("    return np.zeros(tuple(shape))")
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
                subs = f"{labels1},{labels2}->{out_labels}"
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
        lines.append(f"    inter = compute_intermediates({non_view_args})")
        for info in inter_defs:
            name = info["name"]
            lines.append(f"    {name} = inter['{name}']")
        lines.append("    outputs = {}")
        for output_key in output_order:
            output_name = resolve_output_name(output_key, output_names)
            output_labels = output_labels_map[output_key]
            lines.append(
                f"    {output_name} = zeros_for_output('{output_labels}', o, v)"
            )
            grouped = _group_terms_by_subs(output_structs[output_key])
            for (subs, tensors), coeff in sorted(grouped.items()):
                if coeff == 0.0:
                    continue
                args = []
                for name, labels in tensors:
                    if name in used_intermediates:
                        args.append(name)
                    elif name in view_tensor_names:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    {output_name} += ({coeff}) * np.einsum('{subs}', {', '.join(args)}, optimize=True)"
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
            lines.append(
                f"    out = zeros_for_output('{output_labels}', o, v)"
            )
            grouped = _group_terms_by_subs(output_structs[output_key])
            for (subs, tensors), coeff in sorted(grouped.items()):
                if coeff == 0.0:
                    continue
                args = []
                for name, labels in tensors:
                    if name in view_tensor_names:
                        args.append(f"get_view('{name}', '{labels}')")
                    else:
                        args.append(name)
                lines.append(
                    f"    out += ({coeff}) * np.einsum('{subs}', {', '.join(args)}, optimize=True)"
                )
            lines.append("    return out")
            lines.append("")

    (output_dir / filename).write_text("\n".join(lines) + "\n")


def emit_spin_adapted_wrapper(output_dir, output_names, mode: str = "full"):
    output_dir = Path(output_dir)
    r1_name = resolve_output_name("X1", output_names)
    r2_name = resolve_output_name("X2", output_names)
    inter_flag = mode == "intermediates"
    wrapper = f"""import numpy as np

from pyscf.cc import addons

from . import residuals_spinorb as spinorb

AUTOGEN_SPIN_SUMMED = True
AUTOGEN_INTERMEDIATES = {inter_flag}
VIEW_TENSORS = ('f', 'g')

def _orbspin(nocc, nmo):
    orbspin = np.zeros(2 * nmo, dtype=int)
    orbspin[1::2] = 1
    return orbspin

def _build_spin_orbital_fock(f, nocc):
    nmo = f.shape[0]
    nocc_so = 2 * nocc
    nvir_so = 2 * (nmo - nocc)
    f_so = np.zeros((nocc_so + nvir_so, nocc_so + nvir_so))
    for p in range(nmo):
        for q in range(nmo):
            for spin in (0, 1):
                if p < nocc:
                    p_so = 2 * p + spin
                else:
                    p_so = nocc_so + 2 * (p - nocc) + spin
                if q < nocc:
                    q_so = 2 * q + spin
                else:
                    q_so = nocc_so + 2 * (q - nocc) + spin
                f_so[p_so, q_so] = f[p, q]
    return f_so

def _build_spin_orbital_g(g_raw, nocc):
    nmo = g_raw.shape[0]
    nocc_so = 2 * nocc
    nvir_so = 2 * (nmo - nocc)
    nso = nocc_so + nvir_so
    g_phys = g_raw.transpose(0, 2, 1, 3)
    g_so = np.zeros((nso, nso, nso, nso))
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    val = g_phys[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            for sr in (0, 1):
                                for ss in (0, 1):
                                    if sp != sr or sq != ss:
                                        continue
                                    if p < nocc:
                                        p_so = 2 * p + sp
                                    else:
                                        p_so = nocc_so + 2 * (p - nocc) + sp
                                    if q < nocc:
                                        q_so = 2 * q + sq
                                    else:
                                        q_so = nocc_so + 2 * (q - nocc) + sq
                                    if r < nocc:
                                        r_so = 2 * r + sr
                                    else:
                                        r_so = nocc_so + 2 * (r - nocc) + sr
                                    if s < nocc:
                                        s_so = 2 * s + ss
                                    else:
                                        s_so = nocc_so + 2 * (s - nocc) + ss
                                    g_so[p_so, q_so, r_so, s_so] = val
    g_so_as = g_so - g_so.transpose(0, 1, 3, 2)
    return g_so_as

def compute_outputs(f, g, t1, t2, o, v):
    nocc = len(o)
    nmo = f.shape[0]
    orbspin = _orbspin(nocc, nmo)
    f_so = _build_spin_orbital_fock(f, nocc)
    g_so = _build_spin_orbital_g(g, nocc)
    t1_so = addons.spatial2spin(t1.T, orbspin).T
    t2_so = addons.spatial2spin(t2.transpose(2, 3, 0, 1), orbspin).transpose(2, 3, 0, 1)
    o_so = list(range(2 * nocc))
    v_so = list(range(2 * nocc, 2 * nmo))
    outs = spinorb.compute_outputs(f_so, g_so, t1_so, t2_so, o_so, v_so)
    r1_so = outs['{r1_name}'].T
    r2_so = outs['{r2_name}'].transpose(2, 3, 0, 1)
    r1a, r1b = addons.spin2spatial(r1_so, orbspin)
    r2aa, r2ab, r2bb = addons.spin2spatial(r2_so, orbspin)
    r1 = r1a.T
    r2 = r2ab.transpose(2, 3, 0, 1)
    return {{{r1_name!r}: r1, {r2_name!r}: r2}}

def compute_{r1_name}(f, g, t1, t2, o, v):
    return compute_outputs(f, g, t1, t2, o, v)[{r1_name!r}]

def compute_{r2_name}(f, g, t1, t2, o, v):
    return compute_outputs(f, g, t1, t2, o, v)[{r2_name!r}]
"""
    (output_dir / "residuals.py").write_text(wrapper)


def _module_path_from_dir(output_dir: Path):
    rel = output_dir.relative_to(ROOT)
    return ".".join(rel.parts)


def _validate_ccsd_outputs(spec_terms, output_names):
    output_keys = {term["output_key"] for term in spec_terms}
    if "X1" not in output_keys or "X2" not in output_keys:
        raise ValueError("CCSD solver requires X1 and X2 outputs in the spec.")
    r1_name = resolve_output_name("X1", output_names)
    r2_name = resolve_output_name("X2", output_names)
    return r1_name, r2_name


def emit_ccsd_solver(output_dir, spec_terms, output_names, spin_orbital=False):
    output_dir = Path(output_dir)
    module_path = _module_path_from_dir(output_dir)
    r1_name, r2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append("ROOT = Path(__file__).resolve().parents[2]")
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
    lines.append("")
    lines.append("    def diis_extrapolate(t1_list, t2_list, err_list):")
    lines.append("        n = len(err_list)")
    lines.append("        b = np.empty((n + 1, n + 1))")
    lines.append("        b[-1, :] = -1.0")
    lines.append("        b[:, -1] = -1.0")
    lines.append("        b[-1, -1] = 0.0")
    lines.append("        for i in range(n):")
    lines.append("            for j in range(n):")
    lines.append("                b[i, j] = np.dot(err_list[i], err_list[j])")
    lines.append("        rhs = np.zeros(n + 1)")
    lines.append("        rhs[-1] = -1.0")
    lines.append("        coeff = np.linalg.solve(b, rhs)[:-1]")
    lines.append("        t1_new = sum(c * t for c, t in zip(coeff, t1_list))")
    lines.append("        t2_new = sum(c * t for c, t in zip(coeff, t2_list))")
    lines.append("        return t1_new, t2_new")
    lines.append("")
    lines.append("    for it in range(1, max_iter + 1):")
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


def emit_ccsd_pyscf_test(output_dir, spec_terms, output_names, pyscf_mol=None, spin_orbital=False):
    output_dir = Path(output_dir)
    module_path = _module_path_from_dir(output_dir)
    _r1_name, _r2_name = _validate_ccsd_outputs(spec_terms, output_names)

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append("ROOT = Path(__file__).resolve().parents[2]")
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
    lines.append("ROOT = Path(__file__).resolve().parents[1]")
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
    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append("ROOT = Path(__file__).resolve().parents[1]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("# CCSD energy driver that matches PySCF's spatial-orbital energy formula.")
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_ccsd_amplitudes_ijab, compute_integrals, run_scf")
    lines.append("from generated_code.f1t1_einsum import compute_f1t1")
    lines.append("from generated_code.v2t1t1_einsum import compute_v2t1t1")
    lines.append("from generated_code.v2t2_einsum import compute_v2t2")
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
        lines.append("ROOT = Path(__file__).resolve().parents[2]")
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
            output_labels, tensors = _canonicalize_term_labels(output_labels, tensors)
            x1_struct.append((output_labels, tensors, coeff))
        for term in x2_terms:
            output_labels, tensors, coeff = _term_to_residual_struct(term)
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
    solver.append("ROOT = Path(__file__).resolve().parents[2]")
    solver.append("sys.path.insert(0, str(ROOT))")
    solver.append("")
    solver.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_integrals, run_scf")
    solver.append("from generated_code.ccsd_amplitude import residuals as residuals")
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
    energy.append("ROOT = Path(__file__).resolve().parents[2]")
    energy.append("sys.path.insert(0, str(ROOT))")
    energy.append("")
    energy.append("from generated_code.ccsd_amplitude.solver import solve_ccsd")
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
            "[--full|--intermediates] [--out OUTPUT_DIR] [--tasks solver,pyscf_test] [--quiet]"
        )
        sys.exit(1)
    args = list(sys.argv[1:])
    quiet = False
    mode = None
    subset = "both"
    spec_path = None
    output_override = None
    if "--quiet" in args:
        quiet = True
        args = [arg for arg in args if arg != "--quiet"]
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
        if tasks_override is not None:
            tasks = [task.strip() for task in tasks_override.split(",") if task.strip()]
        if output_override is not None:
            output_dir = output_override
        if output_dir is None:
            output_dir = ROOT / "generated_code" / Path(spec_path).stem
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
        emit_spec_residuals(
            output_dir,
            terms,
            output_names,
            tensor_map,
            view_tensors,
            mode=mode,
            quiet=quiet,
            spin_adapted=spin_adapted,
        )
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
        if "pyscf_test" in tasks:
            emit_ccsd_pyscf_test(
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
        for ops in (["F1", "T1"], ["V2", "T1", "T1"], ["V2", "T2"]):
            terms = build_terms(ops, quiet=quiet)
            expr_name = make_name(ops)
            comp_path = ROOT / "generated_code" / f"{expr_name}_einsum.py"
            emit_einsum_code(ops, terms, comp_path)
        output_path = ROOT / "generated_code" / "ccsd_energy.py"
        emit_ccsd_energy(output_path)
        print(f"Wrote {output_path}")
        return
    if list_char_op == ["CCSD_AMPLITUDE"]:
        if mode is None:
            mode = "runtime"
        emit_ccsd_amplitude(
            ROOT / "generated_code" / "ccsd_amplitude",
            mode=mode,
            subset=subset,
            quiet=quiet,
        )
        print(f"Wrote {ROOT / 'generated_code' / 'ccsd_amplitude'}")
        return
    terms = build_terms(list_char_op, quiet=quiet)
    expr_name = make_name(list_char_op)
    output_path = ROOT / "generated_code" / f"{expr_name}_einsum.py"
    emit_einsum_code(list_char_op, terms, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
