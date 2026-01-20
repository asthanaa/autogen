from __future__ import annotations

# Generate einsum-based evaluators from Autogen contraction output.

import contextlib
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from autogen.main_tools import multi_cont  # noqa: E402
from autogen.library import change_terms  # noqa: E402
from autogen.library import full_con  # noqa: E402
from autogen.library import make_op  # noqa: E402
from autogen.library import compare as cpre  # noqa: E402


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


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


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
        if op.name in {"X1", "X2"}:
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
        raise ValueError("No X1/X2 projector found in amplitude term.")

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
    if mode not in {"full", "runtime"}:
        raise ValueError(f"Unsupported CCSD amplitude mode: {mode}")

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
    else:
        x1_terms, x2_terms = build_ccsd_amplitude_terms(quiet=quiet, subset=subset)
        lines = []
        lines.append("import numpy as np")
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
    solver.append("from generated_code.ccsd_amplitude.residuals import compute_r1, compute_r2")
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
    solver.append("def compute_energy(f, g, t1, t2, o, v):")
    solver.append("    f_ai = f[np.ix_(o, v)].T")
    solver.append("    g_ijab = g[np.ix_(o, o, v, v)]")
    solver.append("    e = np.einsum('ai,ai->', f_ai, t1)")
    solver.append("    e += 0.5 * np.einsum('ijab,ai,bj->', g_ijab, t1, t1)")
    solver.append("    e += 0.25 * np.einsum('ijab,abij->', g_ijab, t2)")
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
    solver.append("    nocc = ints['nocc']")
    solver.append("    nmo = ints['nmo']")
    solver.append("    o = list(range(nocc))")
    solver.append("    v = list(range(nocc, nmo))")
    solver.append("")
    solver.append("    t1, t2, denom_ai, denom_abij = mp2_init(f, g, o, v)")
    solver.append("    energy = compute_energy(f, g, t1, t2, o, v)")
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
    solver.append("        r1 = compute_r1(f, g, t1, t2, o, v)")
    solver.append("        r2 = compute_r2(f, g, t1, t2, o, v)")
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
    solver.append("        new_energy = compute_energy(f, g, t1_new, t2_new, o, v)")
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
            "[--runtime|--full] [--x1-only|--x2-only] [--quiet]"
        )
        sys.exit(1)
    args = list(sys.argv[1:])
    quiet = False
    mode = None
    subset = "both"
    if "--quiet" in args:
        quiet = True
        args = [arg for arg in args if arg != "--quiet"]
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
    list_char_op = args
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
