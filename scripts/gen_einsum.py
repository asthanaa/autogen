from __future__ import annotations

from pathlib import Path
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


def canonicalize_g(labels):
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


def build_terms(list_char_op):
    dict_ind = {}
    lou, dict_ind = make_op.make_op(list_char_op, dict_ind)
    st, co = lou[0].st, lou[0].co
    for i in range(1, len(lou)):
        st, co = multi_cont.multi_cont(st, lou[i].st, co, lou[i].co)

    st, co = full_con.full_con(st, co)
    terms = change_terms.change_terms1(st, co, 1.0, dict_ind, lou)
    for term in terms:
        term.compress()
        term.build_map_org()

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if terms[i].fac != 0.0 and terms[j].fac != 0.0:
                flo = cpre.compare(terms[i], terms[j])
                if flo != 0:
                    terms[i].fac = terms[i].fac + terms[j].fac * flo
                    terms[j].fac = 0.0

    return [term for term in terms if term.fac != 0.0]


def term_to_einsum(term):
    tensors = []
    prefactor = 1.0
    for op, coeff in zip(term.large_op_list, term.coeff_list):
        tensor_name = TENSOR_MAP.get(op.name)
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
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_ccsd_amplitudes, compute_ccsd_amplitudes_ijab, compute_integrals, run_scf")
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
    special_f1t1 = list_char_op == ["F1", "T1"]
    special_v2t1t1 = list_char_op == ["V2", "T1", "T1"]
    special_v2t2 = list_char_op == ["V2", "T2"]
    if special_f1t1:
        lines.append("def compute_f1t1(f, t1_ia, o, v):")
        lines.append("    f_ov = f[np.ix_(o, v)]")
        lines.append("    return 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)")
    elif special_v2t1t1:
        lines.append("def compute_v2t1t1(g_raw, t1_ia, o, v):")
        lines.append("    eris_ovvo = g_raw[np.ix_(o, v, v, o)]")
        lines.append("    tau = np.einsum('ia,jb->ijab', t1_ia, t1_ia)")
        lines.append("    e = 2.0 * np.einsum('ijab,iabj->', tau, eris_ovvo)")
        lines.append("    e -= np.einsum('jiab,iabj->', tau, eris_ovvo)")
        lines.append("    return e")
    elif special_v2t2:
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
    if "t1" in needed or "t2" in needed:
        lines.append("    t1_ai, t2_abij = compute_ccsd_amplitudes(mf)")
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
    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import sys")
    lines.append("")
    lines.append("ROOT = Path(__file__).resolve().parents[1]")
    lines.append("sys.path.insert(0, str(ROOT))")
    lines.append("")
    lines.append("from generated_code.pyscf_integrals import build_h2o_631g, compute_ccsd_amplitudes_ijab, compute_integrals, run_scf")
    lines.append("from generated_code.f1t1_einsum import compute_f1t1")
    lines.append("from generated_code.v2t1t1_einsum import compute_v2t1t1")
    lines.append("from generated_code.v2t2_einsum import compute_v2t2")
    lines.append("")
    lines.append("")
    lines.append("def main():")
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
    lines.append("    t1_ia, t2_ijab = compute_ccsd_amplitudes_ijab(mf)")
    lines.append("")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/gen_einsum.py V2 T1 T1")
        print("   or: python scripts/gen_einsum.py CCSD_ENERGY")
        sys.exit(1)
    list_char_op = sys.argv[1:]
    if list_char_op == ["CCSD_ENERGY"]:
        output_path = ROOT / "generated_code" / "ccsd_energy.py"
        emit_ccsd_energy(output_path)
        print(f"Wrote {output_path}")
        return
    terms = build_terms(list_char_op)
    expr_name = make_name(list_char_op)
    output_path = ROOT / "generated_code" / f"{expr_name}_einsum.py"
    emit_einsum_code(list_char_op, terms, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
