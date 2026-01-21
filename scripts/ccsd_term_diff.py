#!/usr/bin/env python
"""Compare CCSD term-group contributions against PySCF update_amps residuals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _ensure_paths():
    src = ROOT / "src"
    for path in (str(ROOT), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)


def view_tensor(tensor, labels, o, v):
    occ = set("ijklmn")
    virt = set("abcdefgh")
    idx = []
    list_axes = []
    for axis, label in enumerate(labels):
        if label in occ:
            idx.append(o)
            list_axes.append(axis)
        elif label in virt:
            idx.append(v)
            list_axes.append(axis)
        else:
            idx.append(slice(None))
    if not list_axes:
        return tensor[tuple(idx)]
    ix = np.ix_(*[idx[a] for a in list_axes])
    ix_iter = iter(ix)
    full_idx = []
    for axis in range(len(idx)):
        if axis in list_axes:
            full_idx.append(next(ix_iter))
        else:
            full_idx.append(idx[axis])
    return tensor[tuple(full_idx)]


def output_shape(output_labels, o, v):
    occ = set("ijklmn")
    virt = set("abcdefgh")
    shape = []
    for label in output_labels:
        if label in occ:
            shape.append(len(o))
        elif label in virt:
            shape.append(len(v))
        else:
            shape.append(len(o) + len(v))
    return tuple(shape)


def dot(a, b):
    return float(np.dot(a.ravel(), b.ravel()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to spec file.")
    parser.add_argument("--use-g-raw", action="store_true", help="Use raw g integrals.")
    parser.add_argument("--use-h1", action="store_true", help="Use h1 instead of f.")
    parser.add_argument("--output", choices=["X1", "X2", "both"], default="both")
    parser.add_argument("--max-groups", type=int, default=20)
    args = parser.parse_args()

    import sys

    _ensure_paths()
    from pyscf import gto, scf, cc
    from scripts import gen_einsum
    from generated_code import pyscf_integrals

    spec = gen_einsum.load_spec(args.spec)
    (
        terms,
        output_names,
        tensor_map,
        view_tensors,
        _outdir,
        _tasks,
        pyscf_mol,
        _spin,
        _spin_adapted,
    ) = gen_einsum.parse_spec_terms(spec)

    if pyscf_mol:
        mol = gto.M(**pyscf_mol)
    else:
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", charge=0, spin=0)

    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    eris = mycc.ao2mo()
    t1_ref, t2_ref = mycc.update_amps(mycc.t1, mycc.t2, eris)

    ints = pyscf_integrals.compute_integrals(mol, mf=mf)
    f = ints["h1"] if args.use_h1 else ints["f"]
    spin_summed = os.getenv("AUTOGEN_SPIN_SUMMED", "1") != "0"
    if args.use_g_raw:
        g = ints["g_raw"]
    else:
        g = ints["g_raw"] if spin_summed else ints["g"]
    nocc = ints["nocc"]
    nmo = ints["nmo"]
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    t1_ai = mycc.t1.T
    t2_abij = mycc.t2.transpose(2, 3, 0, 1)

    eps = np.diag(ints["f"])
    eps_occ = eps[o]
    eps_virt = eps[v]
    denom_ai = eps_occ[None, :] - eps_virt[:, None]
    denom_abij = (
        eps_occ[None, None, :, None]
        + eps_occ[None, None, None, :]
        - eps_virt[:, None, None, None]
        - eps_virt[None, :, None, None]
    )
    denom_ai = np.where(abs(denom_ai) < 1e-12, 1e-12, denom_ai)
    denom_abij = np.where(abs(denom_abij) < 1e-12, 1e-12, denom_abij)

    r1_ref = (t1_ref.T - t1_ai) * denom_ai
    r2_ref = (t2_ref.transpose(2, 3, 0, 1) - t2_abij) * denom_abij

    group_contrib = []
    total_r1 = np.zeros((len(v), len(o)))
    total_r2 = np.zeros((len(v), len(v), len(o), len(o)))

    for term_spec in terms:
        ops = term_spec["ops"]
        fac = term_spec["fac"]
        output_key = term_spec["output_key"]
        if args.output != "both" and output_key != args.output:
            continue

        term_list = gen_einsum.build_terms(ops, merge=False, quiet=True)
        group_r1 = None
        group_r2 = None
        for term in term_list:
            term.fac *= fac
            out_labels, tensors, coeff = gen_einsum._term_to_residual_struct(
                term, tensor_map=tensor_map, require_output=False
            )
            structs = [(out_labels, tensors, coeff)]
            for out_labels, tensors, coeff in structs:
                if output_key != "scalar" and not out_labels:
                    continue
                subs_in = ",".join(labels for _, labels in tensors)
                subs = f"{subs_in}->{out_labels}"
                args_list = []
                for name, labels in tensors:
                    if name in ("g", "f"):
                        tensor = g if name == "g" else f
                        args_list.append(view_tensor(tensor, labels, o, v))
                    elif name == "t1":
                        args_list.append(t1_ai)
                    elif name == "t2":
                        args_list.append(t2_abij)
                    else:
                        raise ValueError(f"Unsupported tensor '{name}' in term {ops}")
                term_val = coeff * np.einsum(subs, *args_list)
                if out_labels == "ai":
                    if group_r1 is None:
                        group_r1 = np.zeros(output_shape(out_labels, o, v))
                    group_r1 += term_val
                elif out_labels == "abij":
                    if group_r2 is None:
                        group_r2 = np.zeros(output_shape(out_labels, o, v))
                    group_r2 += term_val

        if group_r1 is not None:
            total_r1 += group_r1
            group_contrib.append(
                (
                    "X1",
                    " ".join(ops),
                    np.max(np.abs(group_r1)),
                    float(np.linalg.norm(group_r1)),
                    dot(group_r1, r1_ref),
                    dot(group_r1, group_r1),
                )
            )
        if group_r2 is not None:
            total_r2 += group_r2
            group_contrib.append(
                (
                    "X2",
                    " ".join(ops),
                    np.max(np.abs(group_r2)),
                    float(np.linalg.norm(group_r2)),
                    dot(group_r2, r2_ref),
                    dot(group_r2, group_r2),
                )
            )

    print("Reference residuals:")
    print("  r1 max abs:", np.max(np.abs(r1_ref)))
    print("  r2 max abs:", np.max(np.abs(r2_ref)))
    print("")
    print("Totals from generated terms:")
    print("  r1 max abs:", np.max(np.abs(total_r1)))
    print("  r2 max abs:", np.max(np.abs(total_r2)))
    print("  r1 diff max:", np.max(np.abs(total_r1 - r1_ref)))
    print("  r2 diff max:", np.max(np.abs(total_r2 - r2_ref)))
    print("")
    print("Top term groups by contribution:")
    group_contrib.sort(key=lambda x: x[2], reverse=True)
    for output, name, max_abs, norm, dot_ref, dot_self in group_contrib[: args.max_groups]:
        scale = dot_ref / dot_self if dot_self else 0.0
        print(
            f"{output:>2}  {name:35s}  max={max_abs: .3e}  "
            f"norm={norm: .3e}  scale~={scale: .3e}"
        )


if __name__ == "__main__":
    main()
