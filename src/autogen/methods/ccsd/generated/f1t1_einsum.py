import numpy as np

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ..runtime.integrals import (
    build_h2o_631g,
    compute_ccsd_amplitudes_ijab,
    compute_integrals,
    run_scf,
)

OCC = set('ijklmn')
VIRT = set('abcdefgh')

def view_tensor(tensor, labels, o, v):
    idx = []
    list_axes = []
    for axis, label in enumerate(labels):
        if label in OCC:
            idx.append(o)
            list_axes.append(axis)
        elif label in VIRT:
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

def zeros_for_labels(labels, nocc, nvirt, nmo):
    shape = []
    for label in labels:
        if label in OCC:
            shape.append(nocc)
        elif label in VIRT:
            shape.append(nvirt)
        else:
            shape.append(nmo)
    return np.zeros(tuple(shape))

def compute_f1t1(f, t1_ia, o, v):
    f_ov = f[np.ix_(o, v)]
    return 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)

def main():
    mol = build_h2o_631g()
    mf = run_scf(mol)
    ints = compute_integrals(mol, mf=mf)
    f = ints['f']
    nocc = ints['nocc']
    nmo = ints['nmo']
    nvirt = nmo - nocc
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    t1_ia, t2_ijab = compute_ccsd_amplitudes_ijab(mf)
    t1 = t1_ai
    value = compute_f1t1(f, t1_ia, o, v)
    print('f1t1 value:', value)

if __name__ == '__main__':
    main()
