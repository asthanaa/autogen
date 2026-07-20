import numpy as np

from .integrals import build_h2o_631g, compute_integrals, run_scf
from ..generated import residuals

def mp2_init(f, g, o, v):
    eps = np.diag(f)
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
    g_ijab = g[np.ix_(o, o, v, v)]
    t1 = f[np.ix_(o, v)].T / denom_ai
    t2 = g_ijab.transpose(2, 3, 0, 1) / denom_abij
    return t1, t2, denom_ai, denom_abij

def compute_energy(f, g_raw, t1, t2, o, v, spin_orbital=False):
    if spin_orbital:
        f_ov = f[np.ix_(o, v)]
        g_ijab = g_raw[np.ix_(o, o, v, v)]
        t1_ia = t1.T
        t2_ijab = t2.transpose(2, 3, 0, 1)
        e = np.einsum('ia,ia->', f_ov, t1_ia)
        e += 0.25 * np.einsum('ijab,abij->', g_ijab, t2)
        e += 0.5 * np.einsum('ijab,ai,bj->', g_ijab, t1, t1)
        return e
    f_ov = f[np.ix_(o, v)]
    t1_ia = t1.T
    t2_ijab = t2.transpose(2, 3, 0, 1)
    tau = t2_ijab + np.einsum('ia,jb->ijab', t1_ia, t1_ia)
    eris_ovvo = g_raw[np.ix_(o, v, v, o)]
    e = 2.0 * np.einsum('ia,ia->', f_ov, t1_ia)
    e += 2.0 * np.einsum('ijab,iabj->', tau, eris_ovvo)
    e -= np.einsum('jiab,iabj->', tau, eris_ovvo)
    return e

def solve_ccsd(mol=None, max_iter=50, tol=1e-8, damping=0.0, diis_start=2, max_diis=6):
    if mol is None:
        mol = build_h2o_631g()
    mf = run_scf(mol)
    ints = compute_integrals(mol, mf=mf)
    f = ints['f']
    g_raw = ints['g_raw']
    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is True:
        g = g_raw
    else:
        g = ints['g']
    nocc = ints['nocc']
    nmo = ints['nmo']
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    t1, t2, denom_ai, denom_abij = mp2_init(f, g, o, v)
    energy = compute_energy(f, g_raw, t1, t2, o, v, spin_orbital=False)

    t1_list = []
    t2_list = []
    err_list = []

    def diis_extrapolate(t1_list, t2_list, err_list):
        n = len(err_list)
        b = np.empty((n + 1, n + 1))
        b[-1, :] = -1.0
        b[:, -1] = -1.0
        b[-1, -1] = 0.0
        for i in range(n):
            for j in range(n):
                b[i, j] = np.dot(err_list[i], err_list[j])
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        coeff = np.linalg.solve(b, rhs)[:-1]
        t1_new = sum(c * t for c, t in zip(coeff, t1_list))
        t2_new = sum(c * t for c, t in zip(coeff, t2_list))
        return t1_new, t2_new

    for it in range(1, max_iter + 1):
        if hasattr(residuals, 'compute_outputs'):
            outs = residuals.compute_outputs(f, g, t1, t2, o, v)
            r1 = outs['r1']
            r2 = outs['r2']
        else:
            r1 = residuals.compute_r1(f, g, t1, t2, o, v)
            r2 = residuals.compute_r2(f, g, t1, t2, o, v)
        t1_new = t1 + r1 / denom_ai
        t2_new = t2 + r2 / denom_abij
        if damping > 0.0:
            t1_new = (1.0 - damping) * t1_new + damping * t1
            t2_new = (1.0 - damping) * t2_new + damping * t2
        err = np.concatenate([r1.ravel(), r2.ravel()])
        t1_list.append(t1_new.copy())
        t2_list.append(t2_new.copy())
        err_list.append(err)
        if len(err_list) > max_diis:
            t1_list.pop(0)
            t2_list.pop(0)
            err_list.pop(0)
        if it >= diis_start and len(err_list) >= 2:
            t1_new, t2_new = diis_extrapolate(t1_list, t2_list, err_list)
        new_energy = compute_energy(f, g_raw, t1_new, t2_new, o, v, spin_orbital=False)
        r_norm = max(np.max(np.abs(r1)), np.max(np.abs(r2)))
        e_diff = abs(new_energy - energy)
        print(f'iter {it:3d}  energy {new_energy: .10f}  |R| {r_norm:.3e}  dE {e_diff:.3e}')
        t1, t2, energy = t1_new, t2_new, new_energy
        if r_norm < tol and e_diff < tol:
            break
    return energy, t1, t2

def main():
    energy, _t1, _t2 = solve_ccsd()
    print('CCSD correlation energy (iterative):', energy)

if __name__ == '__main__':
    main()
