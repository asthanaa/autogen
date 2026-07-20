import numpy as np

import os

from autogen.methods.ccsd.runtime.integrals import build_h2o_631g, compute_integrals, run_scf
from ..generated import residuals

SPIN_ORBITAL = False

def _pyscf_eom_modules():
    from pyscf import cc, lib
    from pyscf.cc import eom_rccsd

    return cc, lib, eom_rccsd

def _abij_to_iajb(r2):
    return r2.transpose(2, 0, 3, 1)

def _iajb_to_abij(r2):
    return r2.transpose(1, 3, 0, 2)

def pack(r1, r2, nocc, nvirt):
    r1_ia = r1.T
    r2_iajb = _abij_to_iajb(r2)
    nov = nocc * nvirt
    mat = r2_iajb.reshape(nov, nov)
    tril = np.tril_indices(nov)
    return np.concatenate([r1_ia.ravel(), mat[tril]])

def unpack(vec, nocc, nvirt):
    _cc, lib, _eom_rccsd = _pyscf_eom_modules()
    nov = nocc * nvirt
    n1 = nov
    r1_ia = vec[:n1].reshape(nocc, nvirt)
    mat = lib.unpack_tril(vec[n1:], filltriu=lib.SYMMETRIC)
    r2_iajb = mat.reshape(nocc, nvirt, nocc, nvirt)
    r2 = _iajb_to_abij(r2_iajb)
    return r1_ia.T, r2

def build_reference(mol, mf=None, t1=None, t2=None):
    cc, _lib, _eom_rccsd = _pyscf_eom_modules()
    if mf is None:
        mf = run_scf(mol)
    if t1 is None or t2 is None:
        mycc = cc.CCSD(mf).run()
        t1 = mycc.t1.T
        t2 = mycc.t2.transpose(2, 3, 0, 1)
    ints = compute_integrals(mol, mf=mf)
    f = ints['f']
    g_raw = ints['g_raw']
    mode = getattr(residuals, 'AUTOGEN_SPIN_SUMMED_MODE', 'direct')
    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is True:
        if mode == 'spinorb':
            g = g_raw
        else:
            g = g_raw.transpose(0, 2, 1, 3)
    else:
        g = ints['g']
    nocc = ints['nocc']
    nmo = ints['nmo']
    o = list(range(nocc))
    v = list(range(nocc, nmo))
    return f, g, t1, t2, o, v

def sigma_vector(vec, f, g, t1, t2, o, v):
    if isinstance(vec, (list, tuple)):
        return np.vstack([sigma_vector(vec_row, f, g, t1, t2, o, v) for vec_row in vec])
    if vec.ndim == 2:
        return np.vstack([sigma_vector(vec_row, f, g, t1, t2, o, v) for vec_row in vec])
    nocc = len(o)
    nvirt = len(v)
    r1, r2 = unpack(vec, nocc, nvirt)
    s1 = residuals.compute_s1(f, g, t1, t2, r1, r2, o, v)
    s2 = residuals.compute_s2(f, g, t1, t2, r1, r2, o, v)
    return pack(s1, s2, nocc, nvirt)

def solve_eom_ccsd(mol=None, mf=None, t1=None, t2=None, nroots=3, max_iter=50, tol=1e-8, max_space=20):
    cc, lib, eom_rccsd = _pyscf_eom_modules()
    if mol is None:
        mol = build_h2o_631g()
    if SPIN_ORBITAL:
        raise ValueError('EE-EOM-CCSD solver is spin-summed RHF only.')
    if getattr(residuals, 'AUTOGEN_SPIN_SUMMED', None) is not True:
        raise ValueError('EE-EOM-CCSD solver requires spin-summed residuals.')
    f, g, t1, t2, o, v = build_reference(mol, mf=mf, t1=t1, t2=t2)
    use_pyscf_diag = os.getenv('AUTOGEN_EOM_USE_PYSCF_DIAG', '1') != '0'
    diag = None
    guess = None
    max_memory = 4000
    eff_tol = tol
    if use_pyscf_diag and mf is not None:
        try:
            mycc = cc.CCSD(mf)
            mycc.t1 = t1.T
            mycc.t2 = t2.transpose(2, 3, 0, 1)
            eom = eom_rccsd.EOMEESinglet(mycc)
            imds = eom.make_imds()
            diag = eom.get_diag(imds)
            guess = eom.get_init_guess(nroots, koopmans=False, diag=diag)
            max_memory = max(0, eom.max_memory - lib.current_memory()[0])
            eff_tol = eom.conv_tol
        except Exception:
            diag = None
            guess = None
    if diag is None:
        eps = np.diag(f)
        eps_occ = eps[o]
        eps_virt = eps[v]
        denom_ai = eps_virt[:, None] - eps_occ[None, :]
        denom_abij = (
            eps_virt[:, None, None, None]
            + eps_virt[None, :, None, None]
            - eps_occ[None, None, :, None]
            - eps_occ[None, None, None, :]
        )
        denom_iajb = denom_abij.transpose(2, 0, 3, 1)
        nov = len(o) * len(v)
        denom_mat = denom_iajb.reshape(nov, nov)
        tril = np.tril_indices(nov)
        diag = np.concatenate([denom_ai.T.ravel(), denom_mat[tril]])
        guess_idx = np.argsort(diag)[:nroots]
        guess = []
        for idx in guess_idx:
            v0 = np.zeros(diag.size)
            v0[idx] = 1.0
            guess.append(v0)
    matvec = lambda vec: sigma_vector(vec, f, g, t1, t2, o, v)
    def precond(r, e0, x0):
        return r / (e0 - diag + 1e-12)
    real_system = np.isrealobj(f)
    def pickeig(w, v, nroots, envs):
        real_idx = np.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
    conv, es, _vecs = lib.davidson_nosym1(
        matvec, guess, precond, pick=pickeig,
        tol=eff_tol, max_cycle=max_iter, max_space=max_space, max_memory=max_memory, nroots=nroots,
    )
    return np.real_if_close(es)

def main():
    eigvals = solve_eom_ccsd()
    print('EE-EOM-CCSD excitation energies:', eigvals)

if __name__ == '__main__':
    main()
