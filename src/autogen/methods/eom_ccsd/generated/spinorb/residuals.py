import numpy as np

from pyscf.cc import addons, eom_rccsd, eom_uccsd

from . import residuals_spinorb as spinorb

AUTOGEN_SPIN_SUMMED = True
AUTOGEN_SPIN_SUMMED_MODE = 'spinorb'
AUTOGEN_INTERMEDIATES = True
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

def compute_outputs(f, g, t1, t2, r1, r2, o, v):
    nocc = len(o)
    nmo = f.shape[0]
    orbspin = _orbspin(nocc, nmo)
    f_so = _build_spin_orbital_fock(f, nocc)
    g_so = _build_spin_orbital_g(g, nocc)
    t1_so = addons.spatial2spin(t1.T, orbspin).T
    t2_so = addons.spatial2spin(
        t2.transpose(2, 3, 0, 1), orbspin
    ).transpose(2, 3, 0, 1)
    r1_so = eom_rccsd.spatial2spin_singlet(r1.T, orbspin).T
    r2_so = eom_rccsd.spatial2spin_singlet(
        r2.transpose(2, 3, 0, 1), orbspin
    ).transpose(2, 3, 0, 1)
    o_so = list(range(2 * nocc))
    v_so = list(range(2 * nocc, 2 * nmo))
    outs = spinorb.compute_outputs(f_so, g_so, t1_so, t2_so, r1_so, r2_so, o_so, v_so)
    s1_so = outs['s1'].T
    s2_so = outs['s2'].transpose(2, 3, 0, 1)
    s1a, s1b = eom_uccsd.spin2spatial_eomee(s1_so, orbspin)
    s2aa, s2ab, s2bb = eom_uccsd.spin2spatial_eomee(s2_so, orbspin)
    sqrt2 = 2 ** 0.5
    s1 = (0.5 * (s1a + s1b) * sqrt2).T
    s2 = s2ab.transpose(2, 3, 0, 1) * sqrt2
    return {'s1': s1, 's2': s2}

def compute_s1(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s1']

def compute_s2(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s2']
