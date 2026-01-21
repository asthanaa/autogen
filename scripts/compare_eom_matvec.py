import numpy as np

from pyscf import gto, scf, cc
from pyscf.cc import eom_rccsd

from generated_code.pyscf_integrals import compute_integrals


def _build_molecule():
    return gto.M(
        atom="O 0 0 0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
    )


def _pack_vec(eom, s1_ai, s2_abij):
    s1_ia = s1_ai.T
    s2_ijab = s2_abij.transpose(2, 3, 0, 1)
    return eom.amplitudes_to_vector(s1_ia, s2_ijab)


def _compare(label, vec_ref, vec_auto):
    diff = np.linalg.norm(vec_auto - vec_ref)
    ref_norm = np.linalg.norm(vec_ref)
    rel = diff / (ref_norm + 1e-12)
    scale = np.dot(vec_auto, vec_ref) / (np.dot(vec_auto, vec_auto) + 1e-12)
    print(f"{label}: diff={diff:.6e} rel={rel:.6e} scale={scale:.6e}")


def main():
    from generated_code.methods.eom_ccsd import residuals as eom_resid

    if getattr(eom_resid, "AUTOGEN_SPIN_SUMMED", None) is not True:
        raise RuntimeError("EOM residuals must be generated with AUTOGEN_SPIN_SUMMED=1")

    mol = _build_molecule()
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    eom = eom_rccsd.EOMEESinglet(mycc)
    imds = eom.make_imds()

    vec = eom.get_init_guess(nroots=1, koopmans=True)[0]
    r1_ia, r2_ijab = eom.vector_to_amplitudes(vec, eom.nmo, eom.nocc)

    r1_ai = r1_ia.T
    r2_abij = r2_ijab.transpose(2, 3, 0, 1)

    t1_ai = mycc.t1.T
    t2_abij = mycc.t2.transpose(2, 3, 0, 1)

    ints = compute_integrals(mol, mf=mf)
    f = ints["f"]
    g_raw = ints["g_raw"]
    g_as = ints["g"]
    nocc = ints["nocc"]
    nmo = ints["nmo"]
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    vec_ref = eom.matvec(vec, imds)

    s1_raw = eom_resid.compute_s1(f, g_raw, t1_ai, t2_abij, r1_ai, r2_abij, o, v)
    s2_raw = eom_resid.compute_s2(f, g_raw, t1_ai, t2_abij, r1_ai, r2_abij, o, v)
    vec_raw = _pack_vec(eom, s1_raw, s2_raw)
    _compare("g_raw", vec_ref, vec_raw)

    s1_as = eom_resid.compute_s1(f, g_as, t1_ai, t2_abij, r1_ai, r2_abij, o, v)
    s2_as = eom_resid.compute_s2(f, g_as, t1_ai, t2_abij, r1_ai, r2_abij, o, v)
    vec_as = _pack_vec(eom, s1_as, s2_as)
    _compare("g_as", vec_ref, vec_as)

    e0 = mycc.e_corr
    vec_shift = vec_raw - e0 * vec
    _compare("g_raw - e0*vec", vec_ref, vec_shift)


if __name__ == "__main__":
    main()
