import os

import numpy as np
import pytest


pyscf = pytest.importorskip("pyscf")


def _skip_if_not_slow():
    if os.environ.get("RUN_SLOW", "0") != "1":
        pytest.skip("Set RUN_SLOW=1 to run PySCF verification")


def _build_molecule():
    from pyscf import gto

    return gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
    )


@pytest.mark.slow
def test_ccsd_energy_matches_pyscf():
    _skip_if_not_slow()
    from pyscf import scf, cc

    mol = _build_molecule()
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    from generated_code.pyscf_integrals import compute_integrals
    from generated_code.ccsd_amplitude.solver import compute_energy

    ints = compute_integrals(mol, mf=mf)
    f = ints["f"]
    g_raw = ints["g_raw"]
    nocc = ints["nocc"]
    nmo = ints["nmo"]
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    t1_ai = mycc.t1.T
    t2_abij = mycc.t2.transpose(2, 3, 0, 1)
    e_autogen = compute_energy(f, g_raw, t1_ai, t2_abij, o, v)

    assert abs(e_autogen - mycc.e_corr) < 1e-7


@pytest.mark.slow
def test_ccsd_residuals_match_pyscf_update(monkeypatch):
    _skip_if_not_slow()
    from pyscf import scf, cc

    monkeypatch.setenv("AUTOGEN_QUIET", "1")
    monkeypatch.setenv("AUTOGEN_SPIN_SUMMED", "1")

    mol = _build_molecule()
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    mycc.kernel()

    try:
        eris = mycc.ao2mo()
        t1_ref, t2_ref = mycc.update_amps(mycc.t1, mycc.t2, eris)
    except Exception as exc:
        pytest.skip(f"PySCF update_amps not available: {exc}")

    import importlib
    import autogen.pkg.fix_uv as fix_uv

    importlib.reload(fix_uv)

    from generated_code.pyscf_integrals import compute_integrals
    from generated_code.ccsd_amplitude import residuals as autogen_resid

    autogen_resid = importlib.reload(autogen_resid)
    assert not hasattr(autogen_resid, "build_terms"), (
        "Generate precomputed residuals with "
        "`python scripts/gen_einsum.py CCSD_AMPLITUDE --full --quiet` "
        "or `python scripts/gen_einsum.py CCSD_AMPLITUDE --intermediates --quiet`."
    )
    if getattr(autogen_resid, "AUTOGEN_SPIN_SUMMED", None) is not True:
        pytest.skip(
            "Precomputed residuals were not generated with spin-summed mode. "
            "Regenerate with `AUTOGEN_SPIN_SUMMED=1 python scripts/gen_einsum.py "
            "CCSD_AMPLITUDE --full --quiet` (or add `--intermediates`)."
        )
    if getattr(autogen_resid, "AUTOGEN_INTERMEDIATES", None) is not True:
        pytest.skip(
            "Precomputed residuals are not intermediate-optimized. "
            "Regenerate with `python scripts/gen_einsum.py CCSD_AMPLITUDE "
            "--intermediates --quiet`."
        )

    ints = compute_integrals(mol, mf=mf)
    f = ints["f"]
    g = ints["g_raw"] if autogen_resid.AUTOGEN_SPIN_SUMMED else ints["g"]
    nocc = ints["nocc"]
    nmo = ints["nmo"]
    o = list(range(nocc))
    v = list(range(nocc, nmo))

    t1_ai = mycc.t1.T
    t2_abij = mycc.t2.transpose(2, 3, 0, 1)

    r1 = autogen_resid.compute_r1(f, g, t1_ai, t2_abij, o, v)
    r2 = autogen_resid.compute_r2(f, g, t1_ai, t2_abij, o, v)

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

    t1_new_ai = t1_ai + r1 / denom_ai
    t2_new_abij = t2_abij + r2 / denom_abij
    t1_new_ia = t1_new_ai.T
    t2_new_ijab = t2_new_abij.transpose(2, 3, 0, 1)

    assert np.max(np.abs(t1_new_ia - t1_ref)) < 1e-6
    assert np.max(np.abs(t2_new_ijab - t2_ref)) < 1e-6
