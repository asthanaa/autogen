import os

import numpy as np
import pytest

from tests.molecules.cases import CASES, build_molecule


pyscf = pytest.importorskip("pyscf")


def _skip_if_not_slow():
    if os.environ.get("RUN_SLOW", "0") != "1":
        pytest.skip("Set RUN_SLOW=1 to run PySCF verification")


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_eom_ccsd_singlet_matches_pyscf(case):
    _skip_if_not_slow()
    from pyscf import scf, cc
    from pyscf.cc import eom_rccsd

    try:
        from generated_code.methods.eom_ccsd import residuals as eom_resid
        from generated_code.methods.eom_ccsd.eom_solver import solve_eom_ccsd
    except Exception as exc:
        pytest.skip(f"EOM-CCSD code not generated: {exc}")

    if getattr(eom_resid, "AUTOGEN_SPIN_SUMMED", None) is not True:
        pytest.skip(
            "EOM residuals were not generated with spin-summed mode. "
            "Regenerate with `AUTOGEN_SPIN_SUMMED=1 python scripts/gen_einsum.py "
            "--spec method_inputs/eom_ccsd/ee_eom_ccsd_spec.py --intermediates --quiet`."
        )

    mol = build_molecule(case)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    e_pyscf, _vecs = eom_rccsd.EOMEESinglet(mycc).kernel(nroots=3)
    e_autogen = solve_eom_ccsd(
        mol=mol,
        mf=mf,
        t1=mycc.t1.T,
        t2=mycc.t2.transpose(2, 3, 0, 1),
        nroots=3,
    )

    e_pyscf = np.sort(np.array(e_pyscf))
    e_autogen = np.sort(np.array(e_autogen))
    nroots = min(len(e_pyscf), len(e_autogen))
    diff = np.max(np.abs(e_pyscf[:nroots] - e_autogen[:nroots]))
    assert diff < 1e-6
