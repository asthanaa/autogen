"""Minimal PySCF adapter used by the validated CCSD-family solvers."""

from __future__ import annotations

import numpy as np


def _pyscf_modules():
    try:
        from pyscf import ao2mo, gto, scf
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "PySCF is required to run the molecular CCSD and EOM-CCSD solvers."
        ) from exc
    return ao2mo, gto, scf


def build_h2o_631g():
    """Build the historical default H2O/6-31G molecule."""

    _ao2mo, gto, _scf = _pyscf_modules()
    return gto.M(
        atom="O 0 0 0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587",
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
    )


def run_scf(mol):
    """Run the RHF reference calculation used by the legacy solvers."""

    _ao2mo, _gto, scf = _pyscf_modules()
    return scf.RHF(mol).run()


def compute_integrals(mol, mf=None):
    """Build the validated spatial MO-basis integral dictionary."""

    ao2mo, _gto, _scf = _pyscf_modules()
    if mf is None:
        mf = run_scf(mol)
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    hcore_ao = mf.get_hcore()
    h1 = mo_coeff.T @ hcore_ao @ mo_coeff
    fock_ao = mf.get_fock()
    f = mo_coeff.T @ fock_ao @ mo_coeff
    eri_mo = ao2mo.kernel(mol, mo_coeff, aosym="s1")
    g_raw = ao2mo.restore(1, eri_mo, nmo)
    g_phys = g_raw.transpose(0, 2, 1, 3)
    g_as = g_phys - g_phys.transpose(0, 1, 3, 2)
    return {
        "h1": h1,
        "f": f,
        "g": g_as,
        "g_raw": g_raw,
        "g_as": g_as,
        "nocc": nocc,
        "nmo": nmo,
        "mo_coeff": mo_coeff,
    }


def compute_ccsd_amplitudes_ijab(mf):
    """Return converged CCSD amplitudes in PySCF's native layout."""

    try:
        from pyscf import cc
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("PySCF is required to compute CCSD amplitudes.") from exc
    mycc = cc.CCSD(mf).run()
    return mycc.t1, mycc.t2
