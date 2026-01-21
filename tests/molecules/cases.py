from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeCase:
    name: str
    atom: str
    basis: str = "sto-3g"
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0


CASES = [
    MoleculeCase(
        name="h2_sto3g",
        atom="H 0 0 0; H 0 0 0.74",
    ),
    MoleculeCase(
        name="lih_sto3g",
        atom="Li 0 0 0; H 0 0 1.6",
    ),
    MoleculeCase(
        name="h2o_sto3g",
        atom="O 0 0 0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587",
    ),
    MoleculeCase(
        name="n2_sto3g_1p5a",
        atom="N 0 0 0; N 0 0 1.5",
    ),
    MoleculeCase(
        name="n2_sto3g_3p0a",
        atom="N 0 0 0; N 0 0 3.0",
    ),
]


def build_molecule(case: MoleculeCase):
    from pyscf import gto

    return gto.M(
        atom=case.atom,
        basis=case.basis,
        unit=case.unit,
        charge=case.charge,
        spin=case.spin,
    )
