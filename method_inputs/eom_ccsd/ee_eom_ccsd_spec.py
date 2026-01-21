"""EE-EOM-CCSD (spin-summed RHF) sigma spec."""

OUTPUT_DIR = "generated_code/methods/eom_ccsd"
OUTPUTS = {"X1": "s1", "X2": "s2"}
TASKS = ["eom_solver", "eom_pyscf_test"]
PYSCF_MOL = {
    "atom": "O 0 0 0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587",
    "basis": "sto-3g",
    "unit": "Angstrom",
    "charge": 0,
    "spin": 0,
}
SPIN_ORBITAL = False
SPIN_ADAPTED = True

# Use exact BCH nested commutator expansion for Hbar, then apply R.
EOM_BCH = True
EOM_MAX_ORDER = 4
EOM_T1_LABELS = ("T1", "T11", "T12", "T13")
EOM_T2_LABELS = ("T2", "T21")
EOM_H_OPS = ("F1", "V2")

# Placeholder for spec parser; BCH terms are generated in scripts/gen_einsum.py.
TERMS = []
