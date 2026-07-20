"""Generalized CCSD (non-number-conserving H) amplitude/energy spec."""

import math

THEORY_NAME = "ccsd_general"
OUTPUT_DIR = "generated_code/methods/ccsd_general"
OUTPUTS = {"X1": "r1", "X2": "r2", "scalar": "energy"}
TASKS = []
BACKEND_CAPABILITIES = ["standard_bch", "spin_orbital"]
RUNTIME_OPTIONS = {"mode": "full"}

# Non-number-conserving Hamiltonian pieces.
H_OPS = ["H11", "H20", "H02", "H22", "H31", "H13", "H40", "H04"]

# Cluster operator labels (distinct names to represent repeated T's).
T1_LABELS = ["T1", "T11", "T12", "T13"]
T2_LABELS = ["T2", "T21"]

# Truncate e^T at T^4 (CCSD-style).
MAX_T1 = 4
MAX_T2 = 2
MAX_ORDER = 4

# Spin-adapted paths are not defined for pairing Hamiltonians.
SPIN_ORBITAL = True
SPIN_ADAPTED = False

# Treat h** tensors like f/g blocks for occ/virt slicing.
VIEW_TENSORS = ["h11", "h20", "h02", "h22", "h31", "h13", "h40", "h04"]

TERMS = []


def _add_terms(output_key):
    for h_op in H_OPS:
        for n1 in range(MAX_T1 + 1):
            for n2 in range(MAX_T2 + 1):
                if n1 + n2 > MAX_ORDER:
                    continue
                if n1 > len(T1_LABELS) or n2 > len(T2_LABELS):
                    continue
                t1_ops = T1_LABELS[:n1]
                t2_ops = T2_LABELS[:n2]
                ops = ([output_key] if output_key != "scalar" else []) + [h_op] + t1_ops + t2_ops
                fac = 1.0 / (math.factorial(n1) * math.factorial(n2))
                TERMS.append((output_key, fac, ops))


for _out in ("X1", "X2", "scalar"):
    _add_terms(_out)
