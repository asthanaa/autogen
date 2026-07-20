"""Derivation-only Bogoliubov QPCCSD amplitude/energy specification.

The reviewed runtime is the emitted and regression-tested implementation in
``autogen.methods.qpccsd.generated``. This historical generator specification
is retained for algebra provenance and is never imported by production code.
"""

import math

THEORY_NAME = "qp_ccsd"
OUTPUT_DIR = "src/autogen/methods/qpccsd/generated"
OUTPUTS = {"X1": "r1", "X2": "r2", "scalar": "energy"}
# The historical solver emitter does not reproduce the validated OAP and trace
# paths. Keep solver ownership explicit until the template passes those tests.
TASKS = []
BACKEND_CAPABILITIES = ["standard_bch", "qp", "projected_qp", "spin_orbital"]
RUNTIME_OPTIONS = {"mode": "intermediates", "projection_default": True}
PROJECTED_QP = True
ORACLE_FACTORY = "autogen.reference.projected_qp.build_projected_exact_reference"

# Bogoliubov Hamiltonian blocks (including anomalous terms).
H_OPS = ["H11", "H20", "H02", "H22", "H31", "H13", "H40", "H04"]

# Cluster operator labels (distinct names to represent repeated T's).
# QP mode uses "T1qp"/"T2qp" names while keeping the same BCH structure.
T1_LABELS = ["T1qp", "T1qp1", "T1qp2", "T1qp3"]
T2_LABELS = ["T2qp", "T2qp1"]

# Truncate e^T at T^4 (CCSD-style).
MAX_T1 = 4
MAX_T2 = 2
MAX_ORDER = 4

# Use nested-commutator BCH expansion instead of product list in the generator.
BCH = True
BCH_MAX_ORDER = MAX_ORDER
BCH_T1_LABELS = tuple(T1_LABELS)
BCH_T2_LABELS = tuple(T2_LABELS)
BCH_H_OPS = tuple(H_OPS)

# Bogoliubov-transformed expressions are spin-orbital.
SPIN_ORBITAL = True
SPIN_ADAPTED = False
BOGOLIUBOV_QP = True

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
