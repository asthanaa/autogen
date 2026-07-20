"""Reviewed projected-AGP full-space QPCCSD/PAV interface.

Only the production workflow is exported here. Alternative references,
energy conventions, excitation masks, and projected-amplitude solvers live
under :mod:`autogen.methods.qpccsd.variants` and are disabled by default.
"""

from .production.workflow import (
    ProductionConfig,
    ProductionResult,
    build_full_active_qp_space,
    evaluate_direct_pav,
    prepare_projected_agp_reference,
    run_qpccsd_pav,
    solve_direct_qpccsd,
)

__all__ = [
    "ProductionConfig",
    "ProductionResult",
    "prepare_projected_agp_reference",
    "build_full_active_qp_space",
    "solve_direct_qpccsd",
    "evaluate_direct_pav",
    "run_qpccsd_pav",
]
