"""Compatibility shim for the canonical QPCCSD method package.

New code should import :mod:`autogen.methods.qpccsd`. This module deliberately
re-exports only the reviewed production API and does not expose experimental
variants or legacy implementation submodules.
"""

from autogen.methods.qpccsd import (
    ProductionConfig,
    ProductionResult,
    build_full_active_qp_space,
    evaluate_direct_pav,
    prepare_projected_agp_reference,
    run_qpccsd_pav,
    solve_direct_qpccsd,
)
from autogen.methods.qpccsd import __all__
