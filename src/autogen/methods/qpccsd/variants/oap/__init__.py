"""Explicit particle-number optimization-after-projection experiment."""

from __future__ import annotations

from ...production.contracts import EnergyConvention
from ...production.excitation_space import build_symmetry_adapted_qp_space
from ...production.models import CASQPReference
from ...production.projected_metric import ProjectedMetricProjector
from ...production.projection import (
    RichardsonProjectedQPCCSDEvaluator,
    SeriesProjectedQPCCSDEvaluator,
    evaluate_projected_qpccsd,
    solve_projected_qpccsd,
)

METHOD_ID = "pn-oap-qpccsd-experimental"
ENABLED_BY_DEFAULT = False


def solve_oap_qpccsd(
    hamiltonian,
    reference: CASQPReference,
    *,
    include_active_t1_t2: bool = True,
    energy_convention: str | EnergyConvention = EnergyConvention.DIRECT,
    **kwargs,
):
    """Run projected-amplitude optimization under an explicit variant import."""

    convention = EnergyConvention(energy_convention)
    space = build_symmetry_adapted_qp_space(
        reference,
        include_active_t1_t2=include_active_t1_t2,
    )
    selected_reference = (
        reference
        if convention is EnergyConvention.CASSCF_PLUS_DELTA
        else reference.bogoliubov
    )
    return solve_projected_qpccsd(
        hamiltonian,
        selected_reference,
        excitation_space=space,
        **kwargs,
    )


__all__ = [
    "METHOD_ID",
    "ENABLED_BY_DEFAULT",
    "ProjectedMetricProjector",
    "RichardsonProjectedQPCCSDEvaluator",
    "SeriesProjectedQPCCSDEvaluator",
    "evaluate_projected_qpccsd",
    "solve_oap_qpccsd",
    "solve_projected_qpccsd",
]
