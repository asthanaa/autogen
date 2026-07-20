"""Historical CASSCF-plus-QPCCSD-delta energy bookkeeping variant."""

from ...production.contracts import EnergyConvention
from ...production.production import solve_qpccsd

METHOD_ID = "casscf-plus-delta-qpccsd-experimental"
ENABLED_BY_DEFAULT = False


def solve_casscf_plus_delta_qpccsd(hamiltonian, reference, **kwargs):
    """Run the explicitly requested composite-energy convention."""

    return solve_qpccsd(
        hamiltonian,
        reference,
        energy_convention=EnergyConvention.CASSCF_PLUS_DELTA,
        **kwargs,
    )


__all__ = [
    "METHOD_ID",
    "ENABLED_BY_DEFAULT",
    "solve_casscf_plus_delta_qpccsd",
]
