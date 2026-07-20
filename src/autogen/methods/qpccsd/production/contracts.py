"""Stable method identities and serialized-result contracts.

The production route deliberately separates the physical energy convention
from the Python type of the reference.  In particular, passing a
``CASQPReference`` must never silently add the CASSCF energy.
"""

from __future__ import annotations

from enum import Enum


PRODUCTION_RESULT_SCHEMA = "projected-agp-fullspace-qpccsd-pav-v1"
PRODUCTION_REFERENCE_MODE = "projected_agp_2rdm"
PRODUCTION_REFERENCE_PROTOCOL = (
    "cas-natural-orbital-signed-projected-agp-pair-transfer-fit-v1"
)
PRODUCTION_RDM1_CONVENTION = "spin-summed spatial <p^dagger q>"
PRODUCTION_RDM2_CONVENTION = "PySCF spin-free convention"
PRODUCTION_PAIR_TRANSFER_CONVENTION = "active_rdm2[p,q,p,q]"
PRODUCTION_REFERENCE_FIT_DIAGNOSTICS = (
    "fit_success",
    "fit_message",
    "fit_cost",
    "rdm1_max_abs_error",
    "pair_rdm2_max_abs_error",
    "pair_subspace_fidelity",
    "active_average_number_error",
)
PRODUCTION_EXCITATION_SPACE = "full-symmetry-adapted-active-t1-t2"
PRODUCTION_PROJECTION = "fixed-amplitude-pn-pav-ser2-w2"


class EnergyConvention(str, Enum):
    """Supported total-energy bookkeeping conventions."""

    DIRECT = "direct"
    CASSCF_PLUS_DELTA = "casscf_plus_delta"

    @property
    def description(self) -> str:
        if self is EnergyConvention.DIRECT:
            return "E_QP(T) in the correlated molecular Hamiltonian"
        return "E_CASSCF + E_QP(T) - E_QP(0)"


def normalize_energy_convention(
    value: str | EnergyConvention,
) -> EnergyConvention:
    """Return a validated energy convention with a concise error message."""

    try:
        return value if isinstance(value, EnergyConvention) else EnergyConvention(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in EnergyConvention)
        raise ValueError(f"energy_convention must be one of: {allowed}") from error


__all__ = [
    "EnergyConvention",
    "PRODUCTION_EXCITATION_SPACE",
    "PRODUCTION_PAIR_TRANSFER_CONVENTION",
    "PRODUCTION_PROJECTION",
    "PRODUCTION_RDM1_CONVENTION",
    "PRODUCTION_RDM2_CONVENTION",
    "PRODUCTION_REFERENCE_FIT_DIAGNOSTICS",
    "PRODUCTION_REFERENCE_MODE",
    "PRODUCTION_REFERENCE_PROTOCOL",
    "PRODUCTION_RESULT_SCHEMA",
    "normalize_energy_convention",
]
