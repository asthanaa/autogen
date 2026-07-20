"""Reference-only boundary for the proposed Sokolov--Chan comparison.

Only the historical linear one-RDM quasiparticle reference helper is retained
here.  The planned CASSCF-plus-QPCCSD energy route is not implemented, and this
module deliberately exposes no solver or energy evaluator.
"""

from ...production.cas_reference import build_cas_qp_reference_from_rdms

METHOD_ID = "sokolov-linear-1rdm-qpccsd-experimental"
ENABLED_BY_DEFAULT = False
REFERENCE_HELPER_IMPLEMENTED = True
ENERGY_ROUTE_IMPLEMENTED = False


def build_sokolov_linear_reference(**kwargs):
    """Build the explicitly requested one-RDM-only reference diagnostic."""

    kwargs["reference_mode"] = "sokolov_linear_1rdm"
    return build_cas_qp_reference_from_rdms(**kwargs)


__all__ = [
    "METHOD_ID",
    "ENABLED_BY_DEFAULT",
    "REFERENCE_HELPER_IMPLEMENTED",
    "ENERGY_ROUTE_IMPLEMENTED",
    "build_sokolov_linear_reference",
]
