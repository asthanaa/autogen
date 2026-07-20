"""Historical excitation spaces that omit pure-active amplitudes."""

from ...production.excitation_space import (
    build_external_qp_space,
    build_ms0_qp_space,
    build_symmetry_adapted_qp_space,
)

METHOD_ID = "masked-active-qpccsd-experimental"
ENABLED_BY_DEFAULT = False


def build_masked_active_qp_space(reference):
    """Build the symmetry-adapted space with pure-active T1/T2 removed."""

    return build_symmetry_adapted_qp_space(
        reference,
        include_active_t1_t2=False,
    )


__all__ = [
    "METHOD_ID",
    "ENABLED_BY_DEFAULT",
    "build_external_qp_space",
    "build_masked_active_qp_space",
    "build_ms0_qp_space",
]
