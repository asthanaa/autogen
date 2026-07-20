from __future__ import annotations

from typing import Any

import numpy as np

from .models import ActiveSpaceState, CASContractedReference, CASQPReference


def build_cas_contracted_reference(
    reference: CASQPReference,
    *,
    active_state: ActiveSpaceState | None = None,
) -> CASContractedReference:
    """Build the exact-active-state boundary for a contracted CC method.

    The CI vector must be represented in the same natural-orbital basis as
    the stored RDMs and full-space integrals. Reference preparation performs
    that rotation explicitly; source-basis CI data are intentionally rejected.
    """

    if active_state is None:
        try:
            coefficients = reference.metadata["active_ci_coefficients_natural"]
            electron_count = reference.metadata["active_nelec"]
        except KeyError as error:
            raise ValueError(
                "CAS reference lacks a natural-orbital active CI vector; "
                "rebuild it with an exact active solver"
            ) from error
        basis = reference.metadata.get("active_ci_orbital_basis")
        if basis != "cas_natural_orbitals":
            raise ValueError(
                "active CI basis is not aligned with the CAS-QP natural orbitals"
            )
        active_state = ActiveSpaceState(
            ci_coefficients=np.asarray(coefficients),
            electron_count=tuple(int(value) for value in electron_count),
            rdm1=reference.active_rdm1,
            rdm2=reference.active_rdm2,
            orbital_basis=basis,
            rdm3=reference.metadata.get("active_rdm3_natural"),
            rdm4=reference.metadata.get("active_rdm4_natural"),
            metadata={
                "source": "state-specific CASSCF active FCI",
                "ci_norm": reference.metadata.get("active_ci_norm"),
            },
        )
    elif active_state.orbital_basis != "cas_natural_orbitals":
        raise ValueError("contracted reference requires the CAS natural-orbital basis")

    try:
        h1 = np.asarray(reference.metadata["h1_spatial"])
        eri = np.asarray(reference.metadata["eri_spatial"])
        constant = float(reference.metadata["constant_energy"])
    except KeyError as error:
        raise ValueError("CAS reference lacks the full correlated-space Hamiltonian") from error

    return CASContractedReference(
        active_state=active_state,
        casscf_energy=reference.casscf_energy,
        h1_spatial=h1,
        eri_spatial=eri,
        constant_energy=constant,
        inactive_spatial_indices=reference.inactive_spatial_indices,
        active_spatial_indices=reference.active_spatial_indices,
        external_spatial_indices=reference.external_spatial_indices,
        target_number=reference.target_number,
        metadata={
            "method_family": "number-conserving exact-CAS contracted CC",
            "qp_reference_mode": reference.reference_mode,
            "source_rdm_metadata": dict(reference.source_rdm_metadata),
            "full_system_determinants_permitted": False,
        },
    )


def solve_cas_contracted_ccsd(
    reference: CASContractedReference,
    **options: Any,
):
    """Run the independent exact-CAS contracted-CCSD diagnostic backend.

    This entry point is lazy so importing production QPCCSD never imports an
    active FCI or molecular package. The benchmark backend owns all exact-CAS
    contractions and must not construct a determinant basis for the full
    correlated orbital space.
    """

    from .benchmarks.cas_contracted import solve_cas_contracted_ccsd as implementation

    return implementation(reference, **options)


__all__ = ["build_cas_contracted_reference", "solve_cas_contracted_ccsd"]
