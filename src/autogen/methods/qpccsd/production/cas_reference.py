from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from .contracts import (
    PRODUCTION_PAIR_TRANSFER_CONVENTION,
    PRODUCTION_RDM1_CONVENTION,
    PRODUCTION_RDM2_CONVENTION,
    PRODUCTION_REFERENCE_PROTOCOL,
)
from .hamiltonian import build_qp_hamiltonian
from .models import BogoliubovReference, CASQPReference, QPHamiltonian


def _fix_column_phases(coefficients: np.ndarray) -> np.ndarray:
    result = np.asarray(coefficients, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _active_natural_rotation(
    rdm1: np.ndarray,
    labels: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    rdm1 = 0.5 * (np.asarray(rdm1, dtype=float) + np.asarray(rdm1, dtype=float).T)
    nactive = rdm1.shape[0]
    if rdm1.shape != (nactive, nactive):
        raise ValueError("active 1-RDM must be square")
    labels = tuple("active" for _ in range(nactive)) if labels is None else tuple(labels)
    if len(labels) != nactive:
        raise ValueError("active labels and active 1-RDM dimensions differ")

    rotation = np.zeros((nactive, nactive), dtype=float)
    occupations = np.zeros(nactive, dtype=float)
    ordered_labels = tuple(dict.fromkeys(labels))
    for label in ordered_labels:
        indices = np.asarray([index for index, value in enumerate(labels) if value == label])
        block = rdm1[np.ix_(indices, indices)]
        values, vectors = np.linalg.eigh(block)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = _fix_column_phases(vectors[:, order])
        rotation[np.ix_(indices, indices)] = vectors
        occupations[indices] = values
    return rotation, occupations


def _align_natural_orbitals(
    rotation: np.ndarray,
    occupations: np.ndarray,
    mo_coeff: np.ndarray,
    active_indices: tuple[int, ...],
    active_labels: tuple[str, ...],
    previous_reference: CASQPReference | None,
    cross_overlap: np.ndarray | None,
    degeneracy_tolerance: float,
) -> np.ndarray:
    if previous_reference is None or cross_overlap is None:
        return rotation
    if previous_reference.active_spatial_indices != active_indices:
        raise ValueError("previous and current active orbital partitions differ")
    cross_overlap = np.asarray(cross_overlap, dtype=float)
    if cross_overlap.shape != (
        previous_reference.mo_coeff.shape[0],
        mo_coeff.shape[0],
    ):
        raise ValueError("cross_overlap has an incompatible AO shape")

    result = rotation.copy()
    current_active = mo_coeff[:, active_indices] @ result
    previous_active = previous_reference.mo_coeff[:, active_indices]
    visited: set[int] = set()
    for index in range(len(active_indices)):
        if index in visited:
            continue
        group = tuple(
            candidate
            for candidate in range(len(active_indices))
            if active_labels[candidate] == active_labels[index]
            and abs(occupations[candidate] - occupations[index]) <= degeneracy_tolerance
        )
        visited.update(group)
        overlap = (
            previous_active[:, group].T
            @ cross_overlap
            @ current_active[:, group]
        )
        if len(group) == 1:
            if overlap[0, 0] < 0.0:
                result[:, group[0]] *= -1.0
            continue
        left, _singular_values, right_t = np.linalg.svd(overlap, full_matrices=False)
        alignment = right_t.T @ left.T
        result[:, group] = result[:, group] @ alignment
    return result


def rotate_spatial_integrals(
    h1_spatial: np.ndarray,
    eri_spatial: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a chemist-ordered spatial Hamiltonian."""

    h1_spatial = np.asarray(h1_spatial)
    eri_spatial = np.asarray(eri_spatial)
    rotation = np.asarray(rotation)
    nspatial = h1_spatial.shape[0]
    if h1_spatial.shape != (nspatial, nspatial):
        raise ValueError("h1_spatial must be square")
    if eri_spatial.shape != (nspatial,) * 4 or rotation.shape != (nspatial, nspatial):
        raise ValueError("spatial integral and rotation dimensions differ")
    h1_rotated = rotation.T @ h1_spatial @ rotation
    eri_rotated = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        eri_spatial,
        rotation,
        rotation,
        rotation,
        rotation,
        optimize=True,
    )
    return h1_rotated, eri_rotated


def spatial_to_spin_integrals(
    h1_spatial: np.ndarray,
    eri_spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand spatial integrals into interleaved antisymmetrized spin orbitals."""

    h1_spatial = np.asarray(h1_spatial)
    eri_spatial = np.asarray(eri_spatial)
    nspatial = h1_spatial.shape[0]
    if h1_spatial.shape != (nspatial, nspatial) or eri_spatial.shape != (nspatial,) * 4:
        raise ValueError("spatial integral dimensions are inconsistent")
    nspin = 2 * nspatial
    h1 = np.zeros((nspin, nspin), dtype=h1_spatial.dtype)
    for spin in (0, 1):
        h1[spin::2, spin::2] = h1_spatial

    coulomb = np.zeros((nspin,) * 4, dtype=eri_spatial.dtype)
    # g[p,q,r,s] multiplies c_p^dag c_q^dag c_s c_r / 4.
    physicist = eri_spatial.transpose(0, 2, 1, 3)
    for spin_p in (0, 1):
        for spin_q in (0, 1):
            coulomb[
                spin_p::2,
                spin_q::2,
                spin_p::2,
                spin_q::2,
            ] = physicist
    return h1, coulomb - coulomb.swapaxes(2, 3)


def _gaussian_spin_traced_rdm2(reference: BogoliubovReference) -> np.ndarray:
    """Return the spin-traced spatial 2-RDM of the Gaussian QP vacuum."""

    nspatial = reference.nspin // 2
    rho = reference.V @ reference.V.conj().T
    kappa = reference.V @ reference.U.T
    gamma2 = (
        np.einsum("ca,db->abcd", rho, rho, optimize=True)
        - np.einsum("da,cb->abcd", rho, rho, optimize=True)
        + np.einsum("ab,cd->abcd", kappa.conj(), kappa, optimize=True)
    )
    result = np.zeros((nspatial,) * 4, dtype=np.complex128)
    for spin_p in (0, 1):
        for spin_r in (0, 1):
            p = np.arange(nspatial) * 2 + spin_p
            q = np.arange(nspatial) * 2 + spin_p
            r = np.arange(nspatial) * 2 + spin_r
            s = np.arange(nspatial) * 2 + spin_r
            result += gamma2[np.ix_(p, r, q, s)].transpose(0, 2, 1, 3)
    return np.real_if_close(result)


def _projected_agp_pair_rdms(
    geminals: np.ndarray,
    pair_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial 1-RDM diagonals and the spin-traced pair-transfer block."""

    geminals = np.asarray(geminals, dtype=float)
    norb = geminals.size
    if pair_count < 0 or pair_count > norb:
        raise ValueError("AGP pair count lies outside the active orbital space")
    subsets = tuple(combinations(range(norb), pair_count))
    coefficients = {
        subset: float(np.prod(geminals[list(subset)])) if subset else 1.0
        for subset in subsets
    }
    norm = float(sum(value * value for value in coefficients.values()))
    if norm <= 1.0e-30:
        raise ValueError("projected AGP has a numerically zero norm")

    pair_rdm = np.zeros((norb, norb), dtype=float)
    for subset, coefficient in coefficients.items():
        occupied = set(subset)
        probability = coefficient * coefficient / norm
        for q in subset:
            pair_rdm[q, q] += probability
            for p in range(norb):
                if p in occupied:
                    continue
                destination = tuple(sorted((occupied - {q}) | {p}))
                pair_rdm[p, q] += coefficients[destination] * coefficient / norm
    return 2.0 * np.diag(pair_rdm), 2.0 * pair_rdm


def _fit_projected_agp_geminals(
    active_rdm1: np.ndarray,
    active_rdm2: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    from scipy.optimize import least_squares

    occupations = np.clip(np.real(np.diag(active_rdm1)), 0.0, 2.0)
    active_electrons = int(round(float(np.trace(active_rdm1))))
    if active_electrons % 2:
        raise ValueError("the paired AGP reference requires an even active electron count")
    pair_count = active_electrons // 2
    norb = occupations.size
    target_pair = np.asarray(
        [[active_rdm2[p, q, p, q] for q in range(norb)] for p in range(norb)],
        dtype=float,
    )
    target_pair = 0.5 * (target_pair + target_pair.T)

    if pair_count in {0, norb}:
        geminals = np.ones(norb)
        fitted_rdm1, fitted_pair = _projected_agp_pair_rdms(geminals, pair_count)
        return geminals, {
            "fit_success": True,
            "fit_message": "unique empty/full paired determinant",
            "fit_cost": 0.0,
            "rdm1_max_abs_error": float(np.max(np.abs(fitted_rdm1 - occupations), initial=0.0)),
            "pair_rdm2_max_abs_error": float(
                np.max(np.abs(fitted_pair - target_pair), initial=0.0)
            ),
            "pair_subspace_fidelity": 1.0,
        }

    values, vectors = np.linalg.eigh(target_pair)
    phase_vector = vectors[:, int(np.argmax(values))]
    pivot = int(np.argmax(np.abs(phase_vector)))
    if phase_vector[pivot] < 0.0:
        phase_vector = -phase_vector
    signs = np.where(phase_vector < 0.0, -1.0, 1.0)
    magnitudes = np.sqrt(
        np.clip(occupations, 1.0e-12, 2.0 - 1.0e-12)
        / np.clip(2.0 - occupations, 1.0e-12, None)
    )
    pivot = int(np.argmax(magnitudes))
    free = tuple(index for index in range(norb) if index != pivot)
    initial = np.log(np.clip(magnitudes[list(free)] / magnitudes[pivot], 1.0e-12, 1.0e12))

    def unpack(vector: np.ndarray) -> np.ndarray:
        logs = np.zeros(norb)
        logs[list(free)] = vector
        return signs * np.exp(np.clip(logs, -30.0, 30.0))

    def objective(vector: np.ndarray) -> np.ndarray:
        fitted_rdm1, fitted_pair = _projected_agp_pair_rdms(unpack(vector), pair_count)
        return np.concatenate(
            (
                0.5 * (fitted_rdm1 - occupations),
                0.5 * (fitted_pair - target_pair).reshape(-1),
            )
        )

    solution = least_squares(
        objective,
        initial,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=4000,
    )
    geminals = unpack(solution.x)
    fitted_rdm1, fitted_pair = _projected_agp_pair_rdms(geminals, pair_count)
    denominator = float(np.linalg.norm(fitted_pair) * np.linalg.norm(target_pair))
    fidelity = (
        float(np.vdot(fitted_pair, target_pair).real / denominator)
        if denominator > 1.0e-30
        else 1.0
    )
    return geminals, {
        "fit_success": bool(solution.success),
        "fit_message": str(solution.message),
        "fit_cost": float(solution.cost),
        "rdm1_max_abs_error": float(np.max(np.abs(fitted_rdm1 - occupations), initial=0.0)),
        "pair_rdm2_max_abs_error": float(np.max(np.abs(fitted_pair - target_pair), initial=0.0)),
        "pair_subspace_fidelity": fidelity,
    }


def _build_paired_reference_from_signed_geminals(
    spatial_occupations: np.ndarray,
    active_indices: tuple[int, ...],
    signed_geminals: np.ndarray,
    target_number: int,
    quasiparticle_energies: np.ndarray,
    chemical_potential: float,
    metadata: dict[str, Any],
) -> tuple[BogoliubovReference, np.ndarray]:
    from scipy.optimize import brentq

    active_target = int(round(float(np.sum(spatial_occupations[list(active_indices)]))))
    relative = np.asarray(signed_geminals, dtype=float)
    if active_target == 0:
        scale = 0.0
        scaled = np.zeros_like(relative)
    elif active_target == 2 * len(active_indices):
        scale = 1.0e15
        scaled = np.copysign(np.full_like(relative, 1.0e15), relative)
    else:
        squares = relative * relative

        def number_error(log_scale: float) -> float:
            scaled_squares = np.exp(2.0 * log_scale) * squares
            return float(np.sum(2.0 * scaled_squares / (1.0 + scaled_squares)) - active_target)

        scale = np.exp(brentq(number_error, -40.0, 40.0))
        scaled = scale * relative

    nspatial = spatial_occupations.size
    u_spatial = np.ones(nspatial)
    v_spatial = np.zeros(nspatial)
    inactive = tuple(
        index
        for index, occupation in enumerate(spatial_occupations)
        if occupation > 2.0 - 1.0e-10 and index not in active_indices
    )
    u_spatial[list(inactive)] = 0.0
    v_spatial[list(inactive)] = 1.0
    for local, spatial in enumerate(active_indices):
        u_spatial[spatial] = 1.0 / np.sqrt(1.0 + scaled[local] ** 2)
        v_spatial[spatial] = scaled[local] * u_spatial[spatial]

    u = np.repeat(u_spatial, 2)
    v = np.repeat(v_spatial, 2)
    nspin = 2 * nspatial
    partner = np.arange(nspin, dtype=np.int64) ^ 1
    spin_signs = np.where(np.arange(nspin) % 2 == 0, 1.0, -1.0)
    U = np.diag(u.astype(np.complex128))
    V = np.zeros((nspin, nspin), dtype=np.complex128)
    V[np.arange(nspin), partner] = spin_signs * v
    average_number = float(2.0 * np.sum(v_spatial * v_spatial))
    reference = BogoliubovReference(
        U=U,
        V=V,
        u=u,
        v=v,
        partner=partner,
        signs=spin_signs,
        target_number=int(target_number),
        quasiparticle_energies=np.repeat(quasiparticle_energies, 2),
        chemical_potential=float(chemical_potential),
        pairing_collapsed=False,
        metadata={
            "reference_kind": "projection-consistent signed AGP from CAS pair 2-RDM",
            "average_particle_number": average_number,
            "particle_number_error": abs(average_number - target_number),
            "denominator_source": "h11",
            **metadata,
            "global_number_scale": float(scale),
        },
    )
    if max(reference.canonical_errors()) > 1.0e-12:
        raise RuntimeError("constructed signed AGP Bogoliubov transformation is not canonical")
    return reference, scaled


def build_paired_reference_from_occupations(
    spatial_occupations: np.ndarray,
    target_number: int,
    *,
    quasiparticle_energies: np.ndarray | None = None,
    chemical_potential: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> BogoliubovReference:
    spatial_occupations = np.clip(
        np.asarray(spatial_occupations, dtype=float), 0.0, 2.0
    )
    nspatial = spatial_occupations.size
    nspin = 2 * nspatial
    occupation_spin = np.repeat(0.5 * spatial_occupations, 2)
    u = np.sqrt(np.clip(1.0 - occupation_spin, 0.0, 1.0))
    v = np.sqrt(np.clip(occupation_spin, 0.0, 1.0))
    partner = np.arange(nspin, dtype=np.int64) ^ 1
    signs = np.where(np.arange(nspin) % 2 == 0, 1.0, -1.0)
    U = np.diag(u.astype(np.complex128))
    V = np.zeros((nspin, nspin), dtype=np.complex128)
    V[np.arange(nspin), partner] = signs * v
    if quasiparticle_energies is None:
        energies = np.ones(nspin)
    else:
        energies = np.asarray(quasiparticle_energies, dtype=float)
        if energies.shape == (nspatial,):
            energies = np.repeat(energies, 2)
        if energies.shape != (nspin,):
            raise ValueError("quasiparticle energies have an incompatible shape")
        energies = np.maximum(np.abs(energies), 1.0e-8)
    number_error = abs(float(np.sum(spatial_occupations)) - int(target_number))
    if number_error > 1.0e-7:
        raise ValueError(
            "spatial occupations do not reproduce the correlated particle target: "
            f"error={number_error:.3e}"
        )
    fractional = spatial_occupations[
        (spatial_occupations > 1.0e-8) & (spatial_occupations < 2.0 - 1.0e-8)
    ]
    result = BogoliubovReference(
        U=U,
        V=V,
        u=u,
        v=v,
        partner=partner,
        signs=signs,
        target_number=int(target_number),
        quasiparticle_energies=energies,
        chemical_potential=float(chemical_potential),
        pairing_collapsed=bool(fractional.size == 0),
        metadata={
            "reference_kind": "Sokolov-Chan linear CAS-1RDM Bogoliubov",
            "average_particle_number": float(np.sum(spatial_occupations)),
            "particle_number_error": number_error,
            "denominator_source": "h11",
            **({} if metadata is None else metadata),
        },
    )
    if max(result.canonical_errors()) > 1.0e-12:
        raise RuntimeError("constructed CAS Bogoliubov transformation is not canonical")
    return result


def build_cas_qp_reference_from_rdms(
    *,
    casscf_energy: float,
    mo_coeff: np.ndarray,
    active_rdm1: np.ndarray,
    active_rdm2: np.ndarray,
    active_spatial_indices: Sequence[int],
    inactive_spatial_indices: Sequence[int],
    external_spatial_indices: Sequence[int],
    correlated_target_number: int,
    physical_target_number: int | None = None,
    frozen_spatial_indices: Sequence[int] = (),
    orbital_labels: Sequence[str] | None = None,
    spatial_fock: np.ndarray | None = None,
    previous_reference: CASQPReference | None = None,
    cross_overlap: np.ndarray | None = None,
    degeneracy_tolerance: float = 1.0e-8,
    reference_mode: str = "projected_agp_2rdm",
    metadata: dict[str, Any] | None = None,
) -> CASQPReference:
    """Construct the projected-AGP CAS-RDM quasiparticle reference.

    ``sokolov_linear_1rdm`` remains available only as an explicit experimental
    value.  The default fits signed active geminals to natural occupations and
    the pair-transfer block of the active two-body RDM.
    """

    mo_coeff = np.asarray(mo_coeff, dtype=float)
    nspatial = mo_coeff.shape[1]
    active = tuple(int(index) for index in active_spatial_indices)
    inactive = tuple(int(index) for index in inactive_spatial_indices)
    external = tuple(int(index) for index in external_spatial_indices)
    labels = (
        tuple("" for _ in range(nspatial))
        if orbital_labels is None
        else tuple(str(value) for value in orbital_labels)
    )
    if len(labels) != nspatial:
        raise ValueError("orbital_labels must have one entry per correlated orbital")
    active_labels = tuple(labels[index] for index in active)
    natural_rotation, active_occupations = _active_natural_rotation(
        active_rdm1,
        active_labels if any(active_labels) else None,
    )
    natural_rotation = _align_natural_orbitals(
        natural_rotation,
        active_occupations,
        mo_coeff,
        active,
        active_labels,
        previous_reference,
        cross_overlap,
        degeneracy_tolerance,
    )
    full_rotation = np.eye(nspatial)
    full_rotation[np.ix_(active, active)] = natural_rotation
    rotated_coefficients = mo_coeff @ full_rotation
    if previous_reference is None or cross_overlap is None:
        rotated_coefficients = _fix_column_phases(rotated_coefficients)

    rdm1_no = natural_rotation.T @ np.asarray(active_rdm1) @ natural_rotation
    rdm2_no = np.einsum(
        "abcd,ap,bq,cr,ds->pqrs",
        np.asarray(active_rdm2),
        natural_rotation,
        natural_rotation,
        natural_rotation,
        natural_rotation,
        optimize=True,
    )
    active_occupations = np.clip(np.real(np.diag(rdm1_no)), 0.0, 2.0)
    spatial_occupations = np.zeros(nspatial)
    spatial_occupations[list(inactive)] = 2.0
    spatial_occupations[list(active)] = active_occupations

    if spatial_fock is None:
        spatial_qp_energies = np.ones(nspatial)
        chemical_potential = 0.0
    else:
        rotated_fock = full_rotation.T @ np.asarray(spatial_fock) @ full_rotation
        diagonal = np.real(np.diag(rotated_fock))
        occupied_diagonal = diagonal[list(inactive)] if inactive else diagonal
        external_diagonal = diagonal[list(external)] if external else diagonal
        chemical_potential = 0.5 * (
            float(np.max(occupied_diagonal)) + float(np.min(external_diagonal))
        )
        spatial_qp_energies = np.maximum(np.abs(diagonal - chemical_potential), 1.0e-4)

    reconstruction_metrics: dict[str, float | bool | str] = {}
    signed_geminals = None
    if reference_mode == "sokolov_linear_1rdm":
        bogoliubov = build_paired_reference_from_occupations(
            spatial_occupations,
            correlated_target_number,
            quasiparticle_energies=spatial_qp_energies,
            chemical_potential=chemical_potential,
            metadata={"active_natural_occupations": active_occupations.tolist()},
        )
    elif reference_mode == "projected_agp_2rdm":
        relative_geminals, reconstruction_metrics = _fit_projected_agp_geminals(
            rdm1_no,
            rdm2_no,
        )
        bogoliubov, signed_geminals = _build_paired_reference_from_signed_geminals(
            spatial_occupations,
            active,
            relative_geminals,
            correlated_target_number,
            spatial_qp_energies,
            chemical_potential,
            {
                "active_natural_occupations": active_occupations.tolist(),
                "relative_signed_geminals": relative_geminals.tolist(),
            },
        )
        reconstruction_metrics["active_average_number_error"] = abs(
            float(2.0 * np.sum(bogoliubov.v[2 * np.asarray(active)] ** 2))
            - int(round(float(np.trace(rdm1_no))))
        )
    else:
        raise ValueError(f"unsupported CAS-QP reference mode {reference_mode!r}")
    active_reference = build_paired_reference_from_occupations(
        active_occupations,
        int(round(float(np.trace(rdm1_no)))),
    )
    gaussian_rdm2 = _gaussian_spin_traced_rdm2(active_reference)
    cumulant_norm = float(np.linalg.norm(rdm2_no - gaussian_rdm2))
    return CASQPReference(
        bogoliubov=bogoliubov,
        casscf_energy=float(casscf_energy),
        mo_coeff=rotated_coefficients,
        spatial_occupations=spatial_occupations,
        active_rdm1=np.real_if_close(rdm1_no),
        active_rdm2=np.real_if_close(rdm2_no),
        active_spatial_indices=active,
        inactive_spatial_indices=inactive,
        external_spatial_indices=external,
        frozen_spatial_indices=tuple(int(index) for index in frozen_spatial_indices),
        physical_target_number=physical_target_number,
        rdm_cumulant_norm=cumulant_norm,
        orbital_labels=labels,
        reference_mode=reference_mode,
        signed_geminals=signed_geminals,
        reconstruction_metrics=reconstruction_metrics,
        source_rdm_metadata={
            "rdm1": PRODUCTION_RDM1_CONVENTION,
            "rdm2": PRODUCTION_RDM2_CONVENTION,
            "pair_transfer_block": PRODUCTION_PAIR_TRANSFER_CONVENTION,
            "reference_protocol": (
                PRODUCTION_REFERENCE_PROTOCOL
                if reference_mode == "projected_agp_2rdm"
                else "experimental-sokolov-linear-1rdm-v1"
            ),
            "complete_active_rdm2_reconstructed": False,
        },
        metadata={
            "spatial_rotation": full_rotation,
            "active_rotation": natural_rotation,
            "gaussian_active_rdm2": gaussian_rdm2,
            **({} if metadata is None else metadata),
        },
    )


def build_fullspace_qp_hamiltonian(
    h1: np.ndarray | CASQPReference,
    g2: np.ndarray | None = None,
    reference: CASQPReference | BogoliubovReference | None = None,
    *,
    constant: float | complex = 0.0,
    spatial_integrals: bool | None = None,
) -> QPHamiltonian:
    """Transform the full correlated-orbital Hamiltonian into QP blocks."""

    if isinstance(h1, CASQPReference):
        if g2 is not None or reference is not None:
            raise ValueError("reference-only construction does not accept separate integrals")
        reference = h1
        try:
            h1 = np.asarray(reference.metadata["h1_spatial"])
            g2 = np.asarray(reference.metadata["eri_spatial"])
            constant = complex(reference.metadata["constant_energy"])
        except KeyError as error:
            raise ValueError("CASQPReference does not contain prepared spatial integrals") from error
        spatial_integrals = True
    if reference is None or g2 is None:
        raise ValueError("h1, g2, and reference are required")
    bogoliubov = reference.bogoliubov if isinstance(reference, CASQPReference) else reference
    h1 = np.asarray(h1)
    g2 = np.asarray(g2)
    if spatial_integrals is None:
        spatial_integrals = h1.shape == (bogoliubov.nspin // 2,) * 2
    if spatial_integrals:
        h1, g2 = spatial_to_spin_integrals(h1, g2)
    return build_qp_hamiltonian(h1, g2, bogoliubov, constant=constant)


def build_casscf_qp_reference(*args: Any, **kwargs: Any):
    """Lazy PySCF preparation entry point; production tensor imports stay clean."""

    from .benchmarks.pyscf_cas import build_casscf_qp_reference as implementation

    return implementation(*args, **kwargs)


def build_projected_agp_reference(*args: Any, **kwargs: Any):
    """Build the projection-consistent signed-AGP CAS reference."""

    kwargs["reference_mode"] = "projected_agp_2rdm"
    return build_cas_qp_reference_from_rdms(*args, **kwargs)


__all__ = [
    "build_cas_qp_reference_from_rdms",
    "build_casscf_qp_reference",
    "build_fullspace_qp_hamiltonian",
    "build_paired_reference_from_occupations",
    "build_projected_agp_reference",
    "rotate_spatial_integrals",
    "spatial_to_spin_integrals",
]
