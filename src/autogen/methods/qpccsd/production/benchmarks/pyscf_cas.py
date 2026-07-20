from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import comb
from pathlib import Path
import json
import time
from typing import Any
import warnings

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf, symm

from ..cas_reference import (
    build_cas_qp_reference_from_rdms,
    spatial_to_spin_integrals,
)
from ..models import BogoliubovReference, CASQPReference


_PI_PI_STAR_TARGETS = (
    ("E1ux", True),
    ("E1uy", True),
    ("E1gx", False),
    ("E1gy", False),
)


@dataclass(frozen=True)
class FullSpaceBenchmark:
    rhf_energy: float
    casscf_energy: float
    ccsd_energy: float | None
    ccsd_converged: bool | None
    fci_energy: float | None
    ccsd_t_correction: float | None = None
    ccsd_t_energy: float | None = None
    fci_determinant_dimension: int | None = None
    method_status: dict[str, str] | None = None
    failure_reasons: dict[str, str] | None = None
    timings_s: dict[str, float] | None = None


@dataclass(frozen=True)
class CASSCFContinuation:
    """Portable data needed to continue CASSCF and natural-orbital tracking."""

    molecule: gto.Mole
    full_mo_coeff: np.ndarray
    active_ci: np.ndarray
    reference: CASQPReference
    source_path: str | None = None


def _fix_orbital_phases(coefficients: np.ndarray) -> np.ndarray:
    result = np.asarray(coefficients, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _n2_active_indices(
    labels: Sequence[str],
    occupations: np.ndarray,
    orbital_energies: np.ndarray,
    cas_norb: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    for label, occupied in _PI_PI_STAR_TARGETS:
        candidates = [
            index
            for index, (candidate, occupation) in enumerate(zip(labels, occupations))
            if candidate == label and bool(occupation > 0.0) is occupied
        ]
        if not candidates:
            raise ValueError(f"failed to find N2 active orbital {label}")
        selected.append(min(candidates, key=lambda index: orbital_energies[index]))
    if cas_norb == 4:
        return tuple(sorted(selected))
    if cas_norb not in (6, 8, 10):
        raise ValueError(
            "deterministic N2 selection supports CAS(4,4), CAS(6,6), "
            "CAS(8,8), or CAS(10,10)"
        )
    occupied_sigma = [
        index
        for index, (label, occupation) in enumerate(zip(labels, occupations))
        if label == "A1g" and occupation > 0.0
    ]
    virtual_sigma = [
        index
        for index, (label, occupation) in enumerate(zip(labels, occupations))
        if label == "A1u" and occupation == 0.0
    ]
    if not occupied_sigma or not virtual_sigma:
        raise ValueError("failed to find the N2 valence sigma/sigma* pair")
    sigma = max(occupied_sigma, key=lambda index: orbital_energies[index])
    sigma_star = min(virtual_sigma, key=lambda index: orbital_energies[index])
    selected.extend((sigma, sigma_star))
    if cas_norb >= 8:
        occupied_a1u = [
            index
            for index, (label, occupation) in enumerate(zip(labels, occupations))
            if label == "A1u" and occupation > 0.0 and index not in selected
        ]
        virtual_a1g = [
            index
            for index, (label, occupation) in enumerate(zip(labels, occupations))
            if label == "A1g" and occupation == 0.0 and index not in selected
        ]
        if not occupied_a1u or not virtual_a1g:
            raise ValueError("failed to extend N2 to the nested CAS(8,8) space")
        selected.extend(
            (
                max(occupied_a1u, key=lambda index: orbital_energies[index]),
                min(virtual_a1g, key=lambda index: orbital_energies[index]),
            )
        )
    if cas_norb >= 10:
        occupied_a1g = [
            index
            for index, (label, occupation) in enumerate(zip(labels, occupations))
            if label == "A1g" and occupation > 0.0 and index not in selected
        ]
        virtual_a1g = [
            index
            for index, (label, occupation) in enumerate(zip(labels, occupations))
            if label == "A1g" and occupation == 0.0 and index not in selected
        ]
        if not occupied_a1g or not virtual_a1g:
            raise ValueError("failed to extend N2 to the nested CAS(10,10) space")
        selected.extend(
            (
                max(occupied_a1g, key=lambda index: orbital_energies[index]),
                min(virtual_a1g, key=lambda index: orbital_energies[index]),
            )
        )
    result = tuple(sorted(selected))
    if len(result) != cas_norb:
        raise RuntimeError("nested N2 active-space selection produced duplicate orbitals")
    return result


def _select_active_indices(
    molecule: gto.Mole,
    mean_field: scf.hf.RHF,
    labels: Sequence[str],
    cas_norb: int,
    cas_nelec: int,
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    ncore = (molecule.nelectron - cas_nelec) // 2
    if ncore < 0 or ncore + cas_norb > mean_field.mo_coeff.shape[1]:
        raise ValueError("requested CAS does not fit the molecular orbital space")
    if requested is not None:
        result = tuple(int(index) for index in requested)
        if len(result) != cas_norb:
            raise ValueError("active_indices length differs from cas_norb")
        return result
    charges = tuple(int(value) for value in molecule.atom_charges())
    if charges == (7, 7) and cas_nelec in (4, 6, 8, 10) and cas_norb == cas_nelec:
        return _n2_active_indices(
            labels,
            mean_field.mo_occ,
            mean_field.mo_energy,
            cas_norb,
        )
    return tuple(range(ncore, ncore + cas_norb))


def _block_semicanonical_rotation(
    fock_mo: np.ndarray,
    labels: Sequence[str],
    blocks: Sequence[Sequence[int]],
) -> np.ndarray:
    rotation = np.eye(fock_mo.shape[0])
    for block in blocks:
        block = tuple(int(index) for index in block)
        for label in dict.fromkeys(labels[index] for index in block):
            indices = tuple(index for index in block if labels[index] == label)
            if not indices:
                continue
            values, vectors = np.linalg.eigh(fock_mo[np.ix_(indices, indices)])
            order = np.argsort(values)
            rotation[np.ix_(indices, indices)] = vectors[:, order]
    return rotation


def _fix_rotated_orbital_phases(
    coefficients: np.ndarray,
    rotation: np.ndarray,
    columns: Sequence[int],
) -> np.ndarray:
    """Make selected semicanonical orbitals invariant to eigensolver signs."""

    result = np.asarray(rotation, dtype=float).copy()
    rotated = np.asarray(coefficients, dtype=float) @ result
    for column in columns:
        column = int(column)
        pivot = int(np.argmax(np.abs(rotated[:, column])))
        if rotated[pivot, column] < 0.0:
            result[:, column] *= -1.0
            rotated[:, column] *= -1.0
    return result


def _frozen_core_hamiltonian(
    hcore: np.ndarray,
    eri: np.ndarray,
    frozen: Sequence[int],
    correlated: Sequence[int],
    nuclear_repulsion: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    frozen = tuple(int(index) for index in frozen)
    correlated = tuple(int(index) for index in correlated)
    h1 = hcore[np.ix_(correlated, correlated)].copy()
    for local_p, p in enumerate(correlated):
        for local_q, q in enumerate(correlated):
            h1[local_p, local_q] += sum(
                2.0 * eri[p, q, i, i] - eri[p, i, i, q] for i in frozen
            )
    constant = float(nuclear_repulsion)
    constant += sum(2.0 * hcore[index, index] for index in frozen)
    constant += sum(
        2.0 * eri[i, i, j, j] - eri[i, j, j, i]
        for i in frozen
        for j in frozen
    )
    eri_correlated = eri[np.ix_(correlated, correlated, correlated, correlated)].copy()
    return h1, eri_correlated, float(constant)


def build_casscf_qp_reference(
    molecule: gto.Mole,
    *,
    cas_norb: int,
    cas_nelec: int,
    active_indices: Sequence[int] | None = None,
    frozen_core_orbitals: int = 0,
    previous_reference: CASQPReference | None = None,
    previous_continuation: CASSCFContinuation | None = None,
    rhf_convergence_tolerance: float = 1.0e-12,
    casscf_convergence_tolerance: float = 1.0e-10,
    reference_mode: str = "projected_agp_2rdm",
    prepare_contracted_rdms: bool = False,
    verbose: int = 0,
) -> CASQPReference:
    """Prepare a state-specific CASSCF/FCI-RDM full-space QP reference."""

    if molecule.spin != 0 or molecule.nelectron % 2:
        raise ValueError("the current CAS-QP builder requires a closed-shell singlet")
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = float(rhf_convergence_tolerance)
    mean_field.max_cycle = 200
    mean_field.verbose = int(verbose)
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("RHF failed while preparing the CAS-QP reference")
    mean_field.mo_coeff = _fix_orbital_phases(mean_field.mo_coeff)

    labels = tuple(
        str(label)
        for label in symm.label_orb_symm(
            molecule,
            molecule.irrep_name,
            molecule.symm_orb,
            mean_field.mo_coeff,
        )
    ) if molecule.symmetry else tuple("A" for _ in range(mean_field.mo_coeff.shape[1]))
    selected = _select_active_indices(
        molecule,
        mean_field,
        labels,
        int(cas_norb),
        int(cas_nelec),
        active_indices,
    )

    casscf = mcscf.CASSCF(mean_field, int(cas_norb), int(cas_nelec))
    casscf.conv_tol = float(casscf_convergence_tolerance)
    casscf.max_cycle_macro = 100
    casscf.max_cycle_micro = 12
    casscf.fcisolver.spin = 0
    casscf.verbose = int(verbose)
    tracking_reference = previous_reference
    previous_molecule = None
    previous_full_mo = None
    previous_ci = None
    if previous_continuation is not None:
        if previous_reference is not None:
            raise ValueError(
                "provide either previous_reference or previous_continuation, not both"
            )
        tracking_reference = previous_continuation.reference
        previous_molecule = previous_continuation.molecule
        previous_full_mo = previous_continuation.full_mo_coeff
        previous_ci = previous_continuation.active_ci
    elif previous_reference is not None:
        previous_molecule = previous_reference.metadata.get("pyscf_molecule")
        previous_full_mo = previous_reference.metadata.get("full_casscf_mo_coeff")
        previous_ci = previous_reference.metadata.get("active_ci_coefficients")

    if previous_full_mo is None:
        initial_orbitals = casscf.sort_mo(list(selected), base=0)
        initial_ci = None
        continuation_mode = "deterministic_rhf_selection"
    else:
        previous_full_mo = np.asarray(previous_full_mo, dtype=float)
        if previous_full_mo.shape[1] != mean_field.mo_coeff.shape[1]:
            raise ValueError("CASSCF continuation orbital dimension changed")
        initial_orbitals = mcscf.addons.project_init_guess(
            casscf,
            previous_full_mo,
            prev_mol=previous_molecule,
            priority="active",
            use_hf_core=True,
        )
        initial_ci = None if previous_ci is None else np.asarray(previous_ci)
        continuation_mode = "projected_previous_casscf"
    # PySCF's cylindrical-symmetry CSF guess evaluates intentionally singular
    # minors; their zero determinants are valid but NumPy emits warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in det",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in det",
            category=RuntimeWarning,
        )
        casscf.kernel(initial_orbitals, ci0=initial_ci)
    if not casscf.converged:
        newton = mcscf.newton(casscf)
        newton.kernel(casscf.mo_coeff, casscf.ci)
        casscf = newton
    if not casscf.converged:
        raise RuntimeError("state-specific CASSCF/FCI failed to converge")

    dm1_active, dm2_active = casscf.fcisolver.make_rdm12(
        casscf.ci,
        casscf.ncas,
        casscf.nelecas,
    )
    mo_coeff = np.asarray(casscf.mo_coeff, dtype=float)
    full_labels = tuple(
        str(label)
        for label in symm.label_orb_symm(
            molecule,
            molecule.irrep_name,
            molecule.symm_orb,
            mo_coeff,
        )
    ) if molecule.symmetry else tuple("A" for _ in range(mo_coeff.shape[1]))
    ncore = int(casscf.ncore)
    active_full = tuple(range(ncore, ncore + casscf.ncas))
    external_full = tuple(range(ncore + casscf.ncas, mo_coeff.shape[1]))
    inactive_full = tuple(range(ncore))
    frozen_count = int(frozen_core_orbitals)
    if frozen_count < 0 or frozen_count > ncore:
        raise ValueError("frozen_core_orbitals must lie within the inactive space")
    frozen_full = inactive_full[:frozen_count]

    fock_ao = casscf.get_fock(mo_coeff=mo_coeff, ci=casscf.ci, casdm1=dm1_active)
    fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
    semicanonical = _block_semicanonical_rotation(
        fock_mo,
        full_labels,
        (inactive_full, external_full),
    )
    semicanonical = _fix_rotated_orbital_phases(
        mo_coeff,
        semicanonical,
        (*inactive_full, *external_full),
    )
    mo_coeff = mo_coeff @ semicanonical
    fock_mo = semicanonical.T @ fock_mo @ semicanonical

    correlated_full = tuple(index for index in range(mo_coeff.shape[1]) if index not in frozen_full)
    correlated_coefficients = mo_coeff[:, correlated_full]
    local_from_full = {full: local for local, full in enumerate(correlated_full)}
    inactive = tuple(local_from_full[index] for index in inactive_full if index not in frozen_full)
    active = tuple(local_from_full[index] for index in active_full)
    external = tuple(local_from_full[index] for index in external_full)
    correlated_labels = tuple(full_labels[index] for index in correlated_full)
    correlated_irrep_ids = tuple(
        int(symm.irrep_name2id(molecule.groupname, label)) % 10
        for label in correlated_labels
    )
    correlated_fock = fock_mo[np.ix_(correlated_full, correlated_full)]

    cross_overlap = None
    if tracking_reference is not None and previous_molecule is not None:
        cross_overlap = gto.intor_cross("int1e_ovlp", previous_molecule, molecule)
    reference = build_cas_qp_reference_from_rdms(
        casscf_energy=float(casscf.e_tot),
        mo_coeff=correlated_coefficients,
        active_rdm1=dm1_active,
        active_rdm2=dm2_active,
        active_spatial_indices=active,
        inactive_spatial_indices=inactive,
        external_spatial_indices=external,
        correlated_target_number=molecule.nelectron - 2 * frozen_count,
        physical_target_number=molecule.nelectron,
        frozen_spatial_indices=frozen_full,
        orbital_labels=correlated_labels,
        spatial_fock=correlated_fock,
        previous_reference=tracking_reference,
        cross_overlap=cross_overlap,
        reference_mode=reference_mode,
        metadata={
            "pyscf_molecule": molecule,
            "full_casscf_mo_coeff": np.asarray(casscf.mo_coeff, dtype=float).copy(),
            "rhf_energy": float(mean_field.e_tot),
            "casscf_converged": bool(casscf.converged),
            "selected_rhf_active_indices": selected,
            "full_casscf_active_indices": active_full,
            "correlated_full_indices": correlated_full,
            "active_ci_coefficients": np.asarray(casscf.ci).copy(),
            "active_nelec": tuple(int(value) for value in casscf.nelecas),
            "casscf_continuation_mode": continuation_mode,
            "orbital_irrep_ids": correlated_irrep_ids,
            "orbital_symmetry_group": molecule.groupname,
        },
    )
    active_rotation = np.asarray(reference.metadata["active_rotation"])
    ci_natural = casscf.fcisolver.transform_ci_for_orbital_rotation(
        casscf.ci,
        casscf.ncas,
        casscf.nelecas,
        active_rotation,
    )
    reference.metadata.update(
        {
            "active_ci_coefficients_natural": np.asarray(ci_natural).copy(),
            "active_ci_orbital_basis": "cas_natural_orbitals",
            "active_ci_norm": float(np.vdot(ci_natural, ci_natural).real),
        }
    )
    if prepare_contracted_rdms:
        check_rdm1, check_rdm2, rdm3, rdm4 = casscf.fcisolver.make_rdm1234(
            ci_natural,
            casscf.ncas,
            casscf.nelecas,
        )
        if np.max(np.abs(check_rdm1 - reference.active_rdm1)) > 1.0e-10:
            raise RuntimeError("natural-orbital active 3/4-RDM preparation changed the 1-RDM")
        if np.max(np.abs(check_rdm2 - reference.active_rdm2)) > 1.0e-10:
            raise RuntimeError("natural-orbital active 3/4-RDM preparation changed the 2-RDM")
        reference.metadata.update(
            {
                "active_rdm3_natural": np.asarray(rdm3),
                "active_rdm4_natural": np.asarray(rdm4),
                "active_higher_rdms_cached": True,
            }
        )

    combined_coefficients = np.column_stack(
        (mo_coeff[:, frozen_full], reference.mo_coeff)
    ) if frozen_full else reference.mo_coeff
    hcore = combined_coefficients.T @ mean_field.get_hcore() @ combined_coefficients
    norb = combined_coefficients.shape[1]
    eri = np.asarray(
        ao2mo.restore(1, ao2mo.full(molecule, combined_coefficients), norb),
        dtype=float,
    )
    frozen_combined = tuple(range(frozen_count))
    correlated_combined = tuple(range(frozen_count, norb))
    h1_spatial, eri_spatial, constant_energy = _frozen_core_hamiltonian(
        hcore,
        eri,
        frozen_combined,
        correlated_combined,
        molecule.energy_nuc(),
    )
    h1_spin, g2_spin = spatial_to_spin_integrals(h1_spatial, eri_spatial)
    reference.metadata.update(
        {
            "h1_spatial": h1_spatial,
            "eri_spatial": eri_spatial,
            "h1_spin": h1_spin,
            "g2_spin": g2_spin,
            "constant_energy": constant_energy,
            "nuclear_repulsion": float(molecule.energy_nuc()),
            "basis": molecule.basis,
            "atom": molecule.atom,
        }
    )
    return reference


def build_n2_casscf_qp_reference(
    distance: float,
    *,
    basis: str = "sto-3g",
    cas_norb: int = 6,
    cas_nelec: int = 6,
    frozen_n1s: bool = True,
    reference_mode: str = "projected_agp_2rdm",
    prepare_contracted_rdms: bool = False,
    previous_reference: CASQPReference | None = None,
    previous_continuation: CASSCFContinuation | None = None,
    verbose: int = 0,
) -> CASQPReference:
    if distance <= 0.0:
        raise ValueError("N2 distance must be positive")
    half = 0.5 * float(distance)
    molecule = gto.M(
        atom=f"N 0 0 {-half:.12f}; N 0 0 {half:.12f}",
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=True,
        verbose=0,
    )
    return build_casscf_qp_reference(
        molecule,
        cas_norb=cas_norb,
        cas_nelec=cas_nelec,
        frozen_core_orbitals=2 if frozen_n1s else 0,
        previous_reference=previous_reference,
        previous_continuation=previous_continuation,
        reference_mode=reference_mode,
        prepare_contracted_rdms=prepare_contracted_rdms,
        verbose=verbose,
    )


def build_h2_casscf_qp_reference(
    distance: float,
    *,
    basis: str = "6-31g",
    reference_mode: str = "projected_agp_2rdm",
    prepare_contracted_rdms: bool = False,
    previous_reference: CASQPReference | None = None,
    previous_continuation: CASSCFContinuation | None = None,
    verbose: int = 0,
) -> CASQPReference:
    if distance <= 0.0:
        raise ValueError("H2 distance must be positive")
    molecule = gto.M(
        atom=f"H 0 0 0; H 0 0 {float(distance):.12f}",
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=True,
        verbose=0,
    )
    return build_casscf_qp_reference(
        molecule,
        cas_norb=2,
        cas_nelec=2,
        frozen_core_orbitals=0,
        previous_reference=previous_reference,
        previous_continuation=previous_continuation,
        reference_mode=reference_mode,
        prepare_contracted_rdms=prepare_contracted_rdms,
        verbose=verbose,
    )


CONTINUATION_SCHEMA = "casscf-continuation-v3"
_SUPPORTED_CONTINUATION_SCHEMAS = frozenset(
    {"casscf-continuation-v1", "casscf-continuation-v2", CONTINUATION_SCHEMA}
)


def save_casscf_continuation(path: str | Path, reference: CASQPReference) -> Path:
    """Atomically save the portable state required by the next geometry."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    molecule = reference.metadata.get("pyscf_molecule")
    full_mo_coeff = reference.metadata.get("full_casscf_mo_coeff")
    active_ci = reference.metadata.get("active_ci_coefficients")
    if molecule is None or full_mo_coeff is None or active_ci is None:
        raise ValueError("reference lacks CASSCF continuation metadata")
    signed_geminals = (
        np.empty(0, dtype=float)
        if reference.signed_geminals is None
        else np.asarray(reference.signed_geminals, dtype=float)
    )
    orbital_irrep_ids = reference.metadata.get("orbital_irrep_ids")
    if orbital_irrep_ids is None:
        orbital_irrep_ids = tuple(
            int(symm.irrep_name2id(molecule.groupname, label)) % 10
            for label in reference.orbital_labels
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            checkpoint_schema=np.asarray(CONTINUATION_SCHEMA),
            atom=np.asarray(str(molecule.atom)),
            basis_json=np.asarray(json.dumps(molecule.basis)),
            unit=np.asarray(str(molecule.unit)),
            charge=np.asarray(int(molecule.charge)),
            spin=np.asarray(int(molecule.spin)),
            symmetry=np.asarray(bool(molecule.symmetry)),
            full_mo_coeff=np.asarray(full_mo_coeff, dtype=float),
            active_ci=np.asarray(active_ci),
            rhf_energy=np.asarray(float(reference.metadata["rhf_energy"])),
            casscf_energy=np.asarray(reference.casscf_energy),
            reference_mo_coeff=np.asarray(reference.mo_coeff, dtype=float),
            spatial_occupations=np.asarray(reference.spatial_occupations, dtype=float),
            active_rdm1=np.asarray(reference.active_rdm1, dtype=float),
            active_rdm2=np.asarray(reference.active_rdm2, dtype=float),
            active_indices=np.asarray(reference.active_spatial_indices, dtype=np.int64),
            inactive_indices=np.asarray(reference.inactive_spatial_indices, dtype=np.int64),
            external_indices=np.asarray(reference.external_spatial_indices, dtype=np.int64),
            frozen_indices=np.asarray(reference.frozen_spatial_indices, dtype=np.int64),
            physical_target_number=np.asarray(reference.physical_target_number),
            rdm_cumulant_norm=np.asarray(reference.rdm_cumulant_norm),
            orbital_labels_json=np.asarray(json.dumps(reference.orbital_labels)),
            orbital_irrep_ids=np.asarray(orbital_irrep_ids, dtype=np.int64),
            orbital_symmetry_group=np.asarray(str(molecule.groupname)),
            source_rdm_metadata_json=np.asarray(
                json.dumps(reference.source_rdm_metadata)
            ),
            reference_mode=np.asarray(reference.reference_mode),
            signed_geminals=signed_geminals,
            reconstruction_metrics_json=np.asarray(
                json.dumps(reference.reconstruction_metrics)
            ),
            U=reference.bogoliubov.U,
            V=reference.bogoliubov.V,
            u=reference.bogoliubov.u,
            v=reference.bogoliubov.v,
            partner=reference.bogoliubov.partner,
            signs=reference.bogoliubov.signs,
            target_number=np.asarray(reference.bogoliubov.target_number),
            quasiparticle_energies=reference.bogoliubov.quasiparticle_energies,
            chemical_potential=np.asarray(reference.bogoliubov.chemical_potential),
            hfb_energy=np.asarray(reference.bogoliubov.hfb_energy),
            bogoliubov_metadata_json=np.asarray(
                json.dumps(reference.bogoliubov.metadata)
            ),
        )
    temporary.replace(path)
    return path


def load_casscf_continuation(path: str | Path) -> CASSCFContinuation:
    """Load a continuation checkpoint without pickle or executable content."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        schema = str(checkpoint["checkpoint_schema"].item())
        if schema not in _SUPPORTED_CONTINUATION_SCHEMAS:
            raise ValueError(
                f"unsupported CASSCF continuation schema {schema!r}; "
                f"supported schemas are {sorted(_SUPPORTED_CONTINUATION_SCHEMAS)!r}"
            )
        molecule = gto.M(
            atom=str(checkpoint["atom"].item()),
            basis=json.loads(str(checkpoint["basis_json"].item())),
            unit=str(checkpoint["unit"].item()),
            charge=int(checkpoint["charge"].item()),
            spin=int(checkpoint["spin"].item()),
            symmetry=bool(checkpoint["symmetry"].item()),
            verbose=0,
        )
        bogoliubov_metadata = (
            json.loads(str(checkpoint["bogoliubov_metadata_json"].item()))
            if "bogoliubov_metadata_json" in checkpoint.files
            else {}
        )
        # Every CAS-RDM reference produced by this module uses the transformed
        # h11 diagonal for QPCCSD denominators. Older checkpoints omitted this
        # metadata and must retain that behavior after restart.
        bogoliubov_metadata.setdefault("denominator_source", "h11")
        bogoliubov = BogoliubovReference(
            U=checkpoint["U"],
            V=checkpoint["V"],
            u=checkpoint["u"],
            v=checkpoint["v"],
            partner=checkpoint["partner"],
            signs=checkpoint["signs"],
            target_number=int(checkpoint["target_number"].item()),
            quasiparticle_energies=checkpoint["quasiparticle_energies"],
            chemical_potential=float(checkpoint["chemical_potential"].item()),
            hfb_energy=float(checkpoint["hfb_energy"].item()),
            metadata=bogoliubov_metadata,
        )
        signed_geminals = (
            np.asarray(checkpoint["signed_geminals"], dtype=float)
            if "signed_geminals" in checkpoint.files
            else np.empty(0, dtype=float)
        )
        orbital_labels = tuple(
            str(value)
            for value in json.loads(str(checkpoint["orbital_labels_json"].item()))
        )
        orbital_irrep_ids = (
            tuple(int(value) for value in checkpoint["orbital_irrep_ids"])
            if "orbital_irrep_ids" in checkpoint.files
            else tuple(
                int(symm.irrep_name2id(molecule.groupname, label)) % 10
                for label in orbital_labels
            )
        )
        reference = CASQPReference(
            bogoliubov=bogoliubov,
            casscf_energy=float(checkpoint["casscf_energy"].item()),
            mo_coeff=checkpoint["reference_mo_coeff"],
            spatial_occupations=checkpoint["spatial_occupations"],
            active_rdm1=checkpoint["active_rdm1"],
            active_rdm2=checkpoint["active_rdm2"],
            active_spatial_indices=tuple(
                int(value) for value in checkpoint["active_indices"]
            ),
            inactive_spatial_indices=tuple(
                int(value) for value in checkpoint["inactive_indices"]
            ),
            external_spatial_indices=tuple(
                int(value) for value in checkpoint["external_indices"]
            ),
            frozen_spatial_indices=tuple(
                int(value) for value in checkpoint["frozen_indices"]
            ),
            physical_target_number=int(
                checkpoint["physical_target_number"].item()
            ),
            rdm_cumulant_norm=float(checkpoint["rdm_cumulant_norm"].item()),
            orbital_labels=orbital_labels,
            reference_mode=str(checkpoint["reference_mode"].item()),
            signed_geminals=(
                None if signed_geminals.size == 0 else signed_geminals
            ),
            reconstruction_metrics=json.loads(
                str(checkpoint["reconstruction_metrics_json"].item())
            ),
            source_rdm_metadata=(
                json.loads(str(checkpoint["source_rdm_metadata_json"].item()))
                if "source_rdm_metadata_json" in checkpoint.files
                else {}
            ),
            metadata={
                "continuation_schema": schema,
                "orbital_irrep_ids": orbital_irrep_ids,
                "orbital_symmetry_group": (
                    str(checkpoint["orbital_symmetry_group"].item())
                    if "orbital_symmetry_group" in checkpoint.files
                    else molecule.groupname
                ),
                **(
                    {"rhf_energy": float(checkpoint["rhf_energy"].item())}
                    if "rhf_energy" in checkpoint.files
                    else {}
                ),
            },
        )
        return CASSCFContinuation(
            molecule=molecule,
            full_mo_coeff=np.asarray(checkpoint["full_mo_coeff"], dtype=float),
            active_ci=np.asarray(checkpoint["active_ci"]),
            reference=reference,
            source_path=str(path.resolve()),
        )


def restore_continuation_integrals(
    continuation: CASSCFContinuation,
) -> CASQPReference:
    """Restore molecular integrals in the checkpoint's exact QP orbital basis.

    Portable continuation files intentionally omit the potentially large two-electron
    integral tensor.  Rebuilding CASSCF can perturb natural occupations and therefore
    the Bogoliubov basis.  This routine instead transforms fresh AO integrals with the
    orbitals stored in the checkpoint, so existing amplitudes remain basis-compatible.
    """

    reference = continuation.reference
    if "h1_spatial" in reference.metadata and "eri_spatial" in reference.metadata:
        return reference
    molecule = continuation.molecule
    frozen_full = tuple(int(index) for index in reference.frozen_spatial_indices)
    frozen_coefficients = np.asarray(continuation.full_mo_coeff)[:, frozen_full]
    combined_coefficients = (
        np.column_stack((frozen_coefficients, reference.mo_coeff))
        if frozen_full
        else np.asarray(reference.mo_coeff)
    )
    mean_field = scf.RHF(molecule)
    rhf_energy = reference.metadata.get("rhf_energy")
    if rhf_energy is None:
        mean_field.conv_tol = 1.0e-12
        mean_field.max_cycle = 200
        mean_field.verbose = 0
        rhf_energy = float(mean_field.kernel())
        if not mean_field.converged:
            mean_field = mean_field.newton().run()
            rhf_energy = float(mean_field.e_tot)
        if not mean_field.converged:
            raise RuntimeError("RHF did not converge while restoring the checkpoint")
    hcore = combined_coefficients.T @ mean_field.get_hcore() @ combined_coefficients
    norb = combined_coefficients.shape[1]
    eri = np.asarray(
        ao2mo.restore(1, ao2mo.full(molecule, combined_coefficients), norb),
        dtype=float,
    )
    frozen_combined = tuple(range(len(frozen_full)))
    correlated_combined = tuple(range(len(frozen_full), norb))
    h1_spatial, eri_spatial, constant_energy = _frozen_core_hamiltonian(
        hcore,
        eri,
        frozen_combined,
        correlated_combined,
        molecule.energy_nuc(),
    )
    h1_spin, g2_spin = spatial_to_spin_integrals(h1_spatial, eri_spatial)
    metadata = {
        **reference.metadata,
        "pyscf_molecule": molecule,
        "full_casscf_mo_coeff": np.asarray(continuation.full_mo_coeff).copy(),
        "active_ci_coefficients": np.asarray(continuation.active_ci).copy(),
        "h1_spatial": h1_spatial,
        "eri_spatial": eri_spatial,
        "h1_spin": h1_spin,
        "g2_spin": g2_spin,
        "constant_energy": constant_energy,
        "nuclear_repulsion": float(molecule.energy_nuc()),
        "rhf_energy": float(rhf_energy),
        "basis": molecule.basis,
        "atom": molecule.atom,
        "integral_restoration": "saved-orbital AO transformation",
    }
    return replace(reference, metadata=metadata)


def compute_fullspace_benchmarks(
    reference: CASQPReference,
    *,
    methods: Sequence[str] = ("ccsd", "fci"),
    fci_policy: str = "auto",
    fci_max_determinants: int = 1_000_000,
) -> FullSpaceBenchmark:
    """Compute conventional comparisons without forcing an intractable FCI."""

    aliases = {
        "ccsd": "ccsd",
        "ccsd(t)": "ccsd(t)",
        "ccsd_t": "ccsd(t)",
        "ccsdt": "ccsd(t)",
        "fci": "fci",
    }
    requested_raw = tuple(str(method).lower() for method in methods)
    unsupported = {method for method in requested_raw if method not in aliases}
    if unsupported:
        raise ValueError(f"unsupported benchmark methods: {sorted(unsupported)}")
    requested = frozenset(aliases[method] for method in requested_raw)
    if fci_policy not in {"auto", "force", "off"}:
        raise ValueError("fci_policy must be 'auto', 'force', or 'off'")
    if fci_max_determinants < 1:
        raise ValueError("fci_max_determinants must be positive")

    ccsd_energy: float | None = None
    ccsd_converged: bool | None = None
    ccsd_t_correction: float | None = None
    ccsd_t_energy: float | None = None
    fci_energy: float | None = None
    method_status: dict[str, str] = {}
    failure_reasons: dict[str, str] = {}
    timings: dict[str, float] = {}
    h1 = np.asarray(reference.metadata["h1_spatial"])
    nelec = int(reference.target_number)
    nalpha = nelec // 2
    nbeta = nelec - nalpha
    determinant_dimension = comb(h1.shape[0], nalpha) * comb(h1.shape[0], nbeta)

    if "ccsd" in requested or "ccsd(t)" in requested:
        from pyscf import cc

        started = time.perf_counter()
        try:
            molecule = reference.metadata.get("pyscf_molecule")
            if molecule is None:
                raise ValueError("the reference does not retain its PySCF molecule")
            mean_field = scf.RHF(molecule)
            mean_field.conv_tol = 1.0e-12
            mean_field.max_cycle = 200
            mean_field.verbose = 0
            mean_field.kernel()
            if not mean_field.converged:
                mean_field = mean_field.newton().run()
            if not mean_field.converged:
                raise RuntimeError("RHF did not converge for the CCSD benchmark")
            frozen = list(range(len(reference.frozen_spatial_indices)))
            solver = cc.CCSD(mean_field, frozen=frozen if frozen else None)
            solver.conv_tol = 1.0e-10
            solver.max_cycle = 200
            solver.verbose = 0
            solver.kernel()
            ccsd_converged = bool(solver.converged)
            ccsd_energy = float(solver.e_tot)
            method_status["ccsd"] = "complete" if solver.converged else "nonconverged"
            if "ccsd(t)" in requested:
                triples_started = time.perf_counter()
                ccsd_t_correction = float(solver.ccsd_t())
                timings["ccsd(t)"] = time.perf_counter() - triples_started
                ccsd_t_energy = ccsd_energy + ccsd_t_correction
                method_status["ccsd(t)"] = (
                    "complete" if solver.converged else "ccsd_nonconverged"
                )
        except Exception as error:
            method_status["ccsd"] = "failed"
            failure_reasons["ccsd"] = f"{type(error).__name__}: {error}"
            if "ccsd(t)" in requested:
                method_status["ccsd(t)"] = "skipped_ccsd_failure"
                failure_reasons["ccsd(t)"] = failure_reasons["ccsd"]
        timings["ccsd"] = time.perf_counter() - started

    if "fci" in requested:
        if fci_policy == "off":
            method_status["fci"] = "skipped_policy"
        elif fci_policy == "auto" and determinant_dimension > fci_max_determinants:
            method_status["fci"] = "skipped_dimension"
            failure_reasons["fci"] = (
                f"determinant dimension {determinant_dimension} exceeds "
                f"limit {fci_max_determinants}"
            )
        else:
            from pyscf import fci

            started = time.perf_counter()
            try:
                eri = np.asarray(reference.metadata["eri_spatial"])
                solver = fci.direct_spin1.FCI()
                solver.conv_tol = 1.0e-12
                solver.verbose = 0
                energy, _ci = solver.kernel(
                    h1,
                    eri,
                    h1.shape[0],
                    (nalpha, nbeta),
                    ecore=float(reference.metadata["constant_energy"]),
                )
                fci_energy = float(energy)
                method_status["fci"] = "complete"
            except Exception as error:
                method_status["fci"] = "failed"
                failure_reasons["fci"] = f"{type(error).__name__}: {error}"
            timings["fci"] = time.perf_counter() - started

    return FullSpaceBenchmark(
        rhf_energy=float(reference.metadata["rhf_energy"]),
        casscf_energy=reference.casscf_energy,
        ccsd_energy=ccsd_energy,
        ccsd_converged=ccsd_converged,
        fci_energy=fci_energy,
        ccsd_t_correction=ccsd_t_correction,
        ccsd_t_energy=ccsd_t_energy,
        fci_determinant_dimension=determinant_dimension,
        method_status=method_status,
        failure_reasons=failure_reasons,
        timings_s=timings,
    )


__all__ = [
    "CASSCFContinuation",
    "CONTINUATION_SCHEMA",
    "FullSpaceBenchmark",
    "build_casscf_qp_reference",
    "build_h2_casscf_qp_reference",
    "build_n2_casscf_qp_reference",
    "compute_fullspace_benchmarks",
    "load_casscf_continuation",
    "restore_continuation_integrals",
    "save_casscf_continuation",
]
