from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, fci, gto, scf, symm


_PI_PI_STAR_TARGETS = (
    ("E1ux", True),
    ("E1uy", True),
    ("E1gx", False),
    ("E1gy", False),
)


@dataclass(frozen=True)
class N2ActiveSpacePoint:
    distance: float
    h1: np.ndarray
    g2: np.ndarray
    h1_spatial: np.ndarray
    eri_spatial: np.ndarray
    core_energy: float
    rhf_energy: float
    fci_energy: float
    active_indices: tuple[int, ...]
    active_labels: tuple[str, ...]
    core_indices: tuple[int, ...]
    active_electrons: int = 4

    @property
    def target_number(self) -> int:
        return self.active_electrons

    @property
    def nspin(self) -> int:
        return int(self.h1.shape[0])


def _fix_orbital_phases(coefficients: np.ndarray) -> np.ndarray:
    result = np.asarray(coefficients, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _select_active_indices(
    labels: list[str],
    occupations: np.ndarray,
    orbital_energies: np.ndarray,
    *,
    include_sigma_pair: bool,
) -> list[int]:
    selected: list[int] = []
    for label, occupied in _PI_PI_STAR_TARGETS:
        candidates = [
            index
            for index, (candidate, occupation) in enumerate(zip(labels, occupations))
            if candidate == label and bool(occupation > 0.0) is occupied
        ]
        if not candidates:
            raise ValueError(
                f"failed to find {label} with occupied={occupied}"
            )
        # Canonical RHF ordering selects the lowest-energy member when the
        # split-valence basis contains more than one virtual shell of an irrep.
        selected.append(candidates[0])
    if include_sigma_pair:
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
            raise ValueError("failed to find the valence sigma/sigma* orbital pair")
        occupied_sigma_index = max(
            occupied_sigma, key=lambda index: orbital_energies[index]
        )
        virtual_sigma_index = min(
            virtual_sigma, key=lambda index: orbital_energies[index]
        )
        return [
            occupied_sigma_index,
            *selected[:2],
            virtual_sigma_index,
            *selected[2:],
        ]
    return sorted(selected)


def _frozen_core_integrals(
    hcore: np.ndarray,
    eri: np.ndarray,
    active: list[int],
    core: list[int],
    nuclear_repulsion: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    h1 = hcore[np.ix_(active, active)].copy()
    for local_p, p in enumerate(active):
        for local_q, q in enumerate(active):
            h1[local_p, local_q] += sum(
                2.0 * eri[p, q, i, i] - eri[p, i, i, q] for i in core
            )
    constant = float(nuclear_repulsion)
    constant += sum(2.0 * hcore[i, i] for i in core)
    constant += sum(
        2.0 * eri[i, i, j, j] - eri[i, j, j, i] for i in core for j in core
    )
    return h1, eri[np.ix_(active, active, active, active)].copy(), constant


def _spin_expand(h1_spatial: np.ndarray, eri_spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nspatial = h1_spatial.shape[0]
    nspin = 2 * nspatial
    h1 = np.zeros((nspin, nspin), dtype=float)
    eri = np.zeros((nspin, nspin, nspin, nspin), dtype=float)
    for p in range(nspin):
        spatial_p, spin_p = divmod(p, 2)
        for q in range(nspin):
            spatial_q, spin_q = divmod(q, 2)
            if spin_p == spin_q:
                h1[p, q] = h1_spatial[spatial_p, spatial_q]
    for p in range(nspin):
        spatial_p, spin_p = divmod(p, 2)
        for q in range(nspin):
            spatial_q, spin_q = divmod(q, 2)
            for r in range(nspin):
                spatial_r, spin_r = divmod(r, 2)
                if spin_p != spin_r:
                    continue
                for s in range(nspin):
                    spatial_s, spin_s = divmod(s, 2)
                    if spin_q == spin_s:
                        eri[p, q, r, s] = eri_spatial[
                            spatial_p, spatial_r, spatial_q, spatial_s
                        ]
    return h1, eri - eri.swapaxes(2, 3)


def _build_n2_active_space_631g(
    distance: float,
    *,
    active_electrons: int,
) -> N2ActiveSpacePoint:
    if distance <= 0.0:
        raise ValueError("N2 distance must be positive")
    if active_electrons not in (4, 6):
        raise ValueError("supported N2 active spaces are CAS(4,4) and CAS(6,6)")
    half = 0.5 * float(distance)
    molecule = gto.M(
        atom=f"N 0 0 {-half:.12f}; N 0 0 {half:.12f}",
        basis="6-31g",
        unit="Angstrom",
        symmetry=True,
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.max_cycle = 200
    mean_field.verbose = 0
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"RHF failed at R={distance:.6f} Angstrom")

    coefficients = _fix_orbital_phases(mean_field.mo_coeff)
    labels = [
        str(label)
        for label in symm.label_orb_symm(
            molecule,
            molecule.irrep_name,
            molecule.symm_orb,
            coefficients,
        )
    ]
    active = _select_active_indices(
        labels,
        mean_field.mo_occ,
        mean_field.mo_energy,
        include_sigma_pair=active_electrons == 6,
    )
    active_occupied = [index for index in active if mean_field.mo_occ[index] > 0.0]
    if len(active_occupied) != active_electrons // 2:
        raise ValueError("the selected valence space has the wrong electron count")
    core = [
        index
        for index, occupation in enumerate(mean_field.mo_occ)
        if occupation > 0.0 and index not in active_occupied
    ]

    hcore = coefficients.T @ mean_field.get_hcore() @ coefficients
    norb = coefficients.shape[1]
    eri = np.asarray(ao2mo.restore(1, ao2mo.full(molecule, coefficients), norb))
    h1_spatial, eri_spatial, core_energy = _frozen_core_integrals(
        hcore,
        eri,
        active,
        core,
        molecule.energy_nuc(),
    )
    h1, g2 = _spin_expand(h1_spatial, eri_spatial)
    fci_solver = fci.direct_spin1.FCI()
    fci_solver.verbose = 0
    fci_energy, _ = fci_solver.kernel(
        h1_spatial,
        eri_spatial,
        active_electrons,
        (active_electrons // 2, active_electrons // 2),
        ecore=core_energy,
    )
    return N2ActiveSpacePoint(
        distance=float(distance),
        h1=h1,
        g2=g2,
        h1_spatial=h1_spatial,
        eri_spatial=eri_spatial,
        core_energy=float(core_energy),
        rhf_energy=float(mean_field.e_tot),
        fci_energy=float(fci_energy),
        active_indices=tuple(active),
        active_labels=tuple(labels[index] for index in active),
        core_indices=tuple(core),
        active_electrons=active_electrons,
    )


def build_n2_4e4o_631g(distance: float) -> N2ActiveSpacePoint:
    """Build deterministic symmetry-selected N2 CAS(4e,4o)/6-31G data."""

    return _build_n2_active_space_631g(distance, active_electrons=4)


def build_n2_6e6o_631g(distance: float) -> N2ActiveSpacePoint:
    """Build deterministic valence N2 CAS(6e,6o)/6-31G benchmark data."""

    return _build_n2_active_space_631g(distance, active_electrons=6)
