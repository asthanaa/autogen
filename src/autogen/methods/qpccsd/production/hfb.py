from __future__ import annotations

import numpy as np

from .models import BogoliubovReference


class _DIIS:
    def __init__(self, size: int) -> None:
        self.size = size
        self.values: list[np.ndarray] = []
        self.errors: list[np.ndarray] = []

    def update(self, value: np.ndarray, error: np.ndarray) -> np.ndarray:
        self.values.append(value.copy())
        self.errors.append(error.copy())
        if len(self.values) > self.size:
            self.values.pop(0)
            self.errors.pop(0)
        if len(self.values) < 2:
            return value
        count = len(self.values)
        matrix = np.empty((count + 1, count + 1), dtype=float)
        matrix[-1, :] = -1.0
        matrix[:, -1] = -1.0
        matrix[-1, -1] = 0.0
        for left in range(count):
            for right in range(count):
                matrix[left, right] = float(np.dot(self.errors[left], self.errors[right]))
        rhs = np.zeros(count + 1)
        rhs[-1] = -1.0
        try:
            coefficients = np.linalg.solve(matrix, rhs)[:-1]
        except np.linalg.LinAlgError:
            return value
        return sum(
            (coefficient * item for coefficient, item in zip(coefficients, self.values)),
            np.zeros_like(value),
        )


def _paired_densities(occupations: np.ndarray, pair_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    npair = occupations.size
    nspin = 2 * npair
    rho = np.zeros((nspin, nspin), dtype=float)
    kappa = np.zeros((nspin, nspin), dtype=float)
    for pair in range(npair):
        alpha = 2 * pair
        beta = alpha + 1
        rho[alpha, alpha] = occupations[pair]
        rho[beta, beta] = occupations[pair]
        kappa[alpha, beta] = pair_tensor[pair]
        kappa[beta, alpha] = -pair_tensor[pair]
    return rho, kappa


def _fields(h1: np.ndarray, g2: np.ndarray, rho: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fock = h1 + np.einsum("prqs,sr->pq", g2, rho, optimize=True)
    pairing = 0.5 * np.einsum("pqrs,rs->pq", g2, kappa, optimize=True)
    return 0.5 * (fock + fock.T), 0.5 * (pairing - pairing.T)


def _solve_chemical_potential(
    orbital_energies: np.ndarray,
    gaps: np.ndarray,
    target_number: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if float(np.max(np.abs(gaps))) < 1.0e-14:
        occupied_count = target_number // 2
        order = np.argsort(orbital_energies)
        occupations = np.zeros_like(orbital_energies)
        occupations[order[:occupied_count]] = 1.0
        if occupied_count == 0:
            mu = float(np.min(orbital_energies) - 1.0)
        elif occupied_count == orbital_energies.size:
            mu = float(np.max(orbital_energies) + 1.0)
        else:
            homo = float(orbital_energies[order[occupied_count - 1]])
            lumo = float(orbital_energies[order[occupied_count]])
            mu = 0.5 * (homo + lumo)
        return mu, occupations, np.zeros_like(gaps), np.abs(orbital_energies - mu)

    def evaluate(mu: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        shifted = orbital_energies - mu
        energies = np.sqrt(shifted * shifted + gaps * gaps)
        safe = np.where(energies < 1.0e-14, 1.0e-14, energies)
        occupations = 0.5 * (1.0 - shifted / safe)
        pair_tensor = -gaps / (2.0 * safe)
        return float(2.0 * np.sum(occupations)), occupations, pair_tensor, energies

    span = max(10.0, float(np.max(np.abs(orbital_energies))) + float(np.max(np.abs(gaps))) + 1.0)
    lower = float(np.min(orbital_energies) - span)
    upper = float(np.max(orbital_energies) + span)
    for _ in range(160):
        middle = 0.5 * (lower + upper)
        number, occupations, pair_tensor, energies = evaluate(middle)
        if number < target_number:
            lower = middle
        else:
            upper = middle
    mu = 0.5 * (lower + upper)
    _number, occupations, pair_tensor, energies = evaluate(mu)
    return mu, occupations, pair_tensor, energies


def _hfb_energy(
    h1: np.ndarray,
    g2: np.ndarray,
    rho: np.ndarray,
    kappa: np.ndarray,
    constant: float,
) -> float:
    one_body = np.einsum("pq,qp->", h1, rho, optimize=True)
    normal = 0.5 * np.einsum("pqrs,rp,sq->", g2, rho, rho, optimize=True)
    pairing = 0.25 * np.einsum("pqrs,pq,rs->", g2, kappa, kappa, optimize=True)
    return float(np.real(one_body + normal + pairing) + constant)


def build_paired_hfb(
    h1: np.ndarray,
    g2: np.ndarray,
    target_number: int,
    *,
    constant: float = 0.0,
    pairing_seed: float = 1.0e-3,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
    damping: float = 0.35,
    diis_start: int = 2,
    diis_size: int = 6,
    collapse_tolerance: float = 1.0e-8,
) -> BogoliubovReference:
    """Solve real diagonal, time-reversal-paired molecular HFB equations."""

    h1 = np.asarray(h1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    nspin = h1.shape[0]
    if h1.shape != (nspin, nspin) or g2.shape != (nspin,) * 4:
        raise ValueError("incompatible one- and two-body integral shapes")
    if nspin % 2 or target_number < 0 or target_number > nspin or target_number % 2:
        raise ValueError("paired HFB requires an even spin dimension and even target number")
    npair = nspin // 2

    diagonal = np.array([0.5 * (h1[2 * p, 2 * p] + h1[2 * p + 1, 2 * p + 1]) for p in range(npair)])
    order = np.argsort(diagonal)
    occupations = np.zeros(npair)
    occupations[order[: target_number // 2]] = 1.0
    pair_tensor = np.full(npair, float(pairing_seed))
    diis = _DIIS(diis_size)
    converged = False
    residual_norm = float("inf")
    chemical_potential = 0.0
    qp_energies_pair = np.ones(npair)

    for iteration in range(1, max_iterations + 1):
        rho, kappa = _paired_densities(occupations, pair_tensor)
        fock, pairing = _fields(h1, g2, rho, kappa)
        orbital_energies = np.array(
            [0.5 * (fock[2 * p, 2 * p] + fock[2 * p + 1, 2 * p + 1]) for p in range(npair)]
        )
        gaps = np.array([pairing[2 * p, 2 * p + 1] for p in range(npair)])
        chemical_potential, occ_new, pair_new, qp_energies_pair = _solve_chemical_potential(
            orbital_energies,
            gaps,
            target_number,
        )
        candidate = np.concatenate((occ_new, pair_new))
        current = np.concatenate((occupations, pair_tensor))
        error = candidate - current
        residual_norm = float(np.max(np.abs(error)))
        if residual_norm < tolerance:
            occupations = occ_new
            pair_tensor = pair_new
            converged = True
            break
        if (
            np.max(np.abs(pair_new), initial=0.0) < collapse_tolerance
            and np.max(np.abs(pair_tensor), initial=0.0) < 10.0 * collapse_tolerance
        ):
            # Once the anomalous branch has vanished, snap to the exact
            # number-conserving fixed point instead of DIIS-oscillating around
            # a zero pairing field for hundreds of iterations.
            zero_pair = np.zeros_like(pair_tensor)
            zero_rho, zero_kappa = _paired_densities(occ_new, zero_pair)
            zero_fock, _zero_pairing = _fields(h1, g2, zero_rho, zero_kappa)
            zero_energies = np.array(
                [
                    0.5
                    * (zero_fock[2 * p, 2 * p] + zero_fock[2 * p + 1, 2 * p + 1])
                    for p in range(npair)
                ]
            )
            chemical_potential, collapsed_occ, _collapsed_pair, qp_energies_pair = (
                _solve_chemical_potential(zero_energies, zero_pair, target_number)
            )
            collapse_residual = float(np.max(np.abs(collapsed_occ - occ_new)))
            if collapse_residual < max(tolerance, collapse_tolerance):
                occupations = collapsed_occ
                pair_tensor = zero_pair
                residual_norm = collapse_residual
                converged = True
                break
        mixed = (1.0 - damping) * current + damping * candidate
        if iteration >= diis_start:
            mixed = diis.update(mixed, error)
        occupations = np.clip(mixed[:npair], 0.0, 1.0)
        pair_tensor = np.clip(mixed[npair:], -0.5, 0.5)

    rho, kappa = _paired_densities(occupations, pair_tensor)
    u_pair = np.sqrt(np.clip(1.0 - occupations, 0.0, 1.0))
    v_sign = np.where(pair_tensor < 0.0, -1.0, 1.0)
    v_pair = v_sign * np.sqrt(np.clip(occupations, 0.0, 1.0))
    u = np.repeat(u_pair, 2)
    v = np.repeat(v_pair, 2)
    partner = np.arange(nspin, dtype=np.int64) ^ 1
    signs = np.where(np.arange(nspin) % 2 == 0, 1.0, -1.0)
    U = np.diag(u.astype(np.complex128))
    V = np.zeros((nspin, nspin), dtype=np.complex128)
    V[np.arange(nspin), partner] = signs * v
    qp_energies = np.repeat(qp_energies_pair, 2)
    anomalous_norm = float(np.max(np.abs(kappa))) if kappa.size else 0.0
    fock, pairing = _fields(h1, g2, rho, kappa)
    identity = np.eye(nspin)
    generalized_density = np.block([[rho, kappa], [-kappa, identity - rho]])
    shifted_fock = fock - chemical_potential * identity
    hfb_matrix = np.block([[shifted_fock, pairing], [-pairing, -shifted_fock]])
    density_idempotency_error = float(
        np.max(np.abs(generalized_density @ generalized_density - generalized_density))
    )
    stationarity_error = float(
        np.max(np.abs(hfb_matrix @ generalized_density - generalized_density @ hfb_matrix))
    )
    average_particle_number = float(np.trace(rho))
    reference = BogoliubovReference(
        U=U,
        V=V,
        u=u,
        v=v,
        partner=partner,
        signs=signs,
        target_number=target_number,
        quasiparticle_energies=qp_energies,
        chemical_potential=chemical_potential,
        hfb_energy=_hfb_energy(h1, g2, rho, kappa, constant),
        converged=converged,
        iterations=iteration if max_iterations else 0,
        residual_norm=residual_norm,
        pairing_collapsed=anomalous_norm < collapse_tolerance,
        metadata={
            "reference_type": "restricted_diagonal_paired_hfb",
            "average_particle_number": average_particle_number,
            "particle_number_error": abs(average_particle_number - target_number),
            "anomalous_norm": anomalous_norm,
            "generalized_density_idempotency_error": density_idempotency_error,
            "hfb_stationarity_error": stationarity_error,
            "pairing_collapse_reason": (
                "self-consistent anomalous density below collapse tolerance"
                if anomalous_norm < collapse_tolerance
                else None
            ),
        },
    )
    canonical_normal, canonical_anomalous = reference.canonical_errors()
    reference.metadata["canonical_normal_error"] = canonical_normal
    reference.metadata["canonical_anomalous_error"] = canonical_anomalous
    return reference
