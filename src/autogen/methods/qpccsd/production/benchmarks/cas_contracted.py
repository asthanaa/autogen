from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import CASContractedCCSDResult, CASContractedReference


class CASContractedKernel(Protocol):
    """Generated exact-active contraction kernel required by the solver."""

    coordinate_count: int

    def __call__(self, amplitudes: np.ndarray) -> tuple[float, np.ndarray]: ...


def solve_cas_contracted_ccsd(
    reference: CASContractedReference,
    *,
    kernel: CASContractedKernel | None = None,
    initial_amplitudes: np.ndarray | None = None,
    residual_tolerance: float = 1.0e-8,
    max_evaluations: int = 500,
) -> CASContractedCCSDResult:
    """Solve generated number-conserving exact-CAS contracted equations.

    A conventional/tailored CCSD backend is deliberately not accepted here:
    it would replace exact active contractions by active T1/T2 amplitudes and
    would not be the diagnostic method specified by this API.
    """

    if kernel is None:
        if not reference.inactive_spatial_indices and not reference.external_spatial_indices:
            return CASContractedCCSDResult(
                converged=True,
                total_energy=reference.casscf_energy,
                dynamic_correlation_energy=0.0,
                residual_norm=0.0,
                iterations=0,
                diagnostics={
                    "backend": "exact active state; no mixed/external coordinates",
                    "full_system_determinants": False,
                },
            )
        raise NotImplementedError(
            "exact-CAS contracted CCSD requires the generated active-RDM "
            "contraction kernel; tailored CCSD and full-system determinant "
            "fallbacks are intentionally prohibited"
        )

    from scipy.optimize import root

    coordinate_count = int(kernel.coordinate_count)
    if coordinate_count < 0:
        raise ValueError("contracted kernel coordinate_count cannot be negative")
    if initial_amplitudes is None:
        initial = np.zeros(coordinate_count)
    else:
        initial = np.asarray(initial_amplitudes, dtype=float)
        if initial.shape != (coordinate_count,):
            raise ValueError("initial contracted amplitudes have an incompatible shape")
    evaluations = 0
    best: tuple[float, float, np.ndarray] | None = None

    def objective(vector: np.ndarray) -> np.ndarray:
        nonlocal evaluations, best
        energy, residual = kernel(np.asarray(vector, dtype=float))
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (coordinate_count,):
            raise ValueError("contracted kernel residual has an incompatible shape")
        evaluations += 1
        norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        if best is None or norm < best[0]:
            best = (norm, float(energy), np.asarray(vector, dtype=float).copy())
        return residual

    solution = root(
        objective,
        initial,
        method="df-sane",
        options={
            "fatol": 0.1 * residual_tolerance,
            "ftol": 1.0e-12,
            "maxfev": int(max_evaluations),
            "line_search": "cruz",
        },
    )
    objective(np.asarray(solution.x, dtype=float))
    assert best is not None
    residual_norm, energy, _amplitudes = best
    return CASContractedCCSDResult(
        converged=bool(residual_norm < residual_tolerance),
        total_energy=energy,
        dynamic_correlation_energy=energy - reference.casscf_energy,
        residual_norm=residual_norm,
        iterations=evaluations,
        diagnostics={
            "backend": "generated exact-active RDM contractions",
            "full_system_determinants": False,
            "optimizer_success": bool(solution.success),
            "optimizer_message": str(solution.message),
        },
    )


__all__ = ["CASContractedKernel", "solve_cas_contracted_ccsd"]
