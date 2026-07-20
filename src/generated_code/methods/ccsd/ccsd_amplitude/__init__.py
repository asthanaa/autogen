"""Deprecated CCSD-amplitude namespace."""

from . import residuals, residuals_spinorb
from .solver import compute_energy, solve_ccsd

__all__ = ["compute_energy", "residuals", "residuals_spinorb", "solve_ccsd"]
