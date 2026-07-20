"""Deprecated EOM-CCSD namespace; use :mod:`autogen.methods.eom_ccsd`."""

from . import residuals, residuals_spinorb
from .eom_solver import solve_eom_ccsd

__all__ = ["residuals", "residuals_spinorb", "solve_eom_ccsd"]
