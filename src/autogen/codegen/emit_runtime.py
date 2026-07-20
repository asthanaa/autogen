"""Stable wrapper module for generated runtime emission."""

from __future__ import annotations

from .cli import emit_qp_ccsd_projected_residuals, emit_spec_residuals, emit_structured_residuals

__all__ = [
    "emit_qp_ccsd_projected_residuals",
    "emit_spec_residuals",
    "emit_structured_residuals",
]
