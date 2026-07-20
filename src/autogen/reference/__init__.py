"""Reference and oracle evaluators used for theory validation.

The historical determinant-space projected oracle depends on research-only
generated modules that are deliberately not part of the installed runtime.
Keep its public names lazy so independent reference helpers, such as
``wick_ref``, remain importable in a clean wheel.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ProjectedExactReference", "build_projected_exact_reference"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import projected_qp

        return getattr(projected_qp, name)
    raise AttributeError(name)
