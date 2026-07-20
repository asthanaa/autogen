"""Fail-closed boundary for the archived determinant-space projected-QP oracle."""

from __future__ import annotations

from typing import Any

__all__ = ["ProjectedExactReference", "build_projected_exact_reference"]


def _archived_route_error() -> RuntimeError:
    return RuntimeError(
        "the determinant-space projected-QP oracle is an archived research "
        "route and is not installed with the production package"
    )


def build_projected_exact_reference(*args: Any, **kwargs: Any):
    """Reject use of the externally archived determinant-space oracle."""

    del args, kwargs
    raise _archived_route_error()


def __getattr__(name: str) -> Any:
    if name == "ProjectedExactReference":
        raise _archived_route_error()
    raise AttributeError(name)
