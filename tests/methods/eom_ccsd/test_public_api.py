from __future__ import annotations

import inspect

from autogen.methods import eom_ccsd
from autogen.methods.eom_ccsd.generated import residuals
from autogen.methods.eom_ccsd.runtime import solver


def test_canonical_eom_ccsd_export_preserves_runtime_signature() -> None:
    assert eom_ccsd.__all__ == ["solve_eom_ccsd"]
    assert eom_ccsd.solve_eom_ccsd is solver.solve_eom_ccsd
    assert str(inspect.signature(eom_ccsd.solve_eom_ccsd)) == (
        "(mol=None, mf=None, t1=None, t2=None, nroots=3, max_iter=50, "
        "tol=1e-08, max_space=20)"
    )


def test_runtime_helpers_remain_available() -> None:
    for name in ("pack", "unpack", "build_reference", "sigma_vector"):
        assert callable(getattr(solver, name))
    assert callable(residuals.compute_outputs)
