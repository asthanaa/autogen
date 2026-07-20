from __future__ import annotations

import inspect

import numpy as np

from autogen.methods import ccsd
from autogen.methods.ccsd.generated import f1t1_einsum, residuals, v2t1t1_einsum, v2t2_einsum
from autogen.methods.ccsd.runtime import solver


def test_canonical_ccsd_export_preserves_runtime_signature() -> None:
    assert ccsd.__all__ == ["solve_ccsd"]
    assert ccsd.solve_ccsd is solver.solve_ccsd
    assert str(inspect.signature(ccsd.solve_ccsd)) == (
        "(mol=None, max_iter=50, tol=1e-08, damping=0.0, "
        "diis_start=2, max_diis=6)"
    )


def test_runtime_helpers_remain_available() -> None:
    assert callable(solver.mp2_init)
    assert callable(solver.compute_energy)
    assert callable(residuals.compute_outputs)


def test_relocated_energy_kernels_preserve_their_equations() -> None:
    rng = np.random.default_rng(314)
    f = rng.normal(size=(4, 4))
    g = rng.normal(size=(4, 4, 4, 4))
    t1 = rng.normal(size=(2, 2))
    t2 = rng.normal(size=(2, 2, 2, 2))
    o = [0, 1]
    v = [2, 3]
    ovvo = g[np.ix_(o, v, v, o)]
    tau = np.einsum("ia,jb->ijab", t1, t1)

    assert np.allclose(
        f1t1_einsum.compute_f1t1(f, t1, o, v),
        2.0 * np.einsum("ia,ia->", f[np.ix_(o, v)], t1),
    )
    assert np.allclose(
        v2t1t1_einsum.compute_v2t1t1(g, t1, o, v),
        2.0 * np.einsum("ijab,iabj->", tau, ovvo)
        - np.einsum("jiab,iabj->", tau, ovvo),
    )
    assert np.allclose(
        v2t2_einsum.compute_v2t2(g, t2, o, v),
        2.0 * np.einsum("ijab,iabj->", t2, ovvo)
        - np.einsum("jiab,iabj->", t2, ovvo),
    )
