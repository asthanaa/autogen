"""Compatibility imports must resolve to the canonical implementations."""


def test_generated_code_ccsd_forwarders_are_thin() -> None:
    from autogen.methods.ccsd.generated import residuals as canonical_residuals
    from autogen.methods.ccsd.runtime.integrals import compute_integrals as canonical_integrals
    from autogen.methods.ccsd.runtime.solver import compute_energy as canonical_energy
    from generated_code.methods.ccsd.ccsd_amplitude import residuals as legacy_residuals
    from generated_code.methods.ccsd.ccsd_amplitude.solver import compute_energy
    from generated_code.pyscf_integrals import compute_integrals

    assert legacy_residuals.compute_r1 is canonical_residuals.compute_r1
    assert legacy_residuals.compute_r2 is canonical_residuals.compute_r2
    assert compute_energy is canonical_energy
    assert compute_integrals is canonical_integrals


def test_generated_code_eom_forwarders_are_thin() -> None:
    from autogen.methods.eom_ccsd.generated import residuals as canonical_residuals
    from autogen.methods.eom_ccsd.runtime.solver import solve_eom_ccsd as canonical_solver
    from generated_code.methods.eom_ccsd import residuals as legacy_residuals
    from generated_code.methods.eom_ccsd.eom_solver import solve_eom_ccsd

    assert legacy_residuals.compute_s1 is canonical_residuals.compute_s1
    assert legacy_residuals.compute_s2 is canonical_residuals.compute_s2
    assert solve_eom_ccsd is canonical_solver
