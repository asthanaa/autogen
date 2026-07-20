from __future__ import annotations


def test_package_root_exposes_only_the_reviewed_production_workflow() -> None:
    import autogen.methods.qpccsd as qpccsd

    assert set(qpccsd.__all__) == {
        "ProductionConfig",
        "ProductionResult",
        "prepare_projected_agp_reference",
        "build_full_active_qp_space",
        "solve_direct_qpccsd",
        "evaluate_direct_pav",
        "run_qpccsd_pav",
    }


def test_experimental_routes_are_not_available_from_the_package_root() -> None:
    import autogen.methods.qpccsd as qpccsd

    experimental_names = {
        "build_cas_contracted_reference",
        "build_external_qp_space",
        "build_paired_hfb",
        "evaluate_projected_qpccsd",
        "RichardsonProjectedQPCCSDEvaluator",
        "solve_cas_contracted_ccsd",
        "solve_projected_qpccsd",
        "solve_projected_lbccsd",
    }
    assert experimental_names.isdisjoint(qpccsd.__all__)
    for name in experimental_names:
        assert not hasattr(qpccsd, name)


def test_legacy_package_is_a_thin_production_api_shim() -> None:
    import autogen.methods.qpccsd as canonical
    import autogen.qpccsd as compatibility

    assert compatibility.__all__ == canonical.__all__
    for name in canonical.__all__:
        assert getattr(compatibility, name) is getattr(canonical, name)


def test_documented_legacy_submodules_forward_without_implementation_copies() -> None:
    from autogen.methods.qpccsd.cli import main as canonical_main
    from autogen.methods.qpccsd.production.cli import main as implementation_main
    from autogen.methods.qpccsd.production.contracts import (
        PRODUCTION_RESULT_SCHEMA as canonical_schema,
    )
    from autogen.methods.qpccsd.production.io import write_json_atomic as canonical_write
    from autogen.methods.qpccsd.production.models import QPAmplitudes as canonical_amplitudes
    from autogen.methods.qpccsd.production.workflow import (
        ProductionConfig as canonical_config,
    )
    from autogen.qpccsd.cli import main as legacy_main
    from autogen.qpccsd.contracts import PRODUCTION_RESULT_SCHEMA as legacy_schema
    from autogen.qpccsd.io import write_json_atomic as legacy_write
    from autogen.qpccsd.models import QPAmplitudes as legacy_amplitudes
    from autogen.qpccsd.workflow import ProductionConfig as legacy_config

    assert legacy_main is canonical_main
    assert canonical_main is implementation_main
    assert legacy_schema == canonical_schema
    assert legacy_write is canonical_write
    assert legacy_amplitudes is canonical_amplitudes
    assert legacy_config is canonical_config
