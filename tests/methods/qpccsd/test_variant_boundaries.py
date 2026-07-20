from __future__ import annotations


def test_unimplemented_variant_boundaries_are_explicitly_disabled() -> None:
    from autogen.methods.qpccsd.variants import ENABLED_BY_DEFAULT
    from autogen.methods.qpccsd.variants import cubic, qiu_u1, sokolov_chan

    assert ENABLED_BY_DEFAULT is False
    assert cubic.ENABLED_BY_DEFAULT is False
    assert cubic.IMPLEMENTED is False
    assert qiu_u1.ENABLED_BY_DEFAULT is False
    assert qiu_u1.IMPLEMENTED is False
    assert sokolov_chan.ENABLED_BY_DEFAULT is False
    assert sokolov_chan.REFERENCE_HELPER_IMPLEMENTED is True
    assert sokolov_chan.ENERGY_ROUTE_IMPLEMENTED is False


def test_implemented_test_routes_still_require_explicit_imports() -> None:
    from autogen.methods.qpccsd.variants import cas_plus_delta, masked_active, oap

    assert cas_plus_delta.ENABLED_BY_DEFAULT is False
    assert masked_active.ENABLED_BY_DEFAULT is False
    assert oap.ENABLED_BY_DEFAULT is False
