from autogen.library.full_con import full_terms
from autogen.main_tools.commutator import comm
from autogen.main_tools.product import prod


def _assert_terms_look_reasonable(terms):
    assert isinstance(terms, list)
    assert len(terms) > 0

    # Lightweight structural sanity checks (avoid over-specifying exact algebra).
    t0 = terms[0]
    assert hasattr(t0, "st")
    assert hasattr(t0, "fac")
    assert hasattr(t0, "co")


def test_commutator_v2_d1_smoke():
    terms = comm(['V2'], ['D1'], last=1)
    _assert_terms_look_reasonable(terms)

    contracted = full_terms(terms)
    assert isinstance(contracted, list)


def test_commutator_v2_d2_smoke():
    terms = comm(['V2'], ['D2'], last=1)
    _assert_terms_look_reasonable(terms)

    contracted = full_terms(terms)
    assert isinstance(contracted, list)


def test_nested_commutator_v2_d1_t1_smoke():
    inner = comm(['V2'], ['D1'], last=0)
    outer = comm(inner, ['T1'], last=1)
    _assert_terms_look_reasonable(outer)


def test_product_x1_with_commutator_v2_d1_smoke():
    vt_d1 = comm(['V2'], ['D1'], last=0)
    terms = prod(['X1'], vt_d1, last=1)
    _assert_terms_look_reasonable(terms)
