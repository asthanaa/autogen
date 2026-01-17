from autogen.main_tools.commutator import comm
from autogen.library.full_con import full_terms


def test_commutator_v2_t2_smoke():
    terms = comm(['V2'], ['T2'], 1)
    assert isinstance(terms, list)
    assert len(terms) > 0

    contracted = full_terms(terms)
    assert isinstance(contracted, list)
