from autogen.library.full_con import full_terms, full_con


class _DummyOp:
    def __init__(self, kind):
        self.kind = kind


class _DummyTerm:
    def __init__(self, has_op: bool):
        self.st = [[_DummyOp('op' if has_op else 'x')]]


def test_full_terms_filters_terms_with_op():
    terms = [_DummyTerm(has_op=True), _DummyTerm(has_op=False), _DummyTerm(has_op=True)]
    out = full_terms(terms)
    assert len(out) == 1


def test_full_con_filters_operator_strings_and_constants():
    st = [[_DummyOp('x')], [_DummyOp('op')], [_DummyOp('x')]]
    co = [1.0, 2.0, 3.0]
    out_st, out_co = full_con(st, co)
    assert out_co == [1.0, 3.0]
    assert len(out_st) == 2
