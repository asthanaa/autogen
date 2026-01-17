from dataclasses import dataclass

from autogen.library.convert_pqr import change_sum, delete_op_from_sum, ia_limit, pqr_present


@dataclass
class _DummyOp:
    kind: str
    upper: list[str]
    lower: list[str]


class _DummyTerm:
    def __init__(self, coeff_list, sum_list, dict_ind, op=None):
        self.coeff_list = coeff_list
        self.sum_list = sum_list
        self.dict_ind = dict_ind
        # st layout used by delete_op_from_sum/change_op
        self.st = [[op]] if op is not None else [[[ _DummyOp('x', [], []) ]]]

    @staticmethod
    def isa(ch: str) -> int:
        return 1 if ('a' <= ch <= 'h') else 0

    @staticmethod
    def isi(ch: str) -> int:
        return 1 if ('i' <= ch <= 'n') else 0


def test_pqr_present_detects_general_indices():
    term = _DummyTerm(coeff_list=[["p0", "i0"]], sum_list=["p0"], dict_ind={"p": object()})
    assert pqr_present(term) == 1

    term2 = _DummyTerm(coeff_list=[["a0", "i0"]], sum_list=["a0"], dict_ind={"a": object()})
    assert pqr_present(term2) == 0


def test_ia_limit_counts_types_from_dict_ind_keys():
    term = _DummyTerm(coeff_list=[], sum_list=[], dict_ind={"a": 1, "b": 2, "i": 3, "p": 4, "q": 5})
    i, a, p = ia_limit(term)
    assert (i, a, p) == (1, 2, 2)


def test_change_sum_replaces_one_symbol():
    term = _DummyTerm(coeff_list=[], sum_list=["p0", "i0"], dict_ind={})
    change_sum(term, "p0", "i0")
    assert term.sum_list == ["i0", "i0"]


def test_delete_op_from_sum_removes_op_indices():
    op = _DummyOp(kind='op', upper=['a0'], lower=['i0'])
    term = _DummyTerm(coeff_list=[], sum_list=['a0', 'i0', 'p0'], dict_ind={}, op=op)
    delete_op_from_sum(term)
    assert term.sum_list == ['p0']
