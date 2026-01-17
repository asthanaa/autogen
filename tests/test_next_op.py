from autogen.library.next_op import next_op


def test_next_op_i_wraps():
    # i has 6 letters: i..n
    assert next_op('i', [0, 0, 0], 0) == 'i'
    assert next_op('i', [5, 0, 0], 0) == 'n'
    assert next_op('i', [6, 0, 0], 0) == 'i1'


def test_next_op_a_wraps():
    # a has 8 letters: a..h
    assert next_op('a', [0, 0, 0], 0) == 'a'
    assert next_op('a', [0, 7, 0], 0) == 'h'
    assert next_op('a', [0, 8, 0], 0) == 'a1'


def test_next_op_p_wraps():
    # p has 5 letters: p..t
    assert next_op('p', [0, 0, 0], 0) == 'p'
    assert next_op('p', [0, 0, 4], 0) == 't'
    assert next_op('p', [0, 0, 5], 0) == 'p1'
