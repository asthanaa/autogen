import pytest

from autogen.pkg.parity import parity, permutation_parity


def test_permutation_parity_examples():
    assert permutation_parity([0, 1, 2]) == 0
    assert permutation_parity([0, 2, 1]) == 1
    assert permutation_parity([1, 0, 2]) == 1
    assert permutation_parity([1, 2, 0]) == 0


def test_relative_parity_examples():
    assert parity([0, 1], [0, 1]) == 0
    assert parity([0, 1], [1, 0]) == 1
    assert parity(["A", "B", "C"], ["A", "C", "B"]) == 1


def test_parity_rejects_invalid():
    with pytest.raises(ValueError):
        parity([0, 1], [0])
    with pytest.raises(ValueError):
        parity([0, 0], [0, 0])
    with pytest.raises(ValueError):
        parity([0, 1], [0, 2])
