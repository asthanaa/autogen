"""Generated antisymmetric-orbit QPCCSD contractions; do not edit."""

from __future__ import annotations

import numpy as np

_BINARY_EINSUM_PATH = ('einsum_path', (0, 1))
_PAIR_INDEX_CACHE = {}
CONTRACTION_BACKEND = 'pair-packed-preplanned-numpy-blas'
TENSOR_NAMES = ('t1', 't2', 'h02', 'h04', 'h11', 'h13', 'h20', 'h22', 'h31', 'h40')
MAX_FORMAL_SCALING = 6
CONTRACTION_COUNT = 84
EXECUTED_CONTRACTION_COUNT = 42
GLOBAL_CSE_INTERMEDIATE_COUNT = 11
MAX_LIVE_GLOBAL_CSE = 8
MAX_LIVE_RANK4_CSE = 5
PAIR_ORBIT_COUNT = 10
QUADRUPLE_ORBIT_COUNT = 14

def _antisymmetrize_pair(tensor):
    return tensor - tensor.T

def _antisymmetrize_rank4(tensor):
    pair_antisymmetric = tensor - tensor.swapaxes(0, 1)
    pair_antisymmetric = pair_antisymmetric - pair_antisymmetric.swapaxes(2, 3)
    return (
        pair_antisymmetric
        - pair_antisymmetric.transpose((0, 2, 1, 3))
        + pair_antisymmetric.transpose((0, 2, 3, 1))
        + pair_antisymmetric.transpose((2, 0, 1, 3))
        - pair_antisymmetric.transpose((2, 0, 3, 1))
        + pair_antisymmetric.transpose((2, 3, 0, 1))
    )

def _pair_pair_contract(left, right):
    dimension = left.shape[0]
    indices = _PAIR_INDEX_CACHE.get(dimension)
    if indices is None:
        indices = np.triu_indices(dimension, 1)
        _PAIR_INDEX_CACHE[dimension] = indices
    first, second = indices
    rows = first[:, None]
    columns = second[:, None]
    packed_left = left[rows, columns, first[None, :], second[None, :]]
    packed_right = right[rows, columns, first[None, :], second[None, :]]
    packed = 2.0 * (packed_left @ packed_right.T)
    result = np.zeros((dimension,) * 4, dtype=np.result_type(left, right))
    result[rows, columns, first[None, :], second[None, :]] = packed
    result[columns, rows, first[None, :], second[None, :]] = -packed
    result[rows, columns, second[None, :], first[None, :]] = -packed
    result[columns, rows, second[None, :], first[None, :]] = packed
    return result

def compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    h02 = np.asarray(h02)
    h04 = np.asarray(h04)
    h11 = np.asarray(h11)
    h13 = np.asarray(h13)
    h20 = np.asarray(h20)
    h22 = np.asarray(h22)
    h31 = np.asarray(h31)
    h40 = np.asarray(h40)
    dtype = np.result_type(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40, np.complex128)
    energy = np.array(0.0 + 0.0j, dtype=dtype)
    r1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)
    r2 = np.zeros((t1.shape[0],) * 4, dtype=dtype)
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.5) * _s0
    del _s0
    _cse0 = np.einsum('abcd,ab->cd', h04, t1, optimize=_BINARY_EINSUM_PATH)
    _s1 = np.einsum('ab,ab->', _cse0, t1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.125) * _s1
    del _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=_BINARY_EINSUM_PATH)
    energy += (0.041666666666666664) * _s0
    del _s0
    _cse1 = np.einsum('ab->ba', t1, optimize=True)
    _cse2 = np.einsum('ab,ca->cb', h02, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cb->ac', _cse2, _cse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-1.0) * _s3
    del _s3
    _cse3 = np.einsum('abcd->cdab', t2, optimize=True)
    _s1 = np.einsum('ab,cdab->cd', h02, _cse3, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.5) * _s1
    del _s1
    _cse4 = np.einsum('ab,ca->cb', _cse0, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s4 = np.einsum('ab,cb->ac', _cse4, _cse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.5) * _s4
    del _s4
    _s2 = np.einsum('ab,cdab->cd', _cse0, _cse3, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.25) * _s2
    del _s2
    _cse5 = np.einsum('abcd->dabc', t2, optimize=True)
    _s1 = np.einsum('abcd,ebcd->ea', h04, _cse5, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cb->ca', _s1, _cse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.16666666666666666) * _s3
    r1 += (0.16666666666666666) * _s3.transpose((1, 0))
    del _s1, _s3
    _s1 = np.einsum('ab,cb->ac', h11, _cse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (1.0) * _s1
    r1 += (-1.0) * _s1.transpose((1, 0))
    del _s1
    _cse6 = np.einsum('abcd,bc->ad', h13, t1, optimize=_BINARY_EINSUM_PATH)
    _s2 = np.einsum('ab,cb->ac', _cse6, _cse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.5) * _s2
    r1 += (0.5) * _s2.transpose((1, 0))
    del _s2
    _s1 = np.einsum('abcd,ebcd->ae', h13, _cse5, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.16666666666666666) * _s1
    r1 += (0.16666666666666666) * _s1.transpose((1, 0))
    del _s1
    r1 += (1.0) * h20
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.5) * _s0
    del _s0
    _cse7 = np.einsum('abcd->bcda', t2, optimize=True)
    _s3 = np.einsum('ab,cdeb->acde', _cse2, _cse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-1.0) * _s3
    r2 += (1.0) * _s3.transpose((1, 0, 2, 3))
    r2 += (-1.0) * _s3.transpose((1, 2, 0, 3))
    r2 += (1.0) * _s3.transpose((1, 2, 3, 0))
    del _s3
    del _cse2
    _s4 = np.einsum('ab,cdeb->acde', _cse4, _cse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s4
    r2 += (0.5) * _s4.transpose((1, 0, 2, 3))
    r2 += (-0.5) * _s4.transpose((1, 2, 0, 3))
    r2 += (0.5) * _s4.transpose((1, 2, 3, 0))
    del _s4
    del _cse0, _cse4
    _cse8 = np.einsum('abcd,ea->ebcd', h04, _cse1, optimize=_BINARY_EINSUM_PATH)
    _cse9 = np.einsum('abcd,eb->aecd', _cse8, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s5 = np.einsum('abcd,ec->abed', _cse9, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s7 = np.einsum('abcd,ed->abce', _s5, _cse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s7
    del _s5, _s7
    _s5 = np.einsum('abcd,efcd->abef', _cse9, _cse3, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s5
    r2 += (0.5) * _s5.transpose((0, 2, 1, 3))
    r2 += (-0.5) * _s5.transpose((0, 2, 3, 1))
    r2 += (-0.5) * _s5.transpose((2, 0, 1, 3))
    r2 += (0.5) * _s5.transpose((2, 0, 3, 1))
    r2 += (-0.5) * _s5.transpose((2, 3, 0, 1))
    del _s5
    del _cse8, _cse9
    _s1 = np.einsum('abcd,eabc->ed', h04, _cse5, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cdeb->acde', _s1, _cse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.16666666666666666) * _s3
    r2 += (0.16666666666666666) * _s3.transpose((1, 0, 2, 3))
    r2 += (-0.16666666666666666) * _s3.transpose((1, 2, 0, 3))
    r2 += (0.16666666666666666) * _s3.transpose((1, 2, 3, 0))
    del _s1, _s3
    del _cse5
    _s1 = _pair_pair_contract(_cse3, h04)
    _s3 = _pair_pair_contract(_s1, _cse3)
    r2 += (0.25) * _s3
    r2 += (-0.25) * _s3.transpose((0, 2, 1, 3))
    r2 += (0.25) * _s3.transpose((0, 2, 3, 1))
    del _s1, _s3
    _s1 = np.einsum('ab,cdeb->acde', h11, _cse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s1
    r2 += (-1.0) * _s1.transpose((1, 0, 2, 3))
    r2 += (1.0) * _s1.transpose((1, 2, 0, 3))
    r2 += (-1.0) * _s1.transpose((1, 2, 3, 0))
    del _s1
    _s2 = np.einsum('ab,cdeb->acde', _cse6, _cse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s2
    r2 += (0.5) * _s2.transpose((1, 0, 2, 3))
    r2 += (-0.5) * _s2.transpose((1, 2, 0, 3))
    r2 += (0.5) * _s2.transpose((1, 2, 3, 0))
    del _s2
    del _cse6, _cse7
    _cse10 = np.einsum('abcd,eb->aecd', h13, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('abcd,ec->abed', _cse10, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s5 = np.einsum('abcd,ed->abce', _s3, _cse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s5
    r2 += (-1.0) * _s5.transpose((1, 0, 2, 3))
    r2 += (1.0) * _s5.transpose((1, 2, 0, 3))
    r2 += (-1.0) * _s5.transpose((1, 2, 3, 0))
    del _s3, _s5
    _s3 = np.einsum('abcd,efcd->abef', _cse10, _cse3, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.25) * _antisymmetrize_rank4(_s3)
    del _s3
    del _cse10
    _s1 = np.einsum('abcd,ec->abed', h22, _cse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('abcd,ed->abce', _s1, _cse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (-1.0) * _s3
    r2 += (1.0) * _s3.transpose((0, 2, 1, 3))
    r2 += (-1.0) * _s3.transpose((0, 2, 3, 1))
    r2 += (-1.0) * _s3.transpose((2, 0, 1, 3))
    r2 += (1.0) * _s3.transpose((2, 0, 3, 1))
    r2 += (-1.0) * _s3.transpose((2, 3, 0, 1))
    del _s1, _s3
    _s1 = _pair_pair_contract(h22, _cse3)
    r2 += (0.5) * _s1
    r2 += (-0.5) * _s1.transpose((0, 2, 1, 3))
    r2 += (0.5) * _s1.transpose((0, 2, 3, 1))
    r2 += (0.5) * _s1.transpose((2, 0, 1, 3))
    r2 += (-0.5) * _s1.transpose((2, 0, 3, 1))
    r2 += (0.5) * _s1.transpose((2, 3, 0, 1))
    del _s1
    del _cse3
    _s1 = np.einsum('abcd,ed->abce', h31, _cse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s1
    r2 += (-1.0) * _s1.transpose((0, 1, 3, 2))
    r2 += (1.0) * _s1.transpose((0, 3, 1, 2))
    r2 += (-1.0) * _s1.transpose((3, 0, 1, 2))
    del _s1
    del _cse1
    r2 += (1.0) * h40
    return {
        'energy': np.real_if_close(energy),
        'r1': np.real_if_close(r1),
        'r2': np.real_if_close(r2),
    }

def compute_outputs_and_jvp(t1, t2, dt1, dt2, h02, h04, h11, h13, h20, h22, h31, h40):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    dt1 = np.asarray(dt1)
    dt2 = np.asarray(dt2)
    h02 = np.asarray(h02)
    h04 = np.asarray(h04)
    h11 = np.asarray(h11)
    h13 = np.asarray(h13)
    h20 = np.asarray(h20)
    h22 = np.asarray(h22)
    h31 = np.asarray(h31)
    h40 = np.asarray(h40)
    dtype = np.result_type(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40, dt1, dt2, np.complex128)
    energy = np.array(0.0 + 0.0j, dtype=dtype)
    r1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)
    r2 = np.zeros((t1.shape[0],) * 4, dtype=dtype)
    energy_jvp = np.array(0.0 + 0.0j, dtype=dtype)
    r1_jvp = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)
    r2_jvp = np.zeros((t1.shape[0],) * 4, dtype=dtype)
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=_BINARY_EINSUM_PATH)
    _ds0 = np.einsum('ab,ab->', h02, dt1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.5) * _s0
    energy_jvp += (0.5) * _ds0
    del _s0, _ds0
    _cse0 = np.einsum('abcd,ab->cd', h04, t1, optimize=_BINARY_EINSUM_PATH)
    _dcse0 = np.einsum('abcd,ab->cd', h04, dt1, optimize=_BINARY_EINSUM_PATH)
    _s1 = np.einsum('ab,ab->', _cse0, t1, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('ab,ab->', _dcse0, t1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,ab->', _cse0, dt1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.125) * _s1
    energy_jvp += (0.125) * _ds1
    del _s1, _ds1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=_BINARY_EINSUM_PATH)
    _ds0 = np.einsum('abcd,abcd->', h04, dt2, optimize=_BINARY_EINSUM_PATH)
    energy += (0.041666666666666664) * _s0
    energy_jvp += (0.041666666666666664) * _ds0
    del _s0, _ds0
    _cse1 = np.einsum('ab->ba', t1, optimize=True)
    _dcse1 = np.einsum('ab->ba', dt1, optimize=True)
    _cse2 = np.einsum('ab,ca->cb', h02, _cse1, optimize=_BINARY_EINSUM_PATH)
    _dcse2 = np.einsum('ab,ca->cb', h02, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cb->ac', _cse2, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('ab,cb->ac', _dcse2, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cb->ac', _cse2, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-1.0) * _s3
    r1_jvp += (-1.0) * _ds3
    del _s3, _ds3
    _cse3 = np.einsum('abcd->cdab', t2, optimize=True)
    _dcse3 = np.einsum('abcd->cdab', dt2, optimize=True)
    _s1 = np.einsum('ab,cdab->cd', h02, _cse3, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('ab,cdab->cd', h02, _dcse3, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.5) * _s1
    r1_jvp += (0.5) * _ds1
    del _s1, _ds1
    _cse4 = np.einsum('ab,ca->cb', _cse0, _cse1, optimize=_BINARY_EINSUM_PATH)
    _dcse4 = np.einsum('ab,ca->cb', _dcse0, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,ca->cb', _cse0, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s4 = np.einsum('ab,cb->ac', _cse4, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds4 = np.einsum('ab,cb->ac', _dcse4, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cb->ac', _cse4, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.5) * _s4
    r1_jvp += (-0.5) * _ds4
    del _s4, _ds4
    _s2 = np.einsum('ab,cdab->cd', _cse0, _cse3, optimize=_BINARY_EINSUM_PATH)
    _ds2 = np.einsum('ab,cdab->cd', _dcse0, _cse3, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cdab->cd', _cse0, _dcse3, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.25) * _s2
    r1_jvp += (0.25) * _ds2
    del _s2, _ds2
    _cse5 = np.einsum('abcd->dabc', t2, optimize=True)
    _dcse5 = np.einsum('abcd->dabc', dt2, optimize=True)
    _s1 = np.einsum('abcd,ebcd->ea', h04, _cse5, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('abcd,ebcd->ea', h04, _dcse5, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cb->ca', _s1, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('ab,cb->ca', _ds1, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cb->ca', _s1, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.16666666666666666) * _s3
    r1_jvp += (-0.16666666666666666) * _ds3
    r1 += (0.16666666666666666) * _s3.transpose((1, 0))
    r1_jvp += (0.16666666666666666) * _ds3.transpose((1, 0))
    del _s1, _s3, _ds1, _ds3
    _s1 = np.einsum('ab,cb->ac', h11, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('ab,cb->ac', h11, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (1.0) * _s1
    r1_jvp += (1.0) * _ds1
    r1 += (-1.0) * _s1.transpose((1, 0))
    r1_jvp += (-1.0) * _ds1.transpose((1, 0))
    del _s1, _ds1
    _cse6 = np.einsum('abcd,bc->ad', h13, t1, optimize=_BINARY_EINSUM_PATH)
    _dcse6 = np.einsum('abcd,bc->ad', h13, dt1, optimize=_BINARY_EINSUM_PATH)
    _s2 = np.einsum('ab,cb->ac', _cse6, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds2 = np.einsum('ab,cb->ac', _dcse6, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cb->ac', _cse6, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.5) * _s2
    r1_jvp += (-0.5) * _ds2
    r1 += (0.5) * _s2.transpose((1, 0))
    r1_jvp += (0.5) * _ds2.transpose((1, 0))
    del _s2, _ds2
    _s1 = np.einsum('abcd,ebcd->ae', h13, _cse5, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('abcd,ebcd->ae', h13, _dcse5, optimize=_BINARY_EINSUM_PATH)
    r1 += (-0.16666666666666666) * _s1
    r1_jvp += (-0.16666666666666666) * _ds1
    r1 += (0.16666666666666666) * _s1.transpose((1, 0))
    r1_jvp += (0.16666666666666666) * _ds1.transpose((1, 0))
    del _s1, _ds1
    r1 += (1.0) * h20
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=_BINARY_EINSUM_PATH)
    _ds0 = np.einsum('abcd,cd->ab', h22, dt1, optimize=_BINARY_EINSUM_PATH)
    r1 += (0.5) * _s0
    r1_jvp += (0.5) * _ds0
    del _s0, _ds0
    _cse7 = np.einsum('abcd->bcda', t2, optimize=True)
    _dcse7 = np.einsum('abcd->bcda', dt2, optimize=True)
    _s3 = np.einsum('ab,cdeb->acde', _cse2, _cse7, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('ab,cdeb->acde', _dcse2, _cse7, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cdeb->acde', _cse2, _dcse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-1.0) * _s3
    r2_jvp += (-1.0) * _ds3
    r2 += (1.0) * _s3.transpose((1, 0, 2, 3))
    r2_jvp += (1.0) * _ds3.transpose((1, 0, 2, 3))
    r2 += (-1.0) * _s3.transpose((1, 2, 0, 3))
    r2_jvp += (-1.0) * _ds3.transpose((1, 2, 0, 3))
    r2 += (1.0) * _s3.transpose((1, 2, 3, 0))
    r2_jvp += (1.0) * _ds3.transpose((1, 2, 3, 0))
    del _s3, _ds3
    del _cse2, _dcse2
    _s4 = np.einsum('ab,cdeb->acde', _cse4, _cse7, optimize=_BINARY_EINSUM_PATH)
    _ds4 = np.einsum('ab,cdeb->acde', _dcse4, _cse7, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cdeb->acde', _cse4, _dcse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s4
    r2_jvp += (-0.5) * _ds4
    r2 += (0.5) * _s4.transpose((1, 0, 2, 3))
    r2_jvp += (0.5) * _ds4.transpose((1, 0, 2, 3))
    r2 += (-0.5) * _s4.transpose((1, 2, 0, 3))
    r2_jvp += (-0.5) * _ds4.transpose((1, 2, 0, 3))
    r2 += (0.5) * _s4.transpose((1, 2, 3, 0))
    r2_jvp += (0.5) * _ds4.transpose((1, 2, 3, 0))
    del _s4, _ds4
    del _cse0, _cse4, _dcse0, _dcse4
    _cse8 = np.einsum('abcd,ea->ebcd', h04, _cse1, optimize=_BINARY_EINSUM_PATH)
    _dcse8 = np.einsum('abcd,ea->ebcd', h04, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _cse9 = np.einsum('abcd,eb->aecd', _cse8, _cse1, optimize=_BINARY_EINSUM_PATH)
    _dcse9 = np.einsum('abcd,eb->aecd', _dcse8, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,eb->aecd', _cse8, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s5 = np.einsum('abcd,ec->abed', _cse9, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds5 = np.einsum('abcd,ec->abed', _dcse9, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,ec->abed', _cse9, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s7 = np.einsum('abcd,ed->abce', _s5, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds7 = np.einsum('abcd,ed->abce', _ds5, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,ed->abce', _s5, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s7
    r2_jvp += (1.0) * _ds7
    del _s5, _s7, _ds5, _ds7
    _s5 = np.einsum('abcd,efcd->abef', _cse9, _cse3, optimize=_BINARY_EINSUM_PATH)
    _ds5 = np.einsum('abcd,efcd->abef', _dcse9, _cse3, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,efcd->abef', _cse9, _dcse3, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s5
    r2_jvp += (-0.5) * _ds5
    r2 += (0.5) * _s5.transpose((0, 2, 1, 3))
    r2_jvp += (0.5) * _ds5.transpose((0, 2, 1, 3))
    r2 += (-0.5) * _s5.transpose((0, 2, 3, 1))
    r2_jvp += (-0.5) * _ds5.transpose((0, 2, 3, 1))
    r2 += (-0.5) * _s5.transpose((2, 0, 1, 3))
    r2_jvp += (-0.5) * _ds5.transpose((2, 0, 1, 3))
    r2 += (0.5) * _s5.transpose((2, 0, 3, 1))
    r2_jvp += (0.5) * _ds5.transpose((2, 0, 3, 1))
    r2 += (-0.5) * _s5.transpose((2, 3, 0, 1))
    r2_jvp += (-0.5) * _ds5.transpose((2, 3, 0, 1))
    del _s5, _ds5
    del _cse8, _cse9, _dcse8, _dcse9
    _s1 = np.einsum('abcd,eabc->ed', h04, _cse5, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('abcd,eabc->ed', h04, _dcse5, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('ab,cdeb->acde', _s1, _cse7, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('ab,cdeb->acde', _ds1, _cse7, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cdeb->acde', _s1, _dcse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.16666666666666666) * _s3
    r2_jvp += (-0.16666666666666666) * _ds3
    r2 += (0.16666666666666666) * _s3.transpose((1, 0, 2, 3))
    r2_jvp += (0.16666666666666666) * _ds3.transpose((1, 0, 2, 3))
    r2 += (-0.16666666666666666) * _s3.transpose((1, 2, 0, 3))
    r2_jvp += (-0.16666666666666666) * _ds3.transpose((1, 2, 0, 3))
    r2 += (0.16666666666666666) * _s3.transpose((1, 2, 3, 0))
    r2_jvp += (0.16666666666666666) * _ds3.transpose((1, 2, 3, 0))
    del _s1, _s3, _ds1, _ds3
    del _cse5, _dcse5
    _s1 = _pair_pair_contract(_cse3, h04)
    _ds1 = _pair_pair_contract(_dcse3, h04)
    _s3 = _pair_pair_contract(_s1, _cse3)
    _ds3 = _pair_pair_contract(_ds1, _cse3) + _pair_pair_contract(_s1, _dcse3)
    r2 += (0.25) * _s3
    r2_jvp += (0.25) * _ds3
    r2 += (-0.25) * _s3.transpose((0, 2, 1, 3))
    r2_jvp += (-0.25) * _ds3.transpose((0, 2, 1, 3))
    r2 += (0.25) * _s3.transpose((0, 2, 3, 1))
    r2_jvp += (0.25) * _ds3.transpose((0, 2, 3, 1))
    del _s1, _s3, _ds1, _ds3
    _s1 = np.einsum('ab,cdeb->acde', h11, _cse7, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('ab,cdeb->acde', h11, _dcse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s1
    r2_jvp += (1.0) * _ds1
    r2 += (-1.0) * _s1.transpose((1, 0, 2, 3))
    r2_jvp += (-1.0) * _ds1.transpose((1, 0, 2, 3))
    r2 += (1.0) * _s1.transpose((1, 2, 0, 3))
    r2_jvp += (1.0) * _ds1.transpose((1, 2, 0, 3))
    r2 += (-1.0) * _s1.transpose((1, 2, 3, 0))
    r2_jvp += (-1.0) * _ds1.transpose((1, 2, 3, 0))
    del _s1, _ds1
    _s2 = np.einsum('ab,cdeb->acde', _cse6, _cse7, optimize=_BINARY_EINSUM_PATH)
    _ds2 = np.einsum('ab,cdeb->acde', _dcse6, _cse7, optimize=_BINARY_EINSUM_PATH) + np.einsum('ab,cdeb->acde', _cse6, _dcse7, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.5) * _s2
    r2_jvp += (-0.5) * _ds2
    r2 += (0.5) * _s2.transpose((1, 0, 2, 3))
    r2_jvp += (0.5) * _ds2.transpose((1, 0, 2, 3))
    r2 += (-0.5) * _s2.transpose((1, 2, 0, 3))
    r2_jvp += (-0.5) * _ds2.transpose((1, 2, 0, 3))
    r2 += (0.5) * _s2.transpose((1, 2, 3, 0))
    r2_jvp += (0.5) * _ds2.transpose((1, 2, 3, 0))
    del _s2, _ds2
    del _cse6, _cse7, _dcse6, _dcse7
    _cse10 = np.einsum('abcd,eb->aecd', h13, _cse1, optimize=_BINARY_EINSUM_PATH)
    _dcse10 = np.einsum('abcd,eb->aecd', h13, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('abcd,ec->abed', _cse10, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('abcd,ec->abed', _dcse10, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,ec->abed', _cse10, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s5 = np.einsum('abcd,ed->abce', _s3, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds5 = np.einsum('abcd,ed->abce', _ds3, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,ed->abce', _s3, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s5
    r2_jvp += (1.0) * _ds5
    r2 += (-1.0) * _s5.transpose((1, 0, 2, 3))
    r2_jvp += (-1.0) * _ds5.transpose((1, 0, 2, 3))
    r2 += (1.0) * _s5.transpose((1, 2, 0, 3))
    r2_jvp += (1.0) * _ds5.transpose((1, 2, 0, 3))
    r2 += (-1.0) * _s5.transpose((1, 2, 3, 0))
    r2_jvp += (-1.0) * _ds5.transpose((1, 2, 3, 0))
    del _s3, _s5, _ds3, _ds5
    _s3 = np.einsum('abcd,efcd->abef', _cse10, _cse3, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('abcd,efcd->abef', _dcse10, _cse3, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,efcd->abef', _cse10, _dcse3, optimize=_BINARY_EINSUM_PATH)
    r2 += (-0.25) * _antisymmetrize_rank4(_s3)
    r2_jvp += (-0.25) * _antisymmetrize_rank4(_ds3)
    del _s3, _ds3
    del _cse10, _dcse10
    _s1 = np.einsum('abcd,ec->abed', h22, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('abcd,ec->abed', h22, _dcse1, optimize=_BINARY_EINSUM_PATH)
    _s3 = np.einsum('abcd,ed->abce', _s1, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds3 = np.einsum('abcd,ed->abce', _ds1, _cse1, optimize=_BINARY_EINSUM_PATH) + np.einsum('abcd,ed->abce', _s1, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (-1.0) * _s3
    r2_jvp += (-1.0) * _ds3
    r2 += (1.0) * _s3.transpose((0, 2, 1, 3))
    r2_jvp += (1.0) * _ds3.transpose((0, 2, 1, 3))
    r2 += (-1.0) * _s3.transpose((0, 2, 3, 1))
    r2_jvp += (-1.0) * _ds3.transpose((0, 2, 3, 1))
    r2 += (-1.0) * _s3.transpose((2, 0, 1, 3))
    r2_jvp += (-1.0) * _ds3.transpose((2, 0, 1, 3))
    r2 += (1.0) * _s3.transpose((2, 0, 3, 1))
    r2_jvp += (1.0) * _ds3.transpose((2, 0, 3, 1))
    r2 += (-1.0) * _s3.transpose((2, 3, 0, 1))
    r2_jvp += (-1.0) * _ds3.transpose((2, 3, 0, 1))
    del _s1, _s3, _ds1, _ds3
    _s1 = _pair_pair_contract(h22, _cse3)
    _ds1 = _pair_pair_contract(h22, _dcse3)
    r2 += (0.5) * _s1
    r2_jvp += (0.5) * _ds1
    r2 += (-0.5) * _s1.transpose((0, 2, 1, 3))
    r2_jvp += (-0.5) * _ds1.transpose((0, 2, 1, 3))
    r2 += (0.5) * _s1.transpose((0, 2, 3, 1))
    r2_jvp += (0.5) * _ds1.transpose((0, 2, 3, 1))
    r2 += (0.5) * _s1.transpose((2, 0, 1, 3))
    r2_jvp += (0.5) * _ds1.transpose((2, 0, 1, 3))
    r2 += (-0.5) * _s1.transpose((2, 0, 3, 1))
    r2_jvp += (-0.5) * _ds1.transpose((2, 0, 3, 1))
    r2 += (0.5) * _s1.transpose((2, 3, 0, 1))
    r2_jvp += (0.5) * _ds1.transpose((2, 3, 0, 1))
    del _s1, _ds1
    del _cse3, _dcse3
    _s1 = np.einsum('abcd,ed->abce', h31, _cse1, optimize=_BINARY_EINSUM_PATH)
    _ds1 = np.einsum('abcd,ed->abce', h31, _dcse1, optimize=_BINARY_EINSUM_PATH)
    r2 += (1.0) * _s1
    r2_jvp += (1.0) * _ds1
    r2 += (-1.0) * _s1.transpose((0, 1, 3, 2))
    r2_jvp += (-1.0) * _ds1.transpose((0, 1, 3, 2))
    r2 += (1.0) * _s1.transpose((0, 3, 1, 2))
    r2_jvp += (1.0) * _ds1.transpose((0, 3, 1, 2))
    r2 += (-1.0) * _s1.transpose((3, 0, 1, 2))
    r2_jvp += (-1.0) * _ds1.transpose((3, 0, 1, 2))
    del _s1, _ds1
    del _cse1, _dcse1
    r2 += (1.0) * h40
    return {
        'energy': np.real_if_close(energy),
        'r1': np.real_if_close(r1),
        'r2': np.real_if_close(r2),
        'energy_jvp': np.real_if_close(energy_jvp),
        'r1_jvp': np.real_if_close(r1_jvp),
        'r2_jvp': np.real_if_close(r2_jvp),
    }

def compute_jvp(t1, t2, dt1, dt2, h02, h04, h11, h13, h20, h22, h31, h40):
    outputs = compute_outputs_and_jvp(t1, t2, dt1, dt2, h02, h04, h11, h13, h20, h22, h31, h40)
    return {
        'energy': outputs['energy_jvp'],
        'r1': outputs['r1_jvp'],
        'r2': outputs['r2_jvp'],
    }

def compute_energy(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    h02 = np.asarray(h02)
    h04 = np.asarray(h04)
    h11 = np.asarray(h11)
    h13 = np.asarray(h13)
    h20 = np.asarray(h20)
    h22 = np.asarray(h22)
    h31 = np.asarray(h31)
    h40 = np.asarray(h40)
    dtype = np.result_type(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40, np.complex128)
    energy = np.array(0.0 + 0.0j, dtype=dtype)
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.5) * _s0
    del _s0
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=_BINARY_EINSUM_PATH)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=_BINARY_EINSUM_PATH)
    energy += (0.125) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=_BINARY_EINSUM_PATH)
    energy += (0.041666666666666664) * _s0
    del _s0
    return np.real_if_close(energy)

def compute_r1(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    return compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40)['r1']

def compute_r2(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    return compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40)['r2']
