"""Generated connected qp-CCSD Wick contractions; do not edit."""

from __future__ import annotations

import numpy as np

TENSOR_NAMES = ('t1', 't2', 'h02', 'h04', 'h11', 'h13', 'h20', 'h22', 'h31', 'h40')
MAX_FORMAL_SCALING = 6
CONTRACTION_COUNT = 2566

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
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    energy += (0.5) * _s0
    del _s0
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    energy += (0.04166666666666668) * _s0
    del _s0
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    energy += (0.041666666666666664) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    energy += (-0.041666666666666664) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    energy += (0.041666666666666664) * _s1
    del _s0, _s1
    r1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)
    r1 += (1.0) * h20
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    r1 += (1.0) * _s0
    del _s0
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    r1 += (-1.0) * _s0
    del _s0
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    r1 += (0.4999999999999998) * _s0
    del _s0
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    r1 += (0.5) * _s0
    del _s0
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    r1 += (-0.1666666666666667) * _s0
    del _s0
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    r1 += (0.1666666666666667) * _s0
    del _s0
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    r1 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    r1 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (-0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (-0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (-0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (0.16666666666666666) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    r1 += (0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    r1 += (-0.04166666666666658) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (0.04166666666666664) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    r1 += (-0.04166666666666664) * _s2
    del _s0, _s1, _s2
    r2 = np.zeros((t1.shape[0], t1.shape[0], t1.shape[0], t1.shape[0]), dtype=dtype)
    r2 += (0.9999999999999996) * h40
    _s0 = np.einsum('ab,cdeb->cdea', h11, t2, optimize=False)
    r2 += (0.9999999999999996) * _s0
    del _s0
    _s0 = np.einsum('ab,cdeb->cdae', h11, t2, optimize=False)
    r2 += (-0.9999999999999996) * _s0
    del _s0
    _s0 = np.einsum('ab,cdeb->cade', h11, t2, optimize=False)
    r2 += (0.9999999999999996) * _s0
    del _s0
    _s0 = np.einsum('ab,cdeb->acde', h11, t2, optimize=False)
    r2 += (-0.9999999999999996) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->efab', h22, t2, optimize=False)
    r2 += (0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->eafb', h22, t2, optimize=False)
    r2 += (-0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->aefb', h22, t2, optimize=False)
    r2 += (0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->eabf', h22, t2, optimize=False)
    r2 += (0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->aebf', h22, t2, optimize=False)
    r2 += (-0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,efcd->abef', h22, t2, optimize=False)
    r2 += (0.5000000000000003) * _s0
    del _s0
    _s0 = np.einsum('abcd,ed->eabc', h31, t1, optimize=False)
    r2 += (1.0) * _s0
    del _s0
    _s0 = np.einsum('abcd,ed->aebc', h31, t1, optimize=False)
    r2 += (-1.0) * _s0
    del _s0
    _s0 = np.einsum('abcd,ed->abec', h31, t1, optimize=False)
    r2 += (1.0) * _s0
    del _s0
    _s0 = np.einsum('abcd,ed->abce', h31, t1, optimize=False)
    r2 += (-1.0) * _s0
    del _s0
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ac', h11, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (1.0) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',abcd->abcd', _s0, t2, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',abcd->abcd', _s0, t2, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,cdab->cd', h02, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-0.4999999999999993) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->abed', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->abec', h22, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abce', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h22, t1, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-0.5) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->efab', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eafb', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->eabf', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->abcd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acbd', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cabd', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->acdb', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cadb', _s0, t1, optimize=False)
    r2 += (0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ae', h13, t2, optimize=False)
    _s1 = np.einsum('ab,cd->cdab', _s0, t1, optimize=False)
    r2 += (-0.16666666666666632) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',abcd->abcd', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->acde', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cade', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdae', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cdeb->cdea', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efcd->efab', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efcd->efab', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efab->efcd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efcd->efab', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efab->efcd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efab->efcd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbd->efac', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbd->efac', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efac->efbd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbd->efac', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efac->efbd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efac->efbd', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbc->efad', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbc->efad', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efad->efbc', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efbc->efad', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aefb', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efad->efbc', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->aebf', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,efad->efbc', h04, t2, optimize=False)
    _s1 = np.einsum('abcd,efcd->abef', _s0, t2, optimize=False)
    r2 += (0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',abcd->abcd', _s0, t2, optimize=False)
    r2 += (-0.041666666666667275) * _s1
    del _s0, _s1
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.9999999999999986) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.9999999999999986) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.9999999999999986) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ca->cb', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,cb->ca', h02, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('ab,ab->', h02, t1, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.5000000000000003) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->aebc', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->abec', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->aebd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->aecd', h13, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->abed', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abce', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h13, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.16666666666666655) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.08333333333333455) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.08333333333333455) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.08333333333333455) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ebcd->ea', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eacd->eb', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->efab', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eafb', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aefb', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->eabf', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->aebf', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,efcd->abef', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabd->ec', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdea', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cdae', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->cade', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cdeb->acde', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ac', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,eabc->ed', h04, t2, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',abcd->abcd', _s1, t2, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cabd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cadb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cdab->cd', _s0, t2, optimize=False)
    _s2 = np.einsum('ab,cd->cdab', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->abcd', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acbd', _s1, t1, optimize=False)
    r2 += (-0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,abcd->', h04, t2, optimize=False)
    _s1 = np.einsum(',ab->ab', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cd->acdb', _s1, t1, optimize=False)
    r2 += (0.0416666666666667) * _s2
    del _s0, _s1, _s2
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.08333333333333472) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ed->eabc', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ed->aebc', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ed->abec', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ec->eabd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,ec->aebd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,eb->eacd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ea->ebcd', h04, t1, optimize=False)
    _s1 = np.einsum('abcd,eb->aecd', _s0, t1, optimize=False)
    _s2 = np.einsum('abcd,ec->abed', _s1, t1, optimize=False)
    _s3 = np.einsum('abcd,ed->abce', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,cd->ab', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bd->ac', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,bc->ad', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cabd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,cb->ca', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cadb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ca->cb', _s0, t1, optimize=False)
    _s2 = np.einsum('ab,cb->ac', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->cdab', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ab->cd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ac->bd', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->abcd', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acbd', _s2, t1, optimize=False)
    r2 += (-0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    _s0 = np.einsum('abcd,ad->bc', h04, t1, optimize=False)
    _s1 = np.einsum('ab,ab->', _s0, t1, optimize=False)
    _s2 = np.einsum(',ab->ab', _s1, t1, optimize=False)
    _s3 = np.einsum('ab,cd->acdb', _s2, t1, optimize=False)
    r2 += (0.041666666666666866) * _s3
    del _s0, _s1, _s2, _s3
    return {'energy': np.real_if_close(energy), 'r1': np.real_if_close(r1), 'r2': np.real_if_close(r2)}

def compute_energy(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    return compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40)['energy']

def compute_r1(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    return compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40)['r1']

def compute_r2(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40):
    return compute_outputs(t1, t2, h02, h04, h11, h13, h20, h22, h31, h40)['r2']
