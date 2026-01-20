import autogen.pkg.func_ewt as func_ewt
import autogen.pkg.fewt as fewt
import copy
import os
from collections import OrderedDict
import autogen.library.make_op as make_op
import autogen.library.print_op as p_op
import autogen.library.full_con as full_con
# Multi-operator contraction engine used by commutator/product flows.
'''
Problems :
    1. the fully uncontrancted expression has wrong ordering in the lower half.
    2. look at fewt there is a problem with adding general states in line 298

'''

#make operators
#X, V, T1 = make_op.make_op()

'''
op1=func_ewt.contractedobj('op', 1, 1)
op1.upper=['i','j']
op1.lower=['a','b']
op3=func_ewt.contractedobj('op', 1, 1)
op3.upper=['p','q']
op3.lower=['r','s']
op2=func_ewt.contractedobj('op', 1, 1)
op2.upper=['c']
op2.lower=['k']
st1=[[op1]]
st2=[[op2]]
stp=[[op3]]
'''
def sp_multi(a,b):
    re=[]
    #print a, b
    for item1, item2 in zip(a,b):
        item3=item1*item2
        re.append(item3)
    return re
def arrange(st, co, new_term_half, new_const_half):
    terms=[]
    const=[]
    tmp_term=[]
    #print 'in arrange', st, new_term_half
    for term, pre in zip(st, co):
        tmp_term = list(new_term_half)
        tmp_term.extend(term)
        tmp_const=sp_multi(pre,new_const_half)
        #print tmp_const
        terms.append(tmp_term)
        const.append(tmp_const)
    #print "after arrange",terms, const
    return terms, const


def _op_name(obj):
    return obj.name if hasattr(obj, "name") else str(obj)


def _sig_item(item):
    if isinstance(item, str):
        return ("s", item)
    return (
        "o",
        getattr(item, "name", None),
        getattr(item, "kind", None),
        getattr(item, "dag", None),
        getattr(item, "pos", None),
        getattr(item, "string", None),
    )


def _term_key(term):
    key=[]
    for op in term:
        upper = tuple(_sig_item(item) for item in getattr(op, "upper", []))
        lower = tuple(_sig_item(item) for item in getattr(op, "lower", []))
        anti = getattr(op, "anti", 0)
        matrix = getattr(op, "matrix", None)
        matrix_sig = tuple(matrix) if matrix else None
        key.append((op.kind, upper, lower, anti, matrix_sig))
    return tuple(key)


def _dedup_terms(terms, consts):
    index = {}
    out_terms = []
    out_consts = []
    for term, const in zip(terms, consts):
        key = _term_key(term)
        if key in index:
            idx = index[key]
            if len(out_consts[idx]) == len(const):
                out_consts[idx] = [a + b for a, b in zip(out_consts[idx], const)]
            else:
                out_terms.append(term)
                out_consts.append(const)
                index[key] = len(out_terms) - 1
        else:
            index[key] = len(out_terms)
            out_terms.append(term)
            out_consts.append(const)
    return out_terms, out_consts


CACHE_ENABLED = os.getenv("AUTOGEN_MULTI_CONT_CACHE", "1") != "0"
try:
    CACHE_SIZE = int(os.getenv("AUTOGEN_MULTI_CONT_CACHE_SIZE", "256"))
except ValueError:
    CACHE_SIZE = 256
_MULTI_CONT_CACHE = OrderedDict()


def _const_key(consts):
    return tuple(tuple(c) for c in consts)


def _st_key(st):
    return tuple(_term_key(term) for term in st)


def _mc_key(st1, st2, const1, const2):
    return (_st_key(st1), _const_key(const1), _st_key(st2), _const_key(const2))


def _cache_get(key):
    cached = _MULTI_CONT_CACHE.get(key)
    if cached is None:
        return None
    _MULTI_CONT_CACHE.move_to_end(key)
    terms, consts = cached
    return list(terms), [list(c) for c in consts]


def _cache_set(key, terms, consts):
    _MULTI_CONT_CACHE[key] = (tuple(terms), tuple(tuple(c) for c in consts))
    _MULTI_CONT_CACHE.move_to_end(key)
    if len(_MULTI_CONT_CACHE) > CACHE_SIZE:
        _MULTI_CONT_CACHE.popitem(last=False)
#assumption : last part of a tern is the working 'op'. 'op' cannot be before any 'de'
def multi_cont(st1, st2, const1, const2):
    # Combine two operator strings and enumerate all contraction patterns.
    if CACHE_ENABLED:
        cache_key = _mc_key(st1, st2, const1, const2)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    flag2=0
    final_terms=[]
    final_const=[]
    #print 'first operator string constant',const1
    for (term1, pre1) in zip(st1, const1):
        new_term_half=[]
        for op11 in term1:
            if op11.kind!='op':
                new_term_half.append(op11)
        for term2, pre2 in zip(st2, const2):
            for op22 in term2:
                if op22.kind!='op':
                    new_term_half.append(op22)
                elif op11.kind=='op':
                    flag2=1
                    new_const_half=sp_multi(pre1,pre2)
                    o,c=fewt.ewt(op11.upper, op11.lower, op22.upper, op22.lower)
                    terms, const=arrange(o,c, new_term_half, new_const_half)
                    final_terms.extend(terms)
                    final_const.extend(const)
                    #p_op.print_op(final_terms,final_const)
                elif op11.kind!='op':
                    new_term_half.append(op22)
                    final_terms.append(new_term_half)
                    final_const.append(sp_multi(pre1,pre2))
                else :
                    print(" there is a case in multi_cont file I am missing")
            flag2=0
    final_terms, final_const = _dedup_terms(final_terms, final_const)
    if CACHE_ENABLED:
        _cache_set(cache_key, final_terms, final_const)
    return final_terms, final_const
'''
a,b = multi_cont(X.st, V.st, X.co, V.co)
p_op.print_op(a,b)
print '---------------'

a,b=  multi_cont(a, T1.st, b,T1.co)

a,b=full_con.full_con(a,b)

p_op.print_op(a,b)
'''
