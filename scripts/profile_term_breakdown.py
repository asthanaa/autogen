import time

import autogen.library as lib
import autogen.library.change_terms as ct
import autogen.library.compare_utils as compare_utils
from autogen.library.compare_test import create_matrices
from autogen.main_tools import multi_cont


def merge_terms(rep, term, flo):
    rep.fac = rep.fac + term.fac * flo
    term.fac = 0.0


def main():
    ops = ["X2", "V2", "T1", "T11", "T12"]
    fc = 1.0 / 6.0

    t0 = time.perf_counter()
    dict_ind = {}
    lou, dict_ind = lib.make_op.make_op(ops, dict_ind)
    print("make_op:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    a, b = lou[0].st, lou[0].co
    for i in range(1, len(lou)):
        a, b = multi_cont.multi_cont(a, lou[i].st, b, lou[i].co)
    print("multi_cont total:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    a, b = lib.full_con.full_con(a, b)
    print("full_con:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    list_terms = ct.change_terms1(a, b, fc, dict_ind, lou)
    print("change_terms1:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for term in list_terms:
        term.compress()
        term.cond_cont(dict_ind)
        term.build_map_org()
    print("compress/cond/build_map_org:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for term in list_terms:
        create_matrices(term)
    compare_utils.reduce_terms(list_terms, compare_utils.fast_compare, merge_terms)
    print("compare+reduce:", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
