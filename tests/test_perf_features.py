import importlib
import io


def _make_spin_ops(func_ewt):
    op1 = func_ewt.operator("ac", "1", 1, "u", 1, None, 1)
    op2 = func_ewt.operator("ac", "0", 2, "v", 1, None, 2)
    op1.pos2 = 1
    op2.pos2 = 2
    return op1, op2


def _run_emit(spin_summed, monkeypatch):
    monkeypatch.setenv("AUTOGEN_SPIN_SUMMED", "1" if spin_summed else "0")
    import autogen.pkg.fix_uv as fix_uv

    fix_uv = importlib.reload(fix_uv)
    fix_uv.parity.parity = lambda *args, **kwargs: False
    fix_uv.func.loop_present = lambda *args, **kwargs: 0.0

    from autogen.pkg import func_ewt

    op1, op2 = _make_spin_ops(func_ewt)
    full = [op1, op2]
    full_pos = [1, 2]
    full_con = []
    const_con = []
    f = io.StringIO()
    fix_uv.emit_contraction([], [op1], [op2], full, f, full_pos, 0, full_con, const_con)
    return const_con[0][1]


def test_spin_summed_toggle(monkeypatch):
    const_on = _run_emit(True, monkeypatch)
    const_off = _run_emit(False, monkeypatch)
    assert const_on == 0.5
    assert const_off == 1.0


def test_fast_compare_matches_full(monkeypatch):
    monkeypatch.setenv("AUTOGEN_COMPARE_MODE", "fast")
    from autogen.main_tools.commutator import comm
    from autogen.library import compare as cmp_full
    from autogen.library import compare_utils

    terms = comm(["V2"], ["T1"], 1)
    assert terms
    term = terms[0]
    fast_val = compare_utils.fast_compare(term, term)
    full_val = cmp_full.compare(term, term)
    assert fast_val == full_val == 1


def test_matching_cache_reuse(monkeypatch):
    monkeypatch.setenv("AUTOGEN_MATCHING_CACHE", "1")
    monkeypatch.setenv("AUTOGEN_NUMBA", "0")
    import autogen.pkg.make_c as make_c

    make_c = importlib.reload(make_c)
    make_c._MATCHING_CACHE.clear()

    calls = []

    def stub_emit(*args, **kwargs):
        calls.append(1)

    monkeypatch.setattr(make_c.fix_temp, "emit_contraction", stub_emit)

    from autogen.pkg import func_ewt

    op1 = func_ewt.operator("ac", "1", 1, "u", 1, None, 1)
    op2 = func_ewt.operator("ac", "0", 2, "v", 1, None, 2)
    full = [op1, op2]
    poss = [[op2], []]
    f = io.StringIO()

    make_c.make_c(2, [], [], [], [], full, poss, f, set(), [1, 2], 0, "2", [], [])
    size1 = len(make_c._MATCHING_CACHE)

    make_c.make_c(2, [], [], [], [], full, poss, f, set(), [1, 2], 0, "2", [], [])
    size2 = len(make_c._MATCHING_CACHE)

    assert size1 == size2 == 1
    assert calls


def test_multi_cont_cache_reuse(monkeypatch):
    monkeypatch.setenv("AUTOGEN_MULTI_CONT_CACHE", "1")
    import autogen.main_tools.multi_cont as mc

    mc = importlib.reload(mc)
    mc._MULTI_CONT_CACHE.clear()

    from autogen.pkg import func_ewt

    def stub_ewt(*args, **kwargs):
        op = func_ewt.contractedobj("op", 1, 1)
        op.upper = ["i"]
        op.lower = ["a"]
        return [[op]], [[1, 1]]

    monkeypatch.setattr(mc.fewt, "ewt", stub_ewt)

    op1 = func_ewt.contractedobj("op", 1, 1)
    op1.upper = ["i"]
    op1.lower = ["a"]
    op2 = func_ewt.contractedobj("op", 1, 1)
    op2.upper = ["j"]
    op2.lower = ["b"]
    st1 = [[op1]]
    st2 = [[op2]]
    c1 = [[1, 1]]
    c2 = [[1, 1]]

    mc.multi_cont(st1, st2, c1, c2)
    size1 = len(mc._MULTI_CONT_CACHE)
    mc.multi_cont(st1, st2, c1, c2)
    size2 = len(mc._MULTI_CONT_CACHE)

    assert size1 == size2 == 1
