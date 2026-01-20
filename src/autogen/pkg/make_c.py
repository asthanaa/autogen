from . import fix_uv
from . import func_ewt
from . import numba_contractions
from collections import deque, OrderedDict
import os

# Build contraction lists for a given operator string.
fix_temp = fix_uv
func = func_ewt
USE_NUMBA = os.getenv("AUTOGEN_NUMBA") == "1" and getattr(numba_contractions, "HAS_NUMBA", False)

MATCHING_CACHE_ENABLED = os.getenv("AUTOGEN_MATCHING_CACHE", "1") != "0"
try:
    MATCHING_CACHE_SIZE = int(os.getenv("AUTOGEN_MATCHING_CACHE_SIZE", "128"))
except ValueError:
    MATCHING_CACHE_SIZE = 128
_MATCHING_CACHE = OrderedDict()

NUMBA_CANDS_CACHE_ENABLED = os.getenv("AUTOGEN_NUMBA_CANDS_CACHE", "1") != "0"
try:
    NUMBA_CANDS_CACHE_SIZE = int(os.getenv("AUTOGEN_NUMBA_CANDS_CACHE_SIZE", "64"))
except ValueError:
    NUMBA_CANDS_CACHE_SIZE = 64
_NUMBA_CANDS_CACHE = OrderedDict()


def _full_pattern(full):
    return tuple((op.kind, op.dag, op.string) for op in full)


def _build_candidate_indices(full, poss):
    pos_to_index = {op.pos: idx for idx, op in enumerate(full)}
    cands = []
    for row in poss:
        row_idx = []
        for item in row:
            idx = pos_to_index.get(item.pos)
            if idx is not None:
                row_idx.append(idx)
        cands.append(tuple(row_idx))
    return tuple(cands)


def _match_cache_get(key):
    if not MATCHING_CACHE_ENABLED:
        return None
    cached = _MATCHING_CACHE.get(key)
    if cached is None:
        return None
    _MATCHING_CACHE.move_to_end(key)
    return cached


def _match_cache_set(key, value):
    if not MATCHING_CACHE_ENABLED:
        return
    _MATCHING_CACHE[key] = value
    _MATCHING_CACHE.move_to_end(key)
    if len(_MATCHING_CACHE) > MATCHING_CACHE_SIZE:
        _MATCHING_CACHE.popitem(last=False)


def _numba_cands_cache_get(key):
    if not NUMBA_CANDS_CACHE_ENABLED:
        return None
    cached = _NUMBA_CANDS_CACHE.get(key)
    if cached is None:
        return None
    _NUMBA_CANDS_CACHE.move_to_end(key)
    return cached


def _numba_cands_cache_set(key, value):
    if not NUMBA_CANDS_CACHE_ENABLED:
        return
    _NUMBA_CANDS_CACHE[key] = value
    _NUMBA_CANDS_CACHE.move_to_end(key)
    if len(_NUMBA_CANDS_CACHE) > NUMBA_CANDS_CACHE_SIZE:
        _NUMBA_CANDS_CACHE.popitem(last=False)


def _enumerate_matchings_py(cands, lim_cnt):
    if lim_cnt == 0:
        return ((),)
    n = len(cands)
    used = [False] * n
    left = [0] * lim_cnt
    right = [0] * lim_cnt
    out = []

    def dfs(start, depth):
        if depth == lim_cnt:
            out.append(tuple((left[k], right[k]) for k in range(lim_cnt)))
            return
        for i in range(start, n):
            if used[i]:
                continue
            cand = cands[i]
            for j in cand:
                if used[j]:
                    continue
                used[i] = True
                used[j] = True
                left[depth] = i
                right[depth] = j
                dfs(i + 1, depth + 1)
                used[i] = False
                used[j] = False

    dfs(0, 0)
    return tuple(out)


def _emit_matchings_from_pairs(matchings, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con):
    if lim_cnt == 0:
        fix_temp.emit_contraction(contracted, [], [], full, f, full_pos, i_c, full_con, const_con)
        return
    contracted_l = [None] * lim_cnt
    contracted_r = [None] * lim_cnt
    for pairs in matchings:
        for idx, pair in enumerate(pairs):
            left_idx, right_idx = pair
            contracted_l[idx] = full[left_idx]
            contracted_r[idx] = full[right_idx]
        fix_temp.emit_contraction(contracted, contracted_l, contracted_r, full, f, full_pos, i_c, full_con, const_con)


def _emit_matchings_py(cands, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con):
    if lim_cnt == 0:
        fix_temp.emit_contraction(contracted, [], [], full, f, full_pos, i_c, full_con, const_con)
        return
    n = len(cands)
    used = [False] * n
    contracted_l = [None] * lim_cnt
    contracted_r = [None] * lim_cnt

    def dfs(start, depth):
        if depth == lim_cnt:
            fix_temp.emit_contraction(contracted, contracted_l, contracted_r, full, f, full_pos, i_c, full_con, const_con)
            return
        for i in range(start, n):
            if used[i]:
                continue
            cand = cands[i]
            for j in cand:
                if used[j]:
                    continue
                used[i] = True
                used[j] = True
                contracted_l[depth] = full[i]
                contracted_r[depth] = full[j]
                dfs(i + 1, depth + 1)
                used[i] = False
                used[j] = False

    dfs(0, 0)


def _numba_matches_to_pairs(matches):
    out = []
    for pairs in matches:
        lim_cnt = pairs.shape[0]
        out.append(tuple((int(pairs[k, 0]), int(pairs[k, 1])) for k in range(lim_cnt)))
    return tuple(out)


def _emit_matchings_numba(matches, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con):
    if lim_cnt == 0:
        fix_temp.emit_contraction(contracted, [], [], full, f, full_pos, i_c, full_con, const_con)
        return
    contracted_l = [None] * lim_cnt
    contracted_r = [None] * lim_cnt
    for pairs in matches:
        for k in range(lim_cnt):
            left_idx = int(pairs[k, 0])
            right_idx = int(pairs[k, 1])
            contracted_l[k] = full[left_idx]
            contracted_r[k] = full[right_idx]
        fix_temp.emit_contraction(contracted, contracted_l, contracted_r, full, f, full_pos, i_c, full_con, const_con)


def _contracted_key(contracted):
    blocks = []
    for block in contracted:
        blocks.append(tuple(item.pos for item in block))
    return tuple(sorted(blocks))


def make_c(lim_cu, contracted, a, i, u, full, poss, f, store_for_repeat, full_pos, i_c, menu, full_con, const_con):
    for n in range(2, lim_cu+1, 2):
        if n>2:
            u_copy = deque([])
            y = list(u)
            # Build partial operator pools for higher-order cumulants.
            for x in range(n):
                u_copy.append(deque(y))
                if y:
                    pos = u_copy[-1][0].pos
                    for idx, item in enumerate(y):
                        if item.pos == pos:
                            del y[idx]
                            break
                else :
                    return   #if there are less no of operators the
                             #fn should return without any result
            while True:
                add = 0
                u_tmp = []
                #pick up all cumulants
                for index in range(len(u_copy)):
                    u_tmp.append(u_copy[index][0])
                for item in u_tmp :
                    if item.dag == '0': add=add-1
                    else : add=add+1
                if add==0:
                    selected_pos = {item.pos for item in u_tmp}
                    u_2 = [item for item in u if item.pos not in selected_pos]
                    full_2 = [item for item in full if item.pos not in selected_pos]
                    flag1 =0

                    contracted.append(u_tmp)
                    p_1=0 # if the charecter from string 1 is present or not
                    p_2=0
                    if (len(contracted)>0): # this stores all the multiple lanbda and stores them in a list, checks them to see redundancy
                        contracted_key = _contracted_key(contracted)
                        if contracted_key in store_for_repeat:
                            flag1 = 1
                        for item in u_tmp:
                            if item.string == 1:
                                p_1 = 1
                            else:
                                p_2 = 1
                        if (flag1 ==0 and p_1==1 and p_2==1) or menu == '1':

                            make_c(n, contracted, a, i, u_2, full_2, poss, f, store_for_repeat, full_pos, i_c, menu, full_con, const_con)#call the function again with smaller u
                            #if not u_2:
                            store_for_repeat.add(contracted_key)
                    contracted.pop()
                flag =1
                x=-1
                tmp_0 = deque([])

                if len(u_copy[0])==n : break # break out of whole function

                while flag == 1:
                    if (u_copy[x]):
                        u_copy[x].popleft()
                        if x>=(-len(u_copy[x])):
                            flag = 0
                            tmp_0 = deque(u_copy[x])
                            while x<-1:
                                x=x+1
                                tmp_0.popleft()
                                u_copy[x]=deque(tmp_0)
                    else :
                        x=x-1
        elif n==2:
            '''
            poss = deque([])
            for operator in full:
                y = deque([])
                if menu == '1':
                    if operator.kind == 'pa' and operator.dag=='0':
                        for item in a:
                            if operator.pos<item.pos and item.dag=='1':
                                y.append(item)

                    elif operator.kind == 'ho' and operator.dag=='1':
                        for item in i:
                            if operator.pos<item.pos and item.dag=='0':
                                y.append(item)

                    elif operator.kind == 'ac':  #because active states will have eta and gamma
                        for item in u:
                            if operator.pos<item.pos and int(item.dag)!=int(operator.dag):
                                y.append(item)
                        #if (y): remember that empty strings are also included
                    poss.append(y) #list of list in dictionary order i.e 1st annhilation -> possible creation then 2nd ...
                else:
                    if operator.kind == 'pa' and operator.dag=='0':
                        for item in a:
                            if operator.pos<item.pos and item.dag=='1' and operator.string!=item.string:
                                y.append(item)

                    elif operator.kind == 'ho' and operator.dag=='1':
                        for item in i:
                            if operator.pos<item.pos and item.dag=='0' and operator.string != item.string:
                                y.append(item)

                    elif operator.kind == 'ac':  #because active states will have eta and gamma
                        for item in u:
                            if operator.pos<item.pos and int(item.dag)!=int(operator.dag) and operator.string!=item.string:
                                y.append(item)
                        #if (y): remember that empty strings are also included
                    poss.append(y) #list of list in dictionary order i.e 1st annhilation -> possible creation then 2nd ...
            '''
            no = len(full)//2
            cands = _build_candidate_indices(full, poss)
            pattern_key = _full_pattern(full)
            use_numba = USE_NUMBA
            typed_cands = None
            if use_numba:
                typed_key = ("typed", pattern_key)
                typed_cands = _numba_cands_cache_get(typed_key)
                if typed_cands is None:
                    try:
                        typed_cands = numba_contractions.to_typed(cands)
                    except Exception:
                        typed_cands = None
                        use_numba = False
                    else:
                        _numba_cands_cache_set(typed_key, typed_cands)
                if typed_cands is None:
                    use_numba = False
            for lim_cnt in range(0, no + 1):
                if lim_cnt == 0:
                    fix_temp.emit_contraction(contracted, [], [], full, f, full_pos, i_c, full_con, const_con)
                    continue
                cache_key = (pattern_key, lim_cnt)
                matchings = _match_cache_get(cache_key)
                if matchings is not None:
                    _emit_matchings_from_pairs(matchings, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con)
                    continue
                if use_numba:
                    try:
                        matches = numba_contractions.enumerate_matchings_typed(typed_cands, lim_cnt)
                    except Exception:
                        matches = None
                        use_numba = False
                    if matches is not None:
                        if MATCHING_CACHE_ENABLED:
                            matchings = _numba_matches_to_pairs(matches)
                            _match_cache_set(cache_key, matchings)
                            _emit_matchings_from_pairs(matchings, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con)
                        else:
                            _emit_matchings_numba(matches, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con)
                        continue
                if MATCHING_CACHE_ENABLED:
                    matchings = _enumerate_matchings_py(cands, lim_cnt)
                    _match_cache_set(cache_key, matchings)
                    _emit_matchings_from_pairs(matchings, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con)
                else:
                    _emit_matchings_py(cands, lim_cnt, full, contracted, f, full_pos, i_c, full_con, const_con)
