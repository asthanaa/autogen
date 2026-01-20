try:
    import numpy as np
    from numba import njit
    from numba.typed import List as NList
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit
    def _dfs(cands, lim_cnt, start, depth, used, left, right, results):
        n = len(cands)
        if depth == lim_cnt:
            pairs = np.empty((lim_cnt, 2), dtype=np.int64)
            for k in range(lim_cnt):
                pairs[k, 0] = left[k]
                pairs[k, 1] = right[k]
            results.append(pairs)
            return
        for i in range(start, n):
            if used[i]:
                continue
            cand = cands[i]
            for idx in range(len(cand)):
                j = cand[idx]
                if used[j]:
                    continue
                used[i] = 1
                used[j] = 1
                left[depth] = i
                right[depth] = j
                _dfs(cands, lim_cnt, i + 1, depth + 1, used, left, right, results)
                used[i] = 0
                used[j] = 0


    @njit
    def _enumerate_matchings(cands, lim_cnt):
        results = NList()
        if lim_cnt == 0:
            return results
        n = len(cands)
        used = np.zeros(n, dtype=np.uint8)
        left = np.empty(lim_cnt, dtype=np.int64)
        right = np.empty(lim_cnt, dtype=np.int64)
        _dfs(cands, lim_cnt, 0, 0, used, left, right, results)
        return results


    def to_typed(cands):
        typed = NList()
        for row in cands:
            typed.append(np.array(row, dtype=np.int64))
        return typed


    def enumerate_matchings(cands, lim_cnt):
        return _enumerate_matchings(to_typed(cands), lim_cnt)


    def enumerate_matchings_typed(typed_cands, lim_cnt):
        return _enumerate_matchings(typed_cands, lim_cnt)
else:
    def to_typed(cands):
        raise RuntimeError("Numba is not available")


    def enumerate_matchings(cands, lim_cnt):
        raise RuntimeError("Numba is not available")


    def enumerate_matchings_typed(typed_cands, lim_cnt):
        raise RuntimeError("Numba is not available")
