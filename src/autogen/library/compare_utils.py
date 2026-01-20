import os
import time


def term_signature(term):
    # Build a coarse signature guaranteed to match for equivalent terms.
    map_org = getattr(term, "map_org", [])
    coeff_list = getattr(term, "coeff_list", [])
    sum_set = set(getattr(term, "sum_list", []))
    op_types = [op.name[:2] for op in map_org]
    op_types.sort()

    coeff_shapes = []
    map_len = len(map_org)
    for idx, coeff in enumerate(coeff_list):
        op_key = "E0" if idx >= map_len else map_org[idx].name[:2]
        dummy_a = dummy_i = dummy_p = dummy_x = 0
        nondummy_a = nondummy_i = nondummy_p = nondummy_x = 0
        for name in coeff:
            is_dummy = name in sum_set
            if term.isa(name):
                if is_dummy:
                    dummy_a += 1
                else:
                    nondummy_a += 1
            elif term.isi(name):
                if is_dummy:
                    dummy_i += 1
                else:
                    nondummy_i += 1
            elif term.isp(name):
                if is_dummy:
                    dummy_p += 1
                else:
                    nondummy_p += 1
            else:
                if is_dummy:
                    dummy_x += 1
                else:
                    nondummy_x += 1
        coeff_shapes.append(
            (
                op_key,
                len(coeff),
                dummy_a,
                dummy_i,
                dummy_p,
                dummy_x,
                nondummy_a,
                nondummy_i,
                nondummy_p,
                nondummy_x,
            )
        )
    coeff_shapes.sort()
    st_len = len(term.st[0]) if getattr(term, "st", None) else 0
    return (len(coeff_list), st_len, tuple(op_types), tuple(coeff_shapes))


def _op_label(map_org, idx):
    if idx >= len(map_org):
        return "E0"
    name = getattr(map_org[idx], "name", "")
    return name[:2]


def _index_kind(term, name):
    if term.isa(name):
        return "a"
    if term.isi(name):
        return "i"
    if term.isp(name):
        return "p"
    return "x"


def index_incidence_signature(term):
    # Encode how indices connect operators, ignoring dummy names.
    map_org = getattr(term, "map_org", [])
    coeff_list = getattr(term, "coeff_list", [])
    sum_set = set(getattr(term, "sum_list", []))
    index_map = {}
    for idx, coeff in enumerate(coeff_list):
        op_label = _op_label(map_org, idx)
        for name in coeff:
            info = index_map.get(name)
            if info is None:
                info = {
                    "dummy": name in sum_set,
                    "kind": _index_kind(term, name),
                    "ops": {},
                }
                index_map[name] = info
            ops = info["ops"]
            ops[op_label] = ops.get(op_label, 0) + 1
    signatures = []
    for name, info in index_map.items():
        name_key = None if info["dummy"] else name
        ops = tuple(sorted(info["ops"].items()))
        signatures.append((info["kind"], info["dummy"], name_key, ops))
    signatures.sort()
    return tuple(signatures)


def _index_descriptors(term):
    map_org = getattr(term, "map_org", [])
    coeff_list = getattr(term, "coeff_list", [])
    sum_set = set(getattr(term, "sum_list", []))
    index_map = {}
    for idx, coeff in enumerate(coeff_list):
        op_label = _op_label(map_org, idx)
        for name in coeff:
            info = index_map.get(name)
            if info is None:
                info = {
                    "dummy": name in sum_set,
                    "kind": _index_kind(term, name),
                    "ops": {},
                }
                index_map[name] = info
            ops = info["ops"]
            ops[op_label] = ops.get(op_label, 0) + 1
    return index_map


def structural_key(term):
    index_map = _index_descriptors(term)
    descriptors = {}
    for name, info in index_map.items():
        name_key = None if info["dummy"] else name
        ops = tuple(sorted(info["ops"].items()))
        desc = (info["kind"], info["dummy"], name_key, ops)
        descriptors[name] = desc
    unique_desc = sorted(set(descriptors.values()))
    desc_id = {desc: idx for idx, desc in enumerate(unique_desc)}

    map_org = getattr(term, "map_org", [])
    coeff_list = getattr(term, "coeff_list", [])
    patterns = []
    for idx, coeff in enumerate(coeff_list):
        op_label = _op_label(map_org, idx)
        ids = [desc_id[descriptors[name]] for name in coeff]
        ids.sort()
        patterns.append((op_label, tuple(ids)))
    patterns.sort()
    return tuple(patterns)


def coarse_key(term):
    return (term_signature(term), index_incidence_signature(term), structural_key(term))


def matrix_key(term):
    return (coarse_key(term), matrix_signature(term))


def bucket_terms(list_terms):
    buckets = {}
    for idx, term in enumerate(list_terms):
        if term.fac == 0:
            continue
        key = term_signature(term)
        buckets.setdefault(key, []).append(idx)
    return list(buckets.values())


def _needs_matrix(term):
    return not getattr(term, "_matrices_ready", False)


def ensure_matrices_for_terms(list_terms):
    try:
        from .compare_test import create_matrices
    except Exception:
        return
    for term in list_terms:
        if term.fac == 0:
            continue
        if _needs_matrix(term):
            create_matrices(term)


def ensure_matrices_for_indices(list_terms, indices):
    try:
        from .compare_test import create_matrices
    except Exception:
        return
    for idx in indices:
        term = list_terms[idx]
        if term.fac == 0:
            continue
        if _needs_matrix(term):
            create_matrices(term)


def _open_index_map(term):
    if not getattr(term, "large_op_list", None):
        return ()
    if "X2" not in term.large_op_list[0].name:
        return ()
    if not term.coeff_list:
        return ()
    open_map = []
    for idx in term.coeff_list[0]:
        targets = []
        for j in range(1, len(term.large_op_list)):
            for item in term.coeff_list[j]:
                if idx == item:
                    name = term.large_op_list[j].name
                    if name and name[0] in ("T", "D"):
                        targets.append(name[0])
                    else:
                        targets.append(name)
                    break
        open_map.append(tuple(targets))
    return tuple(open_map)


def matrix_signature(term):
    if not hasattr(term, "imatrix") or term.imatrix is None:
        return None
    labels = []
    for op in term.large_op_list:
        if op.name and op.name[0] in ("T", "D"):
            labels.append(op.name[0])
        else:
            labels.append(op.name)
    n = len(labels)
    rows = []
    for i in range(n):
        neigh = []
        for j in range(n):
            if i == j:
                continue
            im_val = int(term.imatrix[i][j])
            am_val = int(term.amatrix[i][j])
            if im_val != 0 or am_val != 0:
                neigh.append((labels[j], im_val, am_val))
        neigh.sort()
        rows.append((labels[i], tuple(neigh)))
    rows.sort()
    return (tuple(rows), _open_index_map(term))


def canonical_key(term):
    has_matrix = not _needs_matrix(term)
    cached = getattr(term, "_cmp_key", None)
    cached_has_matrix = getattr(term, "_cmp_key_has_matrix", None)
    if cached is not None and cached_has_matrix == has_matrix:
        return cached
    key = (term_signature(term), matrix_signature(term) if has_matrix else None)
    term._cmp_key = key
    term._cmp_key_has_matrix = has_matrix
    return key


def is_simple_term(term):
    return all(len(coeff) < 4 for coeff in term.coeff_list)


def _get_compare_mode():
    # Use env var to switch compare strategy without code changes.
    mode = os.getenv("AUTOGEN_COMPARE_MODE", "fast").lower().strip()
    if mode not in ("fast", "full", "check"):
        return "fast"
    return mode


def _fast_compare_impl(term1, term2):
    if is_simple_term(term1) and is_simple_term(term2):
        try:
            from .compare_test import compare as cmp_test
            from .compare_test import create_matrices
        except Exception:
            from . import compare as cmp_full
            return cmp_full.compare(term1, term2)
        if _needs_matrix(term1):
            create_matrices(term1)
        if _needs_matrix(term2):
            create_matrices(term2)
        return cmp_test(term1, term2)
    from . import compare as cmp_full
    return cmp_full.compare(term1, term2)


def fast_compare(term1, term2):
    mode = _get_compare_mode()
    if mode == "full":
        from . import compare as cmp_full
        return cmp_full.compare(term1, term2)
    if mode == "check":
        fast_val = _fast_compare_impl(term1, term2)
        from . import compare as cmp_full
        full_val = cmp_full.compare(term1, term2)
        if fast_val != full_val:
            print(
                "compare mismatch: fast={} full={} term1={} term2={}".format(
                    fast_val,
                    full_val,
                    getattr(term1, "coeff_list", None),
                    getattr(term2, "coeff_list", None),
                )
            )
        return full_val
    return _fast_compare_impl(term1, term2)


def reduce_terms(list_terms, compare_func, merge_func, key_func=canonical_key):
    return reduce_terms_two_stage(
        list_terms,
        compare_func,
        merge_func,
        key_func=key_func,
        secondary_key_func=None,
    )


def reduce_terms_two_stage(
    list_terms, compare_func, merge_func, key_func, secondary_key_func
):
    stats = _compare_stats
    start_time = time.perf_counter() if stats is not None else None
    buckets = {}
    for idx, term in enumerate(list_terms):
        if term.fac == 0:
            continue
        key = key_func(term)
        buckets.setdefault(key, []).append(idx)

    if secondary_key_func is not None:
        refined = {}
        for key, indices in buckets.items():
            if len(indices) <= 1:
                refined[(key, None)] = indices
                continue
            ensure_matrices_for_indices(list_terms, indices)
            for idx in indices:
                term = list_terms[idx]
                if term.fac == 0:
                    continue
                key2 = secondary_key_func(term)
                refined.setdefault(key2, []).append(idx)
        buckets = refined

    for indices in buckets.values():
        reps = []
        for idx in indices:
            term = list_terms[idx]
            if term.fac == 0:
                continue
            merged = False
            for rep_idx in reps:
                rep = list_terms[rep_idx]
                if rep.fac == 0:
                    continue
                if stats is not None:
                    stats.compare_calls += 1
                flo = compare_func(rep, term)
                if flo != 0:
                    if stats is not None:
                        stats.merges += 1
                    merge_func(rep, term, flo)
                    merged = True
                    break
            if not merged:
                reps.append(idx)

    if stats is not None:
        bucket_sizes = [len(items) for items in buckets.values()]
        stats.buckets += len(bucket_sizes)
        stats.bucket_pairs += sum(size * (size - 1) // 2 for size in bucket_sizes)
        stats.total_terms += sum(bucket_sizes)
        if bucket_sizes:
            stats.max_bucket = max(stats.max_bucket, max(bucket_sizes))
        stats.reduce_calls += 1
        stats.elapsed += time.perf_counter() - start_time


class CompareStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.compare_calls = 0
        self.merges = 0
        self.buckets = 0
        self.bucket_pairs = 0
        self.total_terms = 0
        self.max_bucket = 0
        self.reduce_calls = 0
        self.elapsed = 0.0


_compare_stats = None


def enable_compare_stats():
    global _compare_stats
    if _compare_stats is None:
        _compare_stats = CompareStats()
    return _compare_stats


def get_compare_stats():
    return _compare_stats


def reset_compare_stats():
    if _compare_stats is not None:
        _compare_stats.reset()
