"""Shared label canonicalization helpers used by codegen and tests."""

from __future__ import annotations

import re


OCC_SET = set("ijklmn")
VIRT_SET = set("abcdefgh")
_LABEL_RE = re.compile(r"[a-z][0-9]*")
_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonicalize_g(labels: str):
    """Normalize antisymmetrized two-electron labels to occupied/virtual order."""
    if len(labels) != 4:
        return labels, 1.0
    occ_idx = [i for i, label in enumerate(labels) if label in OCC_SET]
    virt_idx = [i for i, label in enumerate(labels) if label in VIRT_SET]
    if occ_idx != [0, 1] or virt_idx != [2, 3]:
        return labels, 1.0
    occ_labels = [labels[i] for i in occ_idx]
    virt_labels = [labels[i] for i in virt_idx]
    occ_sorted = sorted(occ_labels)
    virt_sorted = sorted(virt_labels)
    sign = 1.0
    if occ_labels != occ_sorted:
        sign *= -1.0
    if virt_labels != virt_sorted:
        sign *= -1.0
    return "".join(occ_sorted + virt_sorted), sign


def tokenize_labels(labels: str):
    if not labels:
        return []
    return _LABEL_RE.findall(labels)


def safe_einsum_subs(subs: str) -> str:
    if "->" in subs:
        in_part, out_part = subs.split("->", 1)
    else:
        in_part, out_part = subs, ""
    in_labels = [] if not in_part else in_part.split(",")
    tokens = []
    for labels in in_labels:
        tokens.extend(tokenize_labels(labels))
    tokens.extend(tokenize_labels(out_part))
    mapping = {}
    sym_iter = iter(_EINSUM_SYMBOLS)
    for token in tokens:
        if token in mapping:
            continue
        try:
            mapping[token] = next(sym_iter)
        except StopIteration as exc:
            raise ValueError(
                f"Too many unique indices for einsum subs '{subs}'. "
                f"Need <= {len(_EINSUM_SYMBOLS)}."
            ) from exc

    def map_labels(labels):
        return "".join(mapping[token] for token in tokenize_labels(labels))

    mapped_in = ",".join(map_labels(labels) for labels in in_labels) if in_labels else ""
    mapped_out = map_labels(out_part) if out_part else ""
    return f"{mapped_in}->{mapped_out}" if "->" in subs else mapped_in
