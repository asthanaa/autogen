from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any, Iterable


OUTPUTS = ("N0", "N20", "N40", "H0", "H20", "H40")
EQUATION_SCHEMA = "pn-oap-bpn-v2"
PROJECTOR_ORDERING = "B_mu P_N (H-E) exp(T)"
_DUMMY_NAMES = tuple("ijklmnopqrstuvwxyzabcdefgh")
_ANTISYMMETRIC_RANKS = {
    "K": 2,
    "n20": 2,
    "h20": 2,
    "n40": 4,
    "h40": 4,
}


def _index_sort_key(label: str) -> tuple[int, int | str]:
    match = re.search(r"(-?\d+)(?!.*\d)", label)
    return (0, int(match.group(1))) if match else (1, label)


def _permutation_sign(indices: tuple[int, ...]) -> int:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    )
    return -1 if inversions % 2 else 1


def _canonical_labels(
    tensors: Iterable[tuple[str, tuple[str, ...]]],
    external_count: int,
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Canonicalize dummy labels and antisymmetric tensor slots.

    Independent generators need not choose the same dummy names or the same
    ordering of indices on an antisymmetric moment.  Enumerating the at-most
    four dummy labels produces a stable contraction-graph key and its parity.
    """

    tensors = tuple(tensors)
    external_raw: dict[str, str] = {}
    dummy_raw: set[str] = set()
    external_names = "pqrs"[:external_count]
    for _name, labels in tensors:
        for label in labels:
            match = re.search(r"(-?\d+)(?!.*\d)", label)
            if match and int(match.group(1)) >= 90:
                ordinal = int(match.group(1)) - 90
                if 0 <= ordinal < external_count:
                    external_raw[label] = external_names[ordinal]
                    continue
            if label in external_names:
                external_raw[label] = label
            else:
                dummy_raw.add(label)
    raw_order = tuple(sorted(dummy_raw, key=_index_sort_key))
    if len(raw_order) > len(_DUMMY_NAMES):
        raise ValueError("too many dummy labels in a projected moment term")

    candidates: list[tuple[tuple[tuple[str, str], ...], int]] = []
    for assigned in permutations(_DUMMY_NAMES[: len(raw_order)]):
        mapping = {
            **dict(zip(raw_order, assigned)),
            **external_raw,
        }
        sign = 1
        factors: list[tuple[str, str]] = []
        for name, labels in tensors:
            mapped = tuple(mapping[label] for label in labels)
            antisymmetric_rank = _ANTISYMMETRIC_RANKS.get(name)
            if antisymmetric_rank is not None:
                if len(mapped) != antisymmetric_rank:
                    raise ValueError(
                        f"{name} must have rank {antisymmetric_rank}, not {len(mapped)}"
                    )
                if len(set(mapped)) != len(mapped):
                    sign = 0
                    break
                order = tuple(sorted(range(len(mapped)), key=mapped.__getitem__))
                sign *= _permutation_sign(order)
                mapped = tuple(mapped[index] for index in order)
            factors.append((name, "".join(mapped)))
        candidates.append((tuple(sorted(factors)), sign))

    canonical = min(key for key, _sign in candidates)
    signs = {sign for key, sign in candidates if key == canonical}
    if signs == {-1, 1} or 0 in signs:
        return canonical, 0
    return canonical, signs.pop()


def _consolidate(
    terms: Iterable[tuple[tuple[tuple[str, str], ...], Fraction]]
) -> dict[tuple[tuple[str, str], ...], Fraction]:
    result: dict[tuple[tuple[str, str], ...], Fraction] = defaultdict(Fraction)
    for tensors, coefficient in terms:
        result[tensors] += coefficient
    return {tensors: value for tensors, value in result.items() if value}


def canonicalize_autogen(artifact: dict[str, Any]) -> dict[str, dict[Any, Fraction]]:
    if artifact.get("projection_equation_schema") != EQUATION_SCHEMA:
        raise ValueError("autogen projected artifact has the wrong equation schema")
    if artifact.get("projector_ordering") != PROJECTOR_ORDERING:
        raise ValueError("autogen projected artifact has the wrong projector ordering")
    canonical: dict[str, dict[Any, Fraction]] = {}
    for output in OUTPUTS:
        entries = artifact["outputs"][output]["terms"]
        external_count = len(entries[0]["output_labels"]) if entries else 0
        terms = []
        for entry in entries:
            tensors = tuple(
                (tensor["name"], tuple(tensor["labels"]))
                for tensor in entry["tensors"]
            )
            coefficient = entry["coefficient"]
            canonical_labels, sign = _canonical_labels(tensors, external_count)
            terms.append(
                (
                    canonical_labels,
                    sign
                    * Fraction(
                        coefficient["numerator"], coefficient["denominator"]
                    ),
                )
            )
        canonical[output] = _consolidate(terms)
    return canonical


def canonicalize_sequant(artifact: dict[str, Any]) -> dict[str, dict[Any, Fraction]]:
    if artifact.get("projection_equation_schema") != EQUATION_SCHEMA:
        raise ValueError("SeQuant projected artifact has the wrong equation schema")
    if artifact.get("projector_ordering") != PROJECTOR_ORDERING:
        raise ValueError("SeQuant projected artifact has the wrong projector ordering")
    canonical: dict[str, dict[Any, Fraction]] = {}
    for output in OUTPUTS:
        block = artifact["outputs"][output]
        external_count = len(block["external_indices"])
        terms = []
        for entry in block["terms"]:
            tensors = tuple(
                (
                    tensor["name"],
                    tuple((*tensor.get("bra", ()), *tensor.get("ket", ()))),
                )
                for tensor in entry["tensors"]
            )
            coefficient = entry["coefficient"]
            canonical_labels, sign = _canonical_labels(tensors, external_count)
            terms.append(
                (
                    canonical_labels,
                    sign
                    * Fraction(
                        coefficient["numerator"], coefficient["denominator"]
                    ),
                )
            )
        canonical[output] = _consolidate(terms)
    return canonical


def compare_projected_artifacts(
    autogen_artifact: dict[str, Any],
    sequant_artifact: dict[str, Any],
) -> None:
    left = canonicalize_autogen(autogen_artifact)
    right = canonicalize_sequant(sequant_artifact)
    mismatches = []
    for output in OUTPUTS:
        if left[output] != right[output]:
            only_left = left[output].keys() - right[output].keys()
            only_right = right[output].keys() - left[output].keys()
            differing = {
                key
                for key in left[output].keys() & right[output].keys()
                if left[output][key] != right[output][key]
            }
            mismatches.append(
                f"{output}: autogen-only={len(only_left)}, "
                f"sequant-only={len(only_right)}, coefficients={len(differing)}"
            )
    if mismatches:
        raise AssertionError("projected generator mismatch: " + "; ".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare autogen and SeQuant PN-OAP moments exactly."
    )
    parser.add_argument("autogen_json", type=Path)
    parser.add_argument("sequant_json", type=Path)
    args = parser.parse_args()
    compare_projected_artifacts(
        json.loads(args.autogen_json.read_text()),
        json.loads(args.sequant_json.read_text()),
    )
    print("projected autogen/SeQuant exact-rational parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
