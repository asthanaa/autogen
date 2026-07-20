from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


OUTPUTS = ("energy", "r1", "r2")
SEQUANT_REVISION = "db5dd8dae6408c764a84c2b165cd9e413ab91ac6"
_DUMMY_NAMES = tuple("ijklmnopqrstuvwxyzabcdefgh")
_AUTOGEN_LABEL = re.compile(r"u\d+|[pqrs]")
_SEQUANT_TO_AUTOGEN = {
    f"Omega{suffix}": f"h{suffix}"
    for suffix in ("00", "02", "04", "11", "13", "20", "22", "31", "40")
}
_ANTISYMMETRIC_GROUPS = {
    "h02": ((0, 1),),
    "h04": ((0, 1, 2, 3),),
    "h13": ((1, 2, 3),),
    "h20": ((0, 1),),
    "h22": ((0, 1), (2, 3)),
    "h31": ((0, 1, 2),),
    "h40": ((0, 1, 2, 3),),
    "t1": ((0, 1),),
    "t2": ((0, 1, 2, 3),),
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


def _canonical_term(
    tensors: Iterable[tuple[str, tuple[str, ...]]],
    external_count: int,
) -> tuple[tuple[tuple[str, str], ...], int]:
    tensors = tuple(tensors)
    external_names = "pqrs"[:external_count]
    external_raw: dict[str, str] = {}
    dummy_raw: set[str] = set()
    for _name, labels in tensors:
        for label in labels:
            match = re.search(r"(-?\d+)(?!.*\d)", label)
            if match and int(match.group(1)) >= 90:
                ordinal = int(match.group(1)) - 90
                if not 0 <= ordinal < external_count:
                    raise ValueError(f"invalid external SeQuant index {label!r}")
                external_raw[label] = external_names[ordinal]
            elif label in external_names:
                external_raw[label] = label
            else:
                dummy_raw.add(label)

    raw_order = tuple(sorted(dummy_raw, key=_index_sort_key))
    if len(raw_order) > len(_DUMMY_NAMES):
        raise ValueError("too many dummy labels in a QPCCSD term")
    candidates: list[tuple[tuple[tuple[str, str], ...], int]] = []
    for assigned in permutations(_DUMMY_NAMES[: len(raw_order)]):
        mapping = {**dict(zip(raw_order, assigned)), **external_raw}
        sign = 1
        factors: list[tuple[str, str]] = []
        for raw_name, labels in tensors:
            name = _SEQUANT_TO_AUTOGEN.get(raw_name, raw_name)
            mapped = [mapping[label] for label in labels]
            for group in _ANTISYMMETRIC_GROUPS.get(name, ()):
                values = [mapped[index] for index in group]
                if len(set(values)) != len(values):
                    sign = 0
                    break
                order = tuple(sorted(range(len(values)), key=values.__getitem__))
                sign *= _permutation_sign(order)
                for destination, source in zip(group, order):
                    mapped[destination] = values[source]
            factors.append((name, "".join(mapped)))
        candidates.append((tuple(sorted(factors)), sign))

    canonical = min(key for key, _sign in candidates)
    signs = {sign for key, sign in candidates if key == canonical}
    if signs == {-1, 1} or 0 in signs:
        return canonical, 0
    return canonical, signs.pop()


def _consolidate(
    terms: Iterable[tuple[tuple[tuple[str, str], ...], Fraction]],
) -> dict[tuple[tuple[str, str], ...], Fraction]:
    result: dict[tuple[tuple[str, str], ...], Fraction] = defaultdict(Fraction)
    for tensors, coefficient in terms:
        result[tensors] += coefficient
    return {tensors: value for tensors, value in result.items() if value}


def canonicalize_autogen(artifact: dict[str, Any]) -> dict[str, dict[Any, Fraction]]:
    if artifact.get("schema_version") != 2:
        raise ValueError("autogen QPCCSD artifact has the wrong schema")
    if artifact.get("coefficient_encoding") != "exact-rational":
        raise ValueError("autogen QPCCSD coefficients are not exact rational")
    if artifact.get("amplitude_convention") != "T=t1/2!+t2/4!":
        raise ValueError("autogen QPCCSD amplitude convention differs")

    result: dict[str, dict[Any, Fraction]] = {}
    for output in OUTPUTS:
        terms = []
        for entry in artifact["outputs"][output]["terms"]:
            external_count = len(entry["output_labels"])
            tensors = tuple(
                (tensor["name"], tuple(_AUTOGEN_LABEL.findall(tensor["labels"])))
                for tensor in entry["tensors"]
            )
            key, sign = _canonical_term(tensors, external_count)
            coefficient = entry["coefficient"]
            terms.append(
                (
                    key,
                    sign
                    * Fraction(
                        coefficient["numerator"], coefficient["denominator"]
                    ),
                )
            )
        result[output] = _consolidate(terms)
    return result


def canonicalize_sequant(artifact: dict[str, Any]) -> dict[str, dict[Any, Fraction]]:
    if artifact.get("schema_version") != 1:
        raise ValueError("SeQuant raw QPCCSD artifact has the wrong schema")
    if artifact.get("artifact_format") != "sequant-exact-rational-raw-terms":
        raise ValueError("SeQuant QPCCSD artifact is not exact rational")
    if artifact.get("sequant_revision") != SEQUANT_REVISION:
        raise ValueError("SeQuant QPCCSD artifact uses an unexpected revision")
    if artifact.get("amplitude_convention") != "T=t1/2!+t2/4!":
        raise ValueError("SeQuant QPCCSD amplitude convention differs")

    result: dict[str, dict[Any, Fraction]] = {}
    scalar_seen = False
    for output in OUTPUTS:
        external_count = len(artifact["outputs"][output]["external_indices"])
        terms = []
        for entry in artifact["outputs"][output]["terms"]:
            tensors = tuple(
                (
                    tensor["name"],
                    tuple((*tensor.get("bra", ()), *tensor.get("ket", ()))),
                )
                for tensor in entry["tensors"]
            )
            mapped_names = {_SEQUANT_TO_AUTOGEN.get(name, name) for name, _ in tensors}
            if "h00" in mapped_names:
                coefficient = entry["coefficient"]
                if (
                    output != "energy"
                    or tensors != (("Omega00", ()),)
                    or coefficient != {"numerator": 1, "denominator": 1}
                ):
                    raise ValueError("unexpected SeQuant scalar-Hamiltonian term")
                scalar_seen = True
                continue
            key, sign = _canonical_term(tensors, external_count)
            coefficient = entry["coefficient"]
            terms.append(
                (
                    key,
                    sign
                    * Fraction(
                        coefficient["numerator"], coefficient["denominator"]
                    ),
                )
            )
        result[output] = _consolidate(terms)
    if not scalar_seen:
        raise ValueError("SeQuant QPCCSD artifact lacks the unit h00 term")
    return result


def compare_unprojected_artifacts(
    autogen_artifact: dict[str, Any],
    sequant_artifact: dict[str, Any],
) -> None:
    left = canonicalize_autogen(autogen_artifact)
    right = canonicalize_sequant(sequant_artifact)
    mismatches = []
    for output in OUTPUTS:
        if left[output] == right[output]:
            continue
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
        raise AssertionError("unprojected generator mismatch: " + "; ".join(mismatches))


def evaluate_sequant_raw_terms(
    artifact: dict[str, Any],
    tensors: dict[str, np.ndarray | complex | float],
    *,
    skip_tensor_names: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray | complex]:
    """Evaluate a SeQuant raw-term artifact without using generated kernels."""

    nspin = next(
        (
            int(np.asarray(value).shape[0])
            for value in tensors.values()
            if np.asarray(value).ndim
        ),
        None,
    )
    outputs: dict[str, np.ndarray | complex] = {}
    for output, block in artifact["outputs"].items():
        external_count = len(block["external_indices"])
        external = tuple(f"p_{90 + index}" for index in range(external_count))
        accumulator: np.ndarray | complex | None = None
        for entry in block["terms"]:
            if any(
                tensor["name"] in skip_tensor_names
                for tensor in entry["tensors"]
            ):
                continue
            raw_labels = [
                tuple((*tensor.get("bra", ()), *tensor.get("ket", ())))
                for tensor in entry["tensors"]
            ]
            labels = list(dict.fromkeys(label for values in (*raw_labels, external) for label in values))
            label_ids = {label: index for index, label in enumerate(labels)}
            operands: list[Any] = []
            for tensor, indices in zip(entry["tensors"], raw_labels):
                try:
                    value = tensors[tensor["name"]]
                except KeyError as error:
                    raise KeyError(
                        f"missing raw SeQuant tensor {tensor['name']!r}"
                    ) from error
                operands.extend(
                    (np.asarray(value), [label_ids[index] for index in indices])
                )
            operands.append([label_ids[index] for index in external])
            value = np.einsum(*operands, optimize=True)
            coefficient = entry["coefficient"]
            value = (
                coefficient["numerator"] / coefficient["denominator"]
            ) * value
            accumulator = value if accumulator is None else accumulator + value
        if accumulator is None:
            if external_count:
                if nspin is None:
                    raise ValueError("cannot infer the raw-term tensor dimension")
                accumulator = np.zeros((nspin,) * external_count)
            else:
                accumulator = 0.0
        outputs[output] = accumulator
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare autogen and SeQuant QPCCSD BCH terms exactly."
    )
    parser.add_argument("autogen_json", type=Path)
    parser.add_argument("sequant_json", type=Path)
    args = parser.parse_args()
    compare_unprojected_artifacts(
        json.loads(args.autogen_json.read_text()),
        json.loads(args.sequant_json.read_text()),
    )
    print("unprojected autogen/SeQuant exact-rational parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
