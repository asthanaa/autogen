"""Exact-rational canonical artifact for factorized PN-OAP left moments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _tensor(name: str, labels: str) -> dict[str, str]:
    return {"name": name, "labels": labels}


def _term(
    output_labels: str,
    coefficient: int,
    tensors: Iterable[tuple[str, str]],
) -> dict:
    return {
        "output_labels": output_labels,
        "tensors": [_tensor(name, labels) for name, labels in tensors],
        "coefficient": {"numerator": coefficient, "denominator": 1},
    }


def _scalar(name: str) -> list[dict]:
    return [_term("", 1, ((name, ""),))]


def _pair(name0: str, name2: str) -> list[dict]:
    return [
        _term("pq", 1, (("K", "pq"), (name0, ""))),
        _term("pq", 1, (("A", "pi"), ("A", "qj"), (name2, "ij"))),
    ]


def _quad(name0: str, name2: str, name4: str) -> list[dict]:
    terms = [
        _term("pqrs", 1, (("K", "pq"), ("K", "rs"), (name0, ""))),
        _term("pqrs", -1, (("K", "pr"), ("K", "qs"), (name0, ""))),
        _term("pqrs", 1, (("K", "ps"), ("K", "qr"), (name0, ""))),
    ]
    wedge = (
        (1, "pq", "r", "s"),
        (-1, "pr", "q", "s"),
        (1, "ps", "q", "r"),
        (1, "qr", "p", "s"),
        (-1, "qs", "p", "r"),
        (1, "rs", "p", "q"),
    )
    for coefficient, k_labels, left, right in wedge:
        terms.append(
            _term(
                "pqrs",
                coefficient,
                (
                    ("K", k_labels),
                    ("A", f"{left}i"),
                    ("A", f"{right}j"),
                    (name2, "ij"),
                ),
            )
        )
    terms.append(
        _term(
            "pqrs",
            1,
            (
                ("A", "pi"),
                ("A", "qj"),
                ("A", "rk"),
                ("A", "sl"),
                (name4, "ijkl"),
            ),
        )
    )
    return terms


def build_projected_moment_artifact() -> dict:
    outputs = {
        "N0": _scalar("n0"),
        "N20": _pair("n0", "n20"),
        "N40": _quad("n0", "n20", "n40"),
        "H0": _scalar("h0"),
        "H20": _pair("h0", "h20"),
        "H40": _quad("h0", "h20", "h40"),
    }
    return {
        "schema_version": 2,
        "coefficient_encoding": "exact-rational",
        "artifact": "factorized-pn-oap-left-moments",
        "projection_equation_schema": "pn-oap-bpn-v2",
        "projector_ordering": "B_mu P_N (H-E) exp(T)",
        "probe_factorization": "B_ZR=A*beta+C*beta_dagger; K=<B_ZR B_ZR>=Z",
        "maximum_fullspace_tensor_rank": 4,
        "outputs": {
            name: {
                "count_canonical": len(terms),
                "terms": terms,
            }
            for name, terms in outputs.items()
        },
    }


def write_projected_moment_artifact(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_projected_moment_artifact(), indent=2) + "\n")
    return target


if __name__ == "__main__":
    write_projected_moment_artifact(
        Path(__file__).resolve().parents[1]
        / "equations"
        / "projected_moment_canonical_terms.json"
    )
