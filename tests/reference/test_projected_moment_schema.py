from __future__ import annotations

from itertools import permutations
import json
from math import factorial
from pathlib import Path

import numpy as np

from autogen.codegen.qpccsd_projected_moments import build_projected_moment_artifact
from autogen.codegen.compare_projected_generators import compare_projected_artifacts
from autogen.codegen.compare_unprojected_generators import evaluate_sequant_raw_terms
from autogen.methods.qpccsd.generated.generated_projected_moments import (
    projected_left_moments,
)


def _antisymmetrize(tensor: np.ndarray) -> np.ndarray:
    result = np.zeros_like(tensor)
    for permutation in permutations(range(tensor.ndim)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(len(permutation))
            for right in range(left + 1, len(permutation))
        )
        result += (-1 if inversions % 2 else 1) * tensor.transpose(permutation)
    return result / factorial(tensor.ndim)


def _synthetic_sequant_artifact(autogen_artifact: dict) -> dict:
    label_map = {
        "p": "p_90",
        "q": "p_91",
        "r": "p_92",
        "s": "p_93",
        "i": "p_10",
        "j": "p_11",
        "k": "p_12",
        "l": "p_13",
    }
    outputs = {}
    for name, block in autogen_artifact["outputs"].items():
        entries = []
        for term in block["terms"]:
            tensors = []
            for tensor in reversed(term["tensors"]):
                labels = [label_map[label] for label in tensor["labels"]]
                split = 1 if tensor["name"] == "A" else len(labels)
                tensors.append(
                    {
                        "name": tensor["name"],
                        "bra": labels[:split],
                        "ket": labels[split:],
                    }
                )
            entries.append(
                {"coefficient": dict(term["coefficient"]), "tensors": tensors}
            )
        output_labels = block["terms"][0]["output_labels"] if entries else ""
        outputs[name] = {
            "external_indices": list(output_labels),
            "terms": entries,
        }
    return {
        "projection_equation_schema": autogen_artifact[
            "projection_equation_schema"
        ],
        "projector_ordering": autogen_artifact["projector_ordering"],
        "outputs": outputs,
    }


def test_projected_moment_artifact_is_exact_rational_and_current() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "autogen"
        / "methods"
        / "qpccsd"
        / "derivation"
        / "equations"
        / "projected_moment_canonical_terms.json"
    )
    checked_in = json.loads(path.read_text())
    assert checked_in == build_projected_moment_artifact()
    assert checked_in["projector_ordering"].startswith("B_mu P_N")
    assert checked_in["maximum_fullspace_tensor_rank"] == 4
    assert {
        name: output["count_canonical"]
        for name, output in checked_in["outputs"].items()
    } == {"N0": 1, "N20": 2, "N40": 10, "H0": 1, "H20": 2, "H40": 10}
    for output in checked_in["outputs"].values():
        for term in output["terms"]:
            coefficient = term["coefficient"]
            assert isinstance(coefficient["numerator"], int)
            assert coefficient["denominator"] == 1


def test_projected_generator_comparator_canonicalizes_sequant_raw_terms() -> None:
    autogen_artifact = build_projected_moment_artifact()
    sequant_artifact = _synthetic_sequant_artifact(autogen_artifact)
    compare_projected_artifacts(autogen_artifact, sequant_artifact)


def test_fresh_sequant_projected_terms_match_autogen_exactly() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "sequant_projected_moment_raw_terms.json"
    )
    sequant_artifact = json.loads(fixture.read_text())
    assert sequant_artifact["artifact_format"] == "sequant-exact-rational-raw-terms"
    assert (
        sequant_artifact["sequant_revision"]
        == "db5dd8dae6408c764a84c2b165cd9e413ab91ac6"
    )
    compare_projected_artifacts(
        build_projected_moment_artifact(),
        sequant_artifact,
    )


def test_fresh_sequant_projected_terms_match_numpy_kernel_numerically() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "sequant_projected_moment_raw_terms.json"
    )
    artifact = json.loads(fixture.read_text())
    rng = np.random.default_rng(913)
    nspin = 5
    pair = lambda: _antisymmetrize(rng.normal(size=(nspin, nspin)))
    quad = lambda: _antisymmetrize(rng.normal(size=(nspin,) * 4))
    tensors = {
        "A": rng.normal(size=(nspin, nspin)),
        "K": pair(),
        "n0": 0.91,
        "n20": pair(),
        "n40": quad(),
        "h0": -1.17,
        "h20": pair(),
        "h40": quad(),
    }
    raw = evaluate_sequant_raw_terms(artifact, tensors)
    generated = projected_left_moments(
        tensors["A"],
        tensors["K"],
        n0=tensors["n0"],
        n20=tensors["n20"],
        n40=tensors["n40"],
        h0=tensors["h0"],
        h20=tensors["h20"],
        h40=tensors["h40"],
    )
    for output in ("N0", "N20", "N40", "H0", "H20", "H40"):
        np.testing.assert_allclose(
            raw[output],
            getattr(generated, output.lower()),
            atol=1.0e-12,
        )
