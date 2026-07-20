from __future__ import annotations

import json

from autogen.codegen.projected_terms import (
    canonicalize_projected_tensor,
    build_runtime_plan,
    plan_structured_term,
    summarize_runtime_plan,
    write_numpy_runtime_module,
    write_grouped_term_artifact,
)


def test_qp_t2_is_canonicalized_as_a_fully_antisymmetric_tensor():
    labels, sign = canonicalize_projected_tensor("t2", "qprs")

    assert labels == "pqrs"
    assert sign == -1


def test_planner_prefers_lower_scaling_binary_tree():
    term = plan_structured_term(
        "ag",
        [
            ("A", "abcdef"),
            ("B", "gfbd"),
            ("C", "ce"),
        ],
        name="synthetic",
    )

    assert term.formal_scaling == 6
    assert term.steps[0].arg_refs == ("A", "C")
    assert term.steps[0].formal_scaling == 6
    assert term.steps[-1].formal_scaling == 5


def test_grouped_term_artifact_is_deterministic_and_contains_scaling_metadata(tmp_path):
    raw_outputs = {
        "r1": [
            ("ag", [("A", "abcdef"), ("B", "gfbd"), ("C", "ce")], 1.0),
            ("ag", [("A", "abcdef"), ("B", "gfbd"), ("C", "ce")], -0.5),
            ("ag", [("A", "abcdef"), ("C", "ce"), ("B", "gfbd")], 0.5),
        ]
    }
    first = tmp_path / "grouped_a.json"
    second = tmp_path / "grouped_b.json"

    write_grouped_term_artifact(raw_outputs, first)
    write_grouped_term_artifact(raw_outputs, second)

    first_text = first.read_text()
    second_text = second.read_text()
    assert first_text == second_text

    payload = json.loads(first_text)
    assert payload["stats"]["final_call_count"] > 0
    assert payload["outputs"]["r1"]["groups"]
    assert payload["outputs"]["r1"]["terms"][0]["steps"][0]["formal_scaling"] >= 1
    assert "flop_cost" in payload["outputs"]["r1"]["terms"][0]["steps"][0]
    assert "result_size" in payload["outputs"]["r1"]["terms"][0]["steps"][0]


def test_runtime_plan_summary_exposes_scaling_stats_and_worst_groups():
    raw_outputs = {
        "energy": [
            ("ag", [("A", "abcdef"), ("B", "gfbd"), ("C", "ce")], 1.0),
        ],
        "r2": [
            ("pqrs", [("H", "qt"), ("Z1", "pt"), ("Z2", "rs")], 1.0),
        ],
    }

    plan = build_runtime_plan(raw_outputs, reduction="canonical", use_intermediates=True)
    summary = summarize_runtime_plan(plan)

    assert summary["stats"]["final_call_count"] > 0
    assert summary["stats"]["max_formal_scaling"] >= 4
    assert summary["outputs"]["r2"]["max_formal_scaling"] >= 4
    assert summary["worst_groups"]
    assert summary["worst_groups"][0]["max_formal_scaling"] >= 4


def test_numpy_runtime_module_reuses_local_slots(tmp_path):
    plan = build_runtime_plan(
        {"energy": [("", [("a", "pq"), ("b", "pq")], 0.5)]},
        reduction="canonical",
    )
    path = tmp_path / "kernel.py"

    write_numpy_runtime_module(plan, path, tensor_names=("t1", "a", "b"))

    text = path.read_text()
    assert "np.einsum" in text
    assert "del _s0" in text
    assert "autogen" not in text
