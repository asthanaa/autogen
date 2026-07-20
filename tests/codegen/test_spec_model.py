from __future__ import annotations

from pathlib import Path

from autogen.codegen.scaffold import create_theory_scaffold
from autogen.codegen.spec_model import load_method_spec


ROOT = Path(__file__).resolve().parents[2]


def test_qp_ccsd_spec_declares_capabilities_and_oracle():
    spec = load_method_spec(ROOT / "method_inputs" / "qp_ccsd" / "qp_ccsd_spec.py")
    assert spec.theory_name == "qp_ccsd"
    assert "qp" in spec.backend_capabilities
    assert "projected_qp" in spec.backend_capabilities
    assert "spin_orbital" in spec.backend_capabilities
    assert "qp_ccsd_solver" not in spec.tasks
    assert spec.oracle_factory == "autogen.reference.projected_qp.build_projected_exact_reference"


def test_new_theory_scaffold_creates_loadable_spec(tmp_path):
    created = create_theory_scaffold("demo_theory", root=tmp_path)
    assert created["spec"].exists()
    assert created["test"].exists()
    assert created["docs"].exists()
    assert created["generated_init"].exists()

    spec = load_method_spec(created["spec"])
    assert spec.theory_name == "demo_theory"
    assert spec.terms
