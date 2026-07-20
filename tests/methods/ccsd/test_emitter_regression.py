from __future__ import annotations

import ast

from autogen.codegen import cli


def test_emitted_ccsd_solver_defines_loop_state_before_use(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    destination = tmp_path / "candidate"
    terms = [
        {"output_key": "X1", "fac": 1.0, "ops": ["X1", "F1"]},
        {"output_key": "X2", "fac": 1.0, "ops": ["X2", "V2"]},
    ]
    cli.emit_ccsd_solver(destination, terms, {"X1": "r1", "X2": "r2"})
    source = (destination / "solver.py").read_text()
    ast.parse(source)

    tensor_definition = source.index("tensor_args =")
    first_tensor_mutation = source.index("tensor_args['t1']")
    energy_difference = source.index("e_diff = abs(new_energy - energy)")
    energy_print = source.index("dE {e_diff:.3e}")
    convergence_use = source.index("if r_norm < tol and e_diff < tol")
    assert tensor_definition < first_tensor_mutation
    assert energy_difference < energy_print < convergence_use
