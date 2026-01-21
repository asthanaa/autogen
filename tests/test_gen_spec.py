import importlib.util
from pathlib import Path

import pytest


GEN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gen_einsum.py"
SPEC = importlib.util.spec_from_file_location("gen_einsum", GEN_PATH)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


def test_parse_spec_terms_infers_output():
    spec = {"TERMS": [(1.0, ["X1", "F1"]), (0.5, ["F1", "T1"])]}
    (
        terms,
        output_names,
        tensor_map,
        view_tensors,
        output_dir,
        tasks,
        pyscf_mol,
        spin_orbital,
        spin_adapted,
    ) = gen.parse_spec_terms(spec)
    assert terms[0]["output_key"] == "X1"
    assert terms[1]["output_key"] == "scalar"
    assert gen.resolve_output_name("X1", output_names) == "r1"
    assert gen.resolve_output_name("scalar", output_names) == "scalar"
    assert "F1" in tensor_map
    assert view_tensors == ["g", "f"]
    assert output_dir is None
    assert tasks == []
    assert pyscf_mol is None
    assert spin_orbital is False
    assert spin_adapted is False


def test_parse_spec_terms_dict_format():
    spec = {
        "TERMS": [
            {"output": "R2", "fac": 1.0, "ops": ["X2", "V2", "T1"]},
        ],
        "OUTPUTS": {"R2": "r2_custom"},
    }
    (
        terms,
        output_names,
        _tensor_map,
        _view_tensors,
        _output_dir,
        _tasks,
        _pyscf_mol,
        _spin_orbital,
        _spin_adapted,
    ) = gen.parse_spec_terms(spec)
    assert terms[0]["output_key"] == "R2"
    assert gen.resolve_output_name("R2", output_names) == "r2_custom"


def test_parse_spec_terms_missing_terms():
    with pytest.raises(ValueError):
        gen.parse_spec_terms({})
