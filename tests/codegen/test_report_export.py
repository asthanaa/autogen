from __future__ import annotations

from pathlib import Path

from autogen.codegen.emit_reports import render_raw_split_pdfs, write_raw_tct


def _tiny_plan():
    return {
        "outputs": {
            "scalar": {
                "name": "energy",
                "labels": "",
                "raw_terms": [
                    {
                        "output_labels": "",
                        "tensors": [{"name": "h11", "labels": "pq"}, {"name": "z", "labels": "qp"}],
                        "coeff": 0.5,
                    }
                ],
                "emitted_terms": [],
            },
            "X1": {
                "name": "r1",
                "labels": "pq",
                "raw_terms": [
                    {
                        "output_labels": "pq",
                        "tensors": [{"name": "h11", "labels": "pr"}, {"name": "z", "labels": "rq"}],
                        "coeff": -1.0,
                    }
                ],
                "emitted_terms": [],
            },
            "X2": {
                "name": "r2",
                "labels": "pqrs",
                "raw_terms": [
                    {
                        "output_labels": "pqrs",
                        "tensors": [{"name": "h22", "labels": "pqtu"}, {"name": "z", "labels": "ur"}, {"name": "z", "labels": "ts"}],
                        "coeff": 1.0 / 6.0,
                    }
                ],
                "emitted_terms": [],
            },
        }
    }


def test_write_raw_tct_excludes_emitted_sections(tmp_path):
    out_path = tmp_path / "raw_terms.tct"
    report = write_raw_tct(_tiny_plan(), out_path, plan_path=Path("fake_plan.json"))
    text = out_path.read_text()
    assert report["outputs"]["scalar"]["raw_term_count"] == 1
    assert "artifact_type = raw_projection_terms" in text
    assert "emitted terms are intentionally excluded from this artifact" in text
    assert "[energy]" in text
    assert "[r1]" in text
    assert "[r2]" in text
    assert "emitted terms:" not in text
    assert "(+1/2)" in text or "(+1/6)" in text


def test_render_raw_split_pdfs_writes_portrait_index(tmp_path):
    generated = render_raw_split_pdfs(_tiny_plan(), tmp_path / "pdf", terms_per_pdf=1)
    assert generated
    assert generated[0].name == "projected_qp_raw_index.pdf"
    for path in generated:
        assert path.exists()
