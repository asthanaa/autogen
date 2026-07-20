"""Helpers for creating a new theory scaffold."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def create_theory_scaffold(theory_name: str, *, root: Path | None = None, overwrite: bool = False) -> dict[str, Path]:
    """Create a minimal spec, docs page, and test bundle for a new theory."""
    repo_root = ROOT if root is None else Path(root)
    slug = theory_name.strip().replace("-", "_")
    if not slug.isidentifier():
        raise ValueError(f"Invalid theory name '{theory_name}'. Use a valid Python identifier.")

    spec_dir = repo_root / "method_inputs" / slug
    test_dir = repo_root / "tests" / "theories" / slug
    docs_dir = repo_root / "docs" / "theories"
    generated_dir = repo_root / "generated_code" / "methods" / slug

    targets = {
        "spec": spec_dir / f"{slug}_spec.py",
        "test": test_dir / f"test_{slug}_smoke.py",
        "docs": docs_dir / f"{slug}.md",
        "generated_init": generated_dir / "__init__.py",
    }

    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing scaffold target: {path}")

    spec_text = f'''"""Spec for {_title(slug)}."""

THEORY_NAME = "{slug}"
OUTPUT_DIR = "generated_code/methods/{slug}"
OUTPUTS = {{"X1": "r1", "X2": "r2", "scalar": "energy"}}
VIEW_TENSORS = ("g", "f")
TASKS = ["runtime"]
BACKEND_CAPABILITIES = ["standard_bch"]
RUNTIME_OPTIONS = {{"mode": "full"}}

# Replace this with the actual symbolic terms for the theory.
TERMS = [
    (1.0, ["X1", "F1"]),
    (1.0, ["X2", "V2"]),
]
'''

    test_text = f'''from autogen.codegen import load_method_spec


def test_{slug}_spec_loads():
    spec = load_method_spec("method_inputs/{slug}/{slug}_spec.py")
    assert spec.theory_name == "{slug}"
    assert spec.terms
'''

    docs_text = f"""# {_title(slug)}

## Status

This page is the developer-facing entry point for the `{slug}` theory scaffold.

## Required implementation checklist

- Replace the placeholder `TERMS` in `method_inputs/{slug}/{slug}_spec.py`
- Declare any non-default backend capabilities
- Add an oracle if the theory has a tractable exact or reference evaluator
- Add regression tests under `tests/theories/{slug}/`
- Document conventions, limits, and expected reductions
"""

    targets["spec"].write_text(spec_text)
    targets["test"].write_text(test_text)
    targets["docs"].write_text(docs_text)
    targets["generated_init"].write_text("")
    return targets
