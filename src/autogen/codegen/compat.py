"""Compatibility helpers for script-backed codegen entrypoints."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[3]


def load_spec_namespace(spec_path) -> dict:
    """Load a Python spec file into a plain namespace dict."""
    return runpy.run_path(str(spec_path))


def resolve_output_dir(output_dir, spec_path, *, root: Path | None = None) -> Path:
    """Resolve a spec output directory against the repository root."""
    base = ROOT if root is None else Path(root)
    if output_dir is None:
        return base / "generated_code" / "methods" / Path(spec_path).stem
    resolved = Path(output_dir)
    if resolved.is_absolute():
        return resolved
    return base / resolved


def normalize_tasks(tasks_override, spec_tasks) -> list[str]:
    """Apply CLI task overrides on top of spec-declared tasks."""
    if tasks_override is None:
        return [str(task) for task in spec_tasks]
    return [task.strip() for task in str(tasks_override).split(",") if task.strip()]
