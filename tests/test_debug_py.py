import subprocess
import sys
from pathlib import Path


def test_debug_py_runs_as_script(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    debug_py = project_root / "debug.py"

    result = subprocess.run(
        [sys.executable, str(debug_py)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out_file = tmp_path / "latex_output.txt"
    assert out_file.exists()
    assert out_file.stat().st_size > 0
