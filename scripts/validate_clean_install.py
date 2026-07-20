#!/usr/bin/env python3
"""Install a built wheel in an empty environment and verify public surfaces."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


INSTALL_CHECK = r"""
from importlib.metadata import distribution
from importlib.resources import files

import autogen
import autogen.methods.ccsd
import autogen.methods.eom_ccsd
import autogen.methods.qpccsd
import autogen.qpccsd
import autogen.reference
import generated_code.methods.ccsd.ccsd_amplitude.solver
import generated_code.methods.eom_ccsd.eom_solver
import generated_code.pyscf_integrals

dist = distribution("autogen-wick")
assert autogen.__version__ == dist.version
scripts = {ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"}
assert scripts.get("qpccsd") == "autogen.methods.qpccsd.cli:main", scripts
generated = files("autogen.methods.qpccsd.generated")
for name in ("qpccsd_canonical_terms.json", "projected_moment_canonical_terms.json"):
    assert generated.joinpath(name).is_file(), name
print(f"clean install validated: autogen-wick {dist.version}")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--no-deps", action="store_true")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"wheel does not exist: {wheel}")
    with tempfile.TemporaryDirectory(prefix="autogen-wheel-") as temp_name:
        temp = Path(temp_name)
        env_dir = temp / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
        executable_dir = "Scripts" if os.name == "nt" else "bin"
        python = env_dir / executable_dir / ("python.exe" if os.name == "nt" else "python")
        pip = env_dir / executable_dir / ("pip.exe" if os.name == "nt" else "pip")
        install = [str(pip), "install", "--disable-pip-version-check"]
        if args.no_deps:
            install.append("--no-deps")
        install.append(str(wheel))
        subprocess.run(install, check=True, cwd=temp)
        subprocess.run([str(python), "-c", INSTALL_CHECK], check=True, cwd=temp)
        cli = env_dir / executable_dir / ("qpccsd.exe" if os.name == "nt" else "qpccsd")
        subprocess.run([str(cli), "--help"], check=True, cwd=temp, stdout=subprocess.PIPE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
