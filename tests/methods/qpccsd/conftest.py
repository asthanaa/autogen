from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture(scope="session")
def n2_sto3g_r1p10_anchor() -> dict[str, object]:
    path = FIXTURE_ROOT / "n2_sto3g_cas66_r1p10_certified.json"
    return json.loads(path.read_text(encoding="utf-8"))
