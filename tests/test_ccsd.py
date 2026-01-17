import os

import pytest

from . import ccsd_amplitude as ccsd


@pytest.mark.slow
def test_ccsd_amplitude_generation():
	if os.environ.get("RUN_SLOW", "0") != "1":
		pytest.skip("Set RUN_SLOW=1 to run CCSD amplitude generation")
	ccsd.amplitude()
