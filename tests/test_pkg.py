from autogen.pkg.fewt import ewt


def test_ewt_smoke():
	a, b = ewt(['i0'], ['a0'], ['p0', 'q0'], ['r0', 's0'])
	assert a is not None
	assert b is not None
