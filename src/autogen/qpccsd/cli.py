"""Compatibility forwarder to the canonical QPCCSD command-line facade."""

from autogen.methods.qpccsd.cli import *  # noqa: F401,F403
from autogen.methods.qpccsd.cli import main


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI smoke tests
    raise SystemExit(main())
