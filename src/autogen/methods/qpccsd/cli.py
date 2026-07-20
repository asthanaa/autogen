"""Command-line facade for the reviewed production QPCCSD workflow."""

from .production.cli import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke tests
    raise SystemExit(main())
