"""Disabled QPCCSD research variants.

Importing the canonical :mod:`autogen.methods.qpccsd` package never imports
or selects anything in this namespace. Each variant has a distinct method
identity and must be requested explicitly by its full module path.
"""

ENABLED_BY_DEFAULT = False

__all__: list[str] = []
