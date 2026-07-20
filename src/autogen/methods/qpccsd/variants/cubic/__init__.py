"""Boundary for the separately versioned unprojected cubic-QPCCSD project.

The multi-gigabyte research campaign and its independent package are not
vendored into the production method. No cubic transformation is selected or
implemented through this namespace.
"""

METHOD_ID = "cubic-qpccsd-external-experiment"
ENABLED_BY_DEFAULT = False
IMPLEMENTED = False

__all__ = ["METHOD_ID", "ENABLED_BY_DEFAULT", "IMPLEMENTED"]
