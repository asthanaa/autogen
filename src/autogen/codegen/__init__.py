"""Package-backed code generation interfaces.

The long-term goal is to keep the stable CLI entrypoints in ``scripts/`` while
moving supported implementation code under ``src/autogen``. New theory support
should target this package rather than editing ad hoc script files directly.
"""

from .spec_model import MethodSpec, ParsedTerm, default_output_name, load_method_spec, parse_legacy_spec_terms, resolve_output_name

__all__ = [
    "MethodSpec",
    "ParsedTerm",
    "default_output_name",
    "load_method_spec",
    "parse_legacy_spec_terms",
    "resolve_output_name",
]
