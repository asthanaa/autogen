# Autogen documentation

Autogen is an installable Wick-theorem algebra library with method-centered CCSD,
EOM-CCSD, and QPCCSD implementations.

## Start here

- [Overview](overview.md)
- [Installation](installation.md)
- [Usage](usage.md)
- [API guide](api.md)
- [Methods](methods/index.md)
- [Repository layout](repository_layout.md)
- [Testing and validation](testing.md)
- [Provenance and archives](provenance.md)

## Canonical imports

Symbolic APIs use `autogen.library`, `autogen.main_tools`, and `autogen.pkg`. Method APIs
use `autogen.methods.ccsd`, `autogen.methods.eom_ccsd`, and
`autogen.methods.qpccsd`.

The source of truth is always the package under `src/autogen`. Legacy generated trees,
campaign directories, and archive snapshots are not alternative import roots.
