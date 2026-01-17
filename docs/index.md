# Autogen documentation

Autogen is an automatic expression generator using Wick’s theorem for quantum chemistry derivations.

This `docs/` folder is the main documentation. The root README is intentionally short and points here.

## Start here

- Overview: [overview.md](overview.md)
- Install: [installation.md](installation.md)
- Usage (examples): [usage.md](usage.md)
- Concepts/definitions: [concepts.md](concepts.md)
- API guide: [api.md](api.md)
- Input/output: [io.md](io.md)
- Testing: [testing.md](testing.md)

## Imports

All code should import from the canonical package paths:

- `autogen.library`
- `autogen.main_tools`
- `autogen.pkg`

## Repository layout

- `src/autogen/` – the installable Python package
- `tests/` – pytest suite
- `docs/` – documentation (this folder)
