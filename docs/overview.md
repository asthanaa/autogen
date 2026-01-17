# Overview

Autogen is an automatic expression generator for second-quantized many-body expressions using Wick’s theorem.

It is designed to generate algebraic expressions in the same “diagram-style” symbolic form commonly used in electronic structure theory derivations (e.g., coupled cluster and unitary coupled cluster).

## What it produces

- Lists of symbolic “terms” representing operator strings, contractions, delta factors, and prefactors.
- LaTeX-friendly output via the printing utilities.

## Key features

- String-based operator specification (e.g., `['V2']`, `['T1']`) with internal expansion to operator objects.
- Recursive handling of nested commutators (innermost-first).
- Support for de-excitation operators (e.g., `D1`, `D2`) to enable UCC-style algebra.

In particular, features like de-excitation support and nested commutators (e.g. `[[V2, T1], D1]`) make Autogen suitable for unitary coupled cluster (UCC) style derivations.

## Package layout

- `autogen.library` – core term/operator structures and utilities
- `autogen.main_tools` – commutator/product/driver orchestration
- `autogen.pkg` – EWT/GWT utilities

Next: see [usage.md](usage.md) for runnable examples.