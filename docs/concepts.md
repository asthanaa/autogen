# Concepts and definitions

This project uses short operator labels to represent common second-quantized building blocks.

## Operators

| Label | Meaning | Notes |
|---|---|---|
| `X1` | Project onto single excitation on the left | Used for amplitude/projection expressions |
| `X2` | Project onto double excitation on the left | Used for amplitude/projection expressions |
| `T1` | Single excitation cluster operator | $t_i^a a_a^{\dagger} a_i$ |
| `T2` | Double excitation cluster operator | $t_{ij}^{ab} a_a^{\dagger} a_b^{\dagger} a_j a_i$ |
| `D1` | Single de-excitation operator | $d_i^a a_i^{\dagger} a_a$ |
| `D2` | Double de-excitation operator | $d_{ij}^{ab} a_i^{\dagger} a_j^{\dagger} a_b a_a$ |
| `V2` | Two-body fluctuation operator | $\tfrac{1}{4}\langle pq\|rs\rangle a_p^{\dagger} a_q^{\dagger} a_s a_r$ |
| `F1` | One-body (Fock-like) operator | $f_{pq} a_p^{\dagger} a_q$ |

## Terms

A “term” is the internal object representing:

- A specific contraction pattern (operator string with contracted indices)
- Its prefactor and sign
- Summation indices and coefficient structure

Most top-level APIs return lists of terms.

## Commutators and products

- `comm(A, B, last)` computes a commutator (or nested commutator pieces).
- `prod(A, B, last)` computes an operator product.

The `last` flag is used by some workflows to decide when to apply “fully contracted” filtering.

Next: see [api.md](api.md) for the concrete functions and their signatures.