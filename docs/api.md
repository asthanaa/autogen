# API guide

This is a pragmatic guide to the functions most users call directly.

## Commutators

### `autogen.main_tools.commutator.comm`

```python
from autogen.main_tools.commutator import comm

terms = comm(['V2'], ['T2'], last=1)
```

- Inputs are commonly operator-label lists like `['V2']`, `['T1']`, etc.
- `a` and `b` may also be lists of term objects (to build nested expressions).
- Returns a `list` of term objects.

### Nested commutators

Compute innermost-first:

```python
from autogen.main_tools.commutator import comm

inner = comm(['V2'], ['T1'], last=0)
outer = comm(inner, ['D1'], last=1)
```

## Products

### `autogen.main_tools.product.prod`

```python
from autogen.main_tools.product import prod

terms = prod(['X1'], ['V2'], last=1)
```

### Composing products and commutators

You can combine these building blocks to form expressions like $X_1 [V_2, T_1]$:

```python
from autogen.main_tools.commutator import comm
from autogen.main_tools.product import prod

vt1 = comm(['V2'], ['T1'], last=0)
x1_vt1 = prod(['X1'], vt1, last=1)
```

## Filtering fully-contracted terms

### `autogen.library.full_con.full_terms`

```python
from autogen.library.full_con import full_terms

contracted_only = full_terms(terms)
```

Note: some workflows use the `last` parameter as a “this is the outermost call” flag.
When in doubt, use `last=1` only at the outermost level of an expression you intend to print.

## Converting indices (`p,q,r`)

### `autogen.library.convert_pqr.convert_pqr`

```python
from autogen.library.convert_pqr import convert_pqr

converted = convert_pqr(terms)
```

## Printing LaTeX output

### `autogen.library.print_terms.print_terms`

```python
from autogen.library.print_terms import print_terms

print_terms(terms, 'latex_output.txt')
```

## Driver-style generation

### `autogen.main_tools.driv3.driver`

```python
from autogen.main_tools.driv3 import driver

# Example: build and contract a list of labeled operators
terms = driver(1.0, ['X1', 'V2', 'T1'])
```

Note: `driver` prints a lot of diagnostic output; it is closer to a workflow helper than a “pure” library function.

Next: see [usage.md](usage.md) for end-to-end examples.