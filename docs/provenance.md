# Provenance and archive policy

## Authoritative history

The consolidated repository continues the published `asthanaa/autogen` history. Work
from the disconnected `autogen2` checkout was imported as reviewed content; its unrelated
root history was not merged.

The clean projected-AGP QPCCSD/PAV package originated in `autogen2` commit
`2b881e031a4b1fddcb3b36f9c42f463a4428195f`. The scalable predecessor was
`970dc4ac6291048c85bfd3a198196422ee928d42`. These identifiers establish source
provenance without making either snapshot an importable package inside the final tree.

## Excluded generated plan

The unpublished legacy commit `0a851d3469a8476aa36d01b3437021c9c2815ace` contained a
387,064,269-byte projected-codegen plan. That blob exceeds normal GitHub limits and is
not needed by the canonical production QPCCSD/PAV runtime. It is excluded rather than
rewritten into the consolidated branch.

Its path, byte size, SHA-256, legacy Git location, and regeneration command are recorded
in the repository file `provenance/excluded_large_objects.json`.
This compact record is authoritative for artifact identity; copying the plan back into
the repository is prohibited.

## Campaign and calculation archives

Raw molecular calculations remain on approved remote systems or in separately managed
archives. Preserve the configuration, code commit, result/checkpoint hashes, host,
dependency versions, scheduler receipt, residual certification, and projection-grid
validation together. Do not commit whole run directories.

To promote a result into the test suite, create a compact fixture that records those
identifiers and only the values required by the regression contract. Never replace an
expected energy solely to make a test pass.
