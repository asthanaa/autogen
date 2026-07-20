# Canonical equation manifests

These JSON files are the canonical derivation manifests. Byte-identical,
immutable runtime snapshots are packaged beside the generated Python kernels
because those modules locate their manifests relative to themselves. A parity
test prevents the two copies from drifting. Production imports the generated
Python kernels only; it does not regenerate equations while a calculation is
running.
