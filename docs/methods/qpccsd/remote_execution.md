# Remote execution

All molecular integral generation, CASSCF, QPCCSD, PAV, and plot-producing
calculations for this project must run on Medora, Talon, or an authorized
remote desktop. Do not run those calculations on the laptop or submit them to
NDSU.

Local work may inspect code, edit documentation, build a wheel, and run tests
that use synthetic tensors only.

## Prepare a remote checkout

From the clean repository root on the approved host:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[molecular,test]"
python -m autogen.methods.qpccsd.cli show-defaults
```

Use a new result directory for each run. Never overwrite a result that has
already been cited or used as a regression reference.

## Deterministic serial validation

Use this environment for the one-geometry certification run:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m autogen.methods.qpccsd.cli run configs/qpccsd/n2_sto3g.toml
qpccsd validate results/n2_sto3g_r1.10/production_result.json \
  --anchor tests/methods/qpccsd/fixtures/n2_sto3g_cas66_r1p10_certified.json
python -m pytest -m "molecular and remote" -q
```

The configuration records the requested worker and BLAS thread counts. The
result records the hostname plus Python and NumPy versions, the configuration
hash, and the emitted amplitude-file hash. Capture the full environment in
the run receipt when dependency and BLAS provenance is required.

## Scheduler-neutral batch template

The launcher under `scripts/qpccsd/remote/` deliberately does not select a partition,
account, reservation, or wall time. Supply those site-specific scheduler
directives in a wrapper outside the tracked repository, then invoke:

```bash
scripts/qpccsd/remote/run_config.sh configs/qpccsd/n2_sto3g.toml
```

For a scheduler job, request one process for deterministic regression. For
6-311G production points, choose memory and wall time from a previously
measured adjacent geometry; projection cache limits are controlled by
`cache_mib` and should remain below the scheduler memory allocation.

## Transfer and audit

Transfer the complete run directory, not only the final energy. Preserve:

- the input TOML and its SHA-256;
- result JSON and any checkpoint identity/hash;
- standard output and standard error;
- Python, NumPy, SciPy, PySCF, BLAS, and package versions from the job receipt;
- hostname and scheduler job ID;
- reference-fit and canonicality diagnostics;
- the fresh unshifted residual audit; and
- both PAV grid evaluations.

After transfer, `qpccsd validate RESULT.json` checks the serialized production
contract and can compare it with a supplied certified anchor. It does not
rerun contractions or recertify a molecular calculation. Use the molecular
regression test or rerun the numerical validators on an approved compute host.
