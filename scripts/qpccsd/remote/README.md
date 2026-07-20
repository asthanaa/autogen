# Remote launch helpers

`run_config.sh` is the scheduler-neutral entry point for a reviewed TOML
configuration. It sets deterministic thread defaults unless the caller has
already provided them, records basic host/software provenance, validates the
configuration, and then invokes the canonical QPCCSD module CLI.

Use it only on Medora, Talon, or an authorized remote desktop:

```bash
scripts/qpccsd/remote/run_config.sh configs/qpccsd/n2_sto3g.toml
```

Scheduler account, partition, reservation, memory, and wall-time directives
are site-specific and intentionally absent. Put those in an untracked wrapper
or submission script. Do not hard-code home directories or host-specific
Python paths into this repository.

The output path comes from the TOML file. Give each job an immutable output
directory and transfer the entire directory for audit. See
[`docs/methods/qpccsd/remote_execution.md`](../../../docs/methods/qpccsd/remote_execution.md).
