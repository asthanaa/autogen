"""Benchmark-only molecular preparation; never imported by production APIs."""

from .campaign import (
    AMPLITUDE_CHECKPOINT_SCHEMA,
    CAMPAIGN_SCHEMA,
    SCIENTIFIC_TITLE,
    AtomicAmplitudeCheckpointer,
    checkpoint_metadata,
    excitation_space_hash,
    load_amplitude_checkpoint,
    save_amplitude_checkpoint,
    source_tree_hash,
    write_json_atomic,
)

from .n2_active_space import (
    N2ActiveSpacePoint,
    build_n2_4e4o_631g,
    build_n2_6e6o_631g,
)
from .pyscf_cas import (
    CASSCFContinuation,
    CONTINUATION_SCHEMA,
    FullSpaceBenchmark,
    build_casscf_qp_reference,
    build_h2_casscf_qp_reference,
    build_n2_casscf_qp_reference,
    compute_fullspace_benchmarks,
    load_casscf_continuation,
    restore_continuation_integrals,
    save_casscf_continuation,
)

__all__ = [
    "AMPLITUDE_CHECKPOINT_SCHEMA",
    "AtomicAmplitudeCheckpointer",
    "CAMPAIGN_SCHEMA",
    "SCIENTIFIC_TITLE",
    "CASSCFContinuation",
    "CONTINUATION_SCHEMA",
    "FullSpaceBenchmark",
    "N2ActiveSpacePoint",
    "build_casscf_qp_reference",
    "build_h2_casscf_qp_reference",
    "build_n2_4e4o_631g",
    "build_n2_6e6o_631g",
    "build_n2_casscf_qp_reference",
    "checkpoint_metadata",
    "compute_fullspace_benchmarks",
    "excitation_space_hash",
    "load_amplitude_checkpoint",
    "load_casscf_continuation",
    "restore_continuation_integrals",
    "save_amplitude_checkpoint",
    "save_casscf_continuation",
    "source_tree_hash",
    "write_json_atomic",
]
