#!/usr/bin/env bash
set -euo pipefail

host="$(hostname | tr '[:upper:]' '[:lower:]')"
case "$host" in
  *medora*|*talon*|undmedaasthanad)
    ;;
  *)
    echo "Refusing molecular calculation on unauthorized host: $host" >&2
    exit 64
    ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec "${PYTHON:-python}" -m autogen.methods.qpccsd.cli run "$@"
