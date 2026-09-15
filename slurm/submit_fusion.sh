#!/bin/bash
# Run a fusion job against a Ray cluster already started by ray_cluster.sbatch.
#
#   slurm/submit_fusion.sh <cluster_jobid> [fuse-mat-tiles args...]
#
#   slurm/submit_fusion.sh 123456 --limit 16
#   slurm/submit_fusion.sh 123456 --block-size 512 512 --out data/full.zarr
#
# The driver runs wherever you invoke this (a login node is fine -- it only
# submits tasks and waits); all the fusion work happens on the cluster's
# workers. Run it under nohup/tmux for long jobs, or wrap it in its own small
# sbatch if your site forbids long-running login-node processes.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <cluster_jobid> [fuse-mat-tiles args...]" >&2
    exit 2
fi

JOB_ID="$1"; shift

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
ADDRESS_FILE="$PROJECT_DIR/ray_cluster_${JOB_ID}.address"

if [[ ! -f "$ADDRESS_FILE" ]]; then
    echo "No address file at $ADDRESS_FILE" >&2
    echo "Is cluster job $JOB_ID running and past its startup? Check:" >&2
    echo "  squeue -j $JOB_ID" >&2
    echo "  tail logs/ray-cluster-${JOB_ID}.out" >&2
    exit 1
fi

RAY_ADDRESS="$(cat "$ADDRESS_FILE")"
echo "attaching to Ray cluster $JOB_ID at $RAY_ADDRESS"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export RAY_ADDRESS

exec "$VENV/bin/fuse-mat-tiles" \
    --yaml manifests/test.yaml \
    --ray-address "$RAY_ADDRESS" \
    --cpus-per-task "${CPUS_PER_BLOCK:-2}" \
    "$@"
