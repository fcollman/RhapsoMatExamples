#!/bin/bash
# Run a fusion job against a Ray cluster already started by ray_cluster.sbatch.
#
#   slurm/submit_fusion.sh <cluster_jobid> [fuse-mat-tiles args...]
#
#   slurm/submit_fusion.sh 123456 --limit 16
#   slurm/submit_fusion.sh 123456 --block-size 512 512 --out data/full.zarr
#
# The driver CANNOT run on a login node. `ray.init(address=...)` reaches the
# GCS over TCP, but the driver then attaches to a raylet through a local Unix
# socket -- and a login node has no raylet, which fails as:
#
#   raylet_ipc_client.cc:85: Failed to connect to socket at address:
#   /tmp/ray-<jobid>/session_*/sockets/raylet
#
# So this script places the driver inside the cluster's own allocation with
# `srun --jobid=<cluster job> --overlap`, on the head node. --overlap is
# required: without it srun waits for resources that the `ray start --block`
# steps are already holding.

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

# RAY_ADDRESS, HEAD_NODE, CLUSTER_JOB_ID, RAY_TMPDIR
# shellcheck source=/dev/null
source "$ADDRESS_FILE"

if ! squeue -j "$JOB_ID" -h -o %T 2>/dev/null | grep -q RUNNING; then
    echo "Cluster job $JOB_ID is not RUNNING; start it with:" >&2
    echo "  sbatch slurm/ray_cluster.sbatch" >&2
    exit 1
fi

echo "cluster job : $JOB_ID"
echo "head node   : $HEAD_NODE"
echo "ray address : $RAY_ADDRESS"
echo "running the driver inside the allocation (srun --jobid --overlap)"

cd "$PROJECT_DIR"

exec srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --export=ALL,PYTHONUNBUFFERED=1,RAY_ADDRESS="$RAY_ADDRESS" \
    "$VENV/bin/fuse-mat-tiles" \
        --yaml manifests/test.yaml \
        --ray-address "$RAY_ADDRESS" \
        --cpus-per-task "${CPUS_PER_BLOCK:-2}" \
        "$@"
