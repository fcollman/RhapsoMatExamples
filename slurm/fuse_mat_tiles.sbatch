#!/bin/bash
# Start a Ray cluster across a SLURM allocation, run the mosaic fusion on it,
# and tear the cluster down.
#
#   sbatch slurm/fuse_mat_tiles.sbatch
#   sbatch --nodes=8 slurm/fuse_mat_tiles.sbatch --limit 64
#
# Anything after the script name is passed straight through to fuse-mat-tiles,
# so --limit / --block-size / --strategy / --out all work as usual.
#
# Ray's own SLURM guidance is the basis for the head/worker dance here:
# one `ray start --head` on the first node, one `ray start --address=...`
# per remaining node, then a driver that attaches with address="auto".
#
# ---------------------------------------------------------------------------
# Adjust these for your site
# ---------------------------------------------------------------------------
#SBATCH --job-name=rhapso-fuse
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhapso-fuse-%j.out
#SBATCH --error=logs/rhapso-fuse-%j.out

set -euo pipefail

# Project checkout. Override with PROJECT_DIR=... sbatch ...
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT_DIR/.venv}"

# Ray writes sockets and logs here. It MUST be node-local disk -- a shared
# filesystem breaks Ray's raylet sockets, and on many clusters /tmp is the only
# node-local path. This is the single most common cause of a cluster that
# starts but never gets workers.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$SLURM_JOB_ID}"

# The fused output, by contrast, MUST be on storage every node can see, since
# workers on different nodes write blocks into the same Zarr store. A
# node-local path silently produces a volume with holes.
OUTPUT="${OUTPUT:-$PROJECT_DIR/data/fused-$SLURM_JOB_ID.zarr}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "job        : $SLURM_JOB_ID"
echo "nodes      : $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
echo "cpus/node  : $SLURM_CPUS_PER_TASK"
echo "project    : $PROJECT_DIR"
echo "output     : $OUTPUT"

# ---------------------------------------------------------------------------
# 1. Head node
# ---------------------------------------------------------------------------
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"

# Resolve an address the workers can reach. Many clusters have a separate
# high-speed interface; prefer the first routable address for the head host.
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address \
          | awk '{print $1}')
PORT=${RAY_PORT:-6379}
RAY_ADDRESS="$HEAD_IP:$PORT"

echo "head       : $HEAD_NODE ($RAY_ADDRESS)"

# Leave a core for the driver and Ray's own overhead.
WORKER_CPUS=$(( SLURM_CPUS_PER_TASK - 1 ))

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "$VENV/bin/ray" start --head \
        --node-ip-address="$HEAD_IP" \
        --port="$PORT" \
        --num-cpus="$WORKER_CPUS" \
        --temp-dir="$RAY_TMPDIR" \
        --include-dashboard=false \
        --block &

# Give the head time to come up before workers try to register.
sleep 20

# ---------------------------------------------------------------------------
# 2. Worker nodes
# ---------------------------------------------------------------------------
for (( i = 1; i < SLURM_JOB_NUM_NODES; i++ )); do
    node="${NODES[$i]}"
    echo "worker     : $node"
    srun --nodes=1 --ntasks=1 -w "$node" \
        "$VENV/bin/ray" start \
            --address="$RAY_ADDRESS" \
            --num-cpus="$WORKER_CPUS" \
            --temp-dir="$RAY_TMPDIR" \
            --block &
    sleep 5
done

# Wait until every node has actually registered. Starting the driver early
# means the first blocks all schedule onto the head node.
echo "waiting for $SLURM_JOB_NUM_NODES nodes to register..."
"$VENV/bin/python" - "$RAY_ADDRESS" "$SLURM_JOB_NUM_NODES" <<'PY'
import sys, time, ray
address, expected = sys.argv[1], int(sys.argv[2])
ray.init(address=address)
deadline = time.time() + 300
while time.time() < deadline:
    alive = [n for n in ray.nodes() if n["Alive"]]
    if len(alive) >= expected:
        break
    time.sleep(5)
alive = [n for n in ray.nodes() if n["Alive"]]
cpus = int(ray.cluster_resources().get("CPU", 0))
print(f"cluster ready: {len(alive)}/{expected} nodes, {cpus} CPUs")
if len(alive) < expected:
    print("WARNING: proceeding with fewer nodes than requested")
PY

# ---------------------------------------------------------------------------
# 3. Run the fusion on the cluster
# ---------------------------------------------------------------------------
# PYTHONUNBUFFERED so progress reaches the SLURM log as it happens rather than
# sitting in a pipe buffer until the job ends.
export PYTHONUNBUFFERED=1
export RAY_ADDRESS

echo "starting fusion"
"$VENV/bin/fuse-mat-tiles" \
    --yaml manifests/test.yaml \
    --out "$OUTPUT" \
    --ray-address "$RAY_ADDRESS" \
    --cpus-per-task "${CPUS_PER_BLOCK:-2}" \
    "$@"
status=$?

# ---------------------------------------------------------------------------
# 4. Tear down
# ---------------------------------------------------------------------------
echo "stopping ray"
for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" "$VENV/bin/ray" stop --force || true
done

echo "done (exit $status)"
exit $status
