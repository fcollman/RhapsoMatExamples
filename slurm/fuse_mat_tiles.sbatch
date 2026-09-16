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
# 16 CPUs / 32 GB is sized from measurement, not habit: this job is bound by
# S3 bandwidth, not CPU, so a node saturates its link with only a handful of
# concurrent blocks. WORKER_CPUS=15 / CPUS_PER_BLOCK=2 = 7 blocks x ~1.4 GB
# peak = ~10 GB, plus a 2 GB object store. Smaller asks also queue sooner --
# spend the savings on --nodes instead.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

# Resolve the head node's IP.
#
# Do NOT use `srun ... hostname --ip-address` here: some compute images ship
# without the hostname binary, which fails as
#   execve(): hostname: No such file or directory
# Resolving from the submit node needs no job step at all, and getent is part
# of glibc so it is present essentially everywhere.
#
# Override with RAY_HEAD_IP=... if your cluster has a separate high-speed
# interface and name resolution returns the wrong one.
resolve_head_ip() {
    local host="$1" ip=""

    ip=$(getent ahostsv4 "$host" 2>/dev/null | awk '{print $1; exit}')
    [ -z "$ip" ] && ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1; exit}')

    # Last resort: ask Python, which is guaranteed present via the venv.
    if [ -z "$ip" ]; then
        ip=$("$VENV/bin/python" -c \
            'import socket,sys; print(socket.gethostbyname(sys.argv[1]))' \
            "$host" 2>/dev/null)
    fi

    if [ -z "$ip" ]; then
        echo "Could not resolve an IP for head node '$host'." >&2
        echo "Set RAY_HEAD_IP=<ip> and resubmit." >&2
        exit 1
    fi
    echo "$ip"
}

HEAD_IP="${RAY_HEAD_IP:-$(resolve_head_ip "$HEAD_NODE")}"
PORT=${RAY_PORT:-6379}
RAY_ADDRESS="$HEAD_IP:$PORT"

echo "head       : $HEAD_NODE ($RAY_ADDRESS)"

# Leave a core for the driver and Ray's own overhead.
WORKER_CPUS=$(( SLURM_CPUS_PER_TASK - 1 ))

# Ray's object store otherwise grabs ~30% of node memory, which counts against
# the same --mem cgroup the fusion blocks need. This pipeline only ships small
# objects through it (the ~3 MB chunk template, the transform dict), so a few
# GB is ample.
OBJECT_STORE_BYTES="${OBJECT_STORE_BYTES:-2000000000}"

# Ray's dashboard is off by default (one less port, one less process). Set
# RAY_DASHBOARD=1 to enable it, then tunnel from your workstation:
#   ssh -L 8265:<head node>:8265 <cluster>
#   open http://localhost:8265
DASHBOARD_ARGS="--include-dashboard=false"
if [[ "${RAY_DASHBOARD:-0}" == "1" ]]; then
    DASHBOARD_ARGS="--include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT:-8265}"
fi

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "$VENV/bin/ray" start --head \
        --node-ip-address="$HEAD_IP" \
        --port="$PORT" \
        --num-cpus="$WORKER_CPUS" \
        --temp-dir="$RAY_TMPDIR" \
        --object-store-memory="$OBJECT_STORE_BYTES" \
        $DASHBOARD_ARGS \
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
            --object-store-memory="$OBJECT_STORE_BYTES" \
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
# `set -e` is off for this one command: a failed fusion must still fall through
# to the teardown below, or the Ray cluster keeps the allocation until walltime.
set +e
"$VENV/bin/fuse-mat-tiles" \
    --yaml manifests/test.yaml \
    --out "$OUTPUT" \
    --ray-address "$RAY_ADDRESS" \
    --cpus-per-task "${CPUS_PER_BLOCK:-2}" \
    "$@"
status=$?
set -e

# ---------------------------------------------------------------------------
# 4. Tear down
# ---------------------------------------------------------------------------
echo "stopping ray"
for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" "$VENV/bin/ray" stop --force || true
done

echo "done (exit $status)"
exit $status
