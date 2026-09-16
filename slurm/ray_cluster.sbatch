#!/bin/bash
# Start a long-lived Ray cluster on a SLURM allocation and hold it open, so you
# can submit several fusion jobs into the same cluster.
#
#   sbatch slurm/ray_cluster.sbatch                 # start it
#   cat ray_cluster_<jobid>.address                 # the address it wrote
#   slurm/submit_fusion.sh <jobid> --limit 16       # run a job on it
#   scancel <jobid>                                 # shut it down
#
# Use slurm/fuse_mat_tiles.sbatch instead when you only have one job to run --
# it starts a cluster, runs the fusion, and tears down in a single allocation.
#
#SBATCH --job-name=ray-cluster
# 16 CPUs / 32 GB is sized from measurement, not habit: this job is bound by
# S3 bandwidth, not CPU, so a node saturates its link with only a handful of
# concurrent blocks. WORKER_CPUS=15 / CPUS_PER_BLOCK=2 = 7 blocks x ~1.4 GB
# peak = ~10 GB, plus a 2 GB object store. Smaller asks also queue sooner --
# spend the savings on --nodes instead.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ray-cluster-%j.out
#SBATCH --error=logs/ray-cluster-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT_DIR/.venv}"

# Node-local, never a shared filesystem -- Ray's raylet sockets live here.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$SLURM_JOB_ID}"

# The address file must be somewhere the submitting host can read, so this one
# does belong on shared storage.
ADDRESS_FILE="$PROJECT_DIR/ray_cluster_${SLURM_JOB_ID}.address"

cd "$PROJECT_DIR"
mkdir -p logs

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

echo "job     : $SLURM_JOB_ID"
echo "head    : $HEAD_NODE ($RAY_ADDRESS)"
echo "nodes   : $SLURM_JOB_NUM_NODES"

cleanup() {
    echo "shutting down ray"
    for node in "${NODES[@]}"; do
        srun --nodes=1 --ntasks=1 -w "$node" "$VENV/bin/ray" stop --force || true
    done
    rm -f "$ADDRESS_FILE"
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "$VENV/bin/ray" start --head \
        --node-ip-address="$HEAD_IP" --port="$PORT" \
        --num-cpus="$WORKER_CPUS" --temp-dir="$RAY_TMPDIR" \
        --object-store-memory="$OBJECT_STORE_BYTES" \
        $DASHBOARD_ARGS --block &
sleep 20

for (( i = 1; i < SLURM_JOB_NUM_NODES; i++ )); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        "$VENV/bin/ray" start --address="$RAY_ADDRESS" \
            --num-cpus="$WORKER_CPUS" --temp-dir="$RAY_TMPDIR" \
            --object-store-memory="$OBJECT_STORE_BYTES" --block &
    sleep 5
done

# Publish the address only once the cluster can actually accept work, so a
# submitter that reads the file immediately does not race the workers.
"$VENV/bin/python" - "$RAY_ADDRESS" "$SLURM_JOB_NUM_NODES" <<'PY'
import sys, time, ray
address, expected = sys.argv[1], int(sys.argv[2])
ray.init(address=address)
deadline = time.time() + 300
while time.time() < deadline:
    if len([n for n in ray.nodes() if n["Alive"]]) >= expected:
        break
    time.sleep(5)
print(f"cluster ready: {len([n for n in ray.nodes() if n['Alive']])}/{expected} "
      f"nodes, {int(ray.cluster_resources().get('CPU', 0))} CPUs")
PY

# Record everything a submitter needs. The head node name matters as much as
# the address: a driver must run ON a node that has a raylet, so submit_fusion
# uses `srun --jobid` to place it inside this allocation.
cat > "$ADDRESS_FILE" <<EOF
RAY_ADDRESS=$RAY_ADDRESS
HEAD_NODE=$HEAD_NODE
CLUSTER_JOB_ID=$SLURM_JOB_ID
RAY_TMPDIR=$RAY_TMPDIR
EOF
echo "address written to $ADDRESS_FILE"
echo "submit work with: slurm/submit_fusion.sh $SLURM_JOB_ID [fuse-mat-tiles args]"

# Hold the allocation until the walltime runs out or the job is cancelled.
sleep infinity
