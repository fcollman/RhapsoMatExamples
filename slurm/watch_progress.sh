#!/bin/bash
# Track a running fusion job from the login node.
#
#   slurm/watch_progress.sh <jobid>          # one snapshot
#   slurm/watch_progress.sh <jobid> --watch  # refresh every 30s
#
# Reports three independent signals, so a stalled job is distinguishable from
# a quiet one:
#
#   1. SLURM     -- is the job still running, and for how long
#   2. the log   -- the driver's own "N/M blocks" progress line
#   3. the store -- shards actually on disk, which is ground truth even when
#                   stdout is buffered or the driver has died
#
# The shard count is the one to trust. If it is climbing, work is happening.

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <jobid> [--watch]" >&2
    exit 2
fi

JOB_ID="$1"; shift
WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

find_log() {
    ls -1t "logs/rhapso-fuse-${JOB_ID}.out" "logs/ray-cluster-${JOB_ID}.out" \
        2>/dev/null | head -1
}

snapshot() {
    local log out total done_blocks shards elapsed rate remain

    echo "=== job $JOB_ID @ $(date '+%H:%M:%S') ==="
    squeue -j "$JOB_ID" -o "%.10i %.9T %.10M %.10L %.6D %R" 2>/dev/null \
        || echo "  not in the queue (finished or cancelled)"

    log="$(find_log)"
    if [[ -z "$log" ]]; then
        echo "  no log found for job $JOB_ID in logs/"
        return
    fi

    # The driver prints these once at startup.
    total=$(grep -m1 "Fusion tasks" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    out=$(grep -m1 "^Output " "$log" 2>/dev/null | awk '{print $3}')

    done_blocks=$(grep -oE '[0-9]+/[0-9]+ blocks \([0-9]+%\)' "$log" 2>/dev/null | tail -1)
    echo "  log      : ${done_blocks:-no progress line yet}  ($log)"

    # ETA comes from the LOG (blocks done / blocks total), because empty
    # regions write no shard -- shard count alone would understate progress.
    # Elapsed comes from SLURM, which is authoritative; file timestamps are not
    # (a copied or rotated log lies about when the job started).
    slurm_elapsed="$(squeue -j "$JOB_ID" -h -o %M 2>/dev/null | tr -d ' ')"
    if [[ -n "$done_blocks" && -n "$slurm_elapsed" ]]; then
        python3 - "$done_blocks" "$slurm_elapsed" <<'PYEOF'
import re, sys
done, total = (int(x) for x in re.match(r"(\d+)/(\d+)", sys.argv[1]).groups())

# SLURM %M is [DD-]HH:MM:SS or MM:SS
text = sys.argv[2]
days, _, rest = text.rpartition("-")
parts = [int(p) for p in rest.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
minutes = (int(days or 0) * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60)

if done and minutes > 0:
    rate = done / minutes
    print(f"  rate     : {rate:.1f} blocks/min  (elapsed {minutes:.0f} min)")
    if total > done and rate > 0:
        print(f"  eta      : ~{(total - done) / rate:.0f} min for "
              f"{total - done} blocks")
PYEOF
    fi

    # The store is the liveness check: is anything still being written?
    if [[ -n "${out:-}" && -d "$out/0/c" ]]; then
        python3 - "$out" <<'PYEOF'
import os, sys, time
out = sys.argv[1]
newest, n = 0.0, 0
for root, _, files in os.walk(os.path.join(out, "0", "c")):
    for f in files:
        n += 1
        newest = max(newest, os.path.getmtime(os.path.join(root, f)))
print(f"  store    : {n} shards in {out}")
if n:
    age = time.time() - newest
    flag = "  <-- STALLED?" if age > 600 else ""
    print(f"  last write: {age:.0f}s ago{flag}")
PYEOF
        du -sh "$out" 2>/dev/null | awk '{print "  size     : "$1}'
    else
        echo "  store    : not created yet${out:+ ($out)}"
    fi

    # Empty regions legitimately write no shard, so shards < blocks is normal.
    echo "  note     : blocks with no overlapping tile write no shard"
}

if (( WATCH )); then
    while true; do
        clear; snapshot
        squeue -j "$JOB_ID" -h >/dev/null 2>&1 || break
        sleep 30
    done
else
    snapshot
fi
