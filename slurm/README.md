# Running on a SLURM cluster

Two patterns, both starting a Ray cluster across a SLURM allocation:

| Script | Use when |
|---|---|
| `fuse_mat_tiles.sbatch` | One fusion run. Starts the cluster, fuses, tears down. |
| `ray_cluster.sbatch` + `submit_fusion.sh` | Several runs against one long-lived cluster. |

---

## 1. Environment setup

Do this once, on a **shared filesystem** every compute node can see (home,
project, or scratch — not `/tmp`).

```bash
# uv, if your site does not already provide it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <your-fork> RhapsoMatExamples
cd RhapsoMatExamples

# Python 3.12+ is required (virtualizarr and zarr both need it).
# uv will fetch a standalone interpreter if the site modules are older.
uv sync --python 3.12
```

Check it resolved:

```bash
./.venv/bin/python -c "
import ray, zarr, virtualizarr, Rhapso
print(ray.__version__, zarr.__version__, virtualizarr.__version__)"
```

If your site provides Python via modules, load it *before* `uv sync` and it
will be used instead:

```bash
module load python/3.12
```

### Data

Nothing to stage. The S3 bucket is public, so **no AWS credentials are
needed** — the reader requests anonymously, and the chunk layout is read
straight from the first tile in the bucket.

The driver parses that layout once (~110s over S3: it walks a 25,000-entry
HDF5 chunk index, which is thousands of small latency-bound reads), then hands
the result to every worker through Ray's object store as a ~3 MB object. So it
is paid once per job, not once per worker — a one-time cost against a run
measured in hours.

If your cluster blocks outbound HTTPS from compute nodes, or you want to avoid
the per-request latency entirely, stage the tile *data* to node-local disk —
see [Performance](#6-performance). That is about throughput, not about the
chunk layout.

---

## 2. Adjust the scripts for your site

Both `.sbatch` files need their headers edited — the defaults are generic:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
```

Most sites also require at least one of:

```bash
#SBATCH --partition=<your-partition>
#SBATCH --account=<your-account>
#SBATCH --qos=<your-qos>
```

Keep `--ntasks-per-node=1`: one Ray process per node, which then manages
`--cpus-per-task` workers itself. Giving SLURM multiple tasks per node fights
with Ray's own scheduling.

---

## 3. Submit

### One-shot

```bash
sbatch slurm/fuse_mat_tiles.sbatch --limit 16
sbatch --nodes=8 --time=12:00:00 slurm/fuse_mat_tiles.sbatch   # whole mosaic
```

Everything after the script name is forwarded to `fuse-mat-tiles`, so
`--limit`, `--block-size`, `--strategy`, `--chunk-size`, `--shard-size`,
`--dtype` and `--out` all work.

Useful environment overrides:

```bash
OUTPUT=/scratch/$USER/fused.zarr sbatch slurm/fuse_mat_tiles.sbatch
CPUS_PER_BLOCK=4 sbatch slurm/fuse_mat_tiles.sbatch    # fewer, fatter blocks
PROJECT_DIR=/path/to/checkout sbatch slurm/fuse_mat_tiles.sbatch
```

### Persistent cluster

```bash
sbatch slurm/ray_cluster.sbatch                  # note the job id
tail -f logs/ray-cluster-<jobid>.out             # wait for "cluster ready"

slurm/submit_fusion.sh <jobid> --limit 16 --out data/a.zarr
slurm/submit_fusion.sh <jobid> --limit 64 --out data/b.zarr

scancel <jobid>                                  # release the nodes
```

`submit_fusion.sh` must be *invoked* from wherever you like, but it places the
driver **inside the cluster's allocation** with
`srun --jobid=<cluster job> --overlap`, on the head node.

That is not optional. `ray.init(address=...)` reaches the GCS over TCP, but the
driver then attaches to a raylet through a **local Unix socket**. A login node
has no raylet, so running the driver there fails with:

```
raylet_ipc_client.cc:85: Failed to connect to socket at address:
/tmp/ray-<jobid>/session_*/sockets/raylet
```

The `--overlap` flag is also required: without it `srun` blocks waiting for
resources that the `ray start --block` steps already hold.

The script itself just waits on the step, so run it under `tmux` or `nohup` for
long jobs.

---

## 4. Monitoring

```bash
squeue -u $USER
tail -f logs/rhapso-fuse-<jobid>.out       # progress prints every 5%
```

The driver prints the cluster it attached to before starting, which is the
quickest way to confirm all your nodes joined:

```
Ray cluster    : 124 CPUs across 4 node(s)
```

If that says 1 node, the workers failed to register — check `RAY_TMPDIR`
(below) first.

---

## 5. Things that bite

**`RAY_TMPDIR` must be node-local.** Ray puts its raylet sockets there, and a
shared filesystem breaks them. The scripts default to `/tmp/ray-$SLURM_JOB_ID`.
If your site's `/tmp` is small or shared, point it at real node-local scratch:

```bash
RAY_TMPDIR=/local/$USER/ray-$SLURM_JOB_ID sbatch slurm/fuse_mat_tiles.sbatch
```

This is the most common cause of a cluster that starts but never gains workers.

**The output must be on shared storage.** Workers on different nodes write
blocks into the same Zarr store by path. A node-local `--out` silently produces
a volume with holes — each node writing its own private copy. Use a
project/scratch path.

**Shards and concurrency.** Output is sharded by default at exactly one shard
per fusion block, so every shard is written once by one task. That property is
what makes multi-node writes safe. If you override `--shard-size`, it must
still divide `--block-size` in X and Y; the tool refuses layouts that would let
two tasks write one shard.

**Memory.** Each block holds coordinate arrays plus source data for every
overlapping tile. With the default `--block-size 256 256` and full depth, budget
roughly 1.5–2 GB per concurrent block. Ray runs `--cpus-per-task` CPUs worth of
blocks at once, so on a 32-core node with `CPUS_PER_BLOCK=2` that is ~16
concurrent blocks — size `--mem` accordingly, or raise `CPUS_PER_BLOCK` to run
fewer at a time.

**Walltime.** If the job is killed mid-run the Zarr store is left partially
written. There is no resume; rerun with a fresh `--out`.

---

## 6. Performance

The dominant cost is reading tile data from S3, and the limit is **aggregate
bandwidth on the node's path to the bucket** — not CPU, and not request count.

Measured from one workstation against this bucket (us-east-2):

| request pattern | throughput |
|---|---|
| 1 x 100 MB contiguous GET | 1.6 MB/s |
| 1 x 16 MB contiguous GET | 2.6 - 3.2 MB/s |
| 4 x 16 MB in parallel | 4.2 MB/s |
| **16 x 4 MB in parallel** | **6.4 MB/s** |
| 64 x 1 MB in parallel | 6.0 MB/s |
| the reader as it ships (chunk by chunk) | 6.2 MB/s |

A single stream gets 2-3 MB/s; the pipe only fills with many requests in
flight, saturating near 6 MB/s. **The reader already sits at that ceiling.**

Two consequences that are easy to get backwards:

- **Fewer, larger reads are not better.** The `.mat` chunks are 98.2%
  contiguous on disk, so merging them into a handful of large reads is very
  possible — and it does not help. Doing so cuts concurrency, which is the only
  thing filling the pipe, so it trends toward the 2.6 MB/s single-stream rate.
  This was measured, not assumed: kerchunk references through fsspec's
  `ReferenceFileSystem` (which merges ranges within `max_gap`, default 64 KB)
  read an identical slab in 18.2s versus 18.2s for the shipping reader, and
  disabling merging entirely (`max_gap=0`) gave 18.6s.
- **Bigger `--block-size` does not reduce bytes read.** It changes memory use
  and how coarsely work is divided, nothing more.

### What actually helps

**More nodes.** Each node has its own path to S3, so aggregate throughput
scales with node count until you hit a site egress limit or an S3 ceiling. On
these numbers that is the single biggest lever: ~25 GB for 16 tiles and ~200 GB
for the full mosaic, divided across nodes.

**Keeping blocks concurrent.** Ray runs `--cpus-per-task` CPUs worth of blocks
at once. Lowering `CPUS_PER_BLOCK` raises the number of concurrent blocks and
so the number of in-flight requests — useful while the job is network-bound,
provided memory allows (see Things that bite).

**Staging tile data to node-local disk**, if your nodes have the scratch space:

```bash
# in your sbatch, before the fusion step
srun --ntasks-per-node=1 bash -c '
  mkdir -p /local/$USER/tiles
  aws s3 sync --no-sign-request --exclude "*" --include "slice_150_tile_0*" \
    s3://apex-connects/CMC/Derivatives/Vlad/PS-OCT/3DTiles/Cross/150/ \
    /local/$USER/tiles/'

fuse-mat-tiles --s3-prefix /local/$USER/tiles ...
```

This wins because `aws s3 sync` transfers many objects with its own multipart
concurrency, and because blocks that re-read overlapping tiles then hit local
disk instead of the network. It costs ~1.6 GB per tile per node, so it suits
deep runs on few nodes better than wide ones.

### Measure it on your own cluster first

The table above is one workstation's link and is almost certainly pessimistic
for a compute node. Get your own number before sizing the job:

```bash
U=https://apex-connects.s3.us-east-2.amazonaws.com/CMC/Derivatives/Vlad/PS-OCT/3DTiles/Cross/150/slice_150_tile_001_Cross.mat
srun --nodes=1 --ntasks=1 bash -c '
  for i in $(seq 0 15); do
    o=$((5104 + i*4194304)); curl -s -o /dev/null -r $o-$((o+4194303)) "'"$U"'" &
  done; time wait'
```

16 parallel 4 MB reads = 67 MB. Divide by the elapsed time for your per-node
ceiling, then: total bytes / (per-node MB/s x nodes) is the transfer floor for
the run. At 6 MB/s per node, the full 126-tile mosaic (~200 GB) is ~9 hours on
one node and roughly an hour on eight.
