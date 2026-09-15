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

# Python 3.12+ is required (icechunk does not build for 3.11).
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

`submit_fusion.sh` runs the driver wherever you invoke it — a login node is
fine, since it only submits tasks and waits. For long runs use `tmux` or
`nohup`, or wrap it in a small single-core sbatch if your site forbids
long-lived login-node processes.

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

The dominant cost is **not** compute. The `.mat` files are chunked
`(41, 1, 191)` — 62 KB, one voxel thick in y — so a 256×256 lateral block needs
~1,792 separate ranged GETs per contributing tile. A 16-tile run is roughly
400,000 individual requests. Measured locally against S3, the Ray workers sat
at under 10% total CPU: the job was latency-bound on tiny reads, not
bandwidth- or compute-bound.

Two levers, in order of effect:

**Stage the tiles to node-local disk first.** One bulk 1.57 GB sequential read
per tile beats 25,000 random ones by a wide margin. If every node has the tiles
locally, point `--s3-prefix` at the local directory:

```bash
# in your sbatch, before the fusion step
srun --ntasks-per-node=1 bash -c '
  mkdir -p /local/$USER/tiles
  aws s3 sync --no-sign-request --exclude "*" --include "slice_150_tile_0*" \
    s3://apex-connects/CMC/Derivatives/Vlad/PS-OCT/3DTiles/Cross/150/ \
    /local/$USER/tiles/'

fuse-mat-tiles --s3-prefix /local/$USER/tiles ...
```

This costs 25 GB per node for 16 tiles, so it suits deep runs on few nodes
better than wide ones.

**More nodes.** Because the bottleneck is request latency rather than
bandwidth, adding nodes does help — each contributes its own concurrent
requests. Scaling is closer to linear here than it would be for a
bandwidth-bound job, up to whatever S3 request ceiling your network imposes.

Raising `--block-size` does *not* reduce the request count (the same chunks are
read either way); it only reduces per-block overhead and raises memory use.
