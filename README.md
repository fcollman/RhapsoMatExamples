# RhapsoMatExamples

Fuse a mosaic of MATLAB v7.3 (`.mat`) tiles into a single Zarr with
[Rhapso](https://github.com/AllenNeuralDynamics/Rhapso), reading the `.mat`
files **in place by byte range** — no download, no conversion, no duplicated
storage.

A v7.3 MAT-file is an HDF5 file with a 512-byte userblock, so VirtualiZarr's
`HDFParser` can map its native HDF5 chunks to Zarr chunks. Reading one Zarr
chunk becomes one ranged GET into the `.mat`, whether it sits on local disk or
in S3.

## Install

```bash
uv sync
```

Rhapso comes from PyPI. To develop against a local checkout instead, add to
`pyproject.toml`:

```toml
[tool.uv.sources]
rhapso = { path = "../Rhapso", editable = true }
```

Rhapso's published pins can't coexist with the VirtualiZarr/Icechunk stack, so
`[tool.uv] override-dependencies` relaxes four of them. Each override is
documented in `pyproject.toml` with the reason it's needed; `bioio` is dropped
outright because Rhapso uses it only for TIFF reading, which affine fusion
never touches.

## Commands

| Command | What it does |
|---|---|
| `mat-info FILE [--verify]` | Inspect a `.mat` as virtual Zarr; `--verify` checks reads against h5py |
| `fuse-mat-tiles` | Fuse a tile mosaic described by a YAML manifest |
| `serve-zarr DIR` | Serve a Zarr store with CORS + Range, for Neuroglancer |
| `make-demo-tiles` / `fuse-demo-tiles` | Synthetic 4-tile example, no `.mat` needed |

Running on SLURM: see [slurm/README.md](slurm/README.md).

### Inspect a tile

```bash
mat-info data/slice_150_tile_001_Cross.mat --verify
```

```
zarr shape   : (1000, 1000, 191)   (HDF5 order)
chunk shape  : (41, 1, 191)        (native HDF5 chunk)
chunk grid   : (25, 1000, 1) -> 25,000 chunks
codecs       : ['BytesCodec', 'Zlib']
referenced   : 1.566 GB of the .mat (1.528 GB uncompressed), 0 copied
```

### Fuse a mosaic

```bash
# first 4 tiles, ~3 GB out
fuse-mat-tiles --yaml manifests/test.yaml --limit 4 --out fused.zarr

# whole mosaic, bigger blocks
fuse-mat-tiles --yaml manifests/test.yaml --block-size 512 512 --out fused.zarr
```

Drop `--limit` to use every tile. `--dry-run` reports the plan (output size,
bytes to read, task and file counts) without fusing.

Output is **sharded by default**, one shard per fusion task — 60 files instead
of 696 for four tiles, 425 instead of 19,500 for the full mosaic. Readers still
see the `--chunk-size` chunks; sharding only changes how they're packed on
disk. Override with `--shard-size X Y Z` or turn it off with `--no-shard`.

### View it

```bash
serve-zarr . --port 9102
```

Then in Neuroglancer: `zarr://http://127.0.0.1:9102/fused.zarr`

The output carries OME-NGFF 0.5 `multiscales` metadata, so the store resolves
as a calibrated image — named `z`/`y`/`x` axes in microns — rather than a bare
unitless array. Values run roughly -21..+57, so set the shader range
accordingly; the default contrast will look flat.

## The manifest

`manifests/test.yaml` gives each tile's position; x/y become the translation of
its affine transform.

```yaml
metadata:
  tile_width: 1000
  tile_height: 1000
  depth: 191
  tile_overlap: 0.1
  resolution_microns: [5.0, 5.0, 3.4]   # x, y, z voxel size
tiles:
  - filepath: slice_150_tile_001_Cross.mat
    tile_number: 1
    x: 0.0
    y: 0.0
```

Tile `x`/`y` are in **voxels**, so fusion runs on a voxel grid and the output
is a 1:1 resampling of the source. `resolution_microns` is applied as the OME
`coordinateTransformations` scale rather than being baked into the affines —
that keeps the data untouched while still reading in physical units. Omit it
and the output is written without OME metadata, as unitless voxels.

## Things worth knowing

**Axis order.** MATLAB is column-major and writes HDF5 dimensions reversed, so
a MATLAB `[a b c]` array appears as `(c, b, a)`. A chunk manifest maps byte
ranges to positions and cannot transpose, so the Zarr array is necessarily in
HDF5 order; `fuse_mat_tiles` presents each tile through a transposing view to
reach Rhapso's `[t, c, z, y, x]`. For this dataset the `.mat` arrays are
`(x, y, depth)` — established by cross-correlating tile 1's intensity profile
against tile 2's, which peaks at lag 899 for a nominal 900 offset.

**Output dtype.** Rhapso's stock `FuseCell` ends with
`np.clip(np.rint(block), 0, 65535).astype(np.uint16)`. This data spans roughly
-21..+57, so that would clip every negative to 0 and round away the fraction.
The fusion step is subclassed here to write **float32** by default.

**Identical layouts.** Every `.mat` in this dataset shares the same internal
HDF5 layout, so the chunk manifest is built once from a reference tile and
stamped onto the rest with `rename_paths()` — a path rewrite, no re-parsing.
The driver does that parse itself and ships the ~3 MB result to the workers
through Ray's object store, so the cost is paid once per job rather than once
per worker.

The reference tile can be the first one in the bucket; nothing needs to be
downloaded. A local copy in `data/` only makes that single parse faster (~0.2s
versus ~110s, since walking a 25,000-entry HDF5 chunk index over S3 is
thousands of tiny latency-bound reads) and is worth having when you iterate,
but it is never required. Use `--template` to point at a specific reference, or
`--no-template` to parse every tile separately for a heterogeneous set.

**Sharding and parallel writes.** A Zarr v3 shard is stored as one object, so
writing any chunk inside it rewrites the whole shard. Fusion tasks run
concurrently, so a shard touched by two tasks would race and one would silently
drop the other's chunks. The default shard is therefore exactly one task's
footprint — full depth, `--block-size` in X and Y — which makes every shard the
property of a single task, written once, with no read-modify-write. An explicit
`--shard-size` must be a multiple of `--chunk-size` and must divide
`--block-size` in X and Y; the tool refuses layouts that would race.

**Overlap blending.** `avg_blend` (default) feathers across a 40-voxel ramp
whose weight reaches exactly 0 on each tile's outermost voxel, leaving a
one-voxel zero shell where no neighbour contributes. `lowest_view_wins` and
`max_blend` use a hard mask instead — but `max_blend` takes a voxelwise maximum
against a zero-filled buffer, so it will clamp negative values to 0.

**Only v7.3 files.** MAT v4/v5/v7 are not HDF5: each variable is a single
zlib-deflated stream with no internal chunk index, so there is no sub-array byte
range for a manifest to point at. Re-save from MATLAB with `save(..., '-v7.3')`.
Cell arrays, structs, and objects are stored as HDF5 object references, which
are pointers rather than byte ranges, and cannot be virtualized either.

## Layout

```
manifests/    tile position manifests (YAML)
data/         .mat files and fused output (gitignored)
src/rhapso_mat_examples/
  mat_io.py          open a v7.3 .mat as virtual Zarr
  mat_info.py        inspect / verify a .mat
  fuse_mat_tiles.py  the mosaic fusion pipeline
  serve_zarr.py      CORS + Range static server
  make_demo_tiles.py / fuse_tiles.py   synthetic example
```
