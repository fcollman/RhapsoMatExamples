"""
Fuse a mosaic of MATLAB v7.3 tiles described by a YAML manifest.

Reads tile data straight out of the .mat files by byte range -- no download and
no conversion -- and writes a single fused Zarr locally.

    # first 4 tiles (one row, ~3 GB out)
    fuse-mat-tiles --yaml manifests/test.yaml --limit 4

    # the whole mosaic (check the size it prints first)
    fuse-mat-tiles --yaml manifests/test.yaml

The YAML supplies the layout; each tile's x/y becomes the translation of its
affine transform:

    metadata:
      tile_width: 1000
      tile_height: 1000
      depth: 191
      tile_overlap: 0.1
    tiles:
      - filepath: slice_150_tile_001_Cross.mat
        tile_number: 1
        x: 0.0
        y: 0.0

---------------------------------------------------------------------------
How the tiles are read
---------------------------------------------------------------------------
Every .mat in this dataset shares an identical internal HDF5 layout, so the
chunk manifest is built ONCE from a reference file and its byte offsets are
stamped onto the other tiles with rename_paths(). Parsing one .mat locally
takes ~0.2s; parsing one over S3 takes ~110s, because walking the HDF5 chunk
index is thousands of tiny latency-bound reads. Stamping turns 126 x 110s into
a fraction of a second. Pass --no-template for a heterogeneous set.

---------------------------------------------------------------------------
Axis order
---------------------------------------------------------------------------
The .mat arrays are (x, y, depth) as HDF5 stores them. That was established by
cross-correlating tile 1's intensity profile against tile 2's, which peaks at
lag 899 for a nominal 900 offset. Rhapso wants [t, c, z, y, x], so each tile is
presented through a small transposing view.

---------------------------------------------------------------------------
Output dtype
---------------------------------------------------------------------------
This data is float64 spanning roughly -21..+57 (dB-like). Rhapso's stock
FuseCell ends with np.clip(np.rint(block), 0, 65535).astype(np.uint16), which
would clip every negative to 0 and round away the fraction, so the fusion step
is subclassed here to write float32 by default. Use --dtype uint16 only if you
know your values are non-negative integers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import ray
import yaml
import zarr

from Rhapso.affine_fusion.compute_grid import ComputeGrid
from Rhapso.affine_fusion.fuse_cell import FuseCell
from Rhapso.affine_fusion.generate_fusion_instructions import (
    GenerateFusionInstructions,
)
from Rhapso.affine_fusion.overlapping_blocks import OverlappingBlocks
from Rhapso.affine_fusion.overlapping_views import OverlappingViews

from .mat_io import bucket_region, as_url, make_registry, open_mat, root_group

DEFAULT_S3_PREFIX = (
    "s3://apex-connects/CMC/Derivatives/Vlad/PS-OCT/3DTiles/Cross/150"
)
MAT_VARIABLE = "Tile_cross"


# ---------------------------------------------------------------------------
# YAML manifest -> transforms
# ---------------------------------------------------------------------------


def load_manifest(path: Path, limit: int | None = None):
    """Read the YAML layout; returns (metadata, tiles) sorted by tile_number."""
    with open(path) as handle:
        document = yaml.safe_load(handle)

    metadata = document.get("metadata", {})
    tiles = sorted(document["tiles"], key=lambda tile: tile.get("tile_number", 0))

    return metadata, (tiles[:limit] if limit else tiles)


def tile_size_xyz(metadata) -> tuple[int, int, int]:
    """Tile shape in Rhapso's XYZ order, from the manifest metadata."""
    return (
        int(metadata["tile_width"]),
        int(metadata["tile_height"]),
        int(metadata["depth"]),
    )


def resolution_microns_xyz(metadata) -> tuple[float, float, float] | None:
    """
    Physical voxel size in microns, XYZ, from the manifest.

    Returns None when the manifest does not declare one, in which case the
    output is written without OME metadata and reads as unitless voxels.
    """
    values = metadata.get("resolution_microns")
    if not values:
        return None

    if len(values) != 3:
        raise SystemExit(
            f"metadata.resolution_microns must have 3 entries (x, y, z), "
            f"got {values!r}"
        )

    return tuple(float(v) for v in values)


def ome_multiscales(resolution_xyz, bb_min_xyz) -> dict:
    """
    OME-NGFF 0.5 multiscales metadata for the fused array.

    The tile offsets in the manifest are in voxels, so fusion runs on a voxel
    grid and the physical calibration lives here rather than in the affines --
    that keeps the output a 1:1 resampling of the source instead of forcing it
    onto a micron grid.

    Version 0.5 nests everything under an `ome` key; that is the pairing
    Neuroglancer expects for a Zarr v3 store.
    """
    res_x, res_y, res_z = (float(v) for v in resolution_xyz)
    min_x, min_y, min_z = (float(v) for v in bb_min_xyz)

    # Array axes are [t, c, z, y, x].
    transformations = [
        {"type": "scale", "scale": [1.0, 1.0, res_z, res_y, res_x]}
    ]

    # The fused volume's origin is bb_min in voxels; express it in microns so
    # the array lands where it belongs in physical space.
    if (min_x, min_y, min_z) != (0.0, 0.0, 0.0):
        transformations.append({
            "type": "translation",
            "translation": [0.0, 0.0, min_z * res_z, min_y * res_y,
                            min_x * res_x],
        })

    return {
        "version": "0.5",
        "multiscales": [
            {
                "name": "fused",
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {"path": "0", "coordinateTransformations": transformations}
                ],
            }
        ],
    }


def build_per_view_transforms(tiles, metadata, s3_prefix: str) -> dict:
    """
    One entry per tile: a pure translation to its x/y, at z = 0.

    Inserted in ascending (timepoint, setup) order, which
    GenerateFusionInstructions requires -- it builds per-view bounding boxes in
    dict insertion order but indexes them against a sorted view list.
    """
    size = tile_size_xyz(metadata)
    per_view_transforms = {}

    for setup, tile in enumerate(tiles):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (float(tile["x"]), float(tile["y"]), 0.0)

        per_view_transforms[(0, setup)] = {
            "transform": transform,
            "size": size,
            "path": f"{s3_prefix.rstrip('/')}/{tile['filepath']}",
            "split_def": None,
        }

    return per_view_transforms


def compute_global_bbox(per_view_transforms):
    """Axis-aligned bounds of all transformed tiles, in global coordinates."""
    gmin = np.full(3, np.inf)
    gmax = np.full(3, -np.inf)

    for info in per_view_transforms.values():
        sx, sy, sz = info["size"]
        corners = np.array(
            [
                [x, y, z, 1.0]
                for x in (0.0, sx - 1.0)
                for y in (0.0, sy - 1.0)
                for z in (0.0, sz - 1.0)
            ],
            dtype=np.float64,
        )
        world = (info["transform"] @ corners.T).T[:, :3]
        gmin = np.minimum(gmin, world.min(axis=0))
        gmax = np.maximum(gmax, world.max(axis=0))

    return np.floor(gmin).astype(np.int64), np.ceil(gmax).astype(np.int64)


# ---------------------------------------------------------------------------
# Reading .mat tiles as Rhapso views
# ---------------------------------------------------------------------------


class MatTileView:
    """
    Presents a .mat array stored as (x, y, z) in Rhapso's [t, c, z, y, x] form.

    FuseCell only ever asks for `.shape` and one `arr[0, 0, zs, ys, xs]` slice,
    so a transposing view is enough. Nothing is copied up front and the reads
    still land on the file's native chunks.
    """

    def __init__(self, array):
        self._array = array

    @property
    def shape(self):
        nx, ny, nz = self._array.shape
        return (1, 1, nz, ny, nx)

    @property
    def dtype(self):
        return self._array.dtype

    def __getitem__(self, key):
        _, _, z_slice, y_slice, x_slice = key
        return np.transpose(self._array[x_slice, y_slice, z_slice], (2, 1, 0))


# Per-process cache of opened tiles. The template itself is parsed once by the
# driver and shipped to the workers, so nothing here re-reads a chunk index.
_TILE_CACHE: dict = {}


def parse_template(template_url: str):
    """
    Parse the reference .mat whose byte offsets every tile shares.

    Called ONCE, in the driver. The resulting ManifestArray pickles to a few
    MB, so Ray ships it to the workers through its object store -- far cheaper
    than having every worker walk the HDF5 chunk index itself, which over S3
    costs ~110s each and would mean thousands of simultaneous range reads at
    startup on a large cluster.
    """
    return root_group(open_mat(template_url)).arrays[MAT_VARIABLE]


def open_tile(url: str, template, region: str | None):
    """
    Open one .mat tile as a zarr array.

    `template` is the pre-parsed ManifestArray, or None to parse this tile's
    own chunk index (correct for a heterogeneous set, much slower over S3).
    """
    if url in _TILE_CACHE:
        return _TILE_CACHE[url]

    registry = make_registry(url, region=region)

    if template is not None:
        from virtualizarr.manifests import ManifestGroup, ManifestStore

        store = ManifestStore(
            ManifestGroup(arrays={MAT_VARIABLE: template.rename_paths(url)}),
            registry=registry,
        )
    else:
        store = open_mat(url, registry)

    _TILE_CACHE[url] = zarr.open_group(store, mode="r")[MAT_VARIABLE]
    return _TILE_CACHE[url]


class MatFuseCell(FuseCell):
    """
    FuseCell that reads .mat tiles and writes without the uint16 coercion.

    Two overrides: where tile data comes from, and how the rendered block is
    stored. Blending, sampling, and the instruction set are stock Rhapso.
    """

    def __init__(self, *args, template=None, region=None,
                 out_dtype="float32", **kwargs):
        super().__init__(*args, **kwargs)
        self.template = template
        self.region = region
        self.out_dtype = np.dtype(out_dtype)

    def open_view_dataset(self, view_id, mode="r"):
        url = self.per_view_transforms[view_id]["path"]
        return MatTileView(open_tile(url, self.template, self.region))

    def run(self):
        block_min = self.grid_block[0]
        interval = self.fusion_max_global - self.fusion_min_global
        block_max = [
            min(int(interval[d]), int(block_min[d] + self.grid_block[1][d] - 1))
            for d in range(len(block_min))
        ]

        fused = self.render_fused_block(
            images_dict=self.image_instructions,
            final_blocks=self.blocks,
            block_min=block_min,
            block_max=block_max,
        )

        if self.out_dtype.kind in "ui":
            info = np.iinfo(self.out_dtype)
            fused = np.clip(np.rint(fused), info.min, info.max)

        self.write_block(fused.astype(self.out_dtype, copy=False),
                         self.grid_block[0])

    def write_block(self, fused_block_zyx, out_offset_xyz):
        out = self.open_zarr_array(self.output_path, mode="r+")
        x0, y0, z0 = map(int, out_offset_xyz)
        nz, ny, nx = fused_block_zyx.shape
        out[0, 0, z0:z0 + nz, y0:y0 + ny, x0:x0 + nx] = fused_block_zyx


# ---------------------------------------------------------------------------
# Fusion driver
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=2)
def fuse_grid_block(grid_block, bb_min, bb_max, per_view_transforms, output_path,
                    strategy, template, region, out_dtype):
    offset = grid_block[0] + bb_min

    views, fused_min, fused_max = OverlappingViews(
        offset, grid_block[1], per_view_transforms
    ).run()
    if not views:
        return 0

    blocks = OverlappingBlocks(
        per_view_transforms, views, offset, fused_min, fused_max, grid_block
    ).run()
    if not any(blocks.values()):
        return 0

    instructions, blocks = GenerateFusionInstructions(
        per_view_transforms, grid_block, bb_min, bb_max, strategy, views
    ).run()

    MatFuseCell(
        instructions, blocks, per_view_transforms, output_path, grid_block,
        bb_min, bb_max, strategy,
        template=template, region=region, out_dtype=out_dtype,
    ).run()

    return len(views)


def validate_sharding(chunk_xyz, shard_xyz, block_size_xyz) -> None:
    """
    Check a shard layout is both legal for Zarr and safe to write in parallel.

    Zarr requires each shard to be a whole number of inner chunks. On top of
    that, a shard is stored as ONE object: writing any chunk inside it
    rewrites the whole shard. Fusion tasks run concurrently in separate Ray
    workers, so if two tasks touched the same shard they would race and one
    would silently drop the other's chunks. Tasks tile the volume at multiples
    of block_size, so requiring the shard to divide the block keeps every
    shard the property of exactly one task.
    """
    axes = ("x", "y")

    for index, axis in enumerate(axes):
        if shard_xyz[index] % chunk_xyz[index]:
            raise SystemExit(
                f"--shard-size {axis}={shard_xyz[index]} is not a multiple of "
                f"--chunk-size {axis}={chunk_xyz[index]}."
            )
        if block_size_xyz[index] % shard_xyz[index]:
            raise SystemExit(
                f"--shard-size {axis}={shard_xyz[index]} does not divide "
                f"--block-size {axis}={block_size_xyz[index]}. Two fusion "
                f"tasks would write the same shard and race. Pick a shard "
                f"that divides the block, or raise --block-size."
            )

    if shard_xyz[2] % chunk_xyz[2]:
        raise SystemExit(
            f"--shard-size z={shard_xyz[2]} is not a multiple of "
            f"--chunk-size z={chunk_xyz[2]}."
        )
    # Every task spans the full depth, so z needs no block-alignment check.


def auto_shard_xyz(chunk_xyz, block_size_xyz, dims_xyz) -> list[int]:
    """
    Default shard layout: exactly one shard per fusion task.

    Each task renders a full-depth block of block_size in X and Y, so making
    the shard match that footprint means every shard is written once, by one
    task, in a single call -- no cross-task races and no read-modify-write.
    Depth is rounded up to a whole number of chunks so the whole column lands
    in one shard.

    If the block is not a whole number of chunks, fall back to one chunk per
    shard on that axis (no grouping) rather than emit an illegal layout.
    """
    shard = []
    for axis in (0, 1):
        chunk, block = int(chunk_xyz[axis]), int(block_size_xyz[axis])
        shard.append(block if block % chunk == 0 else chunk)

    chunk_z = int(chunk_xyz[2])
    shard.append(-(-int(dims_xyz[2]) // chunk_z) * chunk_z)
    return shard


def grid_counts(dims_xyz, cell_xyz) -> int:
    """How many cells of this size tile the volume."""
    return int(np.prod([
        -(-int(dim) // int(cell)) for dim, cell in zip(dims_xyz, cell_xyz)
    ]))


def create_output(path: Path, dims_xyz, chunks_xyz, dtype, shards_xyz=None,
                  resolution_xyz=None, bb_min_xyz=(0, 0, 0)) -> None:
    """
    Create the output group with a 5D array "0", matching Rhapso's convention.

    Rhapso's own InitializeOutputZarr hardcodes uint16, has no sharding, and
    writes no OME metadata (its separate MultiScale stage does that), so the
    array is created here instead.
    """
    if path.exists():
        shutil.rmtree(path)

    nx, ny, nz = (int(v) for v in dims_xyz)
    cx, cy, cz = (int(v) for v in chunks_xyz)

    options = {}
    if shards_xyz is not None:
        sx, sy, sz = (int(v) for v in shards_xyz)
        options["shards"] = (1, 1, sz, sy, sx)

    group = zarr.open_group(store=str(path), mode="w", zarr_format=3)
    group.create_array(
        name="0",
        shape=(1, 1, nz, ny, nx),
        # Without sharding, a chunk bigger than the array just wastes space;
        # with sharding the inner chunk must stay an exact divisor of the shard.
        chunks=(1, 1, cz, cy, cx) if shards_xyz is not None
        else (1, 1, min(cz, nz), min(cy, ny), min(cx, nx)),
        dtype=dtype,
        fill_value=0,
        compressors=[
            zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
        ],
        **options,
    )

    # Physical calibration, so the store resolves as an image rather than a
    # bare array and reads in microns instead of unitless voxels.
    if resolution_xyz is not None:
        group.attrs["ome"] = ome_multiscales(resolution_xyz, bb_min_xyz)


def resolve_template(args, tiles, per_view_transforms) -> str | None:
    """Pick the .mat whose chunk layout gets stamped onto every tile."""
    if args.no_template:
        return None

    if args.template:
        return as_url(args.template)

    for directory in (Path(args.data_dir), Path(args.yaml).parent):
        candidate = directory / tiles[0]["filepath"]
        if candidate.exists():
            return as_url(str(candidate))

    return per_view_transforms[(0, 0)]["path"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse MATLAB v7.3 tiles into a single Zarr."
    )
    parser.add_argument("--yaml", default="manifests/test.yaml",
                        help="Tile manifest.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N tiles. Omit to use all.")
    parser.add_argument("--out", default="fused.zarr",
                        help="Local output Zarr path.")
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX,
                        help="Prefix the YAML filepaths hang off.")
    parser.add_argument("--data-dir", default="data",
                        help="Where to look for a local copy of the template .mat.")
    parser.add_argument("--template", default=None,
                        help="Reference .mat whose HDF5 layout all tiles share.")
    parser.add_argument("--no-template", action="store_true",
                        help="Parse every tile separately (slow over S3).")
    parser.add_argument("--strategy", default="avg_blend",
                        choices=["avg_blend", "lowest_view_wins", "max_blend"],
                        help="Overlap handling.")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64", "uint16"],
                        help="Output dtype. uint16 clips negatives to 0.")
    parser.add_argument("--block-size", type=int, nargs=2, default=[256, 256],
                        metavar=("X", "Y"),
                        help="Fusion task size in X and Y; Z is always full.")
    parser.add_argument("--chunk-size", type=int, nargs=3, default=[128, 128, 64],
                        metavar=("X", "Y", "Z"),
                        help="Output Zarr chunk shape, XYZ.")
    parser.add_argument("--shard-size", type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override the Zarr v3 shard shape, XYZ. Must be a "
                             "multiple of --chunk-size and must divide "
                             "--block-size in X and Y, so each shard is "
                             "written by exactly one task. Defaults to one "
                             "shard per fusion task.")
    parser.add_argument("--no-shard", action="store_true",
                        help="Write unsharded, one file per chunk.")
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS"),
                        help="Attach to an existing Ray cluster instead of "
                             "starting a local one. Use 'auto' on a node that "
                             "is part of the cluster, or host:port. Defaults "
                             "to $RAY_ADDRESS. See slurm/ for a SLURM example.")
    parser.add_argument("--cpus-per-task", type=float, default=2,
                        help="CPUs Ray reserves per fusion block. Controls how "
                             "many blocks run at once, and so peak memory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the plan and exit without fusing.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    metadata, tiles = load_manifest(Path(args.yaml), args.limit)
    total = len(load_manifest(Path(args.yaml))[1])

    per_view_transforms = build_per_view_transforms(
        tiles, metadata, args.s3_prefix
    )
    bb_min, bb_max = compute_global_bbox(per_view_transforms)
    dims = (bb_max - bb_min) + 1

    dtype = np.dtype(args.dtype)
    out_bytes = int(np.prod(dims)) * dtype.itemsize

    resolution = resolution_microns_xyz(metadata)

    print(f"Tiles          : {len(tiles)} of {total} in the manifest")
    print(f"Tile size XYZ  : {tile_size_xyz(metadata)}")
    print(f"Overlap        : {metadata.get('tile_overlap')}")
    print(f"Fused dims XYZ : {tuple(int(v) for v in dims)}")

    if resolution:
        extent = tuple(round(float(d) * r / 1000.0, 2)
                       for d, r in zip(dims, resolution))
        print(f"Voxel size XYZ : {resolution} um  -> OME-NGFF 0.5 calibration")
        print(f"Physical extent: {extent} mm")
    else:
        print("Voxel size XYZ : not declared (add metadata.resolution_microns "
              "for physical units)")
    print(f"Output         : {args.out}  dtype={dtype}  "
          f"~{out_bytes / 1e9:.1f} GB uncompressed")
    print(f"Source to read : ~{len(tiles) * 1.57:.0f} GB from S3")

    if args.dtype == "uint16":
        print("\n  WARNING: this data goes negative (~ -21..+57). uint16 clips "
              "every negative value to 0 and rounds away the fraction.")
    if args.strategy == "max_blend":
        print("\n  WARNING: max_blend takes a voxelwise max against a "
              "zero-filled buffer, so negative values become 0.")

    template_url = resolve_template(args, tiles, per_view_transforms)
    if template_url:
        origin = "local" if template_url.startswith("file://") else "S3"
        print(f"Chunk template : {template_url}  [{origin}]")

    region = (bucket_region(args.s3_prefix.split("/")[2])
              if args.s3_prefix.startswith("s3://") else None)

    block_size = [args.block_size[0], args.block_size[1], int(dims[2])]
    grid = ComputeGrid(dims, block_size).run()
    print(f"Fusion tasks   : {len(grid)} blocks of {block_size} (XYZ)")

    chunk_files = grid_counts(dims, args.chunk_size)
    shard_size = None if args.no_shard else (
        args.shard_size or auto_shard_xyz(args.chunk_size, block_size, dims)
    )

    if shard_size:
        validate_sharding(args.chunk_size, shard_size, block_size)
        shard_files = grid_counts(dims, shard_size)
        origin = "explicit" if args.shard_size else "auto, one per task"
        print(f"Output chunks  : {args.chunk_size} (XYZ), "
              f"sharded into {shard_size} ({origin})")
        print(f"Files on disk  : {shard_files:,} shards "
              f"(vs {chunk_files:,} unsharded, "
              f"{chunk_files / max(shard_files, 1):.1f}x fewer)\n")
    else:
        print(f"Output chunks  : {args.chunk_size} (XYZ), unsharded")
        print(f"Files on disk  : {chunk_files:,}\n")

    if args.dry_run:
        print("Dry run; stopping here.")
        return

    output_path = Path(args.out).expanduser().resolve()
    create_output(output_path, dims, args.chunk_size, dtype, shard_size,
                  resolution_xyz=resolution, bb_min_xyz=bb_min)

    if args.ray_address:
        print(f"Attaching to Ray cluster at {args.ray_address}")
        ray.init(address=args.ray_address)
    else:
        ray.init()

    resources = ray.cluster_resources()
    print(f"Ray cluster    : {int(resources.get('CPU', 0))} CPUs across "
          f"{len(ray.nodes())} node(s)\n")

    # Parse the shared chunk index once here, then hand the workers a
    # reference to it. Every tile in this dataset has the same internal HDF5
    # layout, so one parse plus a path rewrite per tile replaces N chunk-index
    # walks -- the difference between seconds and hours over S3.
    template_ref = None
    if template_url:
        started = time.time()
        template = parse_template(template_url)
        elapsed = time.time() - started
        n_chunks = int(np.prod(template.manifest.shape_chunk_grid))
        print(f"Parsed template: {n_chunks:,} chunks in {elapsed:.1f}s "
              f"-> shared with all workers")
        template_ref = ray.put(template)

    task = fuse_grid_block.options(num_cpus=args.cpus_per_task)
    futures = [
        task.remote(
            grid_block, bb_min, bb_max, per_view_transforms, str(output_path),
            args.strategy, template_ref, region, args.dtype,
        )
        for grid_block in grid
    ]

    done = failed = 0
    step = max(len(grid) // 20, 1)
    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        try:
            ray.get(ready[0])
        except Exception as error:  # noqa: BLE001 - report and keep going
            failed += 1
            print(f"  [ERROR] {type(error).__name__}: {error}")
        done += 1
        if done % step == 0 or not futures:
            print(f"  {done}/{len(grid)} blocks "
                  f"({100 * done / len(grid):.0f}%)"
                  + (f"  failed={failed}" if failed else ""))

    array = zarr.open_group(store=str(output_path), mode="r")["0"]
    print(f"\nFused -> {output_path}")
    print(f"  shape={array.shape} dtype={array.dtype} chunks={array.chunks}")
    if failed:
        print(f"  {failed} block(s) failed; output is incomplete.")


if __name__ == "__main__":
    main()
