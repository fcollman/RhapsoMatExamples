"""
Step 2 of 2: fuse a set of Zarr tiles given their affine transforms.

This is the "I already know where my tiles go" path. It runs no detection, no
matching, and no solver, and there is no SpimData XML anywhere -- we build
Rhapso's per-view dictionary directly and hand it to affine fusion.

    python examples/fuse_tiles.py --dir /tmp/rhapso_demo

To use your own data, edit define_tiles() below. Everything else is boilerplate.

---------------------------------------------------------------------------
The contract
---------------------------------------------------------------------------
Fusion consumes exactly one structure. Rhapso's XML parser (ComputeBBox) exists
only to build it, and every downstream stage -- OverlappingViews,
OverlappingBlocks, GenerateFusionInstructions, FuseCell -- reads nothing else:

    per_view_transforms = {
        (timepoint, setup): {
            "transform": <4x4 float64>,   # source voxels -> global space, XYZ
            "size":      (sx, sy, sz),    # tile shape in XYZ voxels
            "path":      "/path/tile.zarr",
            "split_def": None,            # only used for split tiles
        },
        ...
    }

Two things to know about it:

  - The transform maps source voxel coordinates to the global fused space, in
    XYZ order. The output grid is one voxel per unit of that space, so bake any
    voxel-size or anisotropy scaling into the matrix.
  - Insert keys in ascending (timepoint, setup) order.
    GenerateFusionInstructions builds per-view bounding boxes in dict insertion
    order but indexes them against a sorted view list, so an unsorted dict can
    mis-associate views with their bounds.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import ray
import zarr

from Rhapso.affine_fusion.compute_grid import ComputeGrid
from Rhapso.affine_fusion.fuse_cell import FuseCell
from Rhapso.affine_fusion.generate_fusion_instructions import GenerateFusionInstructions
from Rhapso.affine_fusion.initialize_output_zarr import InitializeOutputZarr
from Rhapso.affine_fusion.overlapping_blocks import OverlappingBlocks
from Rhapso.affine_fusion.overlapping_views import OverlappingViews

# ---------------------------------------------------------------------------
# 1. Describe your tiles
# ---------------------------------------------------------------------------

# Fractional overlap between neighbours, used to lay out the demo grid below.
OVERLAP_FRACTION = 0.10


def translation(tx, ty, tz) -> np.ndarray:
    """The simplest useful transform: a pure shift, in global XYZ units."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (tx, ty, tz)
    return T


def affine(linear_3x3, translation_xyz) -> np.ndarray:
    """A general affine, for rotation / scale / shear plus a shift."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(linear_3x3, dtype=np.float64)
    T[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return T


def define_tiles(tiles_dir: Path):
    """
    Return [(zarr path, 4x4 transform), ...] -- replace this with your own.

    The demo lays four tiles out in a 2x2 grid with OVERLAP_FRACTION overlap,
    deriving the step from the tile shape on disk so the numbers stay honest.
    In real use you would just list your tiles and their known transforms:

        return [
            (Path("/data/tile_a.zarr"), translation(0,    0, 0)),
            (Path("/data/tile_b.zarr"), translation(1843, 0, 0)),
            ...
        ]

    or, with anisotropic voxels and a rotation already baked in:

        return [(Path("/data/tile_a.zarr"), affine(my_3x3, my_shift)), ...]
    """
    paths = sorted(tiles_dir.glob("tile_*.zarr"))
    if not paths:
        raise SystemExit(f"No tile_*.zarr found in {tiles_dir}")

    sx, sy, _sz = tile_size_xyz(paths[0])
    step_x = float(round(sx * (1.0 - OVERLAP_FRACTION)))
    step_y = float(round(sy * (1.0 - OVERLAP_FRACTION)))

    origins = [
        (0.0, 0.0, 0.0),
        (step_x, 0.0, 0.0),
        (0.0, step_y, 0.0),
        (step_x, step_y, 0.0),
    ]

    return [(path, translation(*origin)) for path, origin in zip(paths, origins)]


def tile_size_xyz(path: Path):
    """Read a tile's XYZ shape off disk, so you don't have to hardcode it."""
    array = zarr.open_group(store=str(path), mode="r")["0"]
    nz, ny, nx = array.shape[-3:]          # arrays are [t, c, z, y, x]
    return (int(nx), int(ny), int(nz))     # transforms and sizes are XYZ


def build_per_view_transforms(tiles) -> dict:
    """Turn [(path, transform), ...] into the dictionary fusion consumes."""
    return {
        (0, setup): {
            "transform": np.asarray(T, dtype=np.float64),
            "size": tile_size_xyz(path),
            "path": str(path),
            "split_def": None,
        }
        for setup, (path, T) in enumerate(tiles)
    }


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
# 2. Run fusion
# ---------------------------------------------------------------------------


@ray.remote
def fuse_grid_block(grid_block, bb_min, bb_max, per_view_transforms,
                    output_path, overlap_strategy):
    """
    Render one cell of the output grid.

    This is the body of Rhapso.pipelines.ray.affine_fusion.AffineFusion,
    unchanged except that per_view_transforms comes from us rather than from
    parsing an XML.
    """
    super_block_offset = grid_block[0] + bb_min
    super_block_size = grid_block[1]

    overlapping_views, fused_min, fused_max = OverlappingViews(
        super_block_offset, super_block_size, per_view_transforms
    ).run()
    if not overlapping_views:
        return

    blocks = OverlappingBlocks(
        per_view_transforms, overlapping_views, super_block_offset,
        fused_min, fused_max, grid_block,
    ).run()
    if not any(blocks.values()):
        return

    image_instructions, blocks = GenerateFusionInstructions(
        per_view_transforms, grid_block, bb_min, bb_max,
        overlap_strategy, overlapping_views,
    ).run()

    FuseCell(
        image_instructions, blocks, per_view_transforms, output_path,
        grid_block, bb_min, bb_max, overlap_strategy,
    ).run()


def fuse(per_view_transforms, output_path: Path, strategy: str,
         block_size, output_block_size):
    bb_min, bb_max = compute_global_bbox(per_view_transforms)
    dims = (bb_max - bb_min) + 1

    as_ints = lambda v: tuple(int(i) for i in v)  # noqa: E731
    print(f"Fused bbox: min XYZ={as_ints(bb_min)} max XYZ={as_ints(bb_max)} "
          f"dims XYZ={as_ints(dims)}")

    # Creates the output group with a uint16 array "0" of shape (1,1,Z,Y,X).
    # zarr_input_prefix is only read by an attrs copy that run() never invokes,
    # so any tile path satisfies it.
    InitializeOutputZarr(
        output_path=str(output_path),
        zarr_input_prefix=next(iter(per_view_transforms.values()))["path"],
        dims=dims,
        output_zarr_version=3,
        compressor_cname="zstd",
        compressor_clevel=6,
        compressor_shuffle="bitshuffle",
        output_block_size=output_block_size,
    ).run()

    grid = ComputeGrid(dims, block_size).run()
    print(f"Submitting {len(grid)} fusion tasks (strategy={strategy})")

    ray.get([
        fuse_grid_block.remote(
            grid_block, bb_min, bb_max, per_view_transforms,
            str(output_path), strategy,
        )
        for grid_block in grid
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="/tmp/rhapso_demo",
        help="Directory holding a tiles/ subdirectory; output goes here too.",
    )
    parser.add_argument(
        "--strategy",
        default="avg_blend",
        choices=["avg_blend", "lowest_view_wins", "max_blend"],
        help="How competing voxels in the overlap region are resolved. "
             "avg_blend feathers across a 40-voxel ramp; the other two use a "
             "hard mask.",
    )
    args = parser.parse_args()

    work_dir = Path(args.dir).expanduser().resolve()
    output_path = work_dir / "fused.zarr"

    # InitializeOutputZarr only creates the array if one is not already there,
    # so a leftover output from a previous run with different transforms would
    # be silently reused at its old shape. Start clean.
    if output_path.exists():
        print(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    tiles = define_tiles(work_dir / "tiles")
    for path, T in tiles:
        origin = T[:3, 3]
        print(f"  {path.name:16s} size XYZ={tile_size_xyz(path)} "
              f"origin XYZ=({origin[0]:7.1f},{origin[1]:7.1f},{origin[2]:7.1f})")

    per_view_transforms = build_per_view_transforms(tiles)

    ray.init()
    fuse(
        per_view_transforms,
        output_path,
        args.strategy,
        block_size=[256, 256, 128],        # XYZ, one Ray task per block
        output_block_size=[128, 128, 64],  # XYZ, output Zarr chunk shape
    )

    fused = zarr.open_group(store=str(output_path), mode="r")["0"]
    print(f"\nFused {len(tiles)} tiles -> {output_path}")
    print(f"Output array [t,c,z,y,x]: shape={fused.shape} dtype={fused.dtype}")


if __name__ == "__main__":
    main()
