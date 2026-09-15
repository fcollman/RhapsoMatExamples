"""
Step 1 of 2: create some Zarr tiles to fuse.

Writes a 2x2 grid of 3D tiles that overlap their neighbours by 10%. Each tile
holds a smooth function of *global* position, so a correct fusion looks like one
continuous volume and any misalignment shows up as a visible seam.

    python examples/make_demo_tiles.py --out /tmp/rhapso_demo

Then fuse them with:

    python examples/fuse_tiles.py --dir /tmp/rhapso_demo

Skip this script entirely if you already have tiles. Fusion needs each tile to
be a Zarr group holding a 5D [t, c, z, y, x] array named "0" -- it reads
arr[0, 0, z, y, x], so one timepoint and one channel per fusion run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr

# Tile shape in XYZ voxels. Rhapso is XYZ-ordered in its transform and size
# metadata, while the Zarr arrays themselves are [t, c, z, y, x].
TILE_SIZE_XYZ = (256, 256, 128)

# Fractional overlap between neighbouring tiles in X and Y.
OVERLAP_FRACTION = 0.10


def tile_origins_xyz():
    """Corner positions of a 2x2 grid whose tiles overlap by OVERLAP_FRACTION."""
    sx, sy, _sz = TILE_SIZE_XYZ
    step_x = float(round(sx * (1.0 - OVERLAP_FRACTION)))
    step_y = float(round(sy * (1.0 - OVERLAP_FRACTION)))

    return [
        (0.0, 0.0, 0.0),
        (step_x, 0.0, 0.0),
        (0.0, step_y, 0.0),
        (step_x, step_y, 0.0),
    ]


def global_pattern(gx, gy, gz):
    """A smooth function of global position, scaled into uint16 range."""
    value = (
        0.5
        + 0.25 * np.sin(2.0 * np.pi * gx / 97.0)
        + 0.15 * np.cos(2.0 * np.pi * gy / 61.0)
        + 0.10 * np.sin(2.0 * np.pi * gz / 43.0)
    )
    return np.clip(value, 0.0, 1.0) * 40000.0


def write_tile(path: Path, origin_xyz) -> None:
    """Write one tile, filled with the pattern sampled at its global position."""
    sx, sy, sz = TILE_SIZE_XYZ
    ox, oy, oz = origin_xyz

    z = np.arange(sz, dtype=np.float64)[:, None, None] + oz
    y = np.arange(sy, dtype=np.float64)[None, :, None] + oy
    x = np.arange(sx, dtype=np.float64)[None, None, :] + ox

    data = np.broadcast_to(
        np.rint(global_pattern(x, y, z)).astype(np.uint16),
        (sz, sy, sx),
    )

    group = zarr.open_group(store=str(path), mode="w", zarr_format=3)
    array = group.create_array(
        name="0",
        shape=(1, 1, sz, sy, sx),   # [t, c, z, y, x]
        chunks=(1, 1, 64, 128, 128),
        dtype=np.uint16,
        fill_value=0,
    )
    array[0, 0] = data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="/tmp/rhapso_demo",
        help="Directory to write the tiles into.",
    )
    args = parser.parse_args()

    tiles_dir = Path(args.out).expanduser().resolve() / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)

    origins = tile_origins_xyz()
    print(f"Writing {len(origins)} tiles of size XYZ={TILE_SIZE_XYZ} "
          f"with {OVERLAP_FRACTION:.0%} overlap")

    for index, origin in enumerate(origins):
        path = tiles_dir / f"tile_{index}.zarr"
        write_tile(path, origin)
        print(f"  {path.name:16s} origin XYZ = "
              f"({origin[0]:7.1f}, {origin[1]:7.1f}, {origin[2]:7.1f})")

    print(f"\nTiles written to {tiles_dir}")
    print(f"Now run: python examples/fuse_tiles.py --dir {tiles_dir.parent}")


if __name__ == "__main__":
    main()
