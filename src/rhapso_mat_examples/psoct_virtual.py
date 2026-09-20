"""
Wrap the PS-OCT tiles as virtual OME-Zarr stores, so Rhapso can read them unchanged.

    psoct-virtual --out work/virtual            # all 24 tiles of slice 004
    psoct-virtual --out work/virtual --tiles 4  # just the first 4

Each tile gets one small JSON of chunk references -- no pixels are copied -- that
presents the original S3 array through fsspec's reference filesystem:

    reference::file:///abs/path/slice_004_tile_001_ch_0.ome.zarr.json

Rhapso opens that URL with `zarr.storage.FsspecStore.from_url`, so nothing in its reader
needs to know the data is virtual. Two details make it work:

  * The filename carries `_ch_0.ome.zarr`. Rhapso's XML parser derives the channel by
    string surgery on the path (`file_path.split("_ch_", 1)[1]`) and raises IndexError
    without it. Naming the wrapper to match is cheaper than patching the parser, and the
    value is metadata only -- nothing indexes with it.
  * `storage_options` must carry `remote_protocol`/`remote_options`, since a reference
    URL cannot say where its targets live. STORAGE_OPTIONS below is what to pass.

---------------------------------------------------------------------------
What a virtual wrapper can and cannot fix
---------------------------------------------------------------------------
It CAN add metadata: OME-style group attributes, dimension_names, voxel size. Those are
pure metadata, so they cost a few kB per tile.

It also copes with the slice being heterogeneous, which it is: 23 tiles are float32 of
shape (191, 1000, 1000, 2) with dimension_names (z, x, y, channel), and one -- tile 20 --
is complex64 of shape (191, 1000, 1000) with no channel axis. Both are valid zarr; they
just differ. The chunk grid and dimension_names come from each tile's own metadata, so
the wrapper needs no special case, and downstream channel_mode="magnitude" reduces
either form to the same physical quantity. (Tile 20 still will not render in
Neuroglancer, but because complex64 is not a dtype it supports -- not because anything
is missing.)

It CANNOT reorder the axes. The tiles are stored (z, x, y, channel) with chunk shape
(191, 100, 100, 2), so BOTH channels live inside every chunk, interleaved along the last
axis. A virtual store references whole chunks by byte range, and OME's (t, c, z, y, x)
order would need a different byte layout inside each chunk -- c-major rather than
z-major. No amount of reference rewriting produces that; only re-encoding the pixels
would, which is the 34 GB read this wrapper exists to avoid.

So the axis order is *declared* rather than rearranged: dimension_names says
["z", "x", "y", "channel"], and Rhapso's TileVolumeReader picks that up with
source_axes="auto" and transposes lazily at read time. The data describes itself and
nothing is copied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from Rhapso.matching.tile_volume_reader import is_retriable

BUCKET = "apex-connects"
PREFIX = "CMC/Derivatives/Vlad/PS-OCT/3DTiles/Orientation/004/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# fsspec options a reference store needs, since the URL cannot carry them
STORAGE_OPTIONS = {"remote_protocol": "s3", "remote_options": {"anon": True}}

# the tiles declare this; TileVolumeReader(source_axes="auto") reads it back
SOURCE_AXES = "zxyc"


def fetch(url, attempts=5, backoff=1.5, method="GET"):
    """
    Read a URL, retrying only transient failures.

    Building the wrappers is all metadata, but it is still hundreds of small requests
    over a link that may be flaky, and a single reset should not abandon the run. The
    retriable/not decision is Rhapso's, so the policy is the same here as it is for the
    chunk reads themselves.
    """
    last = None

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method=method), timeout=60
            ) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - re-raised unless transient
            if not is_retriable(exc):
                raise

            last = exc

            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))

    raise RuntimeError(f"{url} failed after {attempts} transient errors: {last}") from last


def list_tiles(bucket: str, prefix: str) -> list[str]:
    """Anonymous listing of the .zarr directories under `prefix`."""
    names: list[str] = []
    token = None

    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/",
                  "max-keys": "1000"}

        if token:
            params["continuation-token"] = token

        url = f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"
        root = ET.fromstring(fetch(url))

        for common in root.findall("s3:CommonPrefixes", S3_NS):
            entry = (common.findtext("s3:Prefix", namespaces=S3_NS) or "")[len(prefix):]

            if entry.rstrip("/").endswith(".zarr"):
                names.append(entry.rstrip("/"))

        if root.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            break

        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)

    return sorted(names, key=lambda s: [int(t) if t.isdigit() else t
                                        for t in re.split(r"(\d+)", s)])


def read_zarr_json(bucket: str, prefix: str, name: str) -> dict:
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(prefix + name)}/zarr.json"

    return json.loads(fetch(url))


def chunk_grid(shape, chunk_shape) -> list[tuple[int, ...]]:
    """Every chunk index of a regular grid, as tuples."""
    counts = [-(-int(s) // int(c)) for s, c in zip(shape, chunk_shape)]
    grid = [()]

    for count in counts:
        grid = [item + (i,) for item in grid for i in range(count)]

    return grid


def virtual_refs(bucket: str, prefix: str, name: str, metadata: dict,
                 voxel_size=(5.0, 5.0, 3.4)) -> dict:
    """
    Kerchunk-style references for one tile: its metadata inline, its chunks by URL.

    The metadata is augmented rather than replaced -- dimension_names and the codec
    chain must stay byte-identical to the source or the referenced chunks will not
    decode. Only `attributes` gains anything.
    """
    metadata = json.loads(json.dumps(metadata))      # don't mutate the caller's copy
    shape = metadata["shape"]
    chunk_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    names = list(metadata.get("dimension_names") or [])

    metadata.setdefault("attributes", {})
    metadata["attributes"].update(
        {
            # OME-style description of what the axes mean. Declarative only: the byte
            # layout is untouched, so the order below is the order on disk, which is
            # NOT OME's canonical t,c,z,y,x -- see the module docstring.
            "axes": [
                {"name": axis,
                 "type": "channel" if axis in ("c", "channel") else "space",
                 "unit": None if axis in ("c", "channel") else "micrometer"}
                for axis in names
            ],
            "voxel_size_xyz_um": list(voxel_size),
            "source": f"s3://{bucket}/{prefix}{name}",
            "rhapso_source_axes": SOURCE_AXES,
        }
    )

    base = f"s3://{bucket}/{prefix}{name}"
    refs = {"zarr.json": json.dumps(metadata)}
    for index in chunk_grid(shape, chunk_shape):
        key = "c/" + "/".join(str(i) for i in index)
        refs[key] = [f"{base}/{key}"]

    return {"version": 1, "refs": refs}


def wrapper_name(tile_name: str) -> str:
    """
    slice_004_tile_001_orientation.zarr -> slice_004_tile_001_orientation_ch_0.ome.zarr.json

    The `_ch_0.ome.zarr` infix is what Rhapso's XML parser needs to find a channel.
    """
    stem = tile_name[:-len(".zarr")] if tile_name.endswith(".zarr") else tile_name

    return f"{stem}_ch_0.ome.zarr.json"


def virtual_url(path: Path) -> str:
    """The fsspec reference URL for a wrapper file."""
    return f"reference::file://{path.resolve()}"


def build(out_dir, bucket=BUCKET, prefix=PREFIX, limit=None, voxel_size=(5.0, 5.0, 3.4)):
    """
    Write one wrapper per tile. Returns [(tile_name, url, shape, dimension_names), ...].
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = list_tiles(bucket, prefix)

    if not names:
        raise SystemExit(f"no .zarr tiles found under s3://{bucket}/{prefix}")

    if limit:
        names = names[:limit]

    built = []
    reference_metadata = None

    for name in names:
        metadata = read_zarr_json(bucket, prefix, name)

        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata["shape"] != reference_metadata["shape"]:
            print(f"warning: {name} shape {metadata['shape']} differs from "
                  f"{reference_metadata['shape']}", file=sys.stderr)

        path = out / wrapper_name(name)
        path.write_text(json.dumps(virtual_refs(bucket, prefix, name, metadata,
                                                voxel_size)))
        built.append(
            (name, virtual_url(path), tuple(metadata["shape"]),
             tuple(metadata.get("dimension_names") or ()))
        )

    return built


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="work/virtual",
                        help="directory for the wrapper files (default work/virtual)")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--tiles", type=int,
                        help="wrap only the first N tiles")
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[5.0, 5.0, 3.4],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--verify", action="store_true",
                        help="read one voxel through each wrapper to prove it resolves")
    args = parser.parse_args(argv)

    built = build(args.out, args.bucket, args.prefix, args.tiles, tuple(args.voxel_size))

    for name, url, shape, names in built:
        print(f"{name}  shape={shape}  dims={names}")

    print(f"\nwrote {len(built)} wrapper(s) to {Path(args.out).resolve()}")
    print(f"storage_options: {STORAGE_OPTIONS}")
    print(f"source_axes: {SOURCE_AXES} (or 'auto', read from dimension_names)")

    if args.verify:
        import numpy as np
        import zarr

        for name, url, shape, _names in built:
            store = zarr.storage.FsspecStore.from_url(
                url, storage_options=STORAGE_OPTIONS, read_only=True
            )
            array = zarr.open(store, mode="r")
            middle = tuple(s // 2 for s in shape)
            value = float(np.abs(np.asarray(array[middle])))
            print(f"  verified {name}: {shape} {array.dtype} "
                  f"centre magnitude {value:+.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
