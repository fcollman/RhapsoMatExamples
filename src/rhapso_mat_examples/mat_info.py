"""
Inspect a v7.3 .mat file as virtual Zarr, and optionally prove the reads.

    mat-info data/slice_150_tile_001_Cross.mat --verify
    mat-info s3://bucket/key.mat

Prints what the chunk manifest covers without reading any array data. --verify
compares virtual reads against h5py reads of the same file, which needs a local
copy since h5py is the ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .mat_io import as_url, make_registry, open_mat, root_group


def chunk_lengths(manifest) -> np.ndarray:
    """Compressed byte length of every chunk, as a flat array."""
    lengths = getattr(manifest, "_lengths", None)
    if lengths is not None:
        return np.asarray(lengths).ravel()

    return np.array(
        [entry["length"] for entry in manifest.dict().values()], dtype=np.uint64
    )


def describe(store) -> None:
    """Report the manifest contents; reads no array data."""
    group = root_group(store)

    for name, array in sorted(group.arrays.items()):
        manifest = array.manifest
        grid = manifest.shape_chunk_grid
        lengths = chunk_lengths(manifest)
        lengths = lengths[lengths > 0]
        logical = int(np.prod(array.shape)) * np.dtype(array.dtype).itemsize

        print(f"\n  variable {name!r}")
        print(f"    zarr shape   : {array.shape}   (HDF5 order)")
        print(f"    matlab shape : {tuple(array.shape[::-1])}   (column-major)")
        print(f"    dtype        : {array.dtype}")
        print(f"    chunk shape  : {array.metadata.chunks}   (native HDF5 chunk)")
        print(f"    chunk grid   : {grid} -> {int(np.prod(grid)):,} chunks")
        print(f"    codecs       : "
              f"{[type(c).__name__ for c in array.metadata.codecs]}")
        if lengths.size:
            print(f"    chunk bytes  : min={lengths.min():,} "
                  f"max={lengths.max():,} mean={int(lengths.mean()):,}")
            print(f"    referenced   : {lengths.sum() / 1e9:.3f} GB of the .mat "
                  f"({logical / 1e9:.3f} GB uncompressed), 0 copied")
        attributes = dict(array.metadata.attributes)
        if attributes:
            print(f"    attributes   : {attributes}")


def verify(store, local_path: str, n_slabs: int = 4, seed: int = 0) -> bool:
    """Compare virtual reads against h5py reads of the same file."""
    import h5py
    import zarr

    root = zarr.open_group(store, mode="r")
    rng = np.random.default_rng(seed)
    all_ok = True

    with h5py.File(local_path, "r") as handle:
        for name in sorted(root.array_keys()):
            array, truth = root[name], handle[name]

            if array.shape != truth.shape:
                print(f"    {name}: SHAPE MISMATCH "
                      f"{array.shape} vs {truth.shape}")
                all_ok = False
                continue

            for _ in range(n_slabs):
                slices = []
                for dim, chunk in zip(array.shape, array.chunks):
                    width = min(dim, max(chunk, 1) * 2)
                    start = int(rng.integers(0, max(dim - width, 0) + 1))
                    slices.append(slice(start, start + width))
                slices = tuple(slices)

                got, want = array[slices], truth[slices]
                label = ", ".join(f"{s.start}:{s.stop}" for s in slices)

                if np.array_equal(got, want):
                    print(f"    {name}[{label}] {got.shape} -> match")
                else:
                    print(f"    {name}[{label}] -> MISMATCH")
                    all_ok = False

    return all_ok


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a v7.3 .mat file as virtual Zarr."
    )
    parser.add_argument("mat", help="Path or s3:// URL of a v7.3 .mat file")
    parser.add_argument("--verify", action="store_true",
                        help="Check virtual reads against h5py (local files).")
    args = parser.parse_args(argv)

    url = as_url(args.mat)
    print(f"Parsing {url}")
    store = open_mat(url, make_registry(url))
    describe(store)

    if args.verify:
        local = args.mat.replace("file://", "")
        if not Path(local).exists():
            print("\n--verify needs a local file; skipping.")
            return
        print("\nVerifying virtual reads against h5py:")
        ok = verify(store, local)
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
