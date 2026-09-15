"""
Reading MATLAB v7.3 (.mat) files as virtual Zarr, by byte range.

A v7.3 MAT-file *is* an HDF5 file with a 512-byte userblock, so VirtualiZarr's
HDFParser already handles it: it reads the chunk offsets, lengths, and filter
pipeline out of the HDF5 chunk index and maps them to Zarr chunks. Nothing here
reimplements that. This module is the thin layer around it:

  - `check_v73`   fail early and clearly on a .mat that is not HDF5-based
  - `bucket_region` / `make_registry`  resolve S3 correctly
  - `open_mat`    the four lines that tie them together

Reading one Zarr chunk is one ranged GET into the .mat file. No download, no
conversion, no chunk rewriting.

Limits
------
Only v7.3 files work. MAT v4/v5/v7 are not HDF5: each variable is a single
zlib-deflated stream with no internal chunk index, so there is no sub-array
byte range for a manifest to point at. Re-save from MATLAB with
`save(..., '-v7.3')`.

Cell arrays, structs, and objects are stored as HDF5 object references into a
`#refs#` group. A reference is a pointer rather than a byte range, so those
variables cannot be virtualized; HDFParser will either skip or fail on them.

Axis order
----------
MATLAB is column-major and writes HDF5 dimensions in the opposite order, so a
MATLAB array of size [a b c] appears here with shape (c, b, a). A chunk
manifest maps byte ranges to positions and cannot transpose, so the Zarr array
is necessarily in HDF5 order. Transpose after reading if you need MATLAB's view.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

# The first 116 bytes of a MAT-file are a text description. v7.3 files say so;
# v5/v7 files say "MATLAB 5.0 MAT-file". v4 files have no text header at all.
MAT73_MARKER = b"MATLAB 7.3 MAT-file"
MAT5_MARKER = b"MATLAB 5.0 MAT-file"
HEADER_BYTES = 128


def as_url(path_or_url: str) -> str:
    """Normalize a filesystem path to a file:// URL; leave real URLs alone."""
    if "://" in path_or_url:
        return path_or_url
    return f"file://{Path(path_or_url).resolve()}"


def bucket_region(bucket: str, default: str = "us-east-1") -> str:
    """
    Ask S3 which region a bucket lives in.

    A request to the wrong region comes back as a bare redirect, which obstore
    surfaces as a misleading "incorrectly configured region", so resolve it up
    front rather than guessing.
    """
    import urllib.request

    request = urllib.request.Request(
        f"https://{bucket}.s3.amazonaws.com/", method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.headers.get("x-amz-bucket-region") or default
    except Exception as error:  # noqa: BLE001 - HTTP errors still carry the header
        headers = getattr(error, "headers", None)
        if headers is not None:
            return headers.get("x-amz-bucket-region") or default
        return default


def make_registry(url: str, anonymous: bool = True, region: str | None = None):
    """
    Build an ObjectStoreRegistry for a local path or an S3 URL.

    Convenience only -- pass your own registry if you need credentials, a
    custom endpoint, or non-default retry behaviour.
    """
    from obspec_utils.registry import ObjectStoreRegistry

    if url.startswith("s3://"):
        from obstore.store import S3Store

        bucket = urlparse(url).netloc
        store = S3Store(
            bucket,
            skip_signature=anonymous,
            region=region or bucket_region(bucket),
        )
        return ObjectStoreRegistry({f"s3://{bucket}": store})

    from obstore.store import LocalStore

    directory = str(Path(url.replace("file://", "")).resolve().parent)
    return ObjectStoreRegistry({f"file://{directory}": LocalStore(directory)})


def check_v73(url: str, registry) -> None:
    """Read the 128-byte header and confirm this is a v7.3 MAT-file."""
    import obstore

    store, path = registry.resolve(url)
    header = bytes(obstore.get_range(store, path, start=0, end=HEADER_BYTES))

    if MAT73_MARKER in header:
        return

    if MAT5_MARKER in header:
        raise ValueError(
            f"{url} is a MAT v5/v7 file, which is not HDF5. Each variable is a "
            "single zlib stream with no internal chunk index, so there are no "
            "sub-array byte ranges for a chunk manifest to reference. Re-save "
            "it from MATLAB with save(..., '-v7.3')."
        )

    raise ValueError(
        f"{url} does not look like a v7.3 MAT-file. First bytes: "
        f"{header[:32]!r}. Only v7.3 (HDF5-based) files can be virtualized."
    )


def open_mat(url: str, registry=None, *, validate: bool = True):
    """
    Open a v7.3 .mat file as a virtual Zarr ManifestStore.

    The store is itself a zarr store, so `zarr.open_group(store)` works with no
    Icechunk or Kerchunk round trip.
    """
    from virtualizarr.parsers import HDFParser

    registry = registry or make_registry(url)
    if validate:
        check_v73(url, registry)

    return HDFParser()(url, registry)


def root_group(store):
    """
    The root ManifestGroup of a ManifestStore.

    VirtualiZarr has no public accessor yet, so try the public name first and
    fall back to the private one.
    """
    for attribute in ("group", "_group"):
        value = getattr(store, attribute, None)
        if value is None:
            continue
        return value() if callable(value) else value

    raise AttributeError("could not find the root group on this ManifestStore")


def open_mat_array(url: str, variable: str, registry=None, *, validate: bool = True):
    """Open one variable of a .mat file as a virtual Zarr array."""
    import zarr

    store = open_mat(url, registry, validate=validate)
    return zarr.open_group(store, mode="r")[variable]
