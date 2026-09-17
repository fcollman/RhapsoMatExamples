"""
Build a Neuroglancer link that lays out a slice's Zarr tiles as one image layer.

Each tile becomes a separate source under a single layer, carrying its own
affine transform: the tile's mosaic offset, the voxel size, and an optional
small tilt that corrects Z drift across the mosaic.

    # best: measured per-tile offsets from register-seams
    register-seams --out offsets.json
    ng-tile-link --offsets offsets.json

    # fallback: a regular grid plus a modelled per-tile z step
    ng-tile-link                                # defaults, as measured
    ng-tile-link --tilt-x -42.6 --tilt-y 42.5   # sweep the z drift
    ng-tile-link --no-tilt                      # uncorrected seams

    # one column only, to check the layout quickly
    ng-tile-link --limit 6 --no-tilt

    # reuse an existing layout instead of generating the grid
    ng-tile-link --manifest manifests/test.yaml --fast-axis x --fast-count 14

---------------------------------------------------------------------------
What the defaults are based on
---------------------------------------------------------------------------
The bucket has no manifest for slice 004, so the layout was measured from the
data. The 10% tile overlap is exactly one 100 px chunk, which makes the seam
directly testable: fetch the high-side edge chunk of one tile and the low-side
edge chunk of a candidate neighbour and correlate their log-magnitude.

  - tiles 1 and 2 abut along the array's *y* axis (ncc +0.63 .. +0.68),
    not x -- so the numbering runs down y first
  - tile 1+6 is tile 1's neighbour along *x* (ncc +0.92, versus ~0.0 for
    offsets of 3, 4 and 8) -- so the run length along y is 6

That gives 24 tiles = 6 along y x 4 along x, which is the default here.

The stride is NOT the nominal 900 px, and worse, there is no single stride.
Registering each seam (see "How the seams were registered") gives per-move
vectors that differ in length AND carry a cross-axis component:

    x moves    1 ->  7:  dx = 867   dy = +19   dz = -11
               7 -> 13:  dx = 890   dy =  -1   dz =  -9
              13 -> 19:  dx = 896   dy =  +2   dz = -11
    y moves    1 ->  2:  dy = 931   dx =  +5   dz =  +7
               2 ->  3:  dy = 932   dx =  +7   dz =  +7

So the stage is not calibrated to move purely along one axis, and its step
length varies. A grid cannot express that. Use ``register-seams`` to measure
every seam and ``--offsets`` to apply the result; the grid below is only a
fallback for when you have not measured.

As a fallback the grid uses stride 893 x 931 -- the means above, ignoring the
867 -- so the overlap is roughly 10.7% in x and 6.9% in y, not a uniform 10%.
``--stride`` overrides it and ``--overlap`` still derives it from a fraction.

---------------------------------------------------------------------------
How the seams were registered
---------------------------------------------------------------------------
Metric: normalised cross-correlation of log1p(|z|) -- the complex amplitude,
not the phase, which is near-random voxel to voxel -- averaged over the
overlapping region.

Getting this right took three attempts, and the failures are instructive:

  1. Scanning all shifts freely over a single 100 px chunk let the in-plane
     shift run to +-80, where only a 20 px sliver still overlapped. NCC over a
     sliver is meaningless and the search happily "won" there.
  2. Pinning the in-plane shift at 900 fixed that but assumed the stride
     rather than measuring it, and the recovered z steps were noisy.
  3. Bounding the in-plane range to +-18 kept the overlap honest but the peak
     then sat ON the bound -- the true stride was outside the window.

What works is full containment: take a small template out of tile B and slide
it entirely inside a larger window from tile A.

    window    tile A x-chunks 8-9  -> A local x 800..999   (200 px)
    template  tile B x-chunk 0, cropped to 50 x 50 x 110

Every position then compares the same voxel count, so the metric is unbiased
and the search range comes from the window size, not from leftover overlap.
All six peaks land 18-24 voxels clear of the window edge with ncc 0.63-0.83.

One FFT correlation supplies every position's numerator; the window's local
mean and variance come from 3-D prefix sums, so each position costs O(1). The
explicit shift-and-resum version took over 10 minutes and did not finish; this
runs in about two.

---------------------------------------------------------------------------
Why the transforms are needed
---------------------------------------------------------------------------
These arrays carry no spatial metadata -- zarr.json says only
``attributes: {units: pixels}`` -- so Neuroglancer would otherwise show 24
tiles stacked on top of each other in index space. Every source therefore gets
an explicit transform whose ``outputDimensions`` name the physical voxel size
(5 x 5 x 3.4 um) and whose matrix places the tile in the mosaic.

Because the output dimensions carry the scale, the translation column works in
*pixels*: 900 means 900 x 5 um. That keeps the tile offsets readable and
identical to the numbers in the YAML manifests.

The trap -- and it fails silently, so it is worth stating plainly -- is that
declaring `outputDimensions` alone is not enough. Neuroglancer resolves the
matrix in physical space:

    physical_out[i] = sum_j matrix[i][j] * physical_in[j]
                      + outputScale[i] * matrix[i][rank]

and `physical_in` comes from the *input* scales. These arrays have no units, so
the input scale defaults to 1 with an empty unit, which Neuroglancer reads as
one metre per voxel. Each tile then resolves to [0, 2e8) output units -- 1000 m
re-expressed in 5 um steps -- and the 900-unit translations are invisible next
to that, so all 24 tiles land on top of each other. Hence `inputDimensions`
spells out the voxel size, and the linear block is left dimensionless.

---------------------------------------------------------------------------
Axis order
---------------------------------------------------------------------------
zarr.json declares ``dimension_names: [z, x, y, channel]``, so the array's
axis 1 is the mosaic X and axis 2 is the mosaic Y. The script reads
dimension_names off the first tile rather than assuming it, and permutes into
Neuroglancer's x, y, z, c^ output order. The trailing size-2 axis is renamed
``c^`` -- the ``^`` suffix is what marks a dimension as a *channel* dimension,
which is what lets the shader address it as getDataValue(0) / getDataValue(1).

---------------------------------------------------------------------------
The tilt
---------------------------------------------------------------------------
Z drifts across the mosaic, and it runs BOTH ways -- negative across X,
positive across Y, comparable in size:

    x-steps   tile  1 ->  7:  dz = -11   (ncc +0.70)
              tile  7 -> 13:  dz =  -9   (ncc +0.82)
              tile 13 -> 19:  dz = -11   (ncc +0.83)   mean -10.33 per 893 px
    y-steps   tile  1 ->  2:  dz =  +7   (ncc +0.80)
              tile  2 ->  3:  dz =  +7   (ncc +0.63)
              tile  3 ->  4:  dz =  +7   (ncc +0.81)   mean  +7.00 per 931 px

A single-axis tilt cannot flatten both sets of seams, hence two knobs:

    --tilt-x N   signed Z drift, in z-pixels, across the full X extent
    --tilt-y N   signed Z drift, in z-pixels, across the full Y extent
    --no-tilt    zero both

Defaults scale the per-step figures to the full extents: -42.6 across X and
+42.5 across Y. The per-step values are consistent (-11/-9/-11 and +7/+7/+7),
which is what a uniform drift looks like -- an earlier pass showed scatter
(-5/-10/-11, +12/+7) but that was registration error, not the stage.

Three ways to apply it, and the choice is not cosmetic:

    --tilt-mode step     (default) per-tile Z offset, tiles stay flat
    --tilt-mode rotate   one true rotation of the whole mosaic
    --tilt-mode shear    one pure Z shear of the whole mosaic

Only `step` can close a seam. `rotate` and `shear` are a single affine applied
to the whole mosaic frame, so they preserve every tile-to-tile relationship by
construction: push a physical point in a seam through either neighbour's
matrix and the two agree to under 1e-9 output px, before and after the tilt.
They reorient the slab; they do not register it. `step` moves tiles relative
to each other and changes the same residual by the full 10.3 px.

That also settles what the drift physically is. The overlap shows the SAME
tissue at different z in two tiles, and a rigid rotation of the sample cannot
produce that -- it would leave the overlap self-consistent. Only the
acquisition changing z between tiles does. So the staircase is the right
model, which is why it is the default.

Whether each tile's interior is ALSO sheared is a separate question, and one
the data cannot answer: inside a tile there is no redundancy, and the overlap
agrees under both models. Locating the tissue surface absolutely in the corner
chunks of six tiles gave intra-tile slopes of -8.24 +- 11.37 (x) and
+9.35 +- 5.62 (y) z-px per 1000 px, against seam-implied values of -11.6 and
+7.5. The point estimates match, but slab topography (surface depth ranges
z=8 to z=90 between tiles) dominates the scatter, leaving it 0.7 sigma from
flat in x and 1.7 in y. More tiles will not help; 24 still leaves it under
2 sigma.

``--offsets FILE`` takes fully measured per-tile (x, y, z) from register-seams
and supersedes all of the above. ``--z-offsets FILE`` adds z-only nudges on
top of a modelled tilt.

Either way it is ONE transform of the whole mosaic frame about a single pivot
(``--tilt-pivot center`` by default, or ``origin``), applied as

    physical_out = R @ (perm @ physical_in + offset - pivot) + pivot

so the tile offset is tilted along with the voxel coordinates. That matters:
because it is a single rigid transform, the 10% overlap is preserved exactly --
a physical point in a seam maps to the same output coordinate through either
neighbour's matrix, verified to 1e-9 output px. The pivot only translates the
whole mosaic; it never changes tile-to-tile alignment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEFAULT_BUCKET = "apex-connects"
DEFAULT_PREFIX = "CMC/Derivatives/Vlad/PS-OCT/3DTiles/Orientation/004/"
DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"

# Registered from the seams by FFT template matching on the amplitude; see
# "The layout" and "The tilt" in the module docstring. The two axes genuinely
# differ, so the nominal 10% overlap is wrong in both directions.
MEASURED_STRIDE = (893.0, 931.0)

# Hue = phase of (channel 0 + i * channel 1); value = |z|, on a log-scaled max.
# The magnitude here is heavy-tailed (median ~0.08, max ~80), hence the log
# slider rather than a linear one.
ORIENTATION_SHADER = """\
#uicontrol float logMagMax slider(min=-3.0, max=3.0, default=0.0, step=0.05)
#uicontrol float gamma slider(min=0.1, max=3.0, default=0.6, step=0.05)
#uicontrol float phaseOffset slider(min=-180.0, max=180.0, default=0.0, step=1.0)
#uicontrol bool constantBrightness checkbox(default=false)

void main() {
  float re = float(toRaw(getDataValue(0)));
  float im = float(toRaw(getDataValue(1)));
  float mag = length(vec2(re, im));

  float hue = fract((mag > 0.0 ? atan(im, re) : 0.0) / 6.283185
                    + radians(phaseOffset) / 6.283185);
  vec3 p = abs(fract(hue + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);

  float v = clamp(mag / pow(10.0, logMagMax), 0.0, 1.0);
  v = pow(v, gamma);
  if (constantBrightness) v = 1.0;

  emitRGBA(vec4(rgb * v, v));
}
"""


# --------------------------------------------------------------------------
# tile discovery
# --------------------------------------------------------------------------
@dataclass
class Tile:
    """One tile: where its data lives and where it sits in the mosaic."""

    url: str
    name: str
    x: float  # mosaic offset, in pixels
    y: float
    z: float = 0.0  # only non-zero when --offsets supplies a measured z


def list_s3_tiles(bucket: str, prefix: str) -> list[str]:
    """Anonymous list of the .zarr directories under `prefix`.

    The bucket is public, so a plain HTTPS ListObjectsV2 works and the script
    stays free of an S3 client dependency.
    """
    names: list[str] = []
    token = None
    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": "/",
            "max-keys": "1000",
        }
        if token:
            params["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            root = ET.fromstring(resp.read())
        for cp in root.findall("s3:CommonPrefixes", S3_NS):
            entry = (cp.findtext("s3:Prefix", namespaces=S3_NS) or "")[len(prefix) :]
            if entry.rstrip("/").endswith(".zarr"):
                names.append(entry.rstrip("/"))
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
    return sorted(names, key=natural_key)


def natural_key(s: str):
    """Sort slice_004_tile_2 before slice_004_tile_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_zarr_json(bucket: str, prefix: str, name: str) -> dict | None:
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(prefix + name)}/zarr.json"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001 - metadata is a nicety, not required
        print(f"note: could not read {name}/zarr.json ({exc})", file=sys.stderr)
        return None


def grid_offsets(
    n: int, fast_axis: str, fast_count: int, stride_x: float, stride_y: float
):
    """Mosaic offsets for `n` tiles in numbering order.

    `fast_axis` is the mosaic axis the tile numbering advances along first and
    `fast_count` is how many tiles it takes before wrapping to the next line.

    For slice 004 that is y and 6: tiles 1..6 march along y, then tile 7 starts
    the next column of x. The slice-150 YAML manifest is the other convention
    (x fastest, 14 per row), so pass --fast-axis x --fast-count 14 for it.
    """
    out = []
    for i in range(n):
        a, b = i % fast_count, i // fast_count
        ix, iy = (b, a) if fast_axis == "y" else (a, b)
        out.append((ix * stride_x, iy * stride_y))
    return out


def load_manifest(path: str, bucket: str, prefix: str) -> list[Tile]:
    """Read tile offsets from a fuse-mat-tiles YAML manifest.

    Only the per-tile x/y are taken; the manifest's own resolution and overlap
    are reported but not applied, so the CLI stays the single source of truth
    for the geometry.
    """
    import yaml  # lazy: only manifests need it

    with open(path) as f:
        doc = yaml.safe_load(f)
    meta = doc.get("metadata", {}) or {}
    if "resolution_microns" in meta:
        print(f"note: manifest says resolution_microns="
              f"{meta['resolution_microns']}", file=sys.stderr)
    tiles = []
    for entry in doc["tiles"]:
        name = entry["filepath"].rsplit(".", 1)[0] + ".zarr"
        tiles.append(
            Tile(url=zarr_url(bucket, prefix, name), name=name,
                 x=float(entry["x"]), y=float(entry["y"]))
        )
    return tiles


def zarr_url(bucket: str, prefix: str, name: str) -> str:
    return f"s3://{bucket}/{prefix}{name}/|zarr3:"


def tile_number(name: str) -> int | None:
    """Pull the tile number out of slice_004_tile_007_orientation.zarr."""
    m = re.search(r"_tile_(\d+)", name)
    return int(m.group(1)) if m else None


def load_z_offsets(path: str) -> dict[int, float]:
    """Per-tile z nudges, in z-pixels: {"7": -2.5, "13": 1.0}."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def load_offsets(path: str) -> dict[int, tuple[float, float, float]]:
    """Per-tile absolute mosaic offsets in pixels: {"7": [867, 19, -11], ...}.

    As produced by register-seams. These supersede the generated grid
    entirely: the stage's moves are neither a constant length nor axis-pure
    (867/+19, 890/-1, 896/+2 for the three x steps), so a stride cannot
    express them.
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if len(v) != 3:
            raise SystemExit(f"offset for tile {k} must be [x, y, z], got {v!r}")
        out[int(k)] = (float(v[0]), float(v[1]), float(v[2]))
    return out


def metres(microns: float) -> float:
    """Microns to metres, without the 5.0*1e-6 -> 4.9999999999999996e-06 litter."""
    return float(f"{microns * 1e-6:.12g}")


def spatial_dims(res_um) -> dict:
    return {
        "x": [metres(res_um[0]), "m"],
        "y": [metres(res_um[1]), "m"],
        "z": [metres(res_um[2]), "m"],
    }


# --------------------------------------------------------------------------
# the transform
# --------------------------------------------------------------------------
def tilt_matrix(kx: float, ky: float, mode: str) -> np.ndarray:
    """The 3x3 tilt in *physical* space, in x, y, z order.

    `kx` and `ky` are the slopes dZ/dX and dZ/dY, both dimensionless. The
    matrix is dimensionless on purpose: Neuroglancer applies it to physical
    coordinates, not to voxel indices (see the note in source_transform), so a
    plain rotation is what belongs here. No conjugation by the voxel size --
    Neuroglancer's own input/output scale ratio does that conversion.

    The two axes are independent and both are needed: the measured drift runs
    one way across X and the other way across Y, so a single-axis tilt cannot
    flatten both sets of seams.
    """
    if kx == 0.0 and ky == 0.0:
        return np.eye(3)

    if mode == "shear":
        m = np.eye(3)
        m[2, 0] = kx  # Z += kx * X
        m[2, 1] = ky  # Z += ky * Y
        return m

    # rotation about Y (tilts the XZ plane), then about X (tilts YZ).
    # At these angles the two commute to well under a voxel.
    cx, cy = np.cos(np.arcsin(kx)), np.cos(np.arcsin(ky))
    r_about_y = np.array([[cx, 0.0, -kx], [0.0, 1.0, 0.0], [kx, 0.0, cx]])
    r_about_x = np.array([[1.0, 0.0, 0.0], [0.0, cy, -ky], [0.0, ky, cy]])
    return r_about_x @ r_about_y


def source_transform(
    tile: Tile,
    dim_names: list[str],
    tilt: np.ndarray,
    pivot: np.ndarray,
    res_um: tuple[float, float, float],
    channel_dim: str,
    z_step_px: float = 0.0,
) -> dict:
    """The per-source `transform` JSON: source index space -> mosaic pixels.

    Composition, right to left: permute the source axes into x, y, z; add the
    tile's mosaic offset; then tilt about `pivot`.

    The one thing that is easy to get wrong here: Neuroglancer resolves the
    matrix in PHYSICAL space, not index space --

        physical_out[i] = sum_j matrix[i][j] * physical_in[j]
                          + outputScale[i] * matrix[i][rank]

    so the linear block is dimensionless and the translation column is in
    output units (pixels, given the output scales below). physical_in comes
    from the *input* scales, which is why `inputDimensions` has to be spelled
    out: these arrays carry no units, so their default scale is 1 with no
    unit, which Neuroglancer reads as 1 metre per voxel. Leave it out and each
    tile resolves to [0, 2e8) output units -- 1000 m expressed in 5 um steps --
    and all 24 tiles pile up at the origin, 900-unit translations being
    invisible at that size.
    """
    rank = len(dim_names)
    spatial = [d for d in dim_names if d != channel_dim]
    if sorted(spatial) != ["x", "y", "z"]:
        raise SystemExit(
            f"expected spatial dimension names x, y, z; got {spatial!r}. "
            "Pass --dim-names to override what zarr.json declares."
        )

    # axis permutation: row i of `perm` picks the source axis feeding output i
    perm = np.zeros((3, rank))
    for i, out_name in enumerate("xyz"):
        perm[i, dim_names.index(out_name)] = 1.0

    # everything below is in x, y, z order; `res` converts px <-> um
    res = np.array(res_um, dtype=float)
    offset_um = np.array([tile.x * res[0], tile.y * res[1], tile.z * res[2]])
    pivot_um = pivot * res

    linear = tilt @ perm  # dimensionless: physical in -> physical out
    # physical_out = tilt @ (perm @ physical_in + offset - pivot) + pivot,
    # so the tile offset is tilted along with everything else.
    translation = (tilt @ offset_um + pivot_um - tilt @ pivot_um) / res
    # step mode adds a flat per-tile z offset instead of shearing the interior
    translation[2] += z_step_px

    # rank rows of rank+1 columns; the channel row is an identity passthrough
    matrix = [[0.0] * (rank + 1) for _ in range(rank)]
    for i in range(3):
        for j in range(rank):
            matrix[i][j] = round(float(linear[i, j]), 12)
        matrix[i][rank] = round(float(translation[i]), 9)
    matrix[3][dim_names.index(channel_dim)] = 1.0

    # inputDimensions keys must be the source's own dimension names, in source
    # order, so Neuroglancer pairs them with the right axes.
    by_name = {"x": res_um[0], "y": res_um[1], "z": res_um[2]}
    input_dims = {
        name: ([1, ""] if name == channel_dim else [metres(by_name[name]), "m"])
        for name in dim_names
    }

    return {
        "sourceRank": rank,
        "inputDimensions": input_dims,
        "outputDimensions": {**spatial_dims(res_um), "c^": [1, ""]},
        "matrix": matrix,
    }


# --------------------------------------------------------------------------
# state assembly
# --------------------------------------------------------------------------
def build_state(args, tiles: list[Tile], dim_names: list[str], extent_px) -> dict:
    res_um = tuple(args.resolution)
    ex, ey, ez = extent_px

    # slopes dZ/dX and dZ/dY, from the requested z drift across each extent
    def slope(drift_px: float, span_px: float, span_res_um: float) -> float:
        if drift_px == 0.0 or span_px == 0.0:
            return 0.0
        return (drift_px * res_um[2]) / (span_px * span_res_um)

    if args.no_tilt:
        kx = ky = 0.0
    else:
        kx = slope(args.tilt_x, ex, res_um[0])
        ky = slope(args.tilt_y, ey, res_um[1])

    pivot = (
        np.array([ex / 2.0, ey / 2.0, ez / 2.0])
        if args.tilt_pivot == "center"
        else np.zeros(3)
    )
    # "step" keeps every tile's interior flat and puts the whole drift into a
    # per-tile z translation. On a regular grid it places tiles IDENTICALLY to
    # "rotate"/"shear" -- the models differ only in the intra-tile shear, which
    # the data cannot resolve (see "The tilt" above). Use it when you believe
    # the drift is stage stepping rather than a tilted sample.
    stepping = args.tilt_mode == "step"
    tilt = np.eye(3) if stepping else tilt_matrix(kx, ky, args.tilt_mode)

    def z_step(t: Tile) -> float:
        if not stepping:
            return 0.0
        # z offset in z-pixels, centred on the pivot so the mosaic stays put
        dx = (t.x - pivot[0]) * res_um[0]
        dy = (t.y - pivot[1]) * res_um[1]
        return (kx * dx + ky * dy) / res_um[2]

    znudge = load_z_offsets(args.z_offsets) if args.z_offsets else {}

    sources = [
        {
            "url": t.url,
            "transform": source_transform(
                t, dim_names, tilt, pivot, res_um, args.channel_dim,
                z_step_px=z_step(t) + znudge.get(tile_number(t.name), 0.0),
            ),
        }
        for t in tiles
    ]

    state = {
        "dimensions": spatial_dims(res_um),
        "position": [ex / 2.0, ey / 2.0, ez / 2.0],
        "crossSectionScale": max(ex, ey) / 1200.0,
        "projectionScale": max(ex, ey) * 1.5,
        "layers": [
            {
                "type": "image",
                "name": args.layer_name,
                "source": sources,
                "shader": ORIENTATION_SHADER,
                "opacity": 1.0,
                "blend": "default",
            }
        ],
        "selectedLayer": {"visible": True, "layer": args.layer_name},
        "layout": "4panel",
    }
    return state, (kx, ky)


def make_url(viewer: str, state: dict) -> str:
    payload = json.dumps(state, separators=(",", ":"))
    return viewer.rstrip("/") + "/#!" + urllib.parse.quote(payload, safe="")


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("tile source")
    src.add_argument("--bucket", default=DEFAULT_BUCKET)
    src.add_argument("--prefix", default=DEFAULT_PREFIX,
                     help="S3 prefix holding the per-tile .zarr directories")
    src.add_argument("--manifest",
                     help="YAML manifest to take tile offsets from, instead of "
                          "generating a grid")
    src.add_argument("--no-list", action="store_true",
                     help="don't contact S3; synthesise tile names from "
                          "--slice/--tiles instead")
    src.add_argument("--slice", type=int, default=4)
    src.add_argument("--tiles", type=int, default=24,
                     help="tile count, used only with --no-list")
    src.add_argument("--name-template",
                     default="slice_{slice:03d}_tile_{tile:03d}_orientation.zarr")
    src.add_argument("--limit", type=int,
                     help="use only the first N tiles (handy while tuning)")

    lay = p.add_argument_group("mosaic layout")
    lay.add_argument("--fast-axis", choices=("x", "y"), default="y",
                     help="mosaic axis the tile numbering advances along first "
                          "(slice 004: y, measured by overlap correlation)")
    lay.add_argument("--fast-count", type=int, default=6,
                     help="tiles before the numbering wraps to the next line "
                          "(slice 004: 6, so 24 tiles = 6 in y x 4 in x)")
    lay.add_argument("--tile-size", type=float, nargs=2, default=[1000.0, 1000.0],
                     metavar=("W", "H"))
    lay.add_argument("--overlap", type=float,
                     help="derive the stride from a fractional tile overlap "
                          "(0.1 -> 900 px). Ignored if --stride is given; if "
                          "neither is given the measured strides are used")
    lay.add_argument("--stride", type=float, nargs=2, metavar=("SX", "SY"),
                     default=None,
                     help=f"stride in pixels (default {MEASURED_STRIDE[0]:g} "
                          f"{MEASURED_STRIDE[1]:g}, registered from the seams "
                          "-- note the two axes differ, so the overlap is not "
                          "a uniform 10%%)")
    lay.add_argument("--depth", type=float, default=191.0)
    lay.add_argument("--resolution", type=float, nargs=3, default=[5.0, 5.0, 3.4],
                     metavar=("X", "Y", "Z"), help="voxel size in microns")

    tl = p.add_argument_group("tilt correction")
    tl.add_argument("--tilt-x", type=float, default=-42.6,
                    help="signed change in Z placement, in z-pixels, across "
                         "the full X extent (default -42.6: -10.33 z-px per "
                         "893 px x-step, registered from the seams). Negative "
                         "tilts Z down as X increases")
    tl.add_argument("--tilt-y", type=float, default=42.5,
                    help="same across the full Y extent (default +42.5: "
                         "+7.00 z-px per 931 px y-step). The "
                         "measured seam offset IS this number -- it is the "
                         "relative translation neighbours need, not its "
                         "negative")
    tl.add_argument("--no-tilt", action="store_true",
                    help="zero both tilts, to see the uncorrected seams")
    tl.add_argument("--tilt-mode", choices=("rotate", "shear", "step"),
                    default="step",
                    help="step (default): per-tile Z offset, flat tiles -- the "
                         "only mode that actually moves tiles relative to each "
                         "other. rotate/shear: one global affine over the whole "
                         "mosaic, which reorients the slab but provably cannot "
                         "close a seam (residual unchanged to 1e-9 px)")
    tl.add_argument("--tilt-pivot", choices=("center", "origin"), default="center")
    tl.add_argument("--offsets", metavar="FILE",
                    help="per-tile absolute mosaic offsets in pixels from "
                         "register-seams, {\"7\": [867, 19, -11], ...}. "
                         "Supersedes the generated grid and implies --no-tilt")
    tl.add_argument("--z-offsets", metavar="FILE",
                    help="JSON of per-tile Z nudges in z-pixels, keyed by tile "
                         'number (e.g. {"7": -2.5, "13": 1.0}), added on top '
                         "of the tilt. For per-move stage error the global "
                         "tilt cannot capture")

    out = p.add_argument_group("output")
    out.add_argument("--viewer", default=DEFAULT_VIEWER)
    out.add_argument("--layer-name", default="orientation")
    out.add_argument("--channel-dim", default="channel",
                     help="source dimension holding the 2 components")
    out.add_argument("--dim-names",
                     help="comma-separated source dimension names, overriding "
                          "zarr.json (e.g. 'z,x,y,channel')")
    out.add_argument("--json", dest="json_out",
                     help="also write the viewer state to this file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.stride:
        stride_x, stride_y = args.stride
    elif args.overlap is not None:
        stride_x = args.tile_size[0] * (1 - args.overlap)
        stride_y = args.tile_size[1] * (1 - args.overlap)
    else:
        stride_x, stride_y = MEASURED_STRIDE

    if args.manifest:
        tiles = load_manifest(args.manifest, args.bucket, args.prefix)
    else:
        if args.no_list:
            names = [
                args.name_template.format(slice=args.slice, tile=i + 1)
                for i in range(args.tiles)
            ]
        else:
            names = list_s3_tiles(args.bucket, args.prefix)
            if not names:
                raise SystemExit(f"no .zarr tiles found under {args.prefix}")
        offsets = grid_offsets(
            len(names), args.fast_axis, args.fast_count, stride_x, stride_y
        )
        tiles = [
            Tile(url=zarr_url(args.bucket, args.prefix, n), name=n, x=ox, y=oy)
            for n, (ox, oy) in zip(names, offsets)
        ]

    # measured per-tile offsets supersede the generated grid entirely
    if args.offsets:
        table = load_offsets(args.offsets)
        placed, missing = [], []
        for t in tiles:
            n = tile_number(t.name)
            if n in table:
                ox, oy, oz = table[n]
                placed.append(Tile(url=t.url, name=t.name, x=ox, y=oy, z=oz))
            else:
                missing.append(t.name)
        if missing:
            print(f"warning: no measured offset for {len(missing)} tile(s), "
                  f"dropped: {', '.join(missing[:4])}"
                  f"{' ...' if len(missing) > 4 else ''}", file=sys.stderr)
        if not placed:
            raise SystemExit(f"no tile in {args.offsets} matched the tile names")
        tiles = placed
        if not args.no_tilt:
            print("note: --offsets already carries the per-tile z, so the "
                  "global tilt is switched off", file=sys.stderr)
            args.no_tilt = True

    if args.limit:
        tiles = tiles[: args.limit]

    # dimension names: from the first tile's zarr.json unless overridden
    if args.dim_names:
        dim_names = args.dim_names.split(",")
        shape = None
    else:
        zj = None if args.no_list else read_zarr_json(args.bucket, args.prefix, tiles[0].name)
        if zj is None:
            dim_names = ["z", "x", "y", "channel"]
            shape = None
            print("note: assuming dimension_names = z,x,y,channel", file=sys.stderr)
        else:
            dim_names = list(zj["dimension_names"])
            shape = list(zj["shape"])
            if args.channel_dim in dim_names:
                n_ch = shape[dim_names.index(args.channel_dim)]
                if n_ch != 2:
                    print(f"warning: channel dimension has size {n_ch}, not 2; "
                          "the phase shader expects 2", file=sys.stderr)

    if args.channel_dim not in dim_names:
        raise SystemExit(
            f"channel dimension {args.channel_dim!r} not in {dim_names!r}; "
            "pass --channel-dim"
        )

    # mosaic extent in pixels, for the pivot and the initial view
    tw, th = args.tile_size
    if shape:
        tw = shape[dim_names.index("x")]
        th = shape[dim_names.index("y")]
        depth = shape[dim_names.index("z")]
    else:
        depth = args.depth
    extent = (
        max(t.x for t in tiles) + tw,
        max(t.y for t in tiles) + th,
        depth,
    )

    state, slopes = build_state(args, tiles, dim_names, extent)
    url = make_url(args.viewer, state)

    print(f"tiles       : {len(tiles)}", file=sys.stderr)
    print(f"dim_names   : {dim_names}", file=sys.stderr)
    print(f"grid        : {args.fast_count} along {args.fast_axis} first, "
          f"stride {stride_x:g} x {stride_y:g} px", file=sys.stderr)
    print(f"extent      : {extent[0]:g} x {extent[1]:g} x {extent[2]:g} px "
          f"= {extent[0]*args.resolution[0]/1000:.2f} x "
          f"{extent[1]*args.resolution[1]/1000:.2f} x "
          f"{extent[2]*args.resolution[2]/1000:.2f} mm", file=sys.stderr)
    kx, ky = slopes
    if kx == 0.0 and ky == 0.0:
        print("tilt        : none", file=sys.stderr)
    else:
        rx, ry, rz = args.resolution
        dx = kx * extent[0] * rx / rz
        dy = ky * extent[1] * ry / rz
        print(f"tilt        : {args.tilt_mode}, pivot {args.tilt_pivot}",
              file=sys.stderr)
        print(f"              X: {np.degrees(np.arcsin(kx)):+.5f} deg "
              f"= {dx:+.2f} z-px across {extent[0]:g} x-px", file=sys.stderr)
        print(f"              Y: {np.degrees(np.arcsin(ky)):+.5f} deg "
              f"= {dy:+.2f} z-px across {extent[1]:g} y-px", file=sys.stderr)
    print(f"url length  : {len(url)} chars", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(state, f, indent=2)
        print(f"state       : {args.json_out}", file=sys.stderr)

    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
