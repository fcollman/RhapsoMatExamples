"""
Register every tile seam of a slice and solve for per-tile (x, y, z) offsets.

    register-seams --out offsets.json              # the full grid
    register-seams --out offsets.json --x-only     # just the x seams
    register-seams --pairs 1-7,7-13 --out o.json   # named seams only

Feed the result to ng-tile-link:

    ng-tile-link --offsets offsets.json

---------------------------------------------------------------------------
Why per-tile offsets rather than a stride plus a tilt
---------------------------------------------------------------------------
Registering all 38 seams of slice 004 shows the stage's move is repeatable but
mis-calibrated, with a distinct signature per move index -- these ranges are
across all six rows / four columns:

    x move 1 (col 0->1):  dx 862..866   dy +19..+19   dz -12..-11
    x move 2 (col 1->2):  dx 890..891   dy  -1.. -1   dz  -9.. -9
    x move 3 (col 2->3):  dx 895..896   dy  +2.. +2   dz -11..-11
    y moves (all 20):     dy 930..932   dx  +5.. +6   dz  +7.. +8

So every move repeats to ~4 px in x and ~1 px in y and z, but different moves
differ, and the x moves carry a real y component (+19 px on the first one).
That is not jitter and not a single stride -- it is a deterministic per-move
error, and only per-tile offsets can express it.

A global rotation cannot help either: a rotation is one affine over the whole
mosaic, so it preserves every tile-to-tile relationship exactly and cannot
close a seam (verified -- applying it moves the seam residual by under
1e-9 px, while a per-tile translation moves it by the full 10 px). Only a
per-tile translation moves tiles relative to each other.

The global solve over all 38 seams closes to an rms residual of 0.24 px across
14 independent loops, so the measurements are mutually consistent.

---------------------------------------------------------------------------
Method
---------------------------------------------------------------------------
Metric: normalised cross-correlation of log1p(|z|), the complex amplitude.
Phase is near-random voxel to voxel and does not correlate.

Geometry: full containment. A small template cropped out of tile B slides
entirely inside a larger window from tile A, so every candidate shift compares
the same voxel count and the correlation is unbiased. Anything that lets the
compared region shrink with the shift will happily "win" on a sliver.

Flyback: x is the fast scan axis and the last FLYBACK columns of each line
carry ringing, so they are dropped from the window. For the seams measured so
far the optimum keeps the template clear of them anyway, but the crop costs
nothing and protects longer strides.

Speed: one FFT correlation gives every position's numerator; the window's
local mean and variance come from 3-D prefix sums, so each position is O(1).

Solve: each seam gives o_b - o_a = m. Stack them and least-squares solve for
all tiles at once with tile 1 pinned at the origin, so inconsistent loops are
distributed rather than accumulated along one path. Per-seam residuals are
reported -- large ones mean that seam's match should not be trusted.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

DEFAULT_BUCKET = "apex-connects"
DEFAULT_PREFIX = "CMC/Derivatives/Vlad/PS-OCT/3DTiles/Orientation/004/"
NAME_TEMPLATE = "slice_{slice:03d}_tile_{tile:03d}_orientation.zarr"

NZ = 191
CHUNK = 100
TILE = 1000
FLYBACK = 25
WIN_BASE = 800          # window starts here in tile A's local coords
ZCROP = (40, 150)       # template z range -> dz search +-40
OFFCROP = (25, 75)      # template range on the non-stepping in-plane axis
STEPCROP = (0, 50)      # template range on the stepping axis
MID = 5                 # in-plane chunk index sampled along the seam


def make_reader(bucket, prefix, slice_no):
    import boto3
    import zstandard as zstd
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    dctx = zstd.ZstdDecompressor()
    cache: dict = {}

    def chunk(tile, xc, yc):
        """One chunk as log amplitude.

        The bucket is inconsistent about chunk keys. zarr.json declares rank 4
        with a 1x10x10x1 grid, so the key should be c/0/x/y/0, and 23 of the
        24 tiles carry BOTH that and a 3-level c/0/x/y (the data is stored
        twice). Tile 20 has only the 3-level form, which means Neuroglancer
        itself cannot read it -- but registration still can, via the fallback.
        """
        k = (tile, xc, yc)
        if k not in cache:
            name = NAME_TEMPLATE.format(slice=slice_no, tile=tile)
            raw = None
            for suffix in (f"/c/0/{xc}/{yc}/0", f"/c/0/{xc}/{yc}"):
                try:
                    raw = s3.get_object(Bucket=bucket,
                                        Key=f"{prefix}{name}{suffix}")["Body"].read()
                    break
                except s3.exceptions.NoSuchKey:
                    continue
            if raw is None:
                raise FileNotFoundError(f"{name} chunk {xc},{yc}")
            a = np.frombuffer(dctx.stream_reader(raw).read(),
                              dtype="<f4").reshape(NZ, CHUNK, CHUNK, 2)
            cache[k] = np.log1p(np.hypot(a[..., 0], a[..., 1])).astype(np.float64)
        return cache[k]

    return chunk, cache


# --------------------------------------------------------------------------
# FFT template matching
# --------------------------------------------------------------------------
def prefix_sums(v):
    p = np.zeros([s + 1 for s in v.shape])
    p[1:, 1:, 1:] = v.cumsum(0).cumsum(1).cumsum(2)
    return p


def window_boxes(p, shape):
    """Sums over every template-sized box position, vectorised."""
    dz, dx, dy = shape
    return (p[dz:, dx:, dy:] - p[:-dz, dx:, dy:] - p[dz:, :-dx, dy:]
            - p[dz:, dx:, :-dy] + p[:-dz, :-dx, dy:] + p[:-dz, dx:, :-dy]
            + p[dz:, :-dx, :-dy] - p[:-dz, :-dx, :-dy])


def match(window, template):
    """NCC of template at every fully contained position in window."""
    tz, tx, ty = template.shape
    n = template.size
    tm = template.mean()
    tv = ((template - tm) ** 2).sum()

    shape = [a + b - 1 for a, b in zip(window.shape, template.shape)]
    axes = (0, 1, 2)
    fw = np.fft.rfftn(window, shape, axes=axes)
    ft = np.fft.rfftn(template[::-1, ::-1, ::-1], shape, axes=axes)
    full = np.fft.irfftn(fw * ft, shape, axes=axes)
    num = full[tz - 1: window.shape[0], tx - 1: window.shape[1],
               ty - 1: window.shape[2]]

    s1 = window_boxes(prefix_sums(window), template.shape)
    s2 = window_boxes(prefix_sums(window * window), template.shape)
    wv = s2 - s1 * s1 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = (num - s1 * tm) / np.sqrt(wv * tv)
    ncc[~np.isfinite(ncc)] = -1.0
    return ncc


def register(chunk, a_tile, b_tile, axis, flyback):
    """Measure tile B's offset relative to tile A. Returns (dx, dy, dz, ncc)."""
    hi = TILE - flyback if axis == "x" else TILE
    if axis == "x":
        win = np.concatenate([chunk(a_tile, 8, MID), chunk(a_tile, 9, MID)], axis=1)
        win = win[:, : hi - WIN_BASE, :]
        tem = chunk(b_tile, 0, MID)[ZCROP[0]:ZCROP[1],
                                    STEPCROP[0]:STEPCROP[1],
                                    OFFCROP[0]:OFFCROP[1]]
    else:
        win = np.concatenate([chunk(a_tile, MID, 8), chunk(a_tile, MID, 9)], axis=2)
        tem = chunk(b_tile, MID, 0)[ZCROP[0]:ZCROP[1],
                                    OFFCROP[0]:OFFCROP[1],
                                    STEPCROP[0]:STEPCROP[1]]

    ncc = match(win, tem)
    z0, x0, y0 = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    peak = float(ncc[z0, x0, y0])
    dz = z0 - ZCROP[0]
    if axis == "x":
        step, cross = WIN_BASE + x0 - STEPCROP[0], y0 - OFFCROP[0]
        dx, dy = step, cross
    else:
        step, cross = WIN_BASE + y0 - STEPCROP[0], x0 - OFFCROP[0]
        dx, dy = cross, step

    # flag peaks sitting on a search bound -- the true shift is outside
    edge = min(z0, x0, y0, *(s - 1 - i for s, i in zip(ncc.shape, (z0, x0, y0))))
    return dx, dy, dz, peak, edge


# --------------------------------------------------------------------------
def tile_at(col, row, fast_count):
    return col * fast_count + row + 1


def build_seams(cols, fast_count, x_only, y_only):
    seams = []
    if not y_only:
        for row in range(fast_count):
            for col in range(cols - 1):
                seams.append((tile_at(col, row, fast_count),
                              tile_at(col + 1, row, fast_count), "x"))
    if not x_only:
        for col in range(cols):
            for row in range(fast_count - 1):
                seams.append((tile_at(col, row, fast_count),
                              tile_at(col, row + 1, fast_count), "y"))
    return seams


def solve(measurements, tiles):
    """Least-squares per-tile offsets from relative measurements."""
    index = {t: i for i, t in enumerate(sorted(tiles))}
    n = len(index)
    rows, rhs = [], []
    for (a, b, _axis), (dx, dy, dz, _p, _e) in measurements.items():
        r = np.zeros(n)
        r[index[b]] = 1.0
        r[index[a]] = -1.0
        rows.append(r)
        rhs.append([dx, dy, dz])
    # gauge: pin the lowest-numbered tile at the origin
    g = np.zeros(n)
    g[0] = 1.0
    rows.append(g)
    rhs.append([0.0, 0.0, 0.0])

    A = np.array(rows)
    B = np.array(rhs, dtype=float)
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    resid = A @ sol - B
    return {t: sol[i] for t, i in index.items()}, resid[:-1]


def parse_pairs(spec):
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--slice", type=int, default=4)
    p.add_argument("--cols", type=int, default=4, help="tile columns (x)")
    p.add_argument("--fast-count", type=int, default=6,
                   help="tiles per column, i.e. along y")
    p.add_argument("--x-only", action="store_true")
    p.add_argument("--y-only", action="store_true")
    p.add_argument("--pairs", help="explicit seams, e.g. 1-7,7-13 (axis "
                                   "inferred from the grid)")
    p.add_argument("--flyback", type=int, default=FLYBACK,
                   help=f"trailing x columns to drop (default {FLYBACK})")
    p.add_argument("--out", required=True, help="where to write the offsets")
    args = p.parse_args(argv)

    chunk, cache = make_reader(args.bucket, args.prefix, args.slice)

    all_seams = build_seams(args.cols, args.fast_count, args.x_only, args.y_only)
    if args.pairs:
        want = set(parse_pairs(args.pairs))
        all_seams = [s for s in all_seams if (s[0], s[1]) in want]
        if not all_seams:
            raise SystemExit("none of --pairs match the grid")

    print(f"{len(all_seams)} seams, {len(all_seams) * 3} chunk reads "
          f"(~{len(all_seams) * 3 * 14} MB)\n", file=sys.stderr)

    measurements = {}
    skipped = []
    for i, (a, b, axis) in enumerate(all_seams, 1):
        try:
            dx, dy, dz, peak, edge = register(chunk, a, b, axis, args.flyback)
        except FileNotFoundError as e:
            skipped.append((a, b, axis, str(e)))
            print(f"[{i:>3}/{len(all_seams)}] {axis} seam {a:>3} -> {b:>3}: "
                  f"SKIPPED, missing {e}", file=sys.stderr, flush=True)
            cache.clear()
            continue
        measurements[(a, b, axis)] = (dx, dy, dz, peak, edge)
        warn = "  <-- peak on search bound" if edge == 0 else ""
        print(f"[{i:>3}/{len(all_seams)}] {axis} seam {a:>3} -> {b:>3}: "
              f"dx {dx:>5} dy {dy:>+4} dz {dz:>+4}  ncc {peak:+.3f}{warn}",
              file=sys.stderr, flush=True)
        cache.clear()   # tiles are 1.4 GB each; do not hoard chunks

    if skipped:
        print(f"\n{len(skipped)} seam(s) skipped for missing chunks",
              file=sys.stderr)
    tiles = {t for a, b, _ in measurements for t in (a, b)}
    offsets, resid = solve(measurements, tiles)

    print("\nper-seam residual after the global solve (px):", file=sys.stderr)
    for (a, b, axis), r in zip(measurements, resid):
        mag = float(np.abs(r).max())
        flag = "  <-- inconsistent" if mag > 5 else ""
        print(f"   {axis} {a:>3} -> {b:>3}: "
              f"({r[0]:+6.2f}, {r[1]:+6.2f}, {r[2]:+6.2f}){flag}",
              file=sys.stderr)
    print(f"\nrms residual: {float(np.sqrt((resid ** 2).mean())):.2f} px",
          file=sys.stderr)

    out = {
        str(t): [round(float(v[0]), 3), round(float(v[1]), 3),
                 round(float(v[2]), 3)]
        for t, v in sorted(offsets.items())
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(out)} tile offsets to {args.out}", file=sys.stderr)
    print(f"  ng-tile-link --offsets {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
