"""
Read a Neuroglancer state and report the stage-step metrics of a tile mosaic.

    ng-state-metrics state.json
    ng-state-metrics state.json --layer orientation
    ng-tile-link --offsets offsets.json | ng-state-metrics -   # a URL on stdin

Reports, per in-plane axis, the mean step in x, y and z between neighbouring
tiles -- so the average Z step per X step and per Y step, which is the number
you want when deciding whether a mosaic has a z drift and how big it is.

---------------------------------------------------------------------------
Where the numbers come from
---------------------------------------------------------------------------
Parsing is done by the `neuroglancer` package: `url_state.parse_url` for a
link, `viewer_state.ViewerState` for a JSON dump. That hands back typed
objects -- ManagedLayer, LayerDataSource, CoordinateSpaceTransform,
CoordinateSpace -- with the matrix already a numpy (rank, rank+1) array and
dimension scales already normalised to SI, so none of that has to be
re-derived here.

Each source's tile origin is the translation column of its transform, in
output pixels. The grid is then inferred from those origins rather than
assumed: the x offsets cluster into columns and the y offsets into rows, with
the cluster gap taken from the data, so no --cols or stride has to be
supplied. Neighbours are adjacent (col, row) cells.

Steps are reported per move index as well as pooled, because pooling can hide
a systematic difference between moves. On slice 004 the three x moves are
-11.67, -9.00 and -11.00 z-px, so the pooled -10.56 +- 1.19 is an average over
three different moves rather than noise around one value.

Angles account for anisotropic voxels: a z step of N pixels at 3.4 um against
an x step at 5 um is atan(N * 3.4 / (dx * 5)), not atan(N / dx).

Intra-tile shear is read off the same matrices and reported separately -- it is
the off-diagonal term in the z output row, which a global rotation or shear
puts there and a per-tile translation does not. A global affine cannot change
tile-to-tile registration at all, so seeing it here means the tilt is
reorienting the slab rather than closing any seam.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np
from neuroglancer import url_state, viewer_state

AXES = ("x", "y", "z")

# Neuroglancer normalises dimension scales to the unit it reports; convert
# whatever it reports into metres so the angle maths is in real units.
UNIT_TO_M = {
    "m": 1.0, "dm": 1e-1, "cm": 1e-2, "mm": 1e-3,
    "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12,
}


def load_state(path: str) -> viewer_state.ViewerState:
    """A JSON dump, a Neuroglancer URL, or '-' for stdin."""
    raw = (sys.stdin.read() if path == "-" else open(path).read()).strip()
    if "#!" in raw or raw.startswith("http"):
        return url_state.parse_url(raw)
    return viewer_state.ViewerState(json.loads(raw))


def pick_layer(state: viewer_state.ViewerState, name: str | None):
    if name:
        for managed in state.layers:
            if managed.name == name:
                return managed
        raise SystemExit(f"no layer named {name!r}; have "
                         f"{[m.name for m in state.layers]}")
    multi = [m for m in state.layers
             if m.layer.source is not None and len(m.layer.source) > 1]
    if not multi:
        raise SystemExit("no layer with multiple sources found")
    if len(multi) > 1:
        print(f"note: using layer {multi[0].name!r} of "
              f"{[m.name for m in multi]}", file=sys.stderr)
    return multi[0]


def metres(space, axis: str) -> float:
    """Scale of one axis of a CoordinateSpace, in metres."""
    i = space.names.index(axis)
    unit = space.units[i]
    if unit not in UNIT_TO_M:
        raise SystemExit(f"dimension {axis!r} has unit {unit!r}, expected a "
                         "length unit")
    return float(space.scales[i]) * UNIT_TO_M[unit]


def tile_label(url: str, fallback: int) -> str:
    m = re.search(r"_tile_(\d+)", url)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"([^/]+)\.zarr", url)
    return m.group(1) if m else f"#{fallback}"


def read_source(source, index: int) -> dict:
    """Tile origin in output px, voxel size in um, and any intra-tile shear."""
    label = tile_label(source.url or "", index)
    t = source.transform
    if t is None:
        return {"label": label, "origin": np.zeros(3), "res_um": None,
                "shear": None}

    out = t.output_dimensions
    if not all(a in out.names for a in AXES):
        raise SystemExit(f"expected x, y, z output dimensions, got {out.names}")
    rank = out.rank
    row = {a: out.names.index(a) for a in AXES}

    matrix = np.asarray(t.matrix, dtype=float) if t.matrix is not None else None
    if matrix is None:
        matrix = np.hstack([np.eye(rank), np.zeros((rank, 1))])

    origin = np.array([matrix[row[a], rank] for a in AXES])
    res_um = np.array([metres(out, a) * 1e6 for a in AXES])

    # intra-tile shear: how the z output depends on the in-plane INPUT dims.
    # The matrix is dimensionless (it acts on physical coordinates), so convert
    # to z-px per in-plane-px with the input/output scale ratio.
    shear = {}
    src_space = t.input_dimensions if t.input_dimensions is not None else out
    z_scale = metres(out, "z")
    for a in ("x", "y"):
        if a in src_space.names:
            coef = matrix[row["z"], src_space.names.index(a)]
            shear[a] = coef * metres(src_space, a) / z_scale
    return {"label": label, "origin": origin, "res_um": res_um, "shear": shear}


def cluster(values: np.ndarray, tol: float | None):
    """Group 1-D positions into lattice indices, splitting on large gaps."""
    order = np.argsort(values)
    gaps = np.diff(values[order])
    if tol is None:
        # within-cell jitter is small and between-cell gaps are ~a tile wide,
        # so a quarter of the largest gap separates them cleanly
        tol = (gaps.max() / 4.0) if len(gaps) and gaps.max() > 0 else 1.0
    idx = np.zeros(len(values), dtype=int)
    cur = 0
    for k in range(1, len(order)):
        if gaps[k - 1] > tol:
            cur += 1
        idx[order[k]] = cur
    return idx, tol


def describe(label: str, deltas: np.ndarray, res_um, axis: int) -> None:
    n = len(deltas)
    sd = lambda c: c.std(ddof=1) if n > 1 else 0.0
    print(f"\n{label} steps  (n={n})")
    for i, a in enumerate(AXES):
        c = deltas[:, i]
        print(f"   d{a}: mean {c.mean():+9.3f}   sd {sd(c):7.3f}"
              f"   range {c.min():+9.2f} .. {c.max():+9.2f}")

    dz = deltas[:, 2].mean()
    step = deltas[:, axis].mean()
    if res_um is not None:
        print(f"   -> average dz per {label} step: {dz:+.3f} z-px "
              f"= {dz * res_um[2]:+.2f} um")
        if step:
            slope = dz / step
            ang = np.degrees(np.arctan(dz * res_um[2] / (step * res_um[axis])))
            print(f"   -> dz/d{AXES[axis]} = {slope:+.6f} z-px/px "
                  f"= {slope * 1000:+.2f} z-px per 1000 px "
                  f"({ang:+.4f} deg, voxel-corrected)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("state", help="viewer state JSON, a Neuroglancer URL, or -")
    p.add_argument("--layer", help="layer name (default: first multi-source layer)")
    p.add_argument("--tol", type=float,
                   help="gap that separates grid rows/columns, in px "
                        "(default: a quarter of the largest gap)")
    p.add_argument("--per-move", action="store_true", default=True,
                   help="break the steps down by move index (default on)")
    p.add_argument("--no-per-move", dest="per_move", action="store_false")
    args = p.parse_args(argv)

    managed = pick_layer(load_state(args.state), args.layer)
    sources = managed.layer.source
    if sources is None or len(sources) < 2:
        raise SystemExit("that layer has fewer than two sources; nothing to compare")

    tiles = [read_source(sources[i], i) for i in range(len(sources))]
    res_um = next((t["res_um"] for t in tiles if t["res_um"] is not None), None)
    origins = np.array([t["origin"] for t in tiles])

    col, tol_x = cluster(origins[:, 0], args.tol)
    rowi, tol_y = cluster(origins[:, 1], args.tol)

    print(f"layer {managed.name!r}: {len(tiles)} sources, "
          f"grid {col.max() + 1} x {rowi.max() + 1} (x by y)")
    print(f"voxel size: {res_um[0]:g} x {res_um[1]:g} x {res_um[2]:g} um"
          if res_um is not None else "voxel size: unknown (no transforms)")
    print(f"cluster tolerance: {tol_x:.1f} px in x, {tol_y:.1f} px in y")

    cell = {}
    for i in range(len(tiles)):
        cell.setdefault((col[i], rowi[i]), i)
    if len(cell) != len(tiles):
        print("warning: two or more tiles share a grid cell; check --tol",
              file=sys.stderr)

    x_steps, y_steps = [], []
    by_move = {"x": defaultdict(list), "y": defaultdict(list)}
    for (c, r), i in cell.items():
        right = cell.get((c + 1, r))
        if right is not None:
            d = origins[right] - origins[i]
            x_steps.append(d)
            by_move["x"][c].append(d)
        down = cell.get((c, r + 1))
        if down is not None:
            d = origins[down] - origins[i]
            y_steps.append(d)
            by_move["y"][r].append(d)

    if not x_steps and not y_steps:
        raise SystemExit("no adjacent tile pairs found; check --tol")
    if x_steps:
        describe("X", np.array(x_steps), res_um, 0)
    if y_steps:
        describe("Y", np.array(y_steps), res_um, 1)

    if args.per_move and max(len(by_move["x"]), len(by_move["y"])) > 1:
        print("\nper move index (pooling hides systematic differences):")
        for name in ("x", "y"):
            for k in sorted(by_move[name]):
                d = np.array(by_move[name][k])
                s = d[:, 2].std(ddof=1) if len(d) > 1 else 0.0
                print(f"   {name} move {k}->{k + 1} (n={len(d):>2}): "
                      f"dx {d[:, 0].mean():+8.2f}  dy {d[:, 1].mean():+7.2f}  "
                      f"dz {d[:, 2].mean():+6.2f}  (dz sd {s:.2f})")

    shears = [t["shear"] for t in tiles if t["shear"]]
    if shears:
        sx = np.array([s.get("x", 0.0) for s in shears])
        sy = np.array([s.get("y", 0.0) for s in shears])
        if max(np.abs(sx).max(), np.abs(sy).max()) > 1e-9:
            print("\nintra-tile shear in the z output row:")
            print(f"   dz/dx {sx.mean() * 1000:+.2f} z-px per 1000 x-px"
                  f"   dz/dy {sy.mean() * 1000:+.2f} per 1000 y-px")
            print("   this comes from a global rotate/shear, which reorients "
                  "the slab but\n   cannot change tile-to-tile registration")
        else:
            print("\nno intra-tile shear: tiles are flat, all offset is "
                  "per-tile translation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
