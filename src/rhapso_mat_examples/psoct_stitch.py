"""
Stitch the PS-OCT slice-004 orientation mosaic with Rhapso's phase-correlation matcher.

    psoct-stitch --out work/stitch                 # all 24 tiles
    psoct-stitch --out work/stitch --tiles 6       # one column, for a quick look
    psoct-stitch --out work/stitch --compare manifests/slice_004_orientation_offsets.json

Start from a naive layout -- a regular grid at the nominal 10% overlap -- then let
Rhapso refine it: phase correlation measures a displacement in every overlap, writes the
results as ordinary point correspondences, and the global solver fits per-tile
transforms. The comparison at the end is against the offsets register_seams measured
independently, which is the answer this should reproduce.

Deliberately thin. Everything that does work lives in Rhapso; this module only
builds the naive XML, sets the options, and reports. The pieces it leans on:

    psoct_virtual.build                        virtual OME-Zarr wrappers, no data copied
    ng_tile_link.grid_offsets                  the naive grid, shared with the NG tooling
    Rhapso PhaseCorrelationMatching            the matcher
    Rhapso Solver                              the global fit

---------------------------------------------------------------------------
Where the options come from
---------------------------------------------------------------------------
Set to match register_seams.py, so the two should agree:

    register_seams                        Rhapso equivalent
    ------------------------------------  ---------------------------------------
    NCC over the overlap                  method="ncc" (grid mode) / always (window mode)
    log1p(hypot(ch0, ch1))                log_transform=True, channel_mode="magnitude"
    FLYBACK = 25 trailing x columns        edge_margin=((0,0,0),(25,0,0))
    template 50 x 50 x 110 (x, y, z)      cutout_size=(100, 100, 110) -- the TEMPLATE
                                           size in window mode, see below
    search band around the stride         max_shift=(40, 40, 25)
    integer argmax                        peak_mode="absolute"
    full resolution                       dsxy=1, dsz=1
    no ambiguity gate                     min_peak_ratio=1.0 (off)
    small template in a larger window     cutout_mode="window" (default)

register_seams slid a small template inside a much larger window, so every candidate
shift compared the same voxel count no matter how far it moved -- no "sliver" bias at
the edges of the search. Rhapso's OverlapCutouts now has that as a real mode:
cutout_mode="window" places cutout_size as the TEMPLATE in tile B, and a window in tile A
sized to cutout_size + 2*max_shift, so the template stays fully contained anywhere the
search will actually look. This is the default here.

cutout_mode="grid" is the older, biased alternative: equal-sized boxes from both tiles,
correlated over +-max_shift, where a large shift shrinks the compared region. Kept for
comparison -- pass --cutout-mode grid to reproduce the earlier behaviour.

One option still cannot be matched exactly, and it matters. register_seams took ONE
correlation per seam and solved a pure translation itself. Rhapso's solver has no
translation-only model -- its cheapest is rigid, fitted by weighted Horn from >= 3 points
-- so a single correspondence per pair would leave it degenerate. Hence grid_counts:
several template positions spread through each overlap (both modes place them the same
way), which is what gives the solver something to fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from rhapso_mat_examples.ng_tile_link import grid_offsets, tile_number
from rhapso_mat_examples.psoct_virtual import STORAGE_OPTIONS, build

TILE_SIZE_XYZ = (1000, 1000, 191)
VOXEL_SIZE_XYZ = (5.0, 5.0, 3.4)
LABEL = "phasecorr"

# the naive guess this pipeline starts from
NOMINAL_OVERLAP = 0.1

# tile numbering runs 6 deep along y, then steps in x (measured by overlap correlation;
# this is layout topology, not the stride, which is what gets refined)
FAST_AXIS = "y"
FAST_COUNT = 6


def affine_text(offset) -> str:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(offset, dtype=float)

    return " ".join(repr(float(v)) for v in matrix[:3, :4].reshape(-1))


def write_xml(path, tiles, size_xyz=TILE_SIZE_XYZ, voxel_xyz=VOXEL_SIZE_XYZ) -> Path:
    """
    A minimal SpimData XML describing the naive layout.

    Only what Rhapso's parsers actually read: zgroups for the image loader, ViewSetups
    for the tile sizes, ViewRegistrations for the starting layout, and one
    ViewInterestPointsFile per view -- that last is how GeneratePairs learns which views
    exist at all, so omitting it yields zero pairs and a silent no-op.
    """
    root = ET.Element("SpimData", {"version": "0.2"})
    ET.SubElement(root, "BasePath", {"type": "relative"}).text = "."

    sequence = ET.SubElement(root, "SequenceDescription")
    setups = ET.SubElement(sequence, "ViewSetups")
    loader = ET.SubElement(sequence, "ImageLoader", {"format": "bdv.multimg.zarr"})
    zgroups = ET.SubElement(loader, "zgroups")

    registrations = ET.SubElement(root, "ViewRegistrations")
    interest_points = ET.SubElement(root, "ViewInterestPoints")

    for setup, tile in enumerate(tiles):
        node = ET.SubElement(setups, "ViewSetup")
        ET.SubElement(node, "id").text = str(setup)
        ET.SubElement(node, "name").text = tile["name"]
        ET.SubElement(node, "size").text = " ".join(str(int(v)) for v in size_xyz)
        voxel = ET.SubElement(node, "voxelSize")
        ET.SubElement(voxel, "unit").text = "micrometer"
        ET.SubElement(voxel, "size").text = " ".join(repr(float(v)) for v in voxel_xyz)
        attributes = ET.SubElement(node, "attributes")
        ET.SubElement(attributes, "tile").text = str(setup)
        ET.SubElement(attributes, "channel").text = "0"

        group = ET.SubElement(zgroups, "zgroup", {"setup": str(setup), "tp": "0",
                                                  "timepoint": "0"})
        ET.SubElement(group, "path").text = tile["url"]

        registration = ET.SubElement(
            registrations, "ViewRegistration", {"timepoint": "0", "setup": str(setup)}
        )
        transform = ET.SubElement(registration, "ViewTransform", {"type": "affine"})
        ET.SubElement(transform, "Name").text = "Naive grid"
        ET.SubElement(transform, "affine").text = affine_text(tile["offset"])

        ET.SubElement(
            interest_points, "ViewInterestPointsFile",
            {"timepoint": "0", "setup": str(setup), "label": LABEL, "params": ""},
        ).text = LABEL

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    return path


def naive_tiles(wrappers, overlap=NOMINAL_OVERLAP, size_xyz=TILE_SIZE_XYZ):
    """Pair each virtual wrapper with its offset under the nominal-overlap grid."""
    stride_x = size_xyz[0] * (1.0 - overlap)
    stride_y = size_xyz[1] * (1.0 - overlap)
    offsets = grid_offsets(len(wrappers), FAST_AXIS, FAST_COUNT, stride_x, stride_y)

    return [
        {"name": name, "url": url, "offset": (ox, oy, 0.0)}
        for (name, url, _shape, _dims), (ox, oy) in zip(wrappers, offsets)
    ]


def solved_offsets(xml_path, tiles):
    """
    Per-tile translation after solving, composed the way the fusion stage composes it.

    SaveResults prepends its transform, and ComputeBBox multiplies left to right, so the
    solved model is the leftmost factor -- applied last to a point.
    """
    root = ET.parse(xml_path).getroot()
    result = {}

    for registration in root.findall(".//ViewRegistration"):
        setup = int(registration.get("setup"))
        matrix = np.eye(4)

        for transform in registration.findall("ViewTransform"):
            values = np.fromstring(
                (transform.findtext("affine") or "").replace(",", " "), sep=" "
            )

            if values.size != 12:
                continue

            step = np.eye(4)
            step[:3, :4] = values.reshape(3, 4)
            matrix = matrix @ step

        name = tiles[setup]["name"] if setup < len(tiles) else str(setup)
        result[tile_number(name) or setup] = matrix

    return result


def compare(solved, reference_path, tiles):
    """
    Report solved offsets against register_seams, after removing the global gauge.

    Both are only defined up to a rigid gauge -- the solver pins one tile and
    register_seams pinned tile 1 -- so the comparison subtracts each set's own mean
    before differencing. What matters is the relative layout, not where the mosaic sits.
    """
    reference = {int(k): np.asarray(v, dtype=float)
                 for k, v in json.load(open(reference_path)).items()}
    shared = sorted(set(solved) & set(reference))

    if not shared:
        print("no tiles in common with the reference offsets", file=sys.stderr)
        return

    ours = np.array([solved[t][:3, 3] for t in shared])
    theirs = np.array([reference[t] for t in shared])
    ours = ours - ours.mean(axis=0)
    theirs = theirs - theirs.mean(axis=0)
    delta = ours - theirs

    print(f"\nversus register_seams, {len(shared)} tiles, gauge removed")
    print(f"{'tile':>5}  {'solved (x, y, z)':>26}  {'reference':>26}  {'delta':>22}")

    for row, tile in enumerate(shared):
        print(f"{tile:>5}  {ours[row][0]:8.1f}{ours[row][1]:9.1f}{ours[row][2]:9.1f}  "
              f"{theirs[row][0]:8.1f}{theirs[row][1]:9.1f}{theirs[row][2]:9.1f}  "
              f"{delta[row][0]:7.2f}{delta[row][1]:8.2f}{delta[row][2]:7.2f}")

    rms = float(np.sqrt((delta ** 2).sum(axis=1).mean()))
    print(f"\nrms disagreement {rms:.2f} px "
          f"(max per-axis {np.abs(delta).max(axis=0).round(2)})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="work/stitch")
    parser.add_argument("--tiles", type=int, help="use only the first N tiles")
    parser.add_argument("--overlap", type=float, default=NOMINAL_OVERLAP,
                        help="the naive guess to start from (default 0.1)")
    parser.add_argument("--cutout-mode", default="window", choices=("grid", "window"),
                        help="grid: same-size boxes from both tiles, correlated over "
                             "+-max_shift -- the compared region shrinks near the "
                             "search boundary (the 'sliver' effect). window: a small "
                             "template from B searched inside a larger window from A, "
                             "sized so the template is always fully contained -- "
                             "register_seams.py's own geometry, bias-free at every "
                             "candidate shift (default)")
    parser.add_argument("--grid-counts", type=int, nargs=3, default=[2, 3, 2],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--cutout-size", type=int, nargs=3, default=[100, 100, 110],
                        metavar=("X", "Y", "Z"),
                        help="grid mode: size of both boxes. window mode: size of the "
                             "template only -- the window is sized from this plus "
                             "--max-shift")
    # Wide enough on BOTH in-plane axes: a seam runs along x or y, and whichever axis it
    # steps along needs room for the whole stride error. The naive 900 stride is ~31 px
    # short in y and ~7..30 px off in x, so a 25 px bound rejects the correct match
    # outright -- which is exactly what a first run of this pipeline did.
    parser.add_argument("--max-shift", type=int, nargs=3, default=[40, 40, 25],
                        metavar=("X", "Y", "Z"))
    # In grid mode this trades artifact rejection against the largest shift the
    # geometry can even express: the box is the estimated overlap minus this margin, and
    # a box W wide cannot represent a lag past +-W/2. On slice 004 the nominal overlap is
    # 100 px, so 25 leaves 75 and caps the search at +-37 -- while the column 1->2 seam
    # needs -34..-38, which is why those cutouts get rejected on the search boundary.
    # Window mode does not have the problem: its window grows past the estimated overlap
    # by max_shift, which is how register_seams could keep the full flyback margin.
    parser.add_argument("--flyback", type=int, default=25,
                        help="trailing x columns trimmed from each tile (default 25, "
                             "matching register_seams). Grid mode caps the usable shift "
                             "at (overlap - flyback) / 2")
    parser.add_argument("--dsxy", type=int, default=1)
    parser.add_argument("--dsz", type=int, default=1)
    parser.add_argument("--min-peak", type=float, default=0.3)
    # register_seams applied no ambiguity gate at all -- it relied on loop closure, and
    # Rhapso's solver does the same job better with median+MAD residual rejection. A
    # ratio gate is also ill-suited to NCC here: the peak is many voxels wide on smooth
    # data, so the best lag outside a small exclusion ball sits inside the same peak and
    # the ratio is ~1.00 even for a perfect match. Off by default, with a wide exclusion
    # radius so that raising it is meaningful.
    parser.add_argument("--min-peak-ratio", type=float, default=1.0)
    parser.add_argument("--exclusion-radius", type=int, default=10)
    parser.add_argument("--method", default="ncc", choices=("ncc", "phase"),
                        help="both score by NCC over the real overlap, so --min-peak "
                             "means the same thing either way; they differ in how the "
                             "lag is found. ncc (default): score every lag in the "
                             "--max-shift box. phase: take the strongest few peaks of "
                             "a whitened cross-power spectrum and score only those -- "
                             "cheaper, and its cost does not grow with --max-shift, "
                             "but it reports no peak ratio, so --min-peak-ratio is "
                             "ignored. Only takes effect with --cutout-mode grid; "
                             "window mode's template-in-window correlator is NCC-only")
    parser.add_argument("--run-type", default="rigid",
                        choices=("translation", "rigid", "affine"),
                        help="translation: no rotation at all, just a per-tile offset "
                             "-- the fewest parameters a linear model can have. rigid: "
                             "translation + rotation (default). affine: rigid "
                             "regularized toward a full affine")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="pairs correlated at once. Each pulls whole multi-MB "
                             "chunks, so an unbounded fan-out exhausts connections and "
                             "reads start failing (default 4)")
    parser.add_argument("--compare", help="register_seams offsets JSON to check against")
    parser.add_argument("--skip-matching", action="store_true",
                        help="reuse an existing match store and only re-solve")
    args = parser.parse_args(argv)

    # Solver builds a boto3 client at construction even for local paths
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

    import ray
    from Rhapso.pipelines.ray.phase_correlation_matching import PhaseCorrelationMatching
    from Rhapso.pipelines.ray.solver import Solver

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wrappers = build(out / "virtual", limit=args.tiles)
    tiles = naive_tiles(wrappers, overlap=args.overlap)
    naive_xml = write_xml(out / "naive.xml", tiles)
    store = str(out / "store")

    print(f"{len(tiles)} tiles, naive stride "
          f"{TILE_SIZE_XYZ[0] * (1 - args.overlap):g} px -> {naive_xml}")

    if not args.skip_matching:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        PhaseCorrelationMatching(
            xml_input_path=str(naive_xml),
            n5_output_path=store,
            input_type="zarr",
            image_file_prefix="",
            match_type=args.run_type,
            label=LABEL,
            method=args.method,
            log_transform=True,
            channel_mode="magnitude",
            # "auto" reads each tile's own dimension_names. Slice 004 is
            # heterogeneous -- 23 tiles are (z, x, y, channel) float32 and tile 20 is
            # (z, x, y) complex64 -- so a single hard-coded order fails on tile 20.
            source_axes="auto",
            storage_options=STORAGE_OPTIONS,
            cutout_mode=args.cutout_mode,
            cutout_size=tuple(args.cutout_size),
            grid_counts=tuple(args.grid_counts),
            edge_margin=((0, 0, 0), (args.flyback, 0, 0)),
            peak_mode="absolute",
            min_peak=args.min_peak,
            min_peak_ratio=args.min_peak_ratio,
            exclusion_radius=args.exclusion_radius,
            max_shift=tuple(args.max_shift),
            dsxy=args.dsxy,
            dsz=args.dsz,
            max_concurrency=args.max_concurrency,
            qc_csv_path=str(out / "phase_correlation_qc.csv"),
        ).run()

    solved_xml = out / f"solved_{args.run_type}.xml"

    Solver(
        xml_file_path_output=str(solved_xml),
        n5_input_path=store,
        xml_file_path=str(naive_xml),
        run_type=args.run_type,
        relative_threshold=3.5,
        absolute_threshold=7.0,
        max_cleanup_rounds=3,
        min_matches=3,
        damp=1.0,
        # unused for translation/rigid (neither touches the regularized slot); only
        # affine actually blends toward rigid by this weight
        regularization_weight=0.05 if args.run_type == "affine" else 1.0,
        max_iterations=10000,
        max_allowed_error=float("inf"),
        max_plateauwidth=200,
        metrics_output_path=str(out / "solver_metrics"),
        fixed_tile=None,
    ).run()

    solved = solved_offsets(solved_xml, tiles)

    print(f"\nsolved layout -> {solved_xml}")

    for tile in sorted(solved):
        translation = solved[tile][:3, 3]
        print(f"  tile {tile:>3}: x {translation[0]:9.2f}  y {translation[1]:9.2f}  "
              f"z {translation[2]:8.2f}")

    if args.compare:
        compare(solved, args.compare, tiles)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
