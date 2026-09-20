"""
Look at one correlation surface: why does peak 0.95 come with a ratio of 1.02?

    psoct-peak-debug --setup-a 2 --setup-b 3 --out work/peak_debug.png

Re-runs a single cutout correlation from a QC row, then plots the surface as three
orthogonal heatmaps through the peak (dx-dy, dx-dz, dy-dz) and marks:

    o  the argmax             -- what correlate() reports
    x  the exclusion-radius runner-up -- what peak_ratio currently divides by
    +  DoG-detected local maxima      -- genuinely separate peaks

The suspicion being tested is that on a broad peak the exclusion ball (radius 3) is far
smaller than the peak itself, so the "runner-up" is a point on the flank of the SAME
peak. A ratio near 1 then means "the peak is wide", not "there are two candidates" --
and in the slice-004 run 106 of 152 accepted cutouts look like that, so the metric is
measuring the wrong thing rather than catching a rare pathology.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.matching.overlap_cutouts import OverlapCutouts
from Rhapso.matching.phase_correlation import PhaseCorrelation
from Rhapso.matching.tile_volume_reader import TileVolumeReader
from Rhapso.matching.xml_parser import XMLParserMatching
from Rhapso.pipelines.ray.phase_correlation_matching import PhaseCorrelationMatching
from rhapso_mat_examples.psoct_virtual import STORAGE_OPTIONS

AXIS_LABEL = ("dz", "dy", "dx")      # surface axes are zyx


def pick_row(qc_path, setup_a=None, setup_b=None, min_peak=0.85, max_ratio=1.15):
    """The first accepted QC row that shows the symptom."""
    best = None

    with open(qc_path) as handle:
        for row in csv.DictReader(handle):
            if row.get("Accepted") != "True":
                continue

            try:
                peak = float(row["Correlation"])
                ratio = float(row["PeakRatio"])
            except (TypeError, ValueError):
                continue

            if peak < min_peak or ratio > max_ratio:
                continue

            if setup_a is not None and f"setupId={setup_a})" not in row["TileA"]:
                continue

            if setup_b is not None and f"setupId={setup_b})" not in row["TileB"]:
                continue

            if best is None or ratio < float(best["PeakRatio"]):
                best = row

    if best is None:
        raise SystemExit("no accepted row matched the high-peak / low-ratio filter")

    return best


def rebuild_surface(xml_path, row, cutout_size, dsxy, dsz, max_shift, log_transform):
    """Read the same two boxes again and return the correlation surface."""
    data_global = XMLParserMatching(str(xml_path), "zarr").run()
    stage = PhaseCorrelationMatching(str(xml_path), None, "zarr", "")

    setup_a = int(row["TileA"].split("setupId=")[1].rstrip(")"))
    setup_b = int(row["TileB"].split("setupId=")[1].rstrip(")"))
    view_a, view_b = (0, setup_a), (0, setup_b)

    centre = np.array([float(row["CenterX"]), float(row["CenterY"]),
                       float(row["CenterZ"])])

    cutouts = OverlapCutouts(
        cutout_mode="grid", cutout_size=cutout_size, grid_counts=(2, 2, 2),
        max_cutout=(512, 512, 256), edge_margin=((0, 0, 0), (25, 0, 0)),
    ).run(
        stage.compose_matrix(data_global["viewRegistrations"], view_a),
        data_global["viewSetup"]["byId"][setup_a]["size"],
        stage.compose_matrix(data_global["viewRegistrations"], view_b),
        data_global["viewSetup"]["byId"][setup_b]["size"],
    )

    if not cutouts:
        raise SystemExit("no cutouts for that pair; check the XML matches the QC run")

    box = min(cutouts, key=lambda b: np.abs(b["center_a"] - centre).sum())
    print(f"pair setup {setup_a} -> {setup_b}, cutout centre "
          f"{np.round(box['center_a'], 1)} (QC row said {centre})")

    reader = TileVolumeReader(
        "zarr", channel_mode="magnitude", source_axes="auto",
        storage_options=STORAGE_OPTIONS, dsxy=dsxy, dsz=dsz,
    )
    factors = np.array([dsxy, dsxy, dsz], dtype=float)

    volumes = []
    for side, view in (("a", view_a), ("b", view_b)):
        path = stage.file_path_for_setup(data_global, view[1])
        low, high = box[f"box_{side}"]
        lower = np.floor(np.asarray(low, dtype=float) / factors).astype(int)
        upper = np.ceil(np.asarray(high, dtype=float) / factors).astype(int)
        volumes.append(reader.run(path, lower, upper))

    shape = min((v.shape for v in volumes), key=lambda s: tuple(s))
    volumes = [v[:shape[0], :shape[1], :shape[2]] for v in volumes]

    correlator = PhaseCorrelation(method="ncc", window=True, peak_mode="absolute",
                                 exclusion_radius=3, log_transform=log_transform)

    # surface axes are zyx, geometry is xyz
    a = correlator.prepare(volumes[0].transpose(2, 1, 0), correlator.window)
    b = correlator.prepare(volumes[1].transpose(2, 1, 0), correlator.window)

    half = np.minimum(
        np.asarray(max_shift, dtype=int)[::-1] // np.array([dsz, dsxy, dsxy]),
        np.asarray(a.shape) - 1,
    )
    surface, half = correlator.ncc_surface(a, b, half)

    return correlator, surface, half


def dog_peaks(surface, sigma=1.2, threshold=0.05):
    """
    Distinct local maxima of the surface, via Rhapso's DoG detector.

    A difference of Gaussians is a bandpass: it suppresses the broad shoulder that makes
    the exclusion-ball runner-up meaningless and keeps structure at the scale of a real
    peak. find_peaks then takes strict 26-neighbourhood maxima, and refine_peaks applies
    the Newton step with its own rejection (Hessian must be negative definite, bounded
    condition number), so what comes back are separated peaks rather than flank samples.
    """
    detector = DifferenceOfGaussian(0.0, 1.0, sigma, threshold, 0, 1)

    scaled = surface - surface.min()
    scaled = scaled / max(scaled.max(), 1e-12)

    # apply_gaussian_blur wants a per-axis sigma
    narrow = detector.apply_gaussian_blur(scaled, (sigma,) * scaled.ndim)
    wide = detector.apply_gaussian_blur(scaled, (sigma * 1.6,) * scaled.ndim)
    dog = narrow - wide

    dog_argmax = np.unravel_index(int(np.argmax(dog)), dog.shape)
    print(f"  DoG max {dog.max():.5f} at index {dog_argmax}; "
          f"ties at max: {int(np.sum(dog == dog.max()))}; "
          f"border distance {min(min(int(i), int(n) - 1 - int(i)) for i, n in zip(dog_argmax, dog.shape))}")

    peaks = detector.find_peaks(dog, threshold * dog.max())

    if len(peaks) == 0:
        # find_peaks skips a 1-voxel border and needs a strict 26-neighbour maximum;
        # report what it saw so a zero is diagnosable rather than mysterious
        for factor in (0.05, 0.01, 0.0):
            trial = detector.find_peaks(dog, factor * dog.max())
            print(f"  find_peaks(threshold={factor * dog.max():+.6f}) -> "
                  f"{len(trial)} peak(s)")

        return np.zeros((0, 3)), np.zeros(0), dog

    print(f"  find_peaks -> {len(peaks)} candidate(s) before refinement")

    positions, scores = detector.refine_peaks(np.asarray(peaks), dog)

    print(f"  refine_peaks kept {len(np.asarray(positions).reshape(-1, 3))} "
          f"of {len(peaks)}")
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    scores = np.asarray(scores, dtype=float).ravel()

    order = np.argsort(-scores)

    return positions[order], scores[order], dog


def plot(surface, half, peak, runner_up, dog_positions, dog_dog, info, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = np.asarray(peak, dtype=int)
    planes = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]   # (row axis, col axis, fixed axis)

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)

    for column, (row_axis, col_axis, fixed_axis) in enumerate(planes):
        for band, (field, name) in enumerate(
            ((surface, "NCC surface"), (dog_dog, "DoG of surface"))
        ):
            axis = axes[band][column]
            plane = np.take(field, peak[fixed_axis], axis=fixed_axis)

            if row_axis > col_axis:
                plane = plane.T

            extent = [
                -half[col_axis] - 0.5, field.shape[col_axis] - half[col_axis] - 0.5,
                field.shape[row_axis] - half[row_axis] - 0.5, -half[row_axis] - 0.5,
            ]
            image = axis.imshow(plane, extent=extent, origin="upper",
                                cmap="magma", aspect="auto", interpolation="nearest")
            figure.colorbar(image, ax=axis, shrink=0.85)

            axis.plot(peak[col_axis] - half[col_axis], peak[row_axis] - half[row_axis],
                      "o", mfc="none", mec="cyan", ms=16, mew=2.2, label="argmax")
            axis.plot(runner_up[col_axis] - half[col_axis],
                      runner_up[row_axis] - half[row_axis],
                      "x", color="lime", ms=15, mew=3,
                      label="exclusion-ball runner-up")

            if len(dog_positions):
                axis.plot(dog_positions[:, col_axis] - half[col_axis],
                          dog_positions[:, row_axis] - half[row_axis],
                          "+", color="white", ms=13, mew=2, label="DoG maxima")

            axis.set_xlabel(f"{AXIS_LABEL[col_axis]} (voxels)")
            axis.set_ylabel(f"{AXIS_LABEL[row_axis]} (voxels)")
            axis.set_title(f"{name}: {AXIS_LABEL[row_axis]} vs {AXIS_LABEL[col_axis]}"
                           f"  @ {AXIS_LABEL[fixed_axis]}="
                           f"{peak[fixed_axis] - half[fixed_axis]:+d}")

            if column == 0 and band == 0:
                axis.legend(loc="upper right", fontsize=8, framealpha=0.85)

    figure.suptitle(info, fontsize=11)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=125)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qc", default="work/final/phase_correlation_qc.csv")
    parser.add_argument("--xml", default="work/final/naive.xml")
    parser.add_argument("--setup-a", type=int)
    parser.add_argument("--setup-b", type=int)
    parser.add_argument("--cutout-size", type=int, nargs=3, default=[100, 100, 110])
    parser.add_argument("--max-shift", type=int, nargs=3, default=[40, 40, 25])
    parser.add_argument("--dsxy", type=int, default=2)
    parser.add_argument("--dsz", type=int, default=2)
    parser.add_argument("--dog-sigma", type=float, default=1.2)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--out", default="work/peak_debug.png")
    args = parser.parse_args(argv)

    row = pick_row(args.qc, args.setup_a, args.setup_b)
    print(f"QC row: peak={row['Correlation']} ratio={row['PeakRatio']} "
          f"edge={row['EdgeDistance']} {row['TileA']} -> {row['TileB']}")

    correlator, surface, half = rebuild_surface(
        args.xml, row, tuple(args.cutout_size), args.dsxy, args.dsz,
        tuple(args.max_shift), not args.no_log,
    )

    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak_value = float(surface[peak])

    # where the current metric's denominator actually comes from
    masked = surface.copy()
    lo = [max(0, int(p) - correlator.exclusion_radius) for p in peak]
    hi = [min(int(s), int(p) + correlator.exclusion_radius + 1)
          for p, s in zip(peak, surface.shape)]
    masked[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = -np.inf
    runner_up = np.unravel_index(int(np.argmax(masked)), masked.shape)
    runner_value = float(masked[runner_up])

    separation = np.abs(np.array(runner_up) - np.array(peak))

    print(f"\nsurface {surface.shape}, lag origin at {tuple(half)}")
    print(f"argmax      lag {tuple(int(p - h) for p, h in zip(peak, half))} "
          f"value {peak_value:.4f}")
    print(f"runner-up   lag {tuple(int(p - h) for p, h in zip(runner_up, half))} "
          f"value {runner_value:.4f}  ratio {peak_value / runner_value:.4f}")
    print(f"            separated from the argmax by {tuple(separation)} voxels "
          f"(exclusion radius {correlator.exclusion_radius})")

    # how wide is the peak really? count voxels above 95% of it
    wide = int(np.sum(surface >= 0.95 * peak_value))
    print(f"voxels within 5% of the peak: {wide} "
          f"({100.0 * wide / surface.size:.2f}% of the surface)")

    positions, scores, dog = dog_peaks(surface, args.dog_sigma)
    print(f"\nDoG detector: {len(positions)} distinct maxima")

    for index, (position, score) in enumerate(zip(positions[:6], scores[:6])):
        lag = tuple(round(float(p - h), 2) for p, h in zip(position, half))
        distance = float(np.linalg.norm(position - np.array(peak)))
        print(f"  {index}: lag {lag} dog={score:.4f} distance from argmax "
              f"{distance:.1f} voxels")

    if len(positions) >= 2:
        print(f"\nDoG-based ratio (strongest / next distinct) "
              f"{scores[0] / scores[1]:.3f}")
    else:
        print("\nDoG found a single distinct peak -- unambiguous by that measure")

    info = (f"{row['TileA']} -> {row['TileB']}  cutout "
            f"({row['CenterX']}, {row['CenterY']}, {row['CenterZ']})   "
            f"NCC peak {peak_value:.4f}, exclusion-ball ratio "
            f"{peak_value / runner_value:.4f}, "
            f"{wide} voxels within 5% of peak, {len(positions)} DoG maxima")

    plot(surface, half, peak, runner_up, positions, dog, info, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
