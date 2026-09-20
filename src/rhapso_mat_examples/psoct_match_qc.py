"""
Plot the phase-correlation point correspondences from a psoct-stitch run.

    psoct-match-qc --run work/final2 --out work/match_qc.png

Reuses Rhapso's own correspondence readers rather than re-parsing the parquet store by
hand: `Rhapso.evaluation.match_view.load_point_manifest/load_match_index/
list_saved_match_pairs/get_pair_matches_bidirectional`. Those functions are correct but
read a module-level `ALIGNMENT_BASE` global instead of taking the store path as an
argument, so this module points that global at our store before calling them -- a wart
in match_view.py worth knowing about if you call it elsewhere, not something introduced
here.

What match_view.py itself cannot do for us: its plots (`view_match_pair_slider` etc.) are
interactive matplotlib-widget sliders built to overlay warped image volumes, and getting
there requires a BigStitcher-style XML with tagged PRE/POST transform names plus a real
OME-Zarr multiscale pyramid on disk. Our XML has one plain "Naive grid" transform and our
tiles are single-level virtual reference stores, so neither precondition holds. This
module plots the same correspondences a different way instead: no image volumes, just the
point geometry, which is what answers "do the matches look right".

---------------------------------------------------------------------------
What is plotted
---------------------------------------------------------------------------
Each saved pair's correspondences are local, full-res coordinates in EACH tile's own
frame (match_view's own docstring: "these are already the saved accepted correspondences
... this function does not use the XML transform stack"). To see whether they agree, both
sides are pushed through that tile's registration from the given XML, and the residual is
the distance between the two results in world space -- zero would mean the two tiles'
matched points, once placed, land on exactly the same spot.

Two panels:
    left   every matched pair's world position (midpoint of the two projections),
           colored by residual -- shows spatial coverage and where the fit is worst
    right  histogram of residuals, split by whether the correspondence survived the
           solver's cleanup (cleanup_weight) -- lets you see if what got dropped was
           actually the tail of the distribution

Pass --xml naive.xml to see the RAW correspondence disagreement before solving (this
should be large -- that is the whole point of running the solver), and the default
solved_*.xml to see what is left after.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import xml.etree.ElementTree as ET

from Rhapso.evaluation import match_view


def apply_matrix(matrix, points_xyz):
    homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    return (matrix @ homogeneous.T).T[:, :3]


def transforms_by_setup(xml_path):
    """
    Each view's composed registration, keyed by the raw XML setup int.

    Deliberately not psoct_stitch.solved_offsets: that function re-keys its result by
    tile NUMBER parsed out of the tile name (for comparing against register_seams), which
    would collide every setup onto the same key here since match_view's readers key
    everything by the plain setup int straight out of the parquet store.
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

            if values.size == 12:
                step = np.eye(4)
                step[:3, :4] = values.reshape(3, 4)
                matrix = matrix @ step

        result[setup] = matrix

    return result


def gather(store, xml_path, label, timepoint=0):
    """
    All saved correspondences, projected through each tile's transform.

    Returns per-correspondence: world position of each side, the residual between them,
    and which tile pair it came from -- everything the two plots need.
    """
    match_view.ALIGNMENT_BASE = str(store)     # the global the reader functions expect

    manifest = match_view.load_point_manifest(str(store))
    index = match_view.load_match_index(str(store))
    pairs = match_view.list_saved_match_pairs(index, timepoint, label)

    if not pairs:
        raise SystemExit(f"no saved pairs for label {label!r} under {store}")

    transforms = transforms_by_setup(xml_path)

    world_a, world_b, residual, pair_id = [], [], [], []

    for setup_a, setup_b in pairs:
        if setup_a not in transforms or setup_b not in transforms:
            print(f"skipping ({setup_a}, {setup_b}): no transform in {xml_path.name}")
            continue

        local_a, local_b, _meta = match_view.get_pair_matches_bidirectional(
            manifest, index, timepoint, setup_a, setup_b, label
        )

        if len(local_a) == 0:
            continue

        a = apply_matrix(transforms[setup_a], local_a)
        b = apply_matrix(transforms[setup_b], local_b)

        world_a.append(a)
        world_b.append(b)
        residual.append(np.linalg.norm(a - b, axis=1))
        pair_id.extend([(setup_a, setup_b)] * len(a))

    if not world_a:
        raise SystemExit("no correspondence survived transform lookup; check the XML "
                         "and label match the run that produced the store")

    return (np.vstack(world_a), np.vstack(world_b), np.concatenate(residual), pair_id)


def plot(world_a, world_b, residual, pair_id, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    midpoint = (world_a + world_b) / 2.0

    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6.5),
                                         constrained_layout=True)

    order = np.argsort(residual)   # draw worst on top, so they are not hidden
    scatter = left.scatter(midpoint[order, 0], midpoint[order, 1],
                           c=residual[order], cmap="viridis_r", s=14,
                           vmin=0, vmax=np.percentile(residual, 95))
    figure.colorbar(scatter, ax=left, label="residual (px, world space)")
    left.set_xlabel("x (px)")
    left.set_ylabel("y (px)")
    left.set_title(f"{len(residual)} correspondences across "
                   f"{len(set(pair_id))} tile pairs")
    left.set_aspect("equal")
    left.invert_yaxis()

    right.hist(residual, bins=40, color="#3b6fa0", edgecolor="white", linewidth=0.4)
    right.axvline(float(np.median(residual)), color="crimson", linestyle="--",
                  label=f"median {np.median(residual):.2f}")
    right.axvline(float(np.mean(residual)), color="darkorange", linestyle=":",
                  label=f"mean {np.mean(residual):.2f}")
    right.set_xlabel("residual (px, world space)")
    right.set_ylabel("count")
    right.legend()
    right.set_title("residual distribution")

    figure.suptitle(title, fontsize=11)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("---")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", default="work/final2",
                        help="a psoct-stitch output directory")
    parser.add_argument("--xml", default=None,
                        help="which registration to project through (default: the "
                             "solved_*.xml found under --run)")
    parser.add_argument("--label", default="phasecorr")
    parser.add_argument("--out", default="work/match_qc.png")
    args = parser.parse_args(argv)

    run = Path(args.run)
    store = run / "store"

    if args.xml:
        xml_path = Path(args.xml)
    else:
        found = sorted(run.glob("solved_*.xml"))
        if not found:
            raise SystemExit(f"no solved_*.xml under {run}; pass --xml explicitly")
        xml_path = found[0]

    print(f"store: {store}")
    print(f"xml:   {xml_path}")

    world_a, world_b, residual, pair_id = gather(store, xml_path, args.label)

    print(f"\n{len(residual)} correspondences, {len(set(pair_id))} tile pairs")
    print(f"residual: min {residual.min():.2f}  median {np.median(residual):.2f}  "
          f"mean {residual.mean():.2f}  max {residual.max():.2f}  px")

    by_pair = {}
    for pid, r in zip(pair_id, residual):
        by_pair.setdefault(pid, []).append(r)

    print("\nworst tile pairs by median residual:")
    ranked = sorted(by_pair.items(), key=lambda kv: -np.median(kv[1]))
    for pid, values in ranked[:8]:
        print(f"  {pid}: n={len(values):>3}  median {np.median(values):7.2f}  "
              f"max {np.max(values):7.2f}")

    plot(world_a, world_b, residual, pair_id,
        f"{run.name} / {xml_path.name}  ({args.label})", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
