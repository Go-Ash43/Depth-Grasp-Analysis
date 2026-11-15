#!/usr/bin/env python3
"""Real-data check of the Jacquard grasp-angle convention used by data/jacquard.py.

The unit test (tests/test_angle.py) only proves the loader matches the reference code. This
script asks the data itself which sign of theta is right, by comparing two conventions on the
same annotated grasps:

  A  (loader)    jaw-closing axis = ( cos t,  sin t) in (x, y-down)   -> grasp_corners(g)
  B  (mirrored)  jaw-closing axis = ( cos t, -sin t)                   -> grasp_corners(g with -t)

Physical expectation for a correct convention: the grasp centre lies on the object, and the two
jaw start points (centre +- opening/2 along the closing axis) lie OFF the object and farther from
the camera than the centre (the jaws close from outside onto the object). With the wrong sign,
jaws on tilted grasps across elongated objects land on the object much more often.

Grasps scored (the SAME set for A and B, filters applied in this order):
  1. unique (x, y, theta, opening): rows repeated with only a different jaw size share the same jaw
     midpoints, so they would add duplicate votes;
  2. |sin 2t| >= --min-sin2t: near 0 and 90 degrees A and B are the same rectangle;
  3. all four jaw points (A and B) inside the image: out-of-frame points would otherwise be clipped
     onto the border, which reads as "off object, deeper" and favours whichever convention leaves
     the frame;
  4. centre on the object.
Removed counts per filter are printed and written to the CSV.

Outputs (in --out, default <manifest_dir>/angle_check; refused if inside data_root):
  <obj>_<view>.png           overlay crops, left = A (loader), right = B (mirrored);
                             red = jaw plates (where the fingers start), green = sides along the
                             closing axis, yellow dot = centre
  angle_check.csv            per-scene rates and filter counts
  angle_check_config.json    arguments and resolved paths of this run
  verdict.json               verdict code + pooled rates
  printed summary            pooled rates and a verdict
Exit status: 0 = loader convention supported, 1 = mirrored fits better, 2 = inconclusive, 3 = no data.

This is a data-format check, not a statistic used for any decision, so it may use any scenes;
it reads only --n scenes (default 12), so it is safe to run while other jobs use the drive.

Usage (from the repo root):
  python scripts/check_angle_convention.py               # 12 random scenes, seed 0
  python scripts/check_angle_convention.py --n 30 --seed 1
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.geometry import in_image, pixel_index  # noqa: E402
from data.jacquard import SCENE_ERRORS, data_root, grasp_corners, load_scene  # noqa: E402

FILTER_KEYS = ("n_rows", "rm_duplicate", "rm_angle", "rm_out_of_frame", "rm_centre_off_object")
# Exit status = verdict, so automation cannot read a failed check as success (also in verdict.json).
EXIT = {"LOADER": 0, "MIRRORED": 1, "INCONCLUSIVE": 2, "NO_DATA": 3}


def write_verdict(out, code, text, stats):
    with open(out / "verdict.json", "w") as f:
        json.dump({"verdict": code, "exit_status": EXIT[code], "text": text, **stats}, f, indent=2)


def in_frame(pts, shape):
    """Pixel-centre convention (data/geometry.py), never clipped."""
    return in_image(pts, shape)


def sample(img, pts):
    """Lookup of the pixel containing each (x, y) point. pixel_index raises ValueError on
    out-of-frame points, so callers must filter with in_frame first."""
    x, y = pixel_index(pts, img.shape)
    return img[y, x]


def jaw_points(corners):
    """Jaw start points from grasp_corners output: midpoints of the two jaw plates.
    Corners are c-L/2a-W/2p, c+L/2a-W/2p, c+L/2a+W/2p, c-L/2a+W/2p (see data/jacquard.py)."""
    e1 = (corners[:, 0] + corners[:, 3]) / 2      # c - L/2 a
    e2 = (corners[:, 1] + corners[:, 2]) / 2      # c + L/2 a
    return e1, e2


def score(grasps, corners, obj_mask, depth, margin_mm):
    """Per-grasp booleans: both jaws off the object; both jaws deeper than the centre by margin."""
    e1, e2 = jaw_points(corners)
    c = grasps[:, 0:2]
    off = ~sample(obj_mask, e1) & ~sample(obj_mask, e2)
    dc = sample(depth, c)
    deeper = (sample(depth, e1) - dc > margin_mm) & (sample(depth, e2) - dc > margin_mm)
    return off, deeper


def select(g, obj_mask, min_sin2t):
    """Boolean mask over rows of g for the common A/B scoring set, plus removed counts per filter."""
    counts = {"n_rows": len(g)}
    _, first = np.unique(g[:, 0:4], axis=0, return_index=True)          # exact: same text -> same float
    keep = np.zeros(len(g), bool)
    keep[first] = True                                                   # 1. unique jaw geometry
    counts["rm_duplicate"] = int(len(g) - keep.sum())

    ang = np.abs(np.sin(2 * np.deg2rad(g[:, 2]))) >= min_sin2t           # 2. A != B
    counts["rm_angle"] = int((keep & ~ang).sum())
    keep &= ang

    gB = g.copy()
    gB[:, 2] = -gB[:, 2]
    frame = in_frame(g[:, 0:2], obj_mask.shape)
    for e in (*jaw_points(grasp_corners(g)), *jaw_points(grasp_corners(gB))):
        frame &= in_frame(e, obj_mask.shape)                             # 3. no clipping, A and B
    counts["rm_out_of_frame"] = int((keep & ~frame).sum())
    keep &= frame

    ctr = np.zeros(len(g), bool)
    ctr[frame] = sample(obj_mask, g[frame, 0:2])                         # 4. centre on object
    counts["centre_on_object_before_filter"] = round(float(ctr[keep].mean()), 3) if keep.any() else float("nan")
    counts["rm_centre_off_object"] = int((keep & ~ctr).sum())
    keep &= ctr
    return keep, counts


def draw(rgb, grasps, corners, idx, obj_mask, pad=40, out_px=512):
    """Draw selected rectangles on a copy of rgb, crop to the object, resize for viewing."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i in idx:
        q = np.rint(corners[i]).astype(np.int32)
        cv2.line(img, tuple(q[0]), tuple(q[1]), (0, 200, 0), 1)   # long sides, along the closing axis
        cv2.line(img, tuple(q[2]), tuple(q[3]), (0, 200, 0), 1)
        cv2.line(img, tuple(q[3]), tuple(q[0]), (0, 0, 255), 2)   # jaw plates at c -/+ L/2 a, width = jaw
        cv2.line(img, tuple(q[1]), tuple(q[2]), (0, 0, 255), 2)
        cv2.circle(img, (int(round(grasps[i, 0])), int(round(grasps[i, 1]))), 2, (0, 255, 255), -1)
    ys, xs = np.nonzero(obj_mask)
    h, w = obj_mask.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    side = min(max(y1 - y0, x1 - x0), h, w)                        # square crop, no distortion
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = min(max(cy - side // 2, 0), h - side)                     # keep the whole square inside the image
    x0 = min(max(cx - side // 2, 0), w - side)
    crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "paths.json"),
                    help="paths config; its data_root AND manifest_dir are both used")
    ap.add_argument("--manifest", default="", help="default: <manifest_dir>/manifest.csv")
    ap.add_argument("--out", default="", help="default: <manifest_dir>/angle_check")
    ap.add_argument("--n", type=int, default=12, help="number of scenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-sin2t", type=float, default=0.5,
                    help="score only grasps with |sin 2theta| >= this (where A and B differ)")
    ap.add_argument("--margin-mm", type=float, default=5.0,
                    help="jaw point must be this much deeper than the centre")
    ap.add_argument("--draw", type=int, default=20, help="rectangles drawn per overlay")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow an --out that already holds results of an earlier run")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["data_root"]).expanduser().resolve()      # from --config, not the repo default
    mdir = Path(cfg["manifest_dir"]).expanduser()
    manifest = Path(args.manifest) if args.manifest else mdir / "manifest.csv"
    out = (Path(args.out) if args.out else mdir / "angle_check").expanduser().resolve()
    # protect the repo-default dataset root too, even when --config points at another dataset
    for protected in {root, Path(data_root()).expanduser().resolve()}:
        if out == protected or protected in out.parents:
            sys.exit(f"refusing to write inside the dataset root {protected}: {out}")

    # validate inputs BEFORE touching --out, so a failed rerun cannot leave a new config beside old results
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"manifest {manifest} has no rows")
    old = [p.name for p in out.glob("*") if p.suffix in (".json", ".csv", ".png")] if out.is_dir() else []
    if old and not args.overwrite:
        sys.exit(f"{out} already holds {len(old)} result files of an earlier run (e.g. {old[:3]}); "
                 f"use a fresh --out, or --overwrite")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "angle_check_config.json", "w") as f:
        json.dump({**vars(args), "data_root": str(root), "manifest_resolved": str(manifest.resolve()),
                   "out_resolved": str(out)}, f, indent=2)

    per_scene, pooled = [], {k: [] for k in ("offA", "offB", "deepA", "deepB")}
    removed = {k: 0 for k in FILTER_KEYS}
    n_load_skipped = 0
    written = set()                     # overlay PNGs written by THIS run
    rates = ("jaws_off_A", "jaws_off_B", "jaws_deeper_A", "jaws_deeper_B")
    for j in pick:
        r = rows[j]
        obj_dir, view = f"{r['part']}/{r['object_id']}", int(r["scene_idx"])
        try:
            s = load_scene(root / obj_dir, view)
        except SCENE_ERRORS as e:  # expected data errors only; code bugs (ValueError, TypeError, ...) crash
            print(f"skip {obj_dir} view {view}: {type(e).__name__}: {e}")
            n_load_skipped += 1
            continue
        g = s["grasps"]
        keep, counts = select(g, s["obj_mask"], args.min_sin2t)
        for k in FILTER_KEYS:
            removed[k] += counts[k]
        if keep.sum() == 0:
            print(f"no grasps left after filters in {obj_dir} view {view}: {counts}")
            per_scene.append({"part": r["part"], "object_id": r["object_id"], "view": view,
                              **counts, "n_scored": 0, **{k: "" for k in rates}, "overlay": ""})
            continue
        cA = s["grasp_corners"][keep]          # exactly what the loader produced
        g = g[keep]
        gB = g.copy()
        gB[:, 2] = -gB[:, 2]
        cB = grasp_corners(gB)
        offA, deepA = score(g, cA, s["obj_mask"], s["depth_perfect"], args.margin_mm)
        offB, deepB = score(g, cB, s["obj_mask"], s["depth_perfect"], args.margin_mm)
        for k, v in (("offA", offA), ("offB", offB), ("deepA", deepA), ("deepB", deepB)):
            pooled[k].append(v)

        idx = rng.choice(len(g), size=min(args.draw, len(g)), replace=False)
        panel = np.concatenate([draw(s["rgb"], g, cA, idx, s["obj_mask"]),
                                np.full((512, 8, 3), 255, np.uint8),
                                draw(s["rgb"], gB, cB, idx, s["obj_mask"])], axis=1)
        cv2.putText(panel, "A loader", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "B mirrored", (528, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        png = f"{r['object_id']}_{view}.png"
        if not cv2.imwrite(str(out / png), panel):
            raise OSError(f"cv2.imwrite failed for {out / png}")
        written.add(png)

        per_scene.append({"part": r["part"], "object_id": r["object_id"], "view": view,
                          **counts, "n_scored": len(g),
                          "jaws_off_A": round(float(offA.mean()), 3), "jaws_off_B": round(float(offB.mean()), 3),
                          "jaws_deeper_A": round(float(deepA.mean()), 3),
                          "jaws_deeper_B": round(float(deepB.mean()), 3), "overlay": png})
        print(f"{obj_dir} view {view}: n={len(g)}  jaws off object A={offA.mean():.2f} B={offB.mean():.2f}"
              f"  jaws deeper A={deepA.mean():.2f} B={deepB.mean():.2f}")

    if per_scene:                       # every loaded scene has a row, scored or not, so CSV sums match
        with open(out / "angle_check.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_scene[0]))
            w.writeheader()
            w.writerows(per_scene)
    scored = [s for s in per_scene if s["n_scored"] > 0]
    if not scored:
        msg = f"No scenes scored ({n_load_skipped} load skips, {len(per_scene)} loaded; removals {removed})."
        write_verdict(out, "NO_DATA", msg, {})
        print(msg, file=sys.stderr)
        sys.exit(EXIT["NO_DATA"])

    P = {k: np.concatenate(v) for k, v in pooled.items()}
    oa, ob, da, db = (float(P[k].mean()) for k in ("offA", "offB", "deepA", "deepB"))
    a_wins = sum(s["jaws_off_A"] > s["jaws_off_B"] for s in scored)
    b_wins = sum(s["jaws_off_B"] > s["jaws_off_A"] for s in scored)
    print("\n=== pooled over", len(scored), "scored scenes,", len(P["offA"]), "scored grasps ===")
    print(f"scenes: {len(pick)} picked, {n_load_skipped} skipped at load, "
          f"{len(per_scene) - len(scored)} with no grasps after filters")
    print("rows and removals (all loaded scenes):", removed)
    print(f"both jaws off object:  A (loader) {oa:.3f}   B (mirrored) {ob:.3f}")
    print(f"both jaws deeper:      A (loader) {da:.3f}   B (mirrored) {db:.3f}")
    print(f"scenes where A > B: {a_wins}   B > A: {b_wins}   ties: {len(scored) - a_wins - b_wins}")
    if oa - ob > 0.1 and da - db > 0.1:
        code, verdict = "LOADER", "LOADER CONVENTION (A) SUPPORTED"
    elif ob - oa > 0.1 and db - da > 0.1:
        code, verdict = "MIRRORED", "MIRRORED (B) FITS BETTER: the loader's theta sign is likely wrong"
    else:
        code, verdict = "INCONCLUSIVE", "INCONCLUSIVE: look at the overlays, or rerun with more scenes (--n 30)"
    print("verdict:", verdict)
    write_verdict(out, code, verdict, {
        "n_scenes_scored": len(scored), "n_grasps_scored": int(len(P["offA"])), "n_load_skipped": n_load_skipped,
        "jaws_off_A": oa, "jaws_off_B": ob, "jaws_deeper_A": da, "jaws_deeper_B": db,
        "scenes_A_wins": int(a_wins), "scenes_B_wins": int(b_wins), "removed": removed})
    print(f"overlays, CSV and config JSON in {out}")
    stale = sorted(p.name for p in out.glob("*.png") if p.name not in written)
    if stale:
        print(f"WARNING: {len(stale)} PNGs in {out} are from an earlier run (not in this run's CSV 'overlay' "
              f"column), e.g. {stale[:3]}; use a fresh --out to avoid mixing them up")
    sys.exit(EXIT[code])


if __name__ == "__main__":
    main()
