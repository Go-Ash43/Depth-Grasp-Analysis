"""Fig. 3 of the Module 1 report: paired grasp overlays for the saved mid-severity validation cases (Table VIII).

For each slot (default 0, 6, 10, 35; seed-1 runs): top row = RGB at 224 with every annotation (thin white),
the A0 prediction (blue, solid) and the M prediction (orange, dashed); bottom row = the completed depth D_c
the RGB-D detectors received (one shared colour scale), native holes M_sensor (grey) and injected holes
M_synth (magenta), with the same two predictions. Cropped around the object, the annotations and both
predictions; one crop per scene for both rows.

The mid-severity input is rebuilt with the evaluation's own dataset code (GraspDataset, eval mode, epoch 0)
and checked against the evaluation sidecar: the m_synth and flag sha256 per slot must equal the recorded ones,
so the figure shows exactly the input that was scored. Predictions and outcomes are read from the saved
per-scene CSVs (never recomputed). Rectangles use data.jacquard.grasp_corners (the scoring polygons), scaled
to 224 with data.geometry.scale_grasps; pixel centres are integers (imshow default extent).

Reads the cache and <eval-dir>/results; writes <out> (PDF or PNG, by suffix) and <out>.json (config,
per-slot records incl. the fraction of each predicted rectangle's cells that are filled).

Usage:
  python scripts/make_fig3.py --eval-dir $R/m1_eval_val --out ~/Downloads/figures/grasp-overlays.pdf
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from data.cache_layout import open_cache  # noqa: E402
from data.dataset import MODELS, GraspDataset  # noqa: E402
from data.geometry import scale_grasps  # noqa: E402
from data.jacquard import grasp_corners  # noqa: E402
from data.paths import load_paths, paths_source  # noqa: E402
from data.provenance import git_fields  # noqa: E402

SRC_HW, DST_HW = (1024, 1024), (224, 224)
GRASP_KEYS = ("x", "y", "theta", "opening", "jaw")
STYLE = {"A0": {"color": "#0072B2", "ls": "-"}, "M": {"color": "#E69F00", "ls": "--"}}
SENSOR_RGBA = (0.35, 0.35, 0.35, 0.80)
SYNTH_RGBA = (0.80, 0.47, 0.65, 0.65)


def _true(v):
    return str(v).strip().lower() in ("1", "true")


def sha_u8(a):
    """Same digest as the evaluation sidecar (scripts/eval_m1.py _h)."""
    return hashlib.sha256(np.ascontiguousarray(a, np.uint8).tobytes()).hexdigest()


def read_rows(path, slots):
    rows = {int(r["slot"]): r for r in csv.DictReader(open(path, newline=""))}
    missing = [s for s in slots if s not in rows]
    if missing:
        raise SystemExit(f"{path}: slots {missing} not found")
    return {s: rows[s] for s in slots}


def pred_grasp(row):
    """(1, 5) predicted grasp at 1024 px from a per-scene CSV row; None for an invalid decode."""
    if not _true(row["valid"]):
        return None
    return np.array([[float(row[k]) for k in GRASP_KEYS]], np.float64)


def outcome(row):
    return f"{'hit' if _true(row['match_primary']) else 'miss'}/{'hit' if _true(row['match_strict']) else 'miss'}"


def corners_224(grasps_1024):
    """(N, 5) grasps at 1024 -> (N, 4, 2) corners at 224 (x right, y down, integer = pixel centre)."""
    return grasp_corners(scale_grasps(grasps_1024, SRC_HW, DST_HW))


def crop_box(obj, pts, margin, hw):
    """Square box (x0, x1, y0, y1) in pixel coordinates around the object cells and the points
    (pixel-centre convention: a box edge sits 0.5 px outside the outermost centre), clipped to the image."""
    ys, xs = np.nonzero(obj)
    px = np.concatenate([xs.astype(float), np.asarray(pts, float).reshape(-1, 2)[:, 0]])
    py = np.concatenate([ys.astype(float), np.asarray(pts, float).reshape(-1, 2)[:, 1]])
    x0, x1 = px.min() - 0.5 - margin, px.max() + 0.5 + margin
    y0, y1 = py.min() - 0.5 - margin, py.max() + 0.5 + margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    lo, hi_x, hi_y = -0.5, hw[1] - 0.5, hw[0] - 0.5
    return max(x0, lo), min(x1, hi_x), max(y0, lo), min(y1, hi_y)


def filled_fraction(corners, mask):
    """Fraction of grid cells whose centre lies inside the rectangle that are set in `mask`."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inside = MplPath(corners).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    n = int(inside.sum())
    return None if n == 0 else float(mask[inside].sum()) / n


def parse_args(argv=None):
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval-dir", default=str(Path(paths["runs_dir"]).expanduser() / "m1_eval_val"),
                   help="eval_m1.py --out of the validation evaluation")
    p.add_argument("--cache-dir", default=paths["cache_dir"])
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--condition", default="mid", choices=["low", "mid", "high"])
    p.add_argument("--run-seed", type=int, default=1, help="which seed's A0 / M runs supply the predictions")
    p.add_argument("--slots", type=int, nargs="+", default=[0, 6, 10, 35])
    p.add_argument("--margin", type=float, default=6.0, help="crop margin, px at 224")
    p.add_argument("--width-in", type=float, default=7.0)
    p.add_argument("--height-in", type=float, default=4.1)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", required=True, help="figure path (.pdf or .png); config goes to <out>.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); recorded by convention")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out).expanduser()
    cfg_path = out.with_name(out.name + ".json")
    if out.suffix.lower() not in (".pdf", ".png"):
        raise SystemExit("--out must end in .pdf or .png")
    if not a.overwrite and (out.exists() or cfg_path.exists()):
        raise SystemExit(f"{out} or {cfg_path} exists; pass --overwrite")
    res = Path(a.eval_dir).expanduser() / "results"
    runs = {m: f"{m}_s{a.run_seed}" for m in ("A0", "M")}
    rows = {m: read_rows(res / r / f"{a.condition}.csv", a.slots) for m, r in runs.items()}
    side = read_rows(res / runs["M"] / f"{a.condition}.sidecar.csv", a.slots)

    cache = open_cache(a.cache_dir, a.split)
    A = cache["arrays"]
    ds = GraspDataset(a.cache_dir, a.split, "M", mode="eval", holes=a.condition)
    depth_ch = MODELS["M"].index("depth") + 2                      # rgb takes channels 0-2
    med, sig = ds.depth_norm

    scenes = []
    for slot in a.slots:
        s = ds.sample(0, slot, return_flag=True)
        synth = s["m_synth"].numpy().astype(bool)
        flag = s["flag_plane"].numpy().astype(bool)
        sensor = np.asarray(A["m_sensor"][slot]).astype(bool)
        if not np.array_equal(flag, sensor | synth):
            raise SystemExit(f"slot {slot}: flag != M_sensor | M_synth")
        if sha_u8(synth) != side[slot]["m_synth_sha256"] or sha_u8(flag) != side[slot]["flag_sha256"]:
            raise SystemExit(f"slot {slot}: rebuilt corruption differs from the evaluation sidecar")
        for m in ("A0", "M"):                                      # the CSVs must describe the same instance
            if not np.isclose(float(rows[m][slot]["hole_frac"]), float(side[slot]["hole_frac"])):
                raise SystemExit(f"slot {slot}: {m} CSV hole_frac differs from the sidecar")
        dc = s["x"][depth_ch].numpy().astype(np.float64) * sig + med
        gt = corners_224(s["gt_grasps"])
        preds = {m: pred_grasp(rows[m][slot]) for m in ("A0", "M")}
        pc = {m: (None if g is None else corners_224(g)[0]) for m, g in preds.items()}
        pts = np.concatenate([gt.reshape(-1, 2)] + [c for c in pc.values() if c is not None])
        obj = np.asarray(A["obj"][slot]).astype(bool)
        box = crop_box(obj, pts, a.margin, DST_HW)
        scenes.append({"slot": slot, "rgb": np.asarray(A["rgb"][slot]), "dc": dc, "sensor": sensor,
                       "synth": synth, "obj": obj, "gt": gt, "pc": pc, "box": box,
                       "rec": {"slot": slot, "hole_frac": float(side[slot]["hole_frac"]),
                               "sidecar_hash_match": True,
                               **{f"{m}_outcome_primary_strict": outcome(rows[m][slot]) for m in ("A0", "M")},
                               **{f"{m}_valid": preds[m] is not None for m in ("A0", "M")},
                               **{f"{m}_target_void": _true(rows[m][slot]["target_void"]) for m in ("A0", "M")},
                               **{f"{m}_target_opening_224": rows[m][slot]["target_opening_224"] or None
                                  for m in ("A0", "M")},
                               **{f"{m}_cfn_match_primary": rows[m][slot]["cfn_match_primary"] or None
                                  for m in ("A0", "M")},
                               **{f"{m}_rect_filled_frac": (None if pc[m] is None
                                                            else filled_fraction(pc[m], flag))
                                  for m in ("A0", "M")},
                               **{f"{m}_rect_injected_frac": (None if pc[m] is None
                                                              else filled_fraction(pc[m], synth))
                                  for m in ("A0", "M")},
                               "n_annotations": int(len(gt))}})

    vals = np.concatenate([sc["dc"][sc["obj"]] for sc in scenes])
    vmin, vmax = np.percentile(vals, [1, 99])

    fig, axes = plt.subplots(2, len(scenes), figsize=(a.width_in, a.height_in), squeeze=False)
    for col, sc in enumerate(scenes):
        x0, x1, y0, y1 = sc["box"]
        top, bot = axes[0, col], axes[1, col]
        top.imshow(sc["rgb"], interpolation="nearest")
        im = bot.imshow(sc["dc"], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        over = np.zeros(sc["dc"].shape + (4,))
        over[sc["sensor"]] = SENSOR_RGBA
        over[sc["synth"]] = SYNTH_RGBA
        bot.imshow(over, interpolation="nearest")
        for c in sc["gt"]:
            top.add_patch(Polygon(c, closed=True, fill=False, ec="white", lw=0.35, alpha=0.7))
        for ax in (top, bot):
            for m in ("A0", "M"):
                if sc["pc"][m] is not None:
                    ax.add_patch(Polygon(sc["pc"][m], closed=True, fill=False, ec=STYLE[m]["color"],
                                         ls=STYLE[m]["ls"], lw=1.3))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
        r = sc["rec"]
        top.set_title(f"Slot {sc['slot']}\nA0 {r['A0_outcome_primary_strict']}   "
                      f"M {r['M_outcome_primary_strict']}", fontsize=7)
    axes[0, 0].set_ylabel("RGB, annotations,\npredictions", fontsize=7)
    axes[1, 0].set_ylabel("Completed depth,\nfilled cells", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.12, wspace=0.05, hspace=0.06)
    cax = fig.add_axes([0.915, 0.14, 0.015, 0.33])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Depth (mm)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    handles = [Line2D([], [], color=STYLE["A0"]["color"], ls="-", lw=1.3, label="A0 prediction"),
               Line2D([], [], color=STYLE["M"]["color"], ls="--", lw=1.3, label="M prediction"),
               Line2D([], [], color="0.6", lw=0.6, label="Annotations"),
               Patch(fc=SENSOR_RGBA, label="Native hole"),
               Patch(fc=SYNTH_RGBA, label="Injected hole")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi)

    cfg = {"script": "scripts/make_fig3.py", "args": vars(a), **git_fields(), "paths_source": paths_source(),
           "runs": runs, "depth_colour_range_mm": [float(vmin), float(vmax)], "slots": [sc["rec"] for sc in scenes]}
    cfg_path.write_text(json.dumps(cfg, indent=1, default=str))
    print(f"wrote {out} and {cfg_path}")
    for sc in scenes:
        print(json.dumps(sc["rec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
