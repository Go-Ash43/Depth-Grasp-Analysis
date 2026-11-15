"""DEMO / INFERENCE SCRIPT: top-1 grasps of the trained R, A0 and M detectors on the shipped sample scenes.

What it does, per scene (the same code path that produced the report's evaluation; nothing re-implemented
that the evaluation does differently):
  1. load the raw Jacquard files (data.jacquard.load_scene): RGB, stereo depth (-1 = undetermined), perfect
     depth, object mask, grasp annotations;
  2. preprocess with the cache builder's own function (scripts/build_cache.py process_scene, frozen values
     from configs/frozen_day1.json): clamp, resize to 224 px, Telea completion over the sensor holes M_sensor;
     arrays cast to the cache dtypes (data.cache_layout.array_specs);
  3. condition --holes none|low|mid|high (the H1 conditions): synthetic holes M_synth from the evaluation's
     corruption stream (data.dataset.stream_rng with the fixed corruption seed, epoch 0, the scene's val
     slot), depth re-completed over M_sensor | M_synth (Protocol A), flag = M_sensor | M_synth: exactly the
     input the evaluation scored (data.dataset.GraspDataset.sample, eval mode, deployed flag, no flip);
  4. normalise as GraspDataset._x and run each detector (artifacts/checkpoints, sha256 checked first);
  5. decode the top-1 grasp (data.decode.decode_from_cfg) and score it against the annotations
     (eval.match.match_both: primary 30 deg / IoU > 0.25, strict 15 deg / IoU > 0.40).

Outputs in --out: predictions.csv (one row per scene x condition x model), one overlay PNG per scene
(rows = conditions; left: RGB, annotations in white, each model's predicted rectangle; right: the completed
depth the RGB-D models received, sensor holes grey, synthetic holes magenta), demo_config.json.

--check compares every prediction with the value the report's validation evaluation recorded for the same
scene, model and condition (stored in <package>/samples.json): same top-1 cell, validity, primary and strict
match and hole fraction; angle, opening and Q within --check-tol (GPU and CPU convolutions differ in the last
bits). Exit status 1 if any comparison fails.

Usage (from the repo root; CPU is enough, ~1 min):
  python demo.py --out demo_out                         # 4 sample scenes, all four conditions
  python demo.py --out demo_out --holes mid --check     # one condition, compared with the recorded evaluation
  python demo.py --out demo_out --scene <Jacquard>/<part>/<object_id> --view 0     # any other scene (clean only)
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# CUDA float32 exactly as in training and evaluation (training/runcfg.py setup_precision): the cuBLAS
# workspace must be fixed before torch initialises CUDA (same guard as scripts/eval_m1.py).
_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG is {os.environ['CUBLAS_WORKSPACE_CONFIG']!r}; unset it or use {_CUBLAS!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from data.cache_layout import array_specs  # noqa: E402
from data.dataset import MODELS, SEVERITY_ORDER, STREAM_HOLES, load_data_cfg, load_frozen, stream_rng  # noqa: E402
from data.decode import decode_from_cfg  # noqa: E402
from data.geometry import scale_grasps  # noqa: E402
from data.holes import synth_holes  # noqa: E402
from data.jacquard import grasp_corners, load_scene  # noqa: E402
from data.labels import load_labels_cfg  # noqa: E402
from data.mask_ops import complete_protocol_a  # noqa: E402
from data.provenance import git_fields  # noqa: E402
from eval.match import match_both  # noqa: E402
from models.grconv_lite import build_model  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.runcfg import load_train_cfg, setup_precision  # noqa: E402

DEFAULT_PACKAGE = REPO / "demo_data"
FROZEN_FILE = REPO / "configs" / "frozen_day1.json"
TRAIN_CFG = REPO / "configs" / "train_v1.json"
CONFIG_FILES = ("configs/frozen_day1.json", "configs/train_data.json", "configs/labels.json", "configs/train_v1.json")
CONDITIONS = ("none",) + SEVERITY_ORDER              # none, low, mid, high (scripts/eval_m1.py CONDITIONS)
DEMO_MODELS = ("R", "A0", "M")                       # A* needs perfect depth: not a deployable detector
SRC_HW, DST_HW = (1024, 1024), (224, 224)
ROW_FIELDS = ("scene", "slot", "model", "seed", "condition", "hole_frac", "valid", "x", "y", "theta", "opening",
              "jaw", "q_top", "cell_j", "cell_i", "match_primary", "match_strict")
STYLE = {"R": {"color": "#009E73", "ls": ":"}, "A0": {"color": "#0072B2", "ls": "-"},
         "M": {"color": "#E69F00", "ls": "--"}}
SENSOR_RGBA = (0.35, 0.35, 0.35, 0.80)
SYNTH_RGBA = (0.80, 0.47, 0.65, 0.65)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _build_cache_module():
    """scripts/build_cache.py is a script, not a package module: load it by path (no side effects on import)."""
    spec = importlib.util.spec_from_file_location("build_cache", REPO / "scripts" / "build_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ steps 1-2: raw scene -> 224-px arrays
def prepare_scene(obj_dir, view, bc=None):
    """Raw Jacquard scene -> (arrays at 224 in the cache dtypes, grasps (n, 5) at 1024, float64).

    Raises ValueError if the cache builder would exclude the scene (R1: no object / no valid depth at 224)."""
    bc = bc or _build_cache_module()
    fcfg = bc.frozen_cfg(FROZEN_FILE)
    scene = load_scene(Path(obj_dir).expanduser().resolve(), view)
    row, arrays, grasps, _ = bc.process_scene(scene, fcfg, abort_on_clamp=False)    # val/test: clamp counted
    if row["status"] != "ok":
        raise ValueError(f"{obj_dir} view {view}: excluded by the cache rules ({row['status']})")
    specs = array_specs(fcfg["size"], fcfg["taus"])
    out = {k: np.ascontiguousarray(arrays[k], dtype=specs[k][0])
           for k in ("rgb", "ds", "ds_comp", "m_sensor", "obj")}                     # D* and M_lie never used
    return out, np.asarray(grasps, np.float64)


# ------------------------------------------------------------------ steps 3-4: condition + detector input
def detector_input(arrays, model, holes, slot, split, data_cfg, frozen):
    """Mirror of GraspDataset.sample (mode eval, flag_setting deployed, no flip) followed by GraspDataset._x.

    -> (x float32 [C, 224, 224], m_synth bool [224, 224], hole_frac, depth plane or None)."""
    ms = arrays["m_sensor"].astype(bool)
    synth, hole_frac = np.zeros_like(ms), 0.0
    if holes != "none":
        if slot is None:
            raise ValueError("synthetic holes need the scene's evaluation slot (sample scenes only)")
        rng_h = stream_rng(int(data_cfg["corruption_seed"]), STREAM_HOLES, 0, int(slot), split)
        m, rec = synth_holes(rng_h, ms, arrays["obj"], float(data_cfg["holes"]["severities"][holes]),
                             data_cfg["holes"])
        synth, hole_frac = m.astype(bool), float(rec["achieved_frac"])
    true_inv = ms | synth
    planes = {"rgb": arrays["rgb"]}
    chans = MODELS[model]
    if "depth" in chans:                                  # Protocol A: complete over the TRUE invalid set
        planes["depth"] = (complete_protocol_a(arrays["ds"], true_inv, float(frozen["telea_radius"]))
                           if synth.any() else arrays["ds_comp"])
    planes["flag"] = true_inv                             # deployed flag; enters x only for M
    rm, rs = float(data_cfg["rgb_norm"]["mean"]), float(data_cfg["rgb_norm"]["std"])
    nm = frozen["normalisation"]["ds_comp_mm"]
    dm, dsig = float(nm["median"]), float(nm["robust_sigma"])
    out = []
    for ch in chans:                                      # GraspDataset._x, channel order of MODELS
        if ch == "rgb":
            out.append(((planes["rgb"].astype(np.float32) / 255.0 - rm) / rs).transpose(2, 0, 1))
        elif ch == "depth":
            out.append(((planes["depth"].astype(np.float64) - dm) / dsig).astype(np.float32)[None])
        elif ch == "flag":
            out.append(planes["flag"].astype(np.float32)[None])
        else:
            raise ValueError(f"channel {ch!r} is not part of a deployable detector")
    x = np.ascontiguousarray(np.concatenate(out, 0))
    if not np.isfinite(x).all():
        raise ValueError("non-finite value in the detector input")
    depth = planes.get("depth")
    if depth is None:                                     # R: draw the A0 / M depth plane anyway
        depth = (complete_protocol_a(arrays["ds"], true_inv, float(frozen["telea_radius"]))
                 if synth.any() else arrays["ds_comp"])
    return x, synth, hole_frac, depth


# ------------------------------------------------------------------ step 4-5: detector, decode, match
def load_detector(path, sha256, model, seed, model_cfg, device):
    st = load_checkpoint(path, sha256)                    # refuses a checkpoint whose sha256 differs
    if (st["meta"]["model"], int(st["meta"]["seed"])) != (model, int(seed)):
        raise RuntimeError(f"{path}: checkpoint is {st['meta']['model']} seed {st['meta']['seed']}, "
                           f"expected {model} seed {seed}")
    net = build_model(model, seed, model_cfg["c"], model_cfg["n_res"])
    net.load_state_dict(st["model"])
    return net.to(device).eval()


def predict(net, x, gt, labels_cfg, device):
    """Same steps as eval.val_epoch.run_val / classify_sample for one scene."""
    with torch.inference_mode():
        pred = net(torch.from_numpy(x)[None].to(device))
    pm = {k: pred[k].float().cpu().numpy()[0, 0] for k in ("q", "cos2", "sin2", "width")}
    dec = decode_from_cfg(pm, labels_cfg)                 # top-1 cell, output frame 1024
    m = match_both(dec["grasp"], dec["valid"], gt, labels_cfg)
    x_, y_, th, op, jaw = (float(v) for v in dec["grasp"])
    return {"valid": dec["valid"], "x": x_, "y": y_, "theta": th, "opening": op, "jaw": jaw, "q_top": dec["q"],
            "cell_j": int(dec["cell_xy"][0]), "cell_i": int(dec["cell_xy"][1]),
            "match_primary": m["primary"]["match"], "match_strict": m["strict"]["match"], "grasp": dec["grasp"]}


# ------------------------------------------------------------------ figure
def _rects(grasps_1024):
    return grasp_corners(scale_grasps(np.asarray(grasps_1024, np.float64).reshape(-1, 5), SRC_HW, DST_HW))


def _crop(obj, rect_sets, pad=12):
    pts = [np.argwhere(obj)[:, ::-1].astype(np.float64)] + [r.reshape(-1, 2) for r in rect_sets if len(r)]
    p = np.concatenate(pts, 0)
    x0, y0 = np.floor(p.min(0)) - pad
    x1, y1 = np.ceil(p.max(0)) + pad
    side = max(x1 - x0, y1 - y0)                          # square crop, inside the image
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = np.clip([cx - side / 2, cy - side / 2], -0.5, None)
    x1, y1 = np.clip([x0 + side, y0 + side], None, DST_HW[1] - 0.5)
    return x0, x1, y0, y1


def draw_scene(path, name, arrays, gt, per_cond):
    """per_cond: [(condition, depth plane, m_synth, {model: prediction row})]."""
    ann = _rects(gt)
    preds = [_rects(p["grasp"]) for _, _, _, pr in per_cond for p in pr.values()]
    x0, x1, y0, y1 = _crop(arrays["obj"].astype(bool), [ann] + preds)
    win = arrays["ds_comp"][int(max(y0, 0)):int(y1) + 1, int(max(x0, 0)):int(x1) + 1]   # the shown crop
    vmin, vmax = np.percentile(win, [2, 98])
    n = len(per_cond)
    fig, axes = plt.subplots(n, 2, figsize=(6.4, 3.2 * n), squeeze=False)
    for r, (cond, depth, synth, pr) in enumerate(per_cond):
        a_rgb, a_d = axes[r]
        a_rgb.imshow(arrays["rgb"])
        a_d.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
        over = np.zeros(DST_HW + (4,))
        over[arrays["m_sensor"].astype(bool)] = SENSOR_RGBA
        over[synth] = SYNTH_RGBA
        a_d.imshow(over)
        for ax in (a_rgb, a_d):
            if ax is a_rgb:                               # annotations on the RGB panel only: keep the holes visible
                for c in ann:
                    ax.add_patch(Polygon(c, closed=True, fill=False, ec="white", lw=0.4, alpha=0.35))
            for m, p in pr.items():
                c = _rects(p["grasp"])[0]
                ax.add_patch(Polygon(c, closed=True, fill=False, ec=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=1.6))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
        hits = "  ".join(f"{m}:{'hit' if p['match_primary'] else 'miss'}" for m, p in pr.items())
        a_rgb.set_title(f"{cond}   {hits}", fontsize=8, loc="left")
        a_d.set_title("completed depth; holes: grey sensor, magenta synthetic", fontsize=7, loc="left")
    handles = [Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=1.6, label=m) for m in DEMO_MODELS]
    handles.append(Line2D([], [], color="0.6", lw=0.8, label="annotations"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(name, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ --check
def check_rows(rows, recorded, tol):
    """rows vs samples.json["recorded"][run][condition][slot] -> (n compared, list of failures, max deltas)."""
    fails, n, dmax = [], 0, {"theta": 0.0, "opening": 0.0, "q_top": 0.0}
    for r in rows:
        rec = recorded.get(f"{r['model']}_s{r['seed']}", {}).get(r["condition"], {}).get(str(r["slot"]))
        if rec is None:
            fails.append(f"{r['scene']} {r['model']} {r['condition']}: no recorded value")
            continue
        n += 1
        tag = f"{r['scene']} {r['model']}_s{r['seed']} {r['condition']}"
        for k in ("cell_j", "cell_i"):
            if int(rec[k]) != r[k]:
                fails.append(f"{tag}: {k} {r[k]} != recorded {rec[k]}")
        for k in ("valid", "match_primary", "match_strict"):
            if (rec[k] == "True") != bool(r[k]):
                fails.append(f"{tag}: {k} {r[k]} != recorded {rec[k]}")
        if np.float32(r["hole_frac"]) != np.float32(float(rec["hole_frac"])):   # eval collate stores float32
            fails.append(f"{tag}: hole_frac {r['hole_frac']} != recorded {rec['hole_frac']} (corruption differs)")
        for k in dmax:
            d = abs(float(rec[k]) - r[k])
            if k == "theta":
                d = min(d, 180.0 - d)                     # theta is defined modulo 180 deg
            dmax[k] = max(dmax[k], d)
            if d > tol:
                fails.append(f"{tag}: |{k} - recorded| = {d:.3g} > {tol}")
    return n, fails, dmax


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (created; must be empty if it exists)")
    p.add_argument("--package", default=str(DEFAULT_PACKAGE),
                   help="demo package folder holding samples.json and sample_data/ (default: demo_data/); the "
                        "checkpoints are the repo's artifacts/checkpoints")
    p.add_argument("--holes", default="all", choices=("all",) + CONDITIONS,
                   help="H1 condition: none = sensor holes only; low/mid/high add synthetic holes (default: all)")
    p.add_argument("--models", default=",".join(DEMO_MODELS), help="comma list from R,A0,M (default: all three)")
    p.add_argument("--seed", type=int, default=1,
                   help="training seed of the shipped checkpoints to use (checkpoints/<model>_s<seed>.pt)")
    p.add_argument("--scene", default=None, help="run on this Jacquard object folder instead of the sample scenes")
    p.add_argument("--view", type=int, default=0, help="view index for --scene (default 0)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--check", action="store_true",
                   help="compare with the recorded validation evaluation (sample scenes only); exit 1 on mismatch")
    p.add_argument("--check-tol", type=float, default=1e-3,
                   help="max |difference| in theta (deg), opening (px at 1024) and Q for --check (default 1e-3)")
    p.add_argument("--no-figures", action="store_true", help="write predictions.csv only")
    return p.parse_args()


def main():
    a = parse_args()
    pkg = Path(a.package).expanduser().resolve()
    man = json.loads((pkg / "samples.json").read_text())
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if not models or any(m not in DEMO_MODELS for m in models):
        raise SystemExit(f"--models must be a comma list from {DEMO_MODELS}, got {a.models!r}")
    conds = list(CONDITIONS) if a.holes == "all" else [a.holes]
    if a.scene is not None:
        if a.check:
            raise SystemExit("--check needs the sample scenes (recorded values exist only for them)")
        if conds != ["none"]:
            if a.holes != "all":
                raise SystemExit("--scene: synthetic holes need an evaluation slot; use --holes none")
            conds = ["none"]
        scenes = [{"name": f"{Path(a.scene).name}_v{a.view}", "dir": Path(a.scene), "view": a.view, "slot": None}]
    else:
        scenes = [{"name": f"{s['object_id'][:8]}_v{s['view']}", "dir": pkg / s["dir"], "view": int(s["view"]),
                   "slot": int(s["slot"])} for s in man["scenes"]]
    split = man["split"]

    # configs must be the ones the package was made with: preprocessing, holes, labels and model all come from them
    cfg_now = {f: sha256_file(REPO / f) for f in CONFIG_FILES}
    cfg_diff = sorted(f for f in CONFIG_FILES if cfg_now[f] != man["configs_sha256"].get(f))
    if cfg_diff:
        msg = f"configs differ from the ones the demo package was made with: {cfg_diff}"
        if a.check:
            raise SystemExit(msg)
        print("WARNING:", msg, file=sys.stderr)

    out = Path(a.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: pick a fresh --out")
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(a.seed)                             # nothing here is random; logged for completeness

    data_cfg, labels_cfg, frozen = load_data_cfg(), load_labels_cfg(), load_frozen()
    train_cfg, _ = load_train_cfg(TRAIN_CFG)
    # CUDA: IEEE float32 (no TF32) and deterministic kernels, as the evaluation; CPU has no TF32
    precision = setup_precision(train_cfg) if torch.device(a.device).type == "cuda" else None
    ckpts = {(c["model"], int(c["seed"])): c for c in man["checkpoints"]}
    nets = {}
    for m in models:
        c = ckpts.get((m, a.seed))
        if c is None:
            raise SystemExit(f"the package has no checkpoint for {m} seed {a.seed}")
        nets[m] = load_detector(REPO / c["file"], c["sha256"], m, a.seed, train_cfg["model"], a.device)

    bc = _build_cache_module()
    rows = []
    for sc in scenes:
        arrays, gt = prepare_scene(sc["dir"], sc["view"], bc)
        per_cond = []
        for cond in conds:
            pr = {}
            for m in models:
                x, synth, hole_frac, depth = detector_input(arrays, m, cond, sc["slot"], split, data_cfg, frozen)
                p = predict(nets[m], x, gt, labels_cfg, a.device)
                pr[m] = p
                rows.append({"scene": sc["name"], "slot": sc["slot"], "model": m, "seed": a.seed, "condition": cond,
                             "hole_frac": hole_frac, **{k: v for k, v in p.items() if k != "grasp"}})
            per_cond.append((cond, depth, synth, pr))
            print(f"{sc['name']:<14} {cond:<5} " + "  ".join(
                f"{m}: {'hit ' if p['match_primary'] else 'miss'} (q {p['q_top']:.3f})" for m, p in pr.items()))
        if not a.no_figures:
            draw_scene(out / f"{sc['name']}.png", sc["name"], arrays, gt, per_cond)

    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r[k] for k in ROW_FIELDS} for r in rows)
    for m in models:
        for cond in conds:
            rs = [r for r in rows if r["model"] == m and r["condition"] == cond]
            print(f"top-1 accuracy {m:<2} {cond:<5} primary {sum(r['match_primary'] for r in rs)}/{len(rs)}"
                  f"   strict {sum(r['match_strict'] for r in rs)}/{len(rs)}")

    report = None
    if a.check:
        n, fails, dmax = check_rows(rows, man["recorded"], a.check_tol)
        report = {"compared": n, "failures": fails, "max_abs_delta": dmax, "tol": a.check_tol,
                  "passed": not fails and n == len(rows)}
        print(f"check vs recorded evaluation: {n} predictions, max |d| theta {dmax['theta']:.2e} deg, "
              f"opening {dmax['opening']:.2e} px, q {dmax['q_top']:.2e}: {'PASS' if report['passed'] else 'FAIL'}")
        for f in fails[:20]:
            print("  ", f)

    cfg = {"script": "demo.py", "args": vars(a), "package": str(pkg), "package_made_at_commit": man.get("git_commit"),
           "configs_sha256": cfg_now, "conditions": conds, "precision": precision, "models": models, "n_scenes": len(scenes),
           "torch": torch.__version__, "python": platform.python_version(), "check": report, **git_fields()}
    (out / "demo_config.json").write_text(json.dumps(cfg, indent=1, default=str) + "\n")
    print(f"wrote {out}/predictions.csv" + ("" if a.no_figures else " and one PNG per scene"))
    if report is not None and not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
