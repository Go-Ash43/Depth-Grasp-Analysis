"""Build the demo package (demo_data/) that demo.py runs on: sample scenes, checkpoints, recorded predictions.

Contents written to --out (default: demo_data/ in the repo):
  sample_data/<object_id>/<view>_<object_id>_{RGB.png, stereo_depth.tiff, perfect_depth.tiff, mask.png, grasps.txt}
      the raw Jacquard v2 files of the chosen validation scenes (default: slots 0, 6, 10, 35, the scenes of the
      report's Fig. 3), copied unchanged;
  samples.json
      scenes (slot, object, view, file sha256), checkpoints (default R, A0, M, seed 1: the repo's copies in
      artifacts/checkpoints, written by scripts/export_artifacts.py, which must run first; their sha256 is
      checked against the run selection record, m1_selection/selected_runs.json), the sha256 of the configs
      the package depends on, and the recorded per-scene predictions of the validation evaluation for every
      chosen run and condition (each CSV checked against the evaluation's eval_record.json first).
No absolute paths and no timestamps are written (the package is meant to be committed to the public repo).

Checks before anything is accepted: for every scene, demo.prepare_scene on the COPIED files must reproduce the
production cache entry bit for bit (rgb, ds, ds_comp, m_sensor, obj, grasps), so the demo's preprocessing is
proven identical to what the detectors were trained and evaluated on. The tree must be clean (the package
records the commit it was made at).

Usage (dataset path from DEPTH_LIES_PATHS or --data-root; R = the production runs folder):
  python scripts/make_demo_package.py --selected $R/m1_selection/selected_runs.json --eval-dir $R/m1_eval_val
Then: python demo.py --out <fresh dir> --check
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import demo  # noqa: E402
from data.cache_layout import open_cache  # noqa: E402
from data.jacquard import data_root, scene_paths  # noqa: E402
from data.paths import load_paths  # noqa: E402
from data.provenance import git_fields  # noqa: E402

ARTIFACTS = REPO / "artifacts"
CACHE_KEYS = ("rgb", "ds", "ds_comp", "m_sensor", "obj")
RECORDED_FIELDS = ("severity", "hole_frac", "valid", "x", "y", "theta", "opening", "q_top", "cell_j", "cell_i",
                   "match_primary", "match_strict")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(cond, msg):
    if not cond:
        raise SystemExit(f"refusing: {msg}")


def copy_checked(src, dst, sha=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    got = sha256_file(dst)
    need(got == sha256_file(src), f"copy of {src.name} differs from its source")
    if sha is not None:
        need(got == sha, f"{src.name}: sha256 {got} != recorded {sha}")
    return got


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", required=True, help="m1_selection/selected_runs.json (the recorded run selection)")
    p.add_argument("--eval-dir", required=True, help="the recorded validation evaluation folder (m1_eval_val)")
    p.add_argument("--cache-dir", default=None, help="production cache (default: cache_dir from the paths config)")
    p.add_argument("--data-root", default=None, help="Jacquard v2 root (default: data_root from the paths config)")
    p.add_argument("--split", default="val", choices=("val",), help="evaluation split of the sample scenes")
    p.add_argument("--slots", default="0,6,10,35", help="cache slots of the sample scenes (default: Fig. 3 scenes)")
    p.add_argument("--models", default="R,A0,M", help="detectors to ship (default R,A0,M)")
    p.add_argument("--seed", type=int, default=1, help="training seed of the runs to ship (default 1)")
    p.add_argument("--out", default=str(REPO / "demo_data"), help="package folder (default: demo_data/ in the repo)")
    p.add_argument("--allow-dirty", action="store_true", help="tests only: allow a dirty git tree")
    return p.parse_args()


def main():
    a = parse_args()
    g = git_fields()
    need(a.allow_dirty or not g["git_dirty"], "dirty git tree: commit first (the package records the commit)")
    out = Path(a.out).expanduser().resolve()
    need(not out.exists() or not any(out.iterdir()), f"{out} exists and is not empty")
    slots = [int(s) for s in a.slots.split(",") if s.strip()]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    need(slots and len(set(slots)) == len(slots), f"--slots {a.slots!r}")
    need(models and all(m in demo.DEMO_MODELS for m in models), f"--models must be from {demo.DEMO_MODELS}")
    cache_dir = Path(a.cache_dir or load_paths()["cache_dir"]).expanduser()
    root = Path(a.data_root or data_root()).expanduser().resolve()

    # recorded selection and evaluation, integrity first
    sel_path, ev = Path(a.selected).expanduser(), Path(a.eval_dir).expanduser() / "results"
    sel = json.loads(sel_path.read_text())
    ev_rec_path = ev / "eval_record.json"
    ev_files = json.loads(ev_rec_path.read_text())["files"]

    cache = open_cache(cache_dir, a.split)                # production cache: identity and splits checked
    out.mkdir(parents=True, exist_ok=True)
    bc = demo._build_cache_module()

    scenes = []
    for slot in slots:
        need(0 <= slot < len(cache["index"]), f"slot {slot} outside the {a.split} cache")
        row = cache["index"][slot]
        need(row["status"] == "ok", f"slot {slot} is excluded by the cache rules ({row['status']})")
        obj, view = row["object_id"], int(row["view"])
        rel = Path("sample_data") / obj
        files = {}
        for src in scene_paths(root / row["part"] / obj, view).values():
            files[src.name] = copy_checked(src, out / rel / src.name)
        arrays, gt = demo.prepare_scene(out / rel, view, bc)          # the demo's path on the COPIED files
        for k in CACHE_KEYS:
            ref = np.asarray(cache["arrays"][k][slot])
            need(arrays[k].dtype == ref.dtype and np.array_equal(arrays[k], ref, equal_nan=(k == "ds")),
                 f"slot {slot}: demo preprocessing of {k} differs from the production cache")
        need(np.array_equal(gt, np.asarray(cache["grasps"](slot), np.float64)),
             f"slot {slot}: parsed grasps differ from the production cache")
        scenes.append({"slot": slot, "part": row["part"], "object_id": obj, "view": view, "dir": str(rel),
                       "files_sha256": files, "matches_cache": list(CACHE_KEYS) + ["grasps"]})
        print(f"slot {slot}: {obj} view {view}: copied, preprocessing identical to the cache")

    checkpoints, recorded = [], {}
    art_idx = json.loads((ARTIFACTS / "artifacts.json").read_text())   # scripts/export_artifacts.py, run first
    shipped = {(c["model"], int(c["seed"])): c for c in art_idx["checkpoints"]}
    for m in models:
        key = f"{m}:{a.seed}"
        need(key in sel["runs"], f"{key} not in the run selection")
        need((m, a.seed) in shipped, f"artifacts/ has no checkpoint for {m} seed {a.seed}")
        r, c = sel["runs"][key], shipped[(m, a.seed)]
        run = Path(r["path"]).name                         # e.g. A0_s1 (folder name only, no machine path)
        rel_ckpt = f"artifacts/{c['file']}"                # the repo's copy; the demo reads it from there
        need(c["sha256"] == r["final_ckpt"]["sha256"] and sha256_file(REPO / rel_ckpt) == c["sha256"],
             f"{rel_ckpt}: not the selected final checkpoint of {key}")
        checkpoints.append({"model": m, "seed": a.seed, "file": rel_ckpt, "sha256": c["sha256"],
                            "run": run, "run_id": r["run_id"], "final_epoch_file": Path(r["final_ckpt"]["path"]).name})
        recorded[f"{m}_s{a.seed}"] = {}
        for cond in demo.CONDITIONS:
            rel = f"{run}/{cond}.csv"
            need(rel in ev_files and sha256_file(ev / rel) == ev_files[rel],
                 f"{rel}: missing or changed since the evaluation recorded it")
            with open(ev / rel, newline="") as fh:
                rows = {int(x["slot"]): x for x in csv.DictReader(fh)}
            need(all(s in rows for s in slots), f"{rel}: a sample slot is missing")
            recorded[f"{m}_s{a.seed}"][cond] = {str(s): {k: rows[s][k] for k in RECORDED_FIELDS} for s in slots}
        print(f"{run}: checkpoint verified (sha256 matches the selection), recorded predictions read")

    man = {"format": 1, "split": a.split, "scenes": scenes, "checkpoints": checkpoints,
           "configs_sha256": {f: sha256_file(REPO / f) for f in demo.CONFIG_FILES},
           "source_sha256": {"selected_runs.json": sha256_file(sel_path), "eval_record.json": sha256_file(ev_rec_path)},
           "recorded": recorded, "git_commit": g["git_commit"],
           "made_by": {"script": "scripts/make_demo_package.py",
                       "args": {"split": a.split, "slots": slots, "models": models, "seed": a.seed}}}
    (out / "samples.json").write_text(json.dumps(man, indent=1) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.1f} MB): {len(scenes)} scenes, {len(checkpoints)} checkpoints")


if __name__ == "__main__":
    main()
