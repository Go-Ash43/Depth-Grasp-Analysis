#!/usr/bin/env python3
"""Freeze the object-level split for Jacquard v2 (Phase 0). Run once; never regenerate.

Reads manifest.csv (from make_manifest.py), groups scenes by object, and assigns whole objects to
  train 80% / val 10% / test 10%, then test -> test_module ~60% / test_sealed ~40%.
Objects are stratified by their number of scenes (1-5), so scene proportions also land near
80/10/10 and single-scene objects cannot pile up in one split.

Determinism: object keys are sorted before shuffling with numpy default_rng(seed), so the result
depends only on the manifest contents and the seed, not on file order.

No scene is excluded here. Degenerate-scene exclusion (docs/RULES_preprocessing.md) happens later,
per scene, and never moves an object between splits.

Output: configs/splits.json with the seed, the manifest SHA-256 (cross-checked against
manifest_summary.json), the git commit, per-split object lists ("part/object_id", usable directly as
load_scene's obj_dir) and counts. Refuses to overwrite an existing splits.json without --overwrite.

Usage (from anywhere):
  python scripts/make_split.py                # seed 0
  python scripts/make_split.py --dry-run      # print counts, write nothing
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.jacquard import data_root  # noqa: E402  (reads configs/paths.json only)
from data.provenance import refuse_inside  # noqa: E402

FRACS = {"train": 0.8, "val": 0.1}          # test = the rest
MODULE_FRAC = 0.6                           # of test objects -> test_module; rest -> test_sealed
SPLITS = ["train", "val", "test_module", "test_sealed"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    """Commit hash and whether the working tree has uncommitted changes (None if not a repo)."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def allocate(keys, rng):
    """Shuffle one stratum's object keys and cut them into the four splits."""
    keys = sorted(keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(FRACS["train"] * n))
    n_val = int(round(FRACS["val"] * n))
    n_val = min(n_val, n - n_train)
    test = keys[n_train + n_val:]
    n_mod = int(round(MODULE_FRAC * len(test)))
    return {"train": keys[:n_train], "val": keys[n_train:n_train + n_val],
            "test_module": test[:n_mod], "test_sealed": test[n_mod:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "paths.json"))
    ap.add_argument("--manifest", default="", help="default: <manifest_dir>/manifest.csv")
    ap.add_argument("--out", default=str(REPO / "configs" / "splits.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing splits.json (the split is meant to be frozen)")
    ap.add_argument("--dry-run", action="store_true", help="print counts only, write nothing")
    args = ap.parse_args()

    cfg_paths = json.loads(Path(args.config).read_text())
    out = refuse_inside(args.out, cfg_paths["data_root"], data_root())   # never write on the dataset drive
    if out.exists() and not args.overwrite and not args.dry_run:
        sys.exit(f"{out} already exists. The split is frozen; pass --overwrite only if you mean it.")

    cfg = json.loads(Path(args.config).read_text())
    mdir = Path(cfg["manifest_dir"]).expanduser()
    manifest = Path(args.manifest).expanduser() if args.manifest else mdir / "manifest.csv"
    m_hash = sha256(manifest)

    # Cross-check against the summary written by make_manifest.py (guards against a stale/other file).
    summary_path = manifest.with_name(manifest.name.replace("manifest", "manifest_summary", 1)
                                      .replace(".csv", ".json"))
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text()).get("manifest_sha256")
        if recorded != m_hash:
            sys.exit(f"manifest SHA-256 {m_hash[:12]}... does not match {summary_path.name} "
                     f"({str(recorded)[:12]}...). Use the manifest the summary describes.")
    else:
        print(f"warning: {summary_path} not found; hash not cross-checked")

    # Group scenes by object. Object IDs are unique across parts (checked by make_manifest).
    scenes = defaultdict(list)
    part_of = {}
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            obj = r["object_id"]
            if obj in part_of and part_of[obj] != r["part"]:
                sys.exit(f"object {obj} appears in parts {part_of[obj]} and {r['part']}")
            part_of[obj] = r["part"]
            scenes[obj].append(int(r["scene_idx"]))
    key = {obj: f"{part_of[obj]}/{obj}" for obj in scenes}

    # Stratify by scenes per object; one generator, strata visited in sorted order.
    strata = defaultdict(list)
    for obj, s in scenes.items():
        strata[len(s)].append(key[obj])
    rng = np.random.default_rng(args.seed)
    split = {name: [] for name in SPLITS}
    for k in sorted(strata):
        for name, keys in allocate(strata[k], rng).items():
            split[name].extend(keys)
    for name in SPLITS:
        split[name].sort()

    # Integrity: disjoint, complete.
    all_keys = [k for name in SPLITS for k in split[name]]
    if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(key.values()):
        sys.exit("internal error: splits are not a partition of the objects")

    n_scenes = {k: len(scenes[k.split("/", 1)[1]]) for k in key.values()}
    tot_obj, tot_sc = len(all_keys), sum(n_scenes.values())
    counts = {}
    for name in SPLITS:
        o = len(split[name])
        sc = sum(n_scenes[k] for k in split[name])
        counts[name] = {"objects": o, "scenes": sc,
                        "objects_pct": round(100 * o / tot_obj, 2),
                        "scenes_pct": round(100 * sc / tot_sc, 2),
                        "scenes_per_object": dict(sorted(Counter(n_scenes[k] for k in split[name]).items()))}

    print(f"manifest {manifest} sha256 {m_hash[:12]}...  objects {tot_obj}  scenes {tot_sc}  seed {args.seed}")
    print(f"{'split':12s} {'objects':>8s} {'%':>6s} {'scenes':>8s} {'%':>6s}  scenes/object")
    for name in SPLITS:
        c = counts[name]
        print(f"{name:12s} {c['objects']:8d} {c['objects_pct']:6.2f} {c['scenes']:8d} {c['scenes_pct']:6.2f}  "
              f"{c['scenes_per_object']}")
    if args.dry_run:
        print("dry run: nothing written")
        return

    head, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes in the repo; commit first so the split points at a clean commit")
    result = {
        "seed": args.seed,
        "fractions": {"train": FRACS["train"], "val": FRACS["val"],
                      "test": round(1 - FRACS["train"] - FRACS["val"], 6),
                      "test_module_of_test": MODULE_FRAC},
        "stratified_by": "scenes per object",
        "manifest": str(manifest),
        "manifest_sha256": m_hash,
        "git_commit": head,
        "git_dirty": dirty,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": counts,
        "splits": split,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}. Commit it now; it is frozen from here on.")


if __name__ == "__main__":
    main()