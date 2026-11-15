#!/usr/bin/env python3
"""Inventory every extracted Jacquard v2 scene (Phase 0, task 1).

For each scene it checks that all five files exist, are non-empty and readable,
that images and depth maps are 1024 x 1024, that both depth maps are float32,
and that every grasp line parses as five numbers. It computes NO depth
statistics: the audit and tau come later, from training objects only, after
the split is frozen.

Outputs (in manifest_dir from configs/paths.json):
  manifest.csv          one row per scene; paths relative to data_root
  manifest_problems.csv only the scenes with problems
  manifest_summary.json counts, scenes per object, problems by type, sha256 of
                        manifest.csv (the split will record this hash)

Usage (from the repo root):
  python scripts/make_manifest.py --limit 200      # quick test
  python scripts/make_manifest.py                  # full run, header checks
  python scripts/make_manifest.py --full           # also decode every file
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

EXPECTED_SCENES = 51601
EXPECTED_HW = (1024, 1024)
KEYS = {
    "RGB.png": "rgb",
    "mask.png": "mask",
    "grasps.txt": "grasps",
    "perfect_depth.tiff": "perfect",
    "stereo_depth.tiff": "stereo",
}
NAME_RE = re.compile(
    r"^(\d+)_(.+)_(RGB\.png|mask\.png|grasps\.txt|perfect_depth\.tiff|stereo_depth\.tiff)$"
)


class Progress:
    """Minimal text progress bar (no extra dependency). Redraws in place on a
    terminal; prints one line per 5% when stdout is a log file (e.g. nohup)."""

    def __init__(self, label, total, width=30):
        self.label, self.total, self.width = label, max(total, 1), width
        self.t0 = time.time()
        self.tty = sys.stdout.isatty()
        self.next_mark = 0  # next percentage to print in log mode

    def update(self, done):
        pct = 100 * done // self.total
        if not self.tty:
            if pct < self.next_mark and done < self.total:
                return
            self.next_mark = pct - pct % 5 + 5
        elapsed = time.time() - self.t0
        eta = elapsed / done * (self.total - done) if done else 0.0
        filled = self.width * done // self.total
        bar = "#" * filled + "-" * (self.width - filled)
        msg = (f"{self.label} [{bar}] {pct:3d}% {done}/{self.total} "
               f"{elapsed:.0f}s eta {eta:.0f}s")
        if self.tty:
            print("\r" + msg, end="\n" if done >= self.total else "", flush=True)
        else:
            print(msg, flush=True)


def discover(root):
    """Group files into scenes keyed by (part, object_id, scene_index)."""
    scenes = defaultdict(dict)
    unexpected, name_mismatch = [], []
    parts = sorted(p for p in root.glob("Jacquard_Dataset_*") if p.is_dir())
    if not parts:
        sys.exit(f"No Jacquard_Dataset_* folders under {root}")
    # Same folders in the same sorted order as before; listed up front so the
    # scan has a known total for the progress bar.
    obj_dirs = [(part_dir, d) for part_dir in parts
                for d in sorted(p for p in part_dir.iterdir() if p.is_dir())]
    bar = Progress("Scanning", len(obj_dirs))
    for i, (part_dir, obj_dir) in enumerate(obj_dirs, 1):
        for f in obj_dir.iterdir():
            m = NAME_RE.match(f.name)
            if not m:
                unexpected.append(str(f.relative_to(root)))
                continue
            idx, obj_in_name, suffix = m.groups()
            if obj_in_name != obj_dir.name:
                name_mismatch.append(str(f.relative_to(root)))
            key = (part_dir.name, obj_dir.name, int(idx))
            scenes[key][KEYS[suffix]] = f
        bar.update(i)
    return parts, scenes, unexpected, name_mismatch


def check_image(path, full):
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
        if full:
            im.load()
    return (h, w), mode


def check_tiff(path, full):
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        shape, dtype = tuple(page.shape), str(page.dtype)
        has_nan = None
        if full:
            arr = t.asarray()
            shape, dtype = arr.shape, str(arr.dtype)
            has_nan = bool(np.isnan(arr).any())
    return shape, dtype, has_nan


def check_grasps(path):
    n_lines = n_bad = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            fields = line.split(";")
            try:
                if len(fields) != 5:
                    raise ValueError
                [float(x) for x in fields]
            except ValueError:
                n_bad += 1
    return n_lines, n_bad


def check_scene(item, root, full):
    (part, obj, idx), files = item
    row = {"part": part, "object_id": obj, "scene_idx": idx}
    problems = []
    for key in KEYS.values():
        p = files.get(key)
        row[f"{key}_path"] = str(p.relative_to(root)) if p else ""
        if p is None:
            problems.append(f"missing:{key}")
            continue
        size = p.stat().st_size
        row[f"{key}_bytes"] = size
        if size == 0:
            problems.append(f"empty:{key}")
            continue
        try:
            if key in ("rgb", "mask"):
                hw, mode = check_image(p, full)
                row[f"{key}_shape"] = f"{hw[0]}x{hw[1]}"
                row[f"{key}_mode"] = mode
                if hw != EXPECTED_HW:
                    problems.append(f"shape:{key}")
            elif key in ("perfect", "stereo"):
                shape, dtype, has_nan = check_tiff(p, full)
                row[f"{key}_shape"] = "x".join(map(str, shape))
                row[f"{key}_dtype"] = dtype
                if shape[:2] != EXPECTED_HW:
                    problems.append(f"shape:{key}")
                if dtype != "float32":
                    problems.append(f"dtype:{key}")
                if has_nan:
                    problems.append(f"nan:{key}")
            else:
                n_lines, n_bad = check_grasps(p)
                row["grasp_lines"] = n_lines
                row["grasp_bad_lines"] = n_bad
                if n_lines == 0:
                    problems.append("no_grasps")
                if n_bad:
                    problems.append("bad_grasp_lines")
        except Exception as e:  # unreadable or truncated file
            problems.append(f"error:{key}:{type(e).__name__}")
    row["status"] = "ok" if not problems else "problem"
    row["problems"] = "|".join(problems)
    return row


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/paths.json")
    ap.add_argument("--full", action="store_true",
                    help="decode every image and depth map (slow, catches truncation)")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N scenes")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["data_root"]).expanduser()
    out_dir = Path(cfg["manifest_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""

    t0 = time.time()
    print(f"Scanning {root} ...", flush=True)
    parts, scenes, unexpected, name_mismatch = discover(root)
    items = sorted(scenes.items())
    if args.limit:
        items = items[: args.limit]
    print(f"Found {len(scenes)} scenes in {len(parts)} parts "
          f"({time.time() - t0:.0f}s). Checking {len(items)} "
          f"({'full decode' if args.full else 'headers'}) ...", flush=True)

    rows = []
    bar = Progress(f"Checking ({'full decode' if args.full else 'headers'})", len(items))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(lambda it: check_scene(it, root, args.full), items), 1):
            rows.append(row)
            bar.update(i)

    fields = ["part", "object_id", "scene_idx", "status", "problems", "grasp_lines",
              "grasp_bad_lines"]
    for key in KEYS.values():
        fields += [f"{key}_path", f"{key}_bytes", f"{key}_shape"]
        if key in ("rgb", "mask"):
            fields.append(f"{key}_mode")
        if key in ("perfect", "stereo"):
            fields.append(f"{key}_dtype")

    manifest = out_dir / f"manifest{tag}.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"] != "ok"]
    with open(out_dir / f"manifest_problems{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(bad)

    # An object id appearing in more than one part would break a split by folder.
    parts_per_obj = defaultdict(set)
    for part, obj, _ in scenes:
        parts_per_obj[obj].add(part)
    dup_objects = sorted(o for o, ps in parts_per_obj.items() if len(ps) > 1)
    per_obj = Counter(obj for _, obj, _ in scenes)

    summary = {
        "data_root": str(root),
        "mode": "full" if args.full else "headers",
        "limit": args.limit or None,
        "scenes_found": len(scenes),
        "scenes_expected": EXPECTED_SCENES,
        "count_matches_expected": len(scenes) == EXPECTED_SCENES,
        "scenes_checked": len(rows),
        "scenes_ok": len(rows) - len(bad),
        "scenes_with_problems": len(bad),
        "problems_by_type": dict(Counter(p for r in bad for p in r["problems"].split("|"))),
        "objects": len(per_obj),
        "objects_in_multiple_parts": len(dup_objects),
        "objects_in_multiple_parts_examples": dup_objects[:20],
        "scenes_per_object": dict(sorted(Counter(per_obj.values()).items())),
        "scenes_per_part": dict(sorted(Counter(p for p, _, _ in scenes).items())),
        "unexpected_files": len(unexpected),
        "unexpected_files_examples": unexpected[:50],
        "object_id_name_mismatches": len(name_mismatch),
        "object_id_name_mismatch_examples": name_mismatch[:50],
        "manifest_sha256": sha256(manifest),
        "seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"manifest_summary{tag}.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in (
        "scenes_found", "count_matches_expected", "scenes_checked", "scenes_ok",
        "scenes_with_problems", "problems_by_type", "objects",
        "objects_in_multiple_parts", "scenes_per_object", "unexpected_files",
        "object_id_name_mismatches", "seconds")}, indent=2))
    print(f"Wrote {manifest}")

    clean = (not bad and not unexpected and not name_mismatch
             and (args.limit or len(scenes) == EXPECTED_SCENES))
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()