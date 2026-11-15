#!/usr/bin/env python3
"""Stage A contact-zone diagnostics + train label ranges (notes/MAJOR_v1.md section 6; traps note
"Contact-zone geometry"). Reads the TRAIN cache only (read-only memmaps); geometry only, no model,
no outcome, so nothing here can be tuned on results.

1. Label ranges over EVERY train annotation row: opening and jaw at 224 (max, p99.9, p99, share
   < 4 px), per-grasp jaw/opening ratio; proposed W_NORM = max opening at 224 rounded up to 10 px
   (a scale, not a cap; written to configs/labels.json only by an approved diff).
2. Zone geometry on --n-scenes seeded ok scenes, ONE seeded random dedup annotation per scene as a
   stand-in for A0's top-1, with jaw := opening / 2 (decoder rule). Per candidate: continuous and
   in-frame union area, binary cells, empty rate (union and per jaw), frac_out, area share on the
   object, zone overlap, G0 binary-cell mismatch, G_theta / G0 / G90 on the real M_sensor at 224.
3. Stability under 8 one-pixel shifts and +-5 deg rotations: geometric change |Z xor Z'| / |Z|
   (continuous) and |dG_theta|, as median / p90 / p99 / max, overall and for zones that a flag
   touches before or after the move; share of flag-touching zones; emptiness transitions.
   Subgroups: all scenes, obj_px_224 >= 200, opening < 4 px.

Outputs in --out: per_scene.csv, summary.json, config.json, overlays/*.png (zones at real 224
resolution, coverage over RGB with M_sensor tinted). Refuses to write inside the dataset root or
the cache directory, and refuses to overwrite without --overwrite.

Usage (after the production cache is COMPLETE):
  python scripts/zone_diagnostics.py --n-scenes 2000 --seed 0     # ~5-15 min, CPU only
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import shapely

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.cache_layout import open_cache  # noqa: E402
from data.geometry import scale_grasps  # noqa: E402
from data.jacquard import data_root  # noqa: E402
from data.paths import load_paths  # noqa: E402
from data.labels import dedup_for_targets, load_labels_cfg  # noqa: E402
from data.provenance import git_fields, refuse_existing, refuse_inside  # noqa: E402
from data.zones import CANDIDATES, coverage, zone_features, zone_polygons, with_predicted_jaw  # noqa: E402

SRC = (1024, 1024)
SHIFTS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
ROTS = (-5.0, 5.0)
STATS = ("median", "p90", "p99", "max")


def dist(v):
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


# ------------------------------------------------------------------ 1. label ranges
def label_ranges(grasps_1024, size=224):
    g = np.asarray(grasps_1024, np.float64).reshape(-1, 5)
    s = size / SRC[1]
    op, jw = g[:, 3] * s, g[:, 4] * s                     # lengths scale by s (data.geometry.scale_grasps)
    ratio = jw / op
    mx = float(op.max())
    return {"n_rows": int(len(g)),
            "opening_224": {"max": mx, "p99.9": float(np.percentile(op, 99.9)), "p99": float(np.percentile(op, 99)),
                            "median": float(np.median(op)), "min": float(op.min()),
                            "share_lt_4px": float((op < 4).mean())},
            "jaw_224": {"max": float(jw.max()), "median": float(np.median(jw)), "p5": float(np.percentile(jw, 5)),
                        "min": float(jw.min()), "share_lt_2px": float((jw < 2).mean())},
            "jaw_over_opening_per_grasp": {"median": float(np.median(ratio)), "p5": float(np.percentile(ratio, 5)),
                                           "p95": float(np.percentile(ratio, 95))},
            "w_norm_proposed_px_224": float(np.ceil(mx / 10.0) * 10.0),
            "w_norm_note": "max train opening at 224 rounded up to 10 px; a scale, not a cap"}


def w_norm_check(configured, ranges):
    """Configured W_NORM vs the train-derived rule value. 'unset' while the config is null; the
    binding pre-training check belongs in train.py (step 5), from the same cache."""
    rule = ranges["w_norm_proposed_px_224"]
    if configured is None:
        status = "unset"
    else:
        status = "match" if float(configured) == float(rule) else "mismatch"
    return {"configured": configured, "rule_value": rule, "status": status,
            "rule": "max train opening at 224 rounded up to 10 px"}


def safe_out(out, production_cache, input_cache):
    """Never write into the dataset, the configured production cache, or the selected input cache."""
    return refuse_inside(out, data_root(), production_cache, input_cache)


OWNED = ("summary.json", "per_scene.csv", "config.json")


def check_outputs(out, overwrite):
    """Refuse if any owned output exists (incl. overlays/) unless overwrite; on overwrite remove ONLY
    this script's own files so a rerun never mixes old and new overlays."""
    od = out / "overlays"
    refuse_existing([out / n for n in OWNED] + [od], overwrite)
    if overwrite:
        for n in OWNED:
            (out / n).unlink(missing_ok=True)
        if od.is_dir():
            for p in od.glob("slot*.png"):
                p.unlink()
            (od / "manifest.json").unlink(missing_ok=True)


# ------------------------------------------------------------------ 2-3. one scene
def perturbed(g, dx=0.0, dy=0.0, dtheta=0.0):
    h = np.array(g, np.float64)
    h[0] += dx
    h[1] += dy
    h[2] += dtheta
    return h


def scene_stats(g224, m_sensor, obj, candidates=CANDIDATES, band_px=2.0, empty_min=0.5, g0_segments=64):
    """Per-candidate base features and perturbation records for one stand-in grasp at 224."""
    out = {}
    objf = np.asarray(obj, np.float64)
    for cand in candidates:
        base = zone_features(g224, m_sensor, cand, band_px, empty_min, g0_segments)
        z0 = shapely.union_all(zone_polygons(g224, cand, band_px))
        # geometry, not G_theta: defined whenever the zone has ANY in-frame area (the 0.5-cell empty
        # threshold applies to G_theta only, so small zones are not dropped from this distribution)
        on_obj = float((base["cov"] * objf).sum() / base["area_in"]) if base["area_in"] > 0 else float("nan")
        rec = {k: base[k] for k in ("g_theta", "g0", "g90", "empty", "area_cont", "area_in", "frac_out",
                                    "overlap_area", "cells_binary", "g0_cells_binary", "flag_area", "jaws")}
        rec.update(jaw_empty_any=any(base["jaw_empty"]), area_on_object=on_obj,
                   g0_cell_mismatch=base["g0_cells_binary"] - base["cells_binary"])
        moves = [("shift", dx, dy, 0.0) for dx, dy in SHIFTS] + [("rot", 0.0, 0.0, r) for r in ROTS]
        pert = []
        for kind, dx, dy, dt in moves:
            h = perturbed(g224, dx, dy, dt)
            f = zone_features(h, m_sensor, cand, band_px, empty_min, g0_segments)
            z1 = shapely.union_all(zone_polygons(h, cand, band_px))
            geo = float(shapely.symmetric_difference(z0, z1).area / z0.area)
            touched = base["flag_area"] > 0 or f["flag_area"] > 0     # direct contact, empty or not
            dg = abs(f["g_theta"] - base["g_theta"]) if not (base["empty"] or f["empty"]) else float("nan")
            pert.append({"kind": kind, "geo": geo, "dg": dg, "touched": bool(touched),
                         "empty_transition": bool(f["empty"] != base["empty"])})
        rec["pert"] = pert                 # no coverage map kept (~1.2 MB/scene/candidate); overlays recompute
        out[cand] = rec
    return out


def summarise(records, candidates):
    s = {}
    for cand in candidates:
        rs = [r[cand] for r in records]
        if not rs:
            s[cand] = {"n_scenes": 0}
            continue
        pert = [p for r in rs for p in r["pert"]]
        geo_shift = [p["geo"] for p in pert if p["kind"] == "shift"]
        geo_rot = [p["geo"] for p in pert if p["kind"] == "rot"]
        dg_all = [p["dg"] for p in pert]
        dg_touch = [p["dg"] for p in pert if p["touched"]]
        jaws = [j for r in rs for j in r["jaws"]]                     # 2 per grasp, pooled
        s[cand] = {
            "n_scenes": len(rs),
            "empty_rate": float(np.mean([r["empty"] for r in rs])),
            "grasp_any_jaw_empty_rate": float(np.mean([r["jaw_empty_any"] for r in rs])),
            "jaw_empty_rate": float(np.mean([j["empty"] for j in jaws])),
            "jaw_area_cont": dist([j["area_cont"] for j in jaws]),
            "jaw_area_in": dist([j["area_in"] for j in jaws]),
            "jaw_cells_binary": dist([j["cells_binary"] for j in jaws]),
            "area_cont": dist([r["area_cont"] for r in rs]), "area_in": dist([r["area_in"] for r in rs]),
            "cells_binary": dist([r["cells_binary"] for r in rs]),
            "frac_out": dist([r["frac_out"] for r in rs]),
            "area_on_object": dist([r["area_on_object"] for r in rs]),
            "overlap_area_max": float(max(r["overlap_area"] for r in rs)),
            "g0_cell_mismatch": dist([abs(r["g0_cell_mismatch"]) for r in rs]),
            "g_theta": dist([r["g_theta"] for r in rs]),
            "share_scenes_flag_in_zone": float(np.mean([r["flag_area"] > 0 for r in rs])),
            "geo_change_shift_1px": dist(geo_shift), "geo_change_rot_5deg": dist(geo_rot),
            "abs_dg_all_moves": dist(dg_all), "abs_dg_flag_touching_moves": dist(dg_touch),
            "share_moves_flag_touching": float(np.mean([p["touched"] for p in pert])),
            "n_empty_transitions": int(sum(p["empty_transition"] for p in pert)),
        }
    return s


REPORT_KEYS = ("empty_rate", "geo_change_shift_1px", "abs_dg_flag_touching_moves")


def console_report(summary):
    """Compact per-subgroup view for the console; empty subgroups are reported as such, never indexed."""
    rep = {}
    for g, s in summary["subgroups"].items():
        if s["n"] == 0:
            rep[g] = {"n": 0, "note": "no scenes in this subgroup"}
            continue
        rep[g] = {"n": s["n"], **{c: {k: v[k] for k in REPORT_KEYS}
                                  for c, v in s["candidates"].items() if v.get("n_scenes", 0) > 0}}
    return rep


# ------------------------------------------------------------------ sampling (seeded, separate streams)
def make_rngs(seed):
    """Independent streams (SeedSequence.spawn): stand-in sampling and overlay selection. Overlay
    choices can therefore never change which scenes or grasps are sampled.
    Sampling change (before any production Stage A run): stand-ins come from the first
    spawned child, not default_rng(seed) as in the first draft, so a given seed draws different scenes."""
    a, b = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


def sample_standins(ok_slots, n_scenes, get_grasps, size, rng):
    """n_scenes seeded ok slots (sorted), one seeded random dedup annotation each, jaw := opening/2,
    scaled to 224. -> list of (slot, g224)."""
    if n_scenes <= 0:
        raise ValueError("n_scenes must be > 0")
    if len(ok_slots) == 0:
        raise ValueError("no eligible (status ok) scenes")
    slots = np.sort(rng.choice(np.asarray(ok_slots), size=min(n_scenes, len(ok_slots)), replace=False))
    out = []
    for slot in slots:
        u = dedup_for_targets(get_grasps(int(slot)))
        pick = u[rng.integers(len(u))]
        g1024 = with_predicted_jaw(np.r_[pick, 0.0])               # jaw := opening / 2 (decoder rule)
        out.append((int(slot), scale_grasps(g1024, SRC, (size, size))[0]))
    return out


JAW_KEYS = ("area_cont", "area_in", "cells_binary", "empty")


def compute_stats(standins, m_sensor, obj, index, cands, band, emin, segs, log_every=0, t0=None):
    """Per-scene rows (CSV), per-scene records and subgroup lists. m_sensor / obj: arrays indexed by slot."""
    rows, recs, groups = [], [], {"all": [], "obj_ge200": [], "opening_lt4": []}
    for n, (slot, g224) in enumerate(standins):
        rec = scene_stats(g224, m_sensor[slot], obj[slot], cands, band, emin, segs)
        recs.append(rec)
        r = index[slot]
        groups["all"].append(rec)
        if str(r["small_obj"]) == "0":
            groups["obj_ge200"].append(rec)
        if g224[3] < 4:
            groups["opening_lt4"].append(rec)
        row = {"slot": int(slot), "part": r["part"], "object_id": r["object_id"], "view": r["view"],
               "obj_px_224": r["obj_px_224"], "x224": g224[0], "y224": g224[1], "theta": g224[2],
               "opening224": g224[3], "jaw224": g224[4]}
        for c in cands:
            q = rec[c]
            row.update({f"{c}:{k}": q[k] for k in ("g_theta", "g0", "g90", "empty", "area_in", "frac_out",
                                                   "area_on_object", "cells_binary", "g0_cell_mismatch")})
            for side, jaw in zip(("plus", "minus"), q["jaws"]):     # per-jaw resolution, per scene
                row.update({f"{c}:jaw_{side}_{k}": jaw[k] for k in JAW_KEYS})
            row[f"{c}:geo_shift_max"] = max(p["geo"] for p in q["pert"] if p["kind"] == "shift")
            row[f"{c}:abs_dg_max"] = max((p["dg"] for p in q["pert"] if np.isfinite(p["dg"])), default=float("nan"))
        rows.append(row)
        if log_every and (n + 1) % log_every == 0:
            print(f"{n + 1}/{len(standins)} scenes  {time.time() - (t0 or 0):.0f} s", flush=True)
    return rows, recs, groups


def select_overlays(openings, k, rng):
    """Exactly min(k, n) distinct row indices: the k // 2 smallest openings, then random others
    (backfilled, so an overlap between the two selections never yields fewer than k)."""
    n = len(openings)
    if k < 0:
        raise ValueError("n_overlays must be >= 0")
    k = min(k, n)
    smallest = list(np.argsort(np.asarray(openings, np.float64), kind="stable")[:k // 2])
    rest = np.array([i for i in range(n) if i not in set(smallest)], dtype=np.int64)
    extra = list(rng.choice(rest, size=k - len(smallest), replace=False)) if k > len(smallest) else []
    return [int(i) for i in smallest + extra]


# ------------------------------------------------------------------ overlays
def draw_overlay(rgb, m_sensor, g224, candidates, band_px, up=4):
    """Coverage recomputed here (not retained per scene); purely deterministic, no random draws."""
    panels = []
    for cand in candidates:
        img = rgb[:, :, ::-1].astype(np.float64).copy()              # BGR for cv2
        ms = m_sensor.astype(bool)
        img[ms] = img[ms] * 0.5 + np.array([0, 0, 127.0])           # red tint = M_sensor
        cov = coverage(shapely.union_all(zone_polygons(g224, cand, band_px)), m_sensor.shape)
        a = 0.7 * cov[..., None]
        img = img * (1 - a) + a * np.array([0, 255.0, 0])           # green = zone coverage
        big = cv2.resize(img.clip(0, 255).astype(np.uint8), None, fx=up, fy=up,
                         interpolation=cv2.INTER_NEAREST)
        for z in zone_polygons(g224, cand, band_px):                 # outline, pixel-centre correct
            pts = (np.asarray(z.exterior.coords)[:-1] + 0.5) * up - 0.5
            cv2.polylines(big, [np.round(pts).astype(np.int32)], True, (255, 255, 255), 1)
        c = np.round((g224[:2] + 0.5) * up - 0.5).astype(int)
        cv2.circle(big, tuple(int(v) for v in c), 3, (255, 0, 255), -1)
        cv2.putText(big, cand, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(big)
    return np.concatenate(panels, axis=1)


# ------------------------------------------------------------------ main
def parse_args():
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=paths["cache_dir"])
    ap.add_argument("--labels", default=str(REPO / "configs" / "labels.json"))
    ap.add_argument("--out", default=str(Path(paths["runs_dir"]).expanduser() / "zone_diag_stageA"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=2000)
    ap.add_argument("--n-overlays", type=int, default=12, help="half random, half the smallest openings")
    ap.add_argument("--allow-smoke", action="store_true", help="accept a smoke cache (code checks only)")
    a = ap.parse_args()
    if a.seed < 0:                                     # SeedSequence needs >= 0; fail before any cleanup
        ap.error("--seed must be >= 0")
    if a.n_scenes <= 0:
        ap.error("--n-scenes must be > 0")
    if a.n_overlays < 0:
        ap.error("--n-overlays must be >= 0")
    return a, paths


def source_provenance(cache_dir, cache_summary):
    """Identify the input cache: its own git state, smoke flag, layout, counts, summary sha256."""
    p = Path(cache_dir).expanduser() / "train" / "summary.json"
    keys = ("git_commit", "git_dirty", "smoke", "layout_version", "complete", "error", "n_ok", "n_slots",
            "n_scenes_in_split", "started")
    return {"path": str(p.parent), "summary_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            **{k: cache_summary.get(k) for k in keys}}


def main():
    args, paths = parse_args()
    out = safe_out(args.out, paths["cache_dir"], args.cache_dir)
    cfg = load_labels_cfg(args.labels)
    size, zc = cfg["size"], cfg["zones"]
    cands, band, emin, segs = zc["candidates"], zc["band_px_224"], zc["empty_min_area_cells"], zc["g0_segments"]
    cache = open_cache(args.cache_dir, "train", allow_smoke=args.allow_smoke)
    if int(cache["summary"]["size"]) != int(size):
        raise SystemExit(f"labels config size {size} != cache size {cache['summary']['size']}")
    A, idx = cache["arrays"], cache["index"]
    ok = [i for i, r in enumerate(idx) if r["status"] == "ok"]
    if not ok:                                                     # before any cleanup of old outputs
        raise SystemExit("no eligible (status ok) train scenes in the cache")
    check_outputs(out, args.overwrite)
    t0 = time.time()

    gall = np.load(Path(args.cache_dir).expanduser() / "train" / "grasps_1024.npy", mmap_mode="r")
    ranges = label_ranges(gall, size)
    wcheck = w_norm_check(cfg["targets"]["w_norm_px_224"], ranges)
    print("label ranges:", json.dumps(ranges, indent=1))
    print("W_NORM check:", json.dumps(wcheck))

    rng_standin, rng_overlay = make_rngs(args.seed)
    standins = sample_standins(ok, args.n_scenes, cache["grasps"], size, rng_standin)
    rows, recs, groups = compute_stats(standins, A["m_sensor"], A["obj"], idx, cands, band, emin, segs,
                                       log_every=200, t0=t0)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"label_ranges": ranges, "w_norm_check": wcheck, "n_scenes": len(standins), "seed": args.seed,
               "stand_in": "one seeded random dedup annotation per scene, jaw := opening/2",
               "moves": {"shifts_px": SHIFTS, "rotations_deg": list(ROTS)},
               "subgroups": {g: {"n": len(v), "candidates": summarise(v, cands)} for g, v in groups.items()},
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "config.json").write_text(json.dumps({**vars(args), **git_fields(), "labels_cfg": cfg,
                                                 "source_cache": source_provenance(args.cache_dir, cache["summary"])},
                                                indent=1, default=str))

    od = out / "overlays"
    od.mkdir(exist_ok=True)
    pick_idx = select_overlays([r["opening224"] for r in rows], args.n_overlays, rng_overlay)
    written = []
    for i in pick_idx:
        row = rows[i]
        g224 = np.array([row["x224"], row["y224"], row["theta"], row["opening224"], row["jaw224"]])
        img = draw_overlay(A["rgb"][row["slot"]], A["m_sensor"][row["slot"]], g224, cands, band)
        png = od / f"slot{row['slot']}_{row['object_id']}_{row['view']}.png"
        if not cv2.imwrite(str(png), img):
            raise OSError(f"cv2.imwrite failed for {png}")
        written.append({"file": png.name, "slot": row["slot"], "opening224": row["opening224"]})
    (od / "manifest.json").write_text(json.dumps({"seed": args.seed, "n_scenes": len(standins),
                                                  "overlays": written}, indent=1))
    print(json.dumps(console_report(summary), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
