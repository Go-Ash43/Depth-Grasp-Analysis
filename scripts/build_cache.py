#!/usr/bin/env python3
"""Build the 224-px preprocessing cache (SPEC_trajectory Module 1 item 2; docs/RULES_preprocessing.md).

Per scene, in the order fixed by RULES R4:
  try_load_scene (SkipBudget, R3)
  -> M_lie at 1024 from RAW D_s and D*, for tau and its 0.5x / 2x sensitivity values (SPEC "Masks")
  -> R4 hard check: clamp_truthful_changed (train: any change aborts; val/test: counted)
  -> clamp valid D_s to the frozen range -> resize_depth to 224 (NaN holes, any-rule M_sensor)
  -> telea_fill (clean A0 input, cached)  ->  D* at 224 (unclamped), RGB, object (majority)
  -> lie masks at 224, any and fraction rule, BEFORE subtracting any flag (workers apply R2)
  -> R1 status (no_object_224 / no_valid_depth_224); excluded scenes keep an all-zero slot.
Layout and reader: data/cache_layout.py. Frozen values come only from configs/frozen_day1.json.

Train only (never val/test, so they cannot inform frozen choices):
  normalisation constants  median and (p84.13 - p15.87) / 2 of ds_comp (clamped, completed D_s at
                           224, all pixels) and of dstar (for A*), 0.1-mm histogram bins;
  noise-floor confirmation background |D_s - D*| p99 at 1024 and the tau the frozen rule would pick
                           (logged only; tau stays frozen).

Usage:
  python scripts/build_cache.py --splits train --max-scenes 200 --out <scratch dir>   # smoke (~1-2 min)
  python scripts/build_cache.py --workers 8                                            # train,val,test_module
The sealed shard is built separately, for the final sealed-test run: --splits test_sealed.
"""
import argparse
import csv
import json
import logging
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.cache_layout import (INDEX_FIELDS, LAYOUT_VERSION, LIE_RULES, array_specs,  # noqa: E402
                               lie_index_fields, lie_name)
from data.jacquard import PERFECT_MAX_MM, SkipBudget, data_root, try_load_scene  # noqa: E402
from data.lies import clamp_ds, clamp_truthful_changed, lie_eff_from_224, m_lie  # noqa: E402
from data.mask_ops import (resize_depth, resize_mask_any, resize_mask_fraction,  # noqa: E402
                           resize_mask_majority, telea_fill)
from data.paths import load_paths, resolve_manifest  # noqa: E402
from data.provenance import git_fields, refuse_inside  # noqa: E402
from scripts.audit_report import quantile as _interp_quantile  # noqa: E402  (estimator that froze tau)
from scripts.audit_train import make_hists, sha256  # noqa: E402  (same histogram code as the frozen audit)

LOG = logging.getLogger("build_cache")
SPLITS = ["train", "val", "test_module", "test_sealed"]
STAT_HISTS = ("abserr_bg_1024", "ds_completed_img_224", "dstar_img_224")
TAU_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]   # audit_train defaults
TAU_FACTOR = 1.5                                                       # audit_report default
NORM_PCTS = (0.1587, 0.5, 0.8413)


def frozen_cfg(path):
    """Everything the cache depends on, read from configs/frozen_day1.json (no CLI overrides)."""
    fz = json.loads(Path(path).read_text())
    if fz["framing"] != "full_frame":
        raise SystemExit(f"framing {fz['framing']!r} not supported by this builder")
    taus = sorted({float(fz["tau_mm"]), *map(float, fz["tau_sensitivity_mm"])})
    lo, hi = map(float, fz["ds_clamp_mm"])
    return {"size": int(fz["working_resolution"]), "tau": float(fz["tau_mm"]), "taus": taus,
            "clamp": [lo, hi], "telea_radius": float(fz["telea_radius"]),
            "min_obj_px": int(fz["placement_min_obj_px_224"]),
            # histogram grid: audit binning, finer depth step for the normalisation quantiles
            "depth_max": 5000.0, "depth_step": 0.1, "serr_lim": 1000.0, "serr_step": 0.1}


def stat_hists(cfg):
    h = make_hists(cfg)
    return {k: h[k] for k in STAT_HISTS}


# ------------------------------------------------------------------ one scene
def process_scene(scene, cfg, abort_on_clamp):
    """Pure function: loaded scene -> (row, arrays or None, grasps or None, stat values).

    stat values: {hist name: 1D array} for the train-only histograms (empty dict if excluded).
    Raises RuntimeError if abort_on_clamp and the clamp changes a truthful pixel (R4, train).
    """
    S = cfg["size"]
    hw = (S, S)
    lo, hi = cfg["clamp"]
    ds_raw, dstar = scene["depth_stereo"], scene["depth_perfect"]     # mm; ds_raw NaN at holes
    ms = scene["m_sensor"]
    obj = scene["obj_mask"]
    if obj.dtype != bool:
        raise TypeError(f"obj_mask dtype {obj.dtype}, expected bool (True = object)")
    valid = ~ms.astype(bool)

    obj224 = resize_mask_majority(obj, hw)
    ms224 = resize_mask_any(ms, hw).astype(bool)
    row = {"n_grasps": int(len(scene["grasps"])), "n_grasps_dropped": int(scene["n_grasps_dropped"]),
           "obj_px_224": int(obj224.sum()), "dstar_min": float(dstar.min()), "dstar_max": float(dstar.max()),
           "ds_valid_max": float(ds_raw[valid].max()) if valid.any() else float("nan")}
    row["small_obj"] = int(row["obj_px_224"] < cfg["min_obj_px"])

    # RULES R1, judged once on clean inputs
    if row["obj_px_224"] == 0:
        return {**row, "status": "no_object_224"}, None, None, {}
    if ms224.all():
        return {**row, "status": "no_valid_depth_224"}, None, None, {}

    # M_lie from RAW values at 1024, before the clamp (R4); one per tau
    lies = {t: m_lie(ds_raw, dstar, ms, t) for t in cfg["taus"]}

    # R4 hard check and the clamp itself (detector input only)
    n_truthful = clamp_truthful_changed(ds_raw, dstar, ms, lo, hi, cfg["tau"])
    if n_truthful and abort_on_clamp:
        raise RuntimeError(f"R4: clamp {cfg['clamp']} changes {n_truthful} valid pixels with "
                           f"|D_s - D*| <= 2 tau on a train scene: cache invalid, widen the range")
    ds_cl, changed = clamp_ds(ds_raw, ms, lo, hi)
    row.update(n_clamped=int(changed.sum()), n_clamped_truthful=int(n_truthful))

    # resize, then complete (Telea at working resolution over the true holes)
    ds224, inv224 = resize_depth(ds_cl, ms, hw)
    if not np.array_equal(inv224.astype(bool), ms224):
        raise RuntimeError("resize_depth hole mask != resize_mask_any(M_sensor)")
    comp = telea_fill(ds224, inv224, radius=cfg["telea_radius"])
    if not np.isfinite(comp).all():
        raise RuntimeError("non-finite value after Telea")
    if not np.array_equal(comp[~ms224], ds224[~ms224]):
        raise RuntimeError("Telea changed a valid pixel")
    dstar224, _ = resize_depth(dstar, np.zeros_like(ms), hw)            # D*: never clamped
    rgb224 = cv2.resize(scene["rgb"], (S, S), interpolation=cv2.INTER_AREA)

    arrays = {"rgb": rgb224, "ds": ds224, "ds_comp": comp, "dstar": dstar224,
              "m_sensor": ms224.astype(np.uint8), "obj": obj224.astype(np.uint8)}
    ms224_u8 = ms224.astype(np.uint8)
    for t, lie in lies.items():
        k = f"{t:g}"
        arrays[lie_name("any", t)] = resize_mask_any(lie, hw)
        arrays[lie_name("fraction", t)] = resize_mask_fraction(lie, hw, 0.5)
        eff = lie_eff_from_224(arrays[lie_name("any", t)], ms224_u8).astype(bool)
        frac = lie_eff_from_224(arrays[lie_name("fraction", t)], ms224_u8).astype(bool)
        row[f"lie_obj_1024@{k}"] = int((lie.astype(bool) & obj).sum())
        row[f"lie_eff_obj_224@{k}"] = int((eff & obj224).sum())
        row[f"lie_frac_obj_224@{k}"] = int((frac & obj224).sum())
    row["m_sensor_obj_224"] = int((ms224 & obj224).sum())

    stats = {"abserr_bg_1024": np.abs(ds_raw[valid & ~obj] - dstar[valid & ~obj]),  # raw: noise floor
             "ds_completed_img_224": comp, "dstar_img_224": dstar224}
    return {**row, "status": "ok"}, arrays, scene["grasps"], stats


def build_chunk(items, cfg, abort_on_clamp, collect_stats):
    """Worker: (slot, obj_dir, view) list -> (rows, arrays per slot, grasps per slot, skips, hists, error).

    A failing scene does not raise here: the chunk stops, and the scenes finished before it are
    returned with error = message, so the main process records them (R3) before it aborts."""
    hists = stat_hists(cfg) if collect_stats else None
    rows, arrs, grs, skips = [], [], [], []
    for slot, obj_dir, view in items:
        part, obj_id = obj_dir.split("/", 1)
        base = {"slot": slot, "part": part, "object_id": obj_id, "view": int(view)}
        try:                                   # loader too: it raises on purpose if the drive is gone
            scene, skip = try_load_scene(obj_dir, view, root=cfg["root"])
            if skip is None:
                row, a, g, st = process_scene(scene, cfg, abort_on_clamp)
        except Exception as e:                 # fatal; name the scene, hand back what is done
            return rows, arrs, grs, skips, hists, (f"cache build failed on {obj_dir} view {view}: "
                                                   f"{type(e).__name__}: {e}")
        skips.append(skip)                     # only once the scene was loaded or skipped
        if skip is not None:
            status = "no_grasps" if skip["error_type"] == "EmptyGraspsError" else "load_skip"
            rows.append({**base, "status": status, "detail": skip["message"][:300]})
            arrs.append(None)
            grs.append(None)
            continue
        rows.append({**base, **row})
        arrs.append(a)
        grs.append(g)
        if collect_stats and st:
            for k, v in st.items():
                hists[k].add(v)
    return rows, arrs, grs, skips, hists, None


def _init_worker():
    cv2.setNumThreads(1)


# ------------------------------------------------------------------ split driver
def split_items(splits, split, manifest):
    """Fixed slot order: sorted objects, sorted ok views from the manifest."""
    objects = sorted(splits["splits"][split])
    views = defaultdict(list)
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                views[f'{r["part"]}/{r["object_id"]}'].append(int(r["scene_idx"]))
    missing = [o for o in objects if o not in views]
    if missing:
        raise SystemExit(f"{len(missing)} {split} objects have no ok scenes in the manifest, e.g. {missing[:3]}")
    return objects, [(o, v) for o in objects for v in sorted(views[o])]


def q_interp(h, q):
    """Linear interpolation inside the bin, exactly as scripts/audit_report.py (the frozen decision).
    Not Hist.quantile, which returns the bin's upper edge: comparing the two would report a tau
    disagreement caused only by the estimator."""
    return _interp_quantile((h.edges, h.counts, h.under, h.over), q)


def quantiles(h, qs):
    return {f"p{100 * q:g}": q_interp(h, q) for q in qs}


def train_stats(hists, cfg, frozen_tau):
    ds_q = quantiles(hists["ds_completed_img_224"], NORM_PCTS)
    d_q = quantiles(hists["dstar_img_224"], NORM_PCTS)

    def norm(q):
        lo, med, hi = q["p15.87"], q["p50"], q["p84.13"]
        if None in (lo, med, hi):
            return None
        return {"median_mm": med, "robust_sigma_mm": (hi - lo) / 2}

    p99 = q_interp(hists["abserr_bg_1024"], 0.99)
    pick = None if p99 is None else next((c for c in TAU_CANDIDATES if c >= TAU_FACTOR * p99), None)
    out = {"normalisation": {"rule": "(D - median) / ((p84.13 - p15.87) / 2), all pixels, train, 224",
                             "bin_mm": cfg["depth_step"],
                             "quantile": "linear interpolation in bin (audit_report.quantile)",
                             "ds_comp": norm(ds_q), "dstar": norm(d_q),
                             "ds_comp_quantiles": ds_q, "dstar_quantiles": d_q},
           "noise_floor": {"bg_abserr_p99_mm": p99, "tau_factor": TAU_FACTOR,
                           "tau_rule_pick_mm": pick, "tau_frozen_mm": frozen_tau,
                           "agrees": pick == frozen_tau, "note": "confirmation only; tau stays frozen"},
           "hist_summaries": {k: h.summary() for k, h in hists.items()}}
    if pick != frozen_tau:
        LOG.warning("noise-floor rule on full train picks tau=%s mm (p99 %.2f), frozen tau=%s: "
                    "logged only, NOT changed", pick, p99 or float("nan"), frozen_tau)
    return out


def build_split(split, args, cfg, splits, manifest, out_root, provenance):
    d = out_root / split
    objects, items = split_items(splits, split, manifest)
    n_split = len(items)
    if args.max_scenes and args.max_scenes < len(items):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(items), size=args.max_scenes, replace=False))
        items = [items[i] for i in keep]
    if d.exists() and any(d.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{d} is not empty; pass --overwrite or choose another --out")
        for f in ("COMPLETE", "summary.json", "index.csv", "grasps_1024.npy", "grasp_offsets.npy"):
            # COMPLETE first; no stale records or grasp files survive a failed rebuild
            (d / f).unlink(missing_ok=True)
    d.mkdir(parents=True, exist_ok=True)

    is_train = split == "train"
    specs = array_specs(cfg["size"], cfg["taus"])
    N = len(items)
    mm = {n: np.lib.format.open_memmap(d / f"{n}.npy", mode="w+", dtype=dt, shape=(N,) + sh)
          for n, (dt, sh) in specs.items()}            # zero-filled: excluded slots stay zero
    rows, grasps, budget = [], [None] * N, SkipBudget()
    hists = stat_hists(cfg) if is_train else None
    work = partial(build_chunk, cfg=cfg, abort_on_clamp=is_train, collect_stats=is_train)
    slotted = [(i, o, v) for i, (o, v) in enumerate(items)]
    chunks = [slotted[i:i + args.chunk] for i in range(0, N, args.chunk)]
    LOG.info("%s: %d / %d scenes of %d objects -> %s", split, N, n_split, len(objects), d)
    t0 = time.time()

    def consume(res, ci):
        r, a, g, sk, h, err = res
        for row, arr, gg in zip(r, a, g):
            i = row["slot"]
            if arr is not None:
                for n in specs:
                    mm[n][i] = arr[n]
                grasps[i] = gg
            rows.append(row)
        if is_train and err is None:
            for k in hists:
                hists[k].merge(h[k])
        abort = None
        for s in sk:                                   # R3: whole chunk in order, then the first abort
            try:
                budget.loaded() if s is None else budget.skipped(s)
            except RuntimeError as e:
                abort = abort or e
        if abort is not None:                          # a budget abort came first in scene order
            raise abort
        if err is not None:
            raise RuntimeError(err)
        if (ci + 1) % 20 == 0 or ci + 1 == len(chunks):
            el = time.time() - t0
            LOG.info("%s %d/%d  %.1f min  %.1f scenes/s  ~%.1f min left  skipped=%d", split, len(rows), N,
                     el / 60, len(rows) / el, el / len(rows) * (N - len(rows)) / 60, len(budget.records))

    error, summary, flush_err = None, None, None
    try:
        if args.workers == 0:
            _init_worker()
            for ci, ch in enumerate(chunks):
                consume(work(ch), ci)
        else:
            with Pool(args.workers, initializer=_init_worker) as pool:
                for ci, res in enumerate(pool.imap(work, chunks)):
                    consume(res, ci)
        got = sorted(r["slot"] for r in rows)          # every slot 0..N-1 exactly once, or no success
        if got != list(range(N)):
            missing = sorted(set(range(N)) - set(got))
            raise RuntimeError(f"{split}: {len(got)} records for {N} slots (missing e.g. {missing[:5]}, "
                               f"duplicates {len(got) - len(set(got))}); not complete")
        # grasps: concatenated at 1024, offsets per slot (excluded slots: none). Inside the try, so a
        # failed save is an error like any other and the summary never claims success before it.
        lens = [0 if g is None else len(g) for g in grasps]
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        gall = np.concatenate([g for g in grasps if g is not None]) if any(lens) else np.zeros((0, 5))
        np.save(d / "grasps_1024.npy", gall.astype(np.float64))
        np.save(d / "grasp_offsets.npy", off)
    except BaseException as e:          # R3/R4 abort, crash or Ctrl-C: keep the records, then re-raise
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        for a in mm.values():           # a flush failure must not skip the records nor mask an earlier error
            try:
                a.flush()
            except Exception as e:      # noqa: BLE001 - recorded below, re-raised if it is the only failure
                flush_err = flush_err or e
        if flush_err is not None:
            LOG.error("memmap flush failed: %s: %s", type(flush_err).__name__, flush_err)
            if error is None:           # flush failure alone becomes the primary failure
                error = f"flush failed: {type(flush_err).__name__}: {flush_err}"
        try:
            summary = write_records(d, split, rows, budget, cfg, N, n_split, len(objects), args, provenance,
                                    hists if is_train and error is None else None, error)
        except Exception:               # never mask the original error with a failure while saving records
            LOG.exception("failed to write index.csv / summary.json")
            if error is None:
                raise
        if error is not None:
            LOG.error("%s: build stopped after %d / %d scenes (%s); records written with complete=false",
                      split, len(rows), N, error)
    del mm
    if flush_err is not None:           # only reached when nothing else failed: no COMPLETE marker
        raise flush_err

    # success only (every output above written, summary says complete=true): the marker, last
    (d / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    LOG.info("%s done in %.1f min: %s", split, (time.time() - t0) / 60,
             json.dumps({k: summary[k] for k in ("n_ok", "excluded_by_reason", "flag_excluded_gt_0p5pct")}))
    return summary


def write_records(d, split, rows, budget, cfg, n, n_split, n_objects, args, provenance, hists, error):
    """index.csv (rows processed so far, slot order) and summary.json, on success AND on failure.
    hists=None leaves out train_stats: a partial histogram must not look like a result."""
    fields = INDEX_FIELDS + lie_index_fields(cfg["taus"])
    with open(d / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["slot"]))
    summary = split_summary(split, rows, budget, cfg, n, n_split, n_objects, args)
    summary.update(provenance)
    summary.update(complete=error is None, error=error, n_processed=len(rows))
    if hists is not None:
        summary["train_stats"] = train_stats(hists, cfg, cfg["tau"])
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def split_summary(split, rows, budget, cfg, n, n_split, n_objects, args):
    ok = [r for r in rows if r["status"] == "ok"]
    excl = [r for r in rows if r["status"] != "ok"]
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[(r["part"], r["object_id"])].append(r["status"] == "ok")
    loaded = [r for r in rows if r["status"] not in ("load_skip", "no_grasps")]
    skips = budget.summary()
    s = {
        "layout_version": LAYOUT_VERSION, "split": split, "size": cfg["size"], "taus": cfg["taus"],
        "lie_rules": list(LIE_RULES), "cfg": {k: v for k, v in cfg.items() if k != "root"},
        "smoke": bool(args.max_scenes), "n_slots": n, "n_scenes_in_split": n_split,
        "n_objects_in_split": n_objects, "n_ok": len(ok),
        "excluded_by_reason": dict(Counter(r["status"] for r in excl)),
        "excluded": [{k: r.get(k) for k in ("slot", "part", "object_id", "view", "status", "detail")}
                     for r in excl],
        "objects_all_excluded": sorted("/".join(k) for k, v in by_obj.items() if not any(v)),
        "n_small_obj_ok": sum(r["small_obj"] for r in ok),
        # bound check for perfect_to_mm (3000 mm) over every loaded scene, plus skips by type
        "dstar_range_mm": [min((r["dstar_min"] for r in loaded), default=None),
                           max((r["dstar_max"] for r in loaded), default=None)],
        "skips_by_error_type": dict(Counter(x["error_type"] for x in skips["skips"])),
        "skips_dstar_bound": sum(f"> {PERFECT_MAX_MM:.0f} mm" in x["message"] for x in skips["skips"]),
        "skip_budget": skips,
        # R4: on train any truthful change aborts the build, so this is 0 there by construction
        "clamp": {"scenes_with_clamped_px": sum(r["n_clamped"] > 0 for r in ok),
                  "n_clamped_px": sum(r["n_clamped"] for r in ok),
                  "scenes_with_truthful_clamped": sum(r["n_clamped_truthful"] > 0 for r in ok),
                  "n_truthful_clamped_px": sum(r["n_clamped_truthful"] for r in ok)},
    }
    s["flag_excluded_gt_0p5pct"] = len(excl) > 0.005 * max(n, 1)   # R1: examine before reading results
    return s


# ------------------------------------------------------------------ main
def parse_args():
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="train,val,test_module",
                    help="comma list from " + ",".join(SPLITS) + "; test_sealed only for the final sealed-test run")
    ap.add_argument("--out", default=paths["cache_dir"])
    ap.add_argument("--overwrite", action="store_true", help="rebuild a non-empty split directory")
    ap.add_argument("--seed", type=int, default=0, help="only used to pick the --max-scenes subsample")
    ap.add_argument("--max-scenes", type=int, default=0,
                    help="smoke run: random subsample per split; marked smoke, refused by open_cache")
    ap.add_argument("--workers", type=int, default=8, help="0 = in-process (debugging)")
    ap.add_argument("--chunk", type=int, default=16, help="scenes per worker task")
    ap.add_argument("--frozen", default=str(REPO / "configs" / "frozen_day1.json"))
    ap.add_argument("--splits-file", default=str(REPO / "configs" / "splits.json"))
    a = ap.parse_args()
    a.split_list = [s.strip() for s in a.splits.split(",") if s.strip()]
    bad = [s for s in a.split_list if s not in SPLITS]
    if bad or not a.split_list:
        ap.error(f"unknown split(s) {bad}; choose from {SPLITS}")
    if a.chunk <= 0:                    # range(0, N, -1) is empty: no work, yet "complete" (review 10)
        ap.error("--chunk must be > 0")
    if a.workers < 0:
        ap.error("--workers must be >= 0")
    if a.max_scenes < 0:
        ap.error("--max-scenes must be >= 0")
    if a.max_scenes and Path(a.out).expanduser().resolve() == Path(paths["cache_dir"]).expanduser().resolve():
        ap.error("--max-scenes (smoke) must not write into the production cache_dir; pass another --out")
    return a


def main():
    args = parse_args()
    droot = Path(data_root()).expanduser().resolve()
    out_root = refuse_inside(args.out, droot)
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_root / f"build_{'_'.join(args.split_list)}.log",
                                                      mode="a")])
    cfg = frozen_cfg(args.frozen)
    cfg["root"] = str(droot)
    splits_path = Path(args.splits_file)
    splits = json.loads(splits_path.read_text())
    manifest = resolve_manifest(splits["manifest"])
    m_hash = sha256(manifest)
    if m_hash != splits["manifest_sha256"]:
        raise SystemExit(f"manifest sha256 {m_hash} != splits.json's {splits['manifest_sha256']}")
    provenance = {"script": "scripts/build_cache.py", "args": {k: v for k, v in vars(args).items()},
                  **git_fields(), "frozen_path": args.frozen, "frozen_sha256": sha256(args.frozen),
                  "splits_path": str(splits_path), "splits_sha256": sha256(splits_path),
                  "manifest": str(manifest), "manifest_sha256": m_hash,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "cv2": cv2.__version__},
                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if provenance["git_dirty"]:
        LOG.warning("git tree is dirty: commit before the production build so the cache is reproducible")
    free = shutil.disk_usage(out_root).free / 1e9
    LOG.info("free space at %s: %.0f GB", out_root, free)
    for split in args.split_list:
        build_split(split, args, cfg, splits, manifest, out_root, provenance)


if __name__ == "__main__":
    main()
