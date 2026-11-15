#!/usr/bin/env python3
"""Training-split audit (Phase 0): the measurements needed before the 224-px cache and any training.

Reads ONLY the frozen training objects (configs/splits.json -> "train"; scene views from the manifest
it was built on). Measures, per scene and pooled:

  1. Depth ranges / units: D* (all, object), valid D_s (all, object) at 1024; at 224 (model
     resolution) D*, valid D_s and Telea-completed D_s -> candidates for FIXED depth normalisation
     (traps: per-image statistics are forbidden).
  2. Sensor holes: M_sensor rate over the image and on the object, at 1024 and at 224
     (any-rule, mask_ops.resize_mask_any; object at 224 by majority, resize_mask_majority),
     per scene and pooled (invalid object pixels / object pixels).
  3. Stereo error |D_s - D*| on valid object (and background) pixels at 1024, plus a tau grid:
     M_lie = 1[M_sensor = 0 and |D_s - D*| > tau]           (SPEC_trajectory "Masks and depth")
     reported at 1024 and as RULES R2 lie_eff_224 = resize_any(M_lie) & ~resize_any(M_sensor)
     (no M_synth in a clean audit), plus the R2 fraction rule (>= 0.5).
  4. Object size at 224: percentiles, counts below size thresholds, disappearing objects, and a
     crop table for full frame vs crop (small_objects.md). For each crop size c (at 1024) and two
     placements: "ctr" = centred in the frame; "obj" = centred on the object's bbox, clamped to the
     frame. "obj" uses the ground-truth object mask, so adopting it would need a test-time way to
     locate the object: it is reported for the decision only. Per crop: object px at 224 after the
     crop (exact: crop, then majority resize; cut-off parts are lost), whether the whole object /
     all grasp centres / all grasp corners stay inside, and the joint fraction. Crop rows are
     computed for every loaded scene, including R1-excluded ones (a crop may recover an object that
     vanishes at full frame).
  5. Unusable scenes: RULES R1 reason codes (load_skip, no_grasps, no_object_224,
     no_valid_depth_224) and the R3 SkipBudget.

Nothing is decided here: tau, normalisation constants and the crop / small-object rule are frozen by
hand from summary.json afterwards.

Pooled quantiles come from fixed-bin histograms (exact counts, quantile = upper edge of the bin
holding it; None if it falls outside the histogram range): depth bins --depth-step mm, |error| bins
100 per decade from 1e-3 mm, signed error bins --serr-step mm. n / mean / std / min / max are exact.
Histograms, moments and pooled rates include only scenes that pass R1 (what training will see).
R1-excluded scenes keep their size, sensor-rate and crop columns in per_scene.csv, but no depth or
error columns; load skips keep only the reason.

Errors raised inside the per-scene audit are NOT caught: the loader already turns data errors into
skip records, so anything left is a code bug or a broken data contract and must stop the run.
On any stop, per_scene.csv and a summary.json marked "complete": false are still written.

Outputs in --out:
  config.json      args, git state, splits/manifest hashes, library versions
  per_scene.csv    one row per audited train scene (status = ok or an R1 reason code)
  summary.json     pooled histograms' summaries, pooled hole rates, per-scene percentiles, tau table,
                   crop table, size thresholds, exclusion list, SkipBudget summary, warnings
  histograms.npz   raw bin edges + counts (to recompute anything without re-reading the drive)
  audit.log

Usage:
  python scripts/audit_train.py --max-scenes 200 --workers 8 --out <dir>   # smoke run (~1 min)
  python scripts/audit_train.py --workers 8                               # full train split
"""
import argparse
import csv
import hashlib
import json
import logging
import platform
import subprocess
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

from data.jacquard import IMG_SIZE, SkipBudget, data_root, try_load_scene  # noqa: E402
from data.paths import load_paths, resolve_manifest  # noqa: E402
from data.mask_ops import (resize_depth, resize_mask_any, resize_mask_fraction,  # noqa: E402
                           resize_mask_majority, telea_fill)

PCTS = (0, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 100)
OUTPUT_FILES = ["config.json", "per_scene.csv", "summary.json", "histograms.npz", "audit.log",
                "freeze_candidates.json"]   # written later by audit_report.py; stale after any rerun
LOG = logging.getLogger("audit_train")


def fmt(v, spec=".3g"):
    return "n/a" if v is None else format(v, spec)


# ------------------------------------------------------------------ histograms
class Hist:
    """Fixed-edge histogram with exact n/sum/sumsq/min/max. Mergeable across workers, so pooled
    quantiles over ~40k scenes need no per-pixel storage. Values < edges[0] -> under,
    >= edges[-1] -> over (both still counted in n and the moments). Uniform edges use direct
    indexing (fast path); others use searchsorted."""

    def __init__(self, edges):
        self.edges = np.asarray(edges, np.float64)
        if self.edges.ndim != 1 or len(self.edges) < 2 or not (np.diff(self.edges) > 0).all():
            raise ValueError("edges must be 1D, strictly increasing, len >= 2")
        nb = len(self.edges) - 1
        lo, hi = self.edges[0], self.edges[-1]
        step = (hi - lo) / nb
        uniform = np.allclose(self.edges, lo + step * np.arange(nb + 1), rtol=0, atol=1e-9 * (hi - lo))
        self.step = float(step) if uniform else None
        self.counts = np.zeros(len(self.edges) - 1, np.int64)
        self.under = self.over = self.n = 0
        self.s = self.ss = 0.0
        self.min, self.max = np.inf, -np.inf

    def add(self, x):
        x = np.asarray(x, np.float64).ravel()
        if x.size == 0:
            return
        if not np.isfinite(x).all():
            raise ValueError("non-finite value reached a histogram (mask/NaN handling bug)")
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float(np.dot(x, x))
        self.min, self.max = min(self.min, float(x.min())), max(self.max, float(x.max()))
        idx = self.index(x)
        nb = len(self.counts)
        self.under += int((idx < 0).sum())
        self.over += int((idx >= nb).sum())
        ok = (idx >= 0) & (idx < nb)
        self.counts += np.bincount(idx[ok], minlength=nb)

    def index(self, x):
        """Bin index k with edges[k] <= x < edges[k+1]; -1 below range, nb at/above the last edge.
        Same result as searchsorted(edges, x, 'right') - 1 (tests/test_audit_train.py). The uniform fast
        path divides by the step, then corrects the at most one-bin float error against the ACTUAL
        edges, so values on an edge (0, +-serr_lim, linspace edges) land exactly where searchsorted puts them."""
        e = self.edges
        nb = len(e) - 1
        if self.step is None:
            return np.searchsorted(e, x, side="right") - 1
        k = np.clip(np.floor((x - e[0]) / self.step), -1, nb).astype(np.int64)
        dn = (k >= 0) & (x < e[np.clip(k, 0, nb)])
        k[dn] -= 1
        up = (k < nb) & (x >= e[np.clip(k + 1, 0, nb)])
        k[up] += 1
        return k

    def merge(self, other):
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("cannot merge histograms with different edges")
        self.counts += other.counts
        self.under += other.under
        self.over += other.over
        self.n += other.n
        self.s += other.s
        self.ss += other.ss
        self.min, self.max = min(self.min, other.min), max(self.max, other.max)

    def quantile(self, q):
        """Upper edge of the bin containing the q-quantile; exact min/max at q = 0/1; None if the
        quantile falls below or above the histogram range (not estimable from the bins)."""
        if self.n == 0:
            return None
        if q <= 0:
            return self.min
        if q >= 1:
            return self.max
        target = q * self.n
        if target <= self.under or target > self.n - self.over:
            return None
        cum = np.cumsum(self.counts)
        i = int(np.searchsorted(cum, target - self.under, side="left"))
        return float(self.edges[min(i, len(self.counts) - 1) + 1])

    def summary(self):
        if self.n == 0:
            return {"n": 0}
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        return {"n": int(self.n), "mean": mean, "std": float(np.sqrt(var)), "min": self.min, "max": self.max,
                "n_under_range": int(self.under), "n_over_range": int(self.over),
                "range": [float(self.edges[0]), float(self.edges[-1])],
                "quantiles": {f"p{p:g}": self.quantile(p / 100) for p in PCTS}}


def make_hists(cfg):
    dmax, dstep = cfg["depth_max"], cfg["depth_step"]
    depth = np.linspace(0.0, dmax, int(round(dmax / dstep)) + 1)
    abserr = np.concatenate([[0.0], np.logspace(-3, 4, 701)])          # 0, 1e-3 .. 1e4 mm, 100/decade
    lim, sstep = cfg["serr_lim"], cfg["serr_step"]
    serr = np.linspace(-lim, lim, int(round(2 * lim / sstep)) + 1)
    names_depth = ["dstar_img_1024", "dstar_obj_1024", "ds_valid_img_1024", "ds_valid_obj_1024",
                   "dstar_img_224", "ds_valid_img_224", "ds_completed_img_224", "ds_completed_obj_224"]
    h = {n: Hist(depth) for n in names_depth}
    h.update({n: Hist(abserr) for n in ["abserr_obj_1024", "abserr_bg_1024", "telea_abserr_objhole_224"]})
    h.update({n: Hist(serr) for n in ["serr_obj_1024", "serr_bg_1024"]})
    return h


# ------------------------------------------------------------------ per-scene audit
def _med(x):
    return float(np.median(x)) if x.size else float("nan")


def _p(x, q):
    return float(np.percentile(x, q)) if x.size else float("nan")


def _inside(pts, lo_x, lo_y, c):
    """All (..., 2) x, y points inside the crop of pixel indices [lo, lo + c) (pixel-centre convention)."""
    x, y = pts[..., 0], pts[..., 1]
    return bool(((x >= lo_x - 0.5) & (x < lo_x + c - 0.5) & (y >= lo_y - 0.5) & (y < lo_y + c - 0.5)).all())


def crop_rows(obj, n_obj, bbox, centres, corners, cfg):
    """Crop-vs-full-frame measurements (small_objects.md) for every crop size and placement."""
    S = cfg["size"]
    x0, x1, y0, y1 = bbox
    out = {}
    for c in cfg["crops"]:
        lo_c = (IMG_SIZE - c) // 2
        lo_ox = int(np.clip(np.floor((x0 + x1 + 1) / 2 - c / 2), 0, IMG_SIZE - c))
        lo_oy = int(np.clip(np.floor((y0 + y1 + 1) / 2 - c / 2), 0, IMG_SIZE - c))
        for mode, lx, ly in (("ctr", lo_c, lo_c), ("obj", lo_ox, lo_oy)):
            oc = obj[ly:ly + c, lx:lx + c]
            k = f"crop{c}_{mode}"
            out[f"{k}_objpx224"] = int(resize_mask_majority(oc, (S, S)).sum())   # clipped area, exact rule
            out[f"{k}_obj_in"] = int(int(oc.sum()) == n_obj)
            out[f"{k}_ctr_in"] = int(_inside(centres, lx, ly, c))
            out[f"{k}_cor_in"] = int(_inside(corners, lx, ly, c))
    return out


def audit_scene(scene, hists, cfg):
    """All per-scene measurements. Returns the per_scene.csv row; adds to `hists` only if R1 passes."""
    S = cfg["size"]
    hw = (S, S)
    sc = S / IMG_SIZE
    dstar, ds = scene["depth_perfect"], scene["depth_stereo"]       # mm; ds is NaN where m_sensor = 1
    ms = scene["m_sensor"].astype(bool)                              # 1 = invalid
    valid = ~ms
    obj = scene["obj_mask"]
    if obj.dtype != bool:                                            # loader contract: True = object
        raise TypeError(f"obj_mask dtype {obj.dtype}, expected bool")
    bg = ~obj
    g = scene["grasps"]                                              # (N,5) x, y, theta_deg, opening, jaw

    obj224 = resize_mask_majority(obj, hw)                           # RULES R1: object at 224 by majority
    ms224 = resize_mask_any(ms, hw).astype(bool)                     # any-rule, as the model sees it
    n_obj, n_obj224 = int(obj.sum()), int(obj224.sum())              # loader guarantees n_obj >= 1
    ys, xs = np.nonzero(obj)
    bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    corners = scene["grasp_corners"]                                 # (N,4,2) in (x, y) order
    if not np.allclose(corners.mean(axis=1), g[:, 0:2], atol=1e-3):   # guards an (x, y) / (y, x) mix-up
        raise RuntimeError("grasp_corners centroid != grasp centre: corner coordinate order changed?")
    row = {
        "status": "ok",
        "n_grasp_rows": int(len(g)),
        "n_grasp_unique": int(len(np.unique(g[:, :4], axis=0))),     # jaw-size duplicates collapsed
        "obj_px_1024": n_obj, "obj_px_224": n_obj224,
        "obj_x0": bbox[0], "obj_x1": bbox[1], "obj_y0": bbox[2], "obj_y1": bbox[3],
        "opening_224_median": _med(g[:, 3] * sc), "jaw_224_median": _med(g[:, 4] * sc),
        "jaw_224_min": float((g[:, 4] * sc).min()),
        "sensor_img_1024": float(ms.mean()), "sensor_obj_1024": float(ms[obj].mean()),
        "sensor_img_224": float(ms224.mean()),
        "sensor_obj_224": float(ms224[obj224].mean()) if n_obj224 else float("nan"),
        "n_valid_1024": int(valid.sum()), "n_valid_obj_1024": int((valid & obj).sum()),
        "n_valid_224": int((~ms224).sum()), "n_valid_obj_224": int((obj224 & ~ms224).sum()),
    }
    if n_obj224:
        y2, x2 = np.nonzero(obj224)
        row["obj_bbox_w_224"] = int(x2.max() - x2.min() + 1)
        row["obj_bbox_h_224"] = int(y2.max() - y2.min() + 1)
    row.update(crop_rows(obj, n_obj, bbox, g[:, 0:2], corners, cfg))  # before R1: crops may recover objects

    # RULES R1, judged once on clean inputs
    if n_obj224 == 0:
        row["status"] = "no_object_224"
        return row
    if ms224.all():
        row["status"] = "no_valid_depth_224"
        return row

    # ---- 1. depth ranges / units (1024)
    hists["dstar_img_1024"].add(dstar)
    hists["dstar_obj_1024"].add(dstar[obj])
    hists["ds_valid_img_1024"].add(ds[valid])
    hists["ds_valid_obj_1024"].add(ds[valid & obj])
    table = _med(dstar[bg])
    row.update({
        "dstar_min": float(dstar.min()), "dstar_max": float(dstar.max()),
        "ds_valid_min": float(ds[valid].min()), "ds_valid_max": float(ds[valid].max()),
        "dstar_bg_median": table, "dstar_obj_median": _med(dstar[obj]),
        "obj_height_mm": table - float(dstar[obj].min()),            # rough: table median - nearest object point
    })

    # ---- 3. stereo error on valid pixels (1024)
    err = ds - dstar                                                  # NaN at holes
    e_obj, e_bg = err[valid & obj], err[valid & bg]
    a_obj, a_bg = np.abs(e_obj), np.abs(e_bg)
    hists["abserr_obj_1024"].add(a_obj)
    hists["abserr_bg_1024"].add(a_bg)
    hists["serr_obj_1024"].add(e_obj)
    hists["serr_bg_1024"].add(e_bg)
    row.update({
        "err_obj_abs_median": _med(a_obj), "err_obj_abs_p95": _p(a_obj, 95),
        "err_obj_signed_mean": float(e_obj.mean()) if e_obj.size else float("nan"),
        "err_bg_abs_median": _med(a_bg), "err_bg_abs_p95": _p(a_bg, 95),
    })
    abs_err0 = np.where(valid, np.abs(np.nan_to_num(err, nan=0.0)), 0.0)  # holes -> 0, never > tau
    for t in cfg["taus"]:
        lie = abs_err0 > t                                            # M_lie (valid by construction)
        lie_eff = resize_mask_any(lie, hw).astype(bool) & ~ms224      # RULES R2 lie_eff_224 (no M_synth)
        lie_frac = resize_mask_fraction(lie, hw, 0.5).astype(bool) & ~ms224   # R2 fraction-rule sensitivity
        k = f"{t:g}"
        row[f"lie_img_1024@{k}"] = int(lie.sum())
        row[f"lie_obj_1024@{k}"] = int((lie & obj).sum())
        row[f"lie_eff_img_224@{k}"] = int(lie_eff.sum())
        row[f"lie_eff_obj_224@{k}"] = int((lie_eff & obj224).sum())
        row[f"lie_frac_obj_224@{k}"] = int((lie_frac & obj224).sum())

    # ---- 1b. depth at model resolution: normalisation candidates
    ds224, inv224 = resize_depth(ds, ms, hw)
    if not np.array_equal(inv224.astype(bool), ms224):
        raise RuntimeError("resize_depth hole mask != resize_mask_any(M_sensor): mask_ops inconsistency")
    comp = telea_fill(ds224, inv224, radius=cfg["telea_radius"])     # A0 clean input before normalisation
    dstar224, _ = resize_depth(dstar, np.zeros_like(ms), hw)
    hists["dstar_img_224"].add(dstar224)
    hists["ds_valid_img_224"].add(ds224[~ms224])
    hists["ds_completed_img_224"].add(comp)
    hists["ds_completed_obj_224"].add(comp[obj224])
    filled_obj = obj224 & ms224
    te = np.abs(comp[filled_obj] - dstar224[filled_obj])              # how wrong Telea is inside object holes
    hists["telea_abserr_objhole_224"].add(te)
    row["telea_objhole_abserr_median_224"] = _med(te)
    row["comp_min_224"], row["comp_max_224"] = float(comp.min()), float(comp.max())
    return row


def audit_chunk(items, cfg):
    """Worker: audit a chunk of (obj_dir, view) in order. Returns rows, statuses and merged hists."""
    hists = make_hists(cfg)
    rows, statuses = [], []
    for obj_dir, view in items:
        part, obj_id = obj_dir.split("/", 1)
        base = {"part": part, "object_id": obj_id, "view": int(view)}
        scene, skip = try_load_scene(obj_dir, view, root=cfg["root"])
        if skip is not None:
            reason = "no_grasps" if skip["error_type"] == "EmptyGraspsError" else "load_skip"
            statuses.append(skip)
            rows.append({**base, "status": reason, "detail": skip["message"][:300]})
            continue
        statuses.append(None)
        sh = make_hists(cfg)                     # scene-local, merged only after the scene completes
        try:
            row = audit_scene(scene, sh, cfg)
        except Exception as e:                   # still fatal: just name the scene in the traceback
            raise RuntimeError(f"audit failed on {obj_dir} view {view}: {type(e).__name__}: {e}") from e
        for k in hists:
            hists[k].merge(sh[k])
        rows.append({**base, **row})
    return rows, statuses, hists


def _init_worker():
    cv2.setNumThreads(1)          # one process per core; avoid oversubscription


# ------------------------------------------------------------------ summary
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def pct(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), **{f"p{p:g}": float(np.percentile(v, p)) for p in PCTS}}


def _mean(bools):
    return float(np.mean(bools)) if len(bools) else None


PER_SCENE_COLS = ["obj_px_1024", "obj_px_224", "obj_bbox_w_224", "obj_bbox_h_224",
                  "sensor_img_1024", "sensor_obj_1024", "sensor_img_224", "sensor_obj_224",
                  "dstar_bg_median", "dstar_obj_median", "obj_height_mm", "dstar_min", "dstar_max",
                  "ds_valid_min", "ds_valid_max", "comp_min_224", "comp_max_224",
                  "err_obj_abs_median", "err_obj_abs_p95", "err_obj_signed_mean",
                  "err_bg_abs_median", "err_bg_abs_p95", "telea_objhole_abserr_median_224",
                  "n_grasp_rows", "n_grasp_unique", "opening_224_median", "jaw_224_median", "jaw_224_min"]


def summarise(rows, hists, budget, cfg, n_objects_split, n_scenes_split):
    ok = [r for r in rows if r["status"] == "ok"]
    loaded = [r for r in rows if r["status"] not in ("load_skip", "no_grasps")]   # have crop columns
    col = lambda name, rs=ok: [r.get(name) for r in rs]   # noqa: E731
    warnings = []

    # ---- exclusions (R1) and objects that drop out entirely
    excluded = [{"part": r["part"], "object_id": r["object_id"], "view": r["view"], "reason": r["status"]}
                for r in rows if r["status"] != "ok"]
    excl_frac = len(excluded) / max(len(rows), 1)
    if excl_frac > 0.005:
        warnings.append(f"R1: {100 * excl_frac:.2f}% of audited train scenes excluded (> 0.5%): examine before use")
    if len(rows) == n_scenes_split:                  # actual coverage, not what was planned
        by_obj = defaultdict(list)
        for r in rows:
            by_obj[f'{r["part"]}/{r["object_id"]}'].append(r["status"] == "ok")
        dropped_objects = sorted(k for k, v in by_obj.items() if not any(v))
    else:
        dropped_objects = "not computed: not every scene of the split was audited (subsample or stopped run)"

    # ---- pooled hole rates (invalid pixels / pixels; plan's "about 1% of object pixels" gate)
    tot = lambda k: sum(r[k] for r in ok)   # noqa: E731
    pooled_holes = {}
    if ok:
        pooled_holes = {
            "obj_1024": 1 - tot("n_valid_obj_1024") / tot("obj_px_1024"),
            "obj_224": 1 - tot("n_valid_obj_224") / tot("obj_px_224"),
            "img_1024": 1 - tot("n_valid_1024") / (len(ok) * IMG_SIZE ** 2),
            "img_224": 1 - tot("n_valid_224") / (len(ok) * cfg["size"] ** 2),
        }

    # ---- tau table (pooled over included scenes)
    tau_table = []
    for t in cfg["taus"]:
        k = f"{t:g}"
        s = lambda name: sum(r[f"{name}@{k}"] for r in ok)   # noqa: E731
        per_scene_obj = [r[f"lie_obj_1024@{k}"] / r["n_valid_obj_1024"] for r in ok if r["n_valid_obj_1024"]]
        per_scene_eff = [r[f"lie_eff_obj_224@{k}"] / r["n_valid_obj_224"] for r in ok if r["n_valid_obj_224"]]
        tau_table.append({
            "tau_mm": t, "is_candidate": t in cfg["tau_candidates"],
            "lie_rate_img_1024": s("lie_img_1024") / max(tot("n_valid_1024"), 1) if ok else None,
            "lie_rate_obj_1024": s("lie_obj_1024") / max(tot("n_valid_obj_1024"), 1) if ok else None,
            "lie_eff_rate_img_224": s("lie_eff_img_224") / max(tot("n_valid_224"), 1) if ok else None,
            "lie_eff_rate_obj_224": s("lie_eff_obj_224") / max(tot("n_valid_obj_224"), 1) if ok else None,
            "lie_frac_rate_obj_224": s("lie_frac_obj_224") / max(tot("n_valid_obj_224"), 1) if ok else None,
            "per_scene_lie_rate_obj_1024": pct(per_scene_obj),
            "per_scene_lie_eff_rate_obj_224": pct(per_scene_eff),
            "frac_scenes_any_obj_lie_1024": _mean([r[f"lie_obj_1024@{k}"] > 0 for r in ok]),
            "frac_scenes_any_obj_lie_eff_224": _mean([r[f"lie_eff_obj_224@{k}"] > 0 for r in ok]),
        })

    # ---- object size at 224 (full frame, R1-included scenes) and small-object thresholds
    size_thr = []
    for t in cfg["small_thresholds"]:
        small = [r for r in ok if r["obj_px_224"] < t]
        size_thr.append({"obj_px_224_below": t, "n_scenes": len(small), "frac_scenes": len(small) / max(len(ok), 1),
                         "n_objects": len({(r["part"], r["object_id"]) for r in small})})

    # ---- crop table over every loaded scene (R1-excluded included: a crop may recover them)
    crop_table = []
    for c in cfg["crops"]:
        for mode in ("ctr", "obj"):
            k = f"crop{c}_{mode}"
            px = col(f"{k}_objpx224", loaded)
            crop_table.append({
                "crop_px_at_1024": c, "placement": mode, "n_scenes": len(loaded),
                "frac_object_inside": _mean(col(f"{k}_obj_in", loaded)),
                "frac_grasp_centres_inside": _mean(col(f"{k}_ctr_in", loaded)),
                "frac_grasp_corners_inside": _mean(col(f"{k}_cor_in", loaded)),
                "frac_object_and_corners_inside": _mean([a and b for a, b in
                                                         zip(col(f"{k}_obj_in", loaded), col(f"{k}_cor_in", loaded))]),
                "frac_object_and_centres_inside": _mean([a and b for a, b in
                                                         zip(col(f"{k}_obj_in", loaded), col(f"{k}_ctr_in", loaded))]),
                "n_objpx224_zero": sum(p == 0 for p in px),
                "objpx224": pct(px),
                "n_objpx224_below": {str(t): sum(p < t for p in px) for t in cfg["small_thresholds"]},
            })

    # ---- sanity / unit warnings
    hs = {k: h.summary() for k, h in hists.items()}
    med = hs["dstar_img_1024"].get("quantiles", {}).get("p50")
    if med is not None and not (300 <= med <= 3000):
        warnings.append(f"D* median {med:.1f} mm is outside 300..3000 mm: check units")
    for r in crop_table:                             # sanity: the full-frame "crop" must contain every object
        if r["crop_px_at_1024"] == IMG_SIZE and r["frac_object_inside"] not in (None, 1.0):
            warnings.append(f"crop1024/{r['placement']}: object-inside fraction {r['frac_object_inside']} != 1")
    for k, h in hists.items():
        if h.n and (h.under or h.over):
            warnings.append(f"{k}: {h.under} values below / {h.over} above histogram range {hs[k]['range']}")

    return {
        "n_objects_in_split": n_objects_split, "n_scenes_in_split": n_scenes_split,
        "n_scenes_audited": len(rows), "n_scenes_loaded": len(loaded), "n_scenes_included": len(ok),
        "exclusions": {"counts": dict(Counter(e["reason"] for e in excluded)), "frac": excl_frac,
                       "scenes": excluded, "objects_dropped_entirely": dropped_objects},
        "skip_budget": budget.summary(),
        "pooled": hs,
        "pooled_hole_rate": pooled_holes,
        "per_scene": {c: pct(col(c)) for c in PER_SCENE_COLS},
        "object_fully_invalid": {
            "n_scenes_1024": sum(r["sensor_obj_1024"] == 1.0 for r in ok),
            "n_scenes_224": sum(r["sensor_obj_224"] == 1.0 for r in ok)},
        "tau_candidates": cfg["tau_candidates"],
        "tau_table": tau_table,
        "small_object_thresholds": size_thr,
        "crop_table": crop_table,
        "warnings": warnings,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_outputs(out, rows, hists, budget, cfg, config, n_objects, n_scenes, complete, t0):
    """Write per_scene.csv, histograms.npz and summary.json. Called on success AND on any stop."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "histograms.npz",
                        **{f"{k}__edges": h.edges for k, h in hists.items()},
                        **{f"{k}__counts": h.counts for k, h in hists.items()},
                        **{f"{k}__under_over": np.array([h.under, h.over]) for k, h in hists.items()})
    summary = summarise(rows, hists, budget, cfg, n_objects, n_scenes)
    summary["complete"] = complete
    summary["config"] = config
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["runtime_min"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    return summary


def report(summary):
    P = summary["pooled"]
    q = lambda k, p: P[k].get("quantiles", {}).get(p)   # noqa: E731
    LOG.info("included %d / %d scenes; exclusions %s; skips %d", summary["n_scenes_included"],
             summary["n_scenes_audited"], summary["exclusions"]["counts"], summary["skip_budget"]["n_skipped"])
    for k in ["dstar_img_1024", "ds_valid_img_1024", "ds_completed_img_224", "ds_completed_obj_224"]:
        if P[k]["n"]:
            LOG.info("%-22s mean %s std %s  min %s p0.5 %s p50 %s p99.5 %s max %s mm", k,
                     *(fmt(v, ".1f") for v in (P[k]["mean"], P[k]["std"], P[k]["min"], q(k, "p0.5"),
                                               q(k, "p50"), q(k, "p99.5"), P[k]["max"])))
    for k in ["abserr_obj_1024", "abserr_bg_1024"]:
        if P[k]["n"]:
            LOG.info("%-22s p50 %s p90 %s p95 %s p99 %s mm", k, *(fmt(q(k, p)) for p in ("p50", "p90", "p95", "p99")))
    ph = summary["pooled_hole_rate"]
    LOG.info("pooled hole rate: obj_1024 %s obj_224 %s img_1024 %s img_224 %s",
             *(fmt(ph.get(k), ".4f") for k in ("obj_1024", "obj_224", "img_1024", "img_224")))
    LOG.info("tau(mm)  lie_obj_1024  lie_eff_obj_224  lie_frac_obj_224  scenes_any_obj_lie_224")
    for t in summary["tau_table"]:
        LOG.info("%7g  %12s  %15s  %16s  %22s", t["tau_mm"], fmt(t["lie_rate_obj_1024"], ".4f"),
                 fmt(t["lie_eff_rate_obj_224"], ".4f"), fmt(t["lie_frac_rate_obj_224"], ".4f"),
                 fmt(t["frac_scenes_any_obj_lie_eff_224"], ".3f"))
    ps = summary["per_scene"]
    LOG.info("obj_px_224 p1 %s p5 %s p10 %s p50 %s | sensor_obj_224 p50 %s p90 %s",
             *(fmt(ps["obj_px_224"].get(p), ".0f") for p in ("p1", "p5", "p10", "p50")),
             fmt(ps["sensor_obj_224"].get("p50"), ".3f"), fmt(ps["sensor_obj_224"].get("p90"), ".3f"))
    LOG.info("crop  place  obj_in  corners_in  obj&corners  objpx_p5  objpx_p50  n_zero")
    for c in summary["crop_table"]:
        LOG.info("%4d  %5s  %6s  %10s  %11s  %8s  %9s  %6d", c["crop_px_at_1024"], c["placement"],
                 fmt(c["frac_object_inside"], ".4f"), fmt(c["frac_grasp_corners_inside"], ".4f"),
                 fmt(c["frac_object_and_corners_inside"], ".4f"), fmt(c["objpx224"].get("p5"), ".0f"),
                 fmt(c["objpx224"].get("p50"), ".0f"), c["n_objpx224_zero"])
    for w in summary["warnings"]:
        LOG.warning("WARNING: %s", w)


# ------------------------------------------------------------------ main
def parse_args():
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default=str(REPO / "configs" / "splits.json"))
    ap.add_argument("--split", default="train", choices=["train"],
                    help="train only: val/test statistics must not inform frozen choices")
    ap.add_argument("--out", default=str(Path(paths["runs_dir"]).expanduser() / "audit_train"))
    ap.add_argument("--overwrite", action="store_true",
                    help="allow a non-empty --out; this script's previous outputs there are deleted first")
    ap.add_argument("--seed", type=int, default=0, help="only used to pick the --max-scenes subsample")
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes of the split (smoke runs: 200)")
    ap.add_argument("--workers", type=int, default=8, help="0 = run in-process (debugging)")
    ap.add_argument("--chunk", type=int, default=32, help="scenes per worker task")
    ap.add_argument("--size", type=int, default=224, choices=[224],
                    help="model working resolution (column names say _224, so locked)")
    ap.add_argument("--telea-radius", type=float, default=3.0, help="must match the cache / dataset setting")
    ap.add_argument("--tau-candidates", type=float, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50],
                    help="selectable tau values (mm); the audited grid adds 0.5x and 2x of each "
                         "(SPEC: every M_lie result is repeated at 0.5 tau and 2 tau)")
    ap.add_argument("--small-thresholds", type=int, nargs="+", default=[25, 50, 100, 150, 200, 300, 400],
                    help="object px at 224; vanished objects (0 px) are counted under exclusions")
    ap.add_argument("--crops", type=int, nargs="+", default=[384, 448, 512, 640, 768, 896, 1024],
                    help="crop sizes at 1024 for the crop-vs-full-frame table")
    ap.add_argument("--depth-max", type=float, default=5000.0, help="depth histogram upper edge, mm")
    ap.add_argument("--depth-step", type=float, default=0.5, help="depth histogram bin width, mm")
    ap.add_argument("--serr-lim", type=float, default=1000.0, help="signed-error histogram range +-, mm")
    ap.add_argument("--serr-step", type=float, default=0.1, help="signed-error histogram bin width, mm")
    a = ap.parse_args()
    if a.chunk <= 0:                    # range(0, n, -1) is empty: no work, yet outputs written (review 10)
        ap.error("--chunk must be > 0")
    if a.workers < 0:
        ap.error("--workers must be >= 0")
    if a.max_scenes < 0:
        ap.error("--max-scenes must be >= 0")
    if any(t <= 0 for t in a.tau_candidates):
        ap.error("tau candidates must be > 0")
    if any(c < a.size or c > IMG_SIZE for c in a.crops):
        ap.error(f"crops must be in {a.size}..{IMG_SIZE} (crop then downsample to --size)")
    a.tau_candidates = sorted(set(a.tau_candidates))
    a.taus = tau_grid(a.tau_candidates)
    return a


def tau_grid(candidates):
    """Every candidate with its 0.5x and 2x partners, so any pick has both sensitivity values."""
    return sorted({t for c in candidates for t in (c / 2, c, 2 * c)})


def main():
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    droot = Path(data_root()).expanduser().resolve()
    if out == droot or droot in out.parents:
        sys.exit(f"--out {out} is inside the dataset directory {droot}; refusing to write there")
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out} is not empty; pass --overwrite or choose another --out")
        for name in OUTPUT_FILES:                     # stale outputs must not look current after a failed rerun
            (out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "audit.log", mode="w")])

    # ---- frozen split + the manifest it was built on (hash cross-check)
    splits_path = Path(args.splits)
    splits = json.load(open(splits_path))
    objects = sorted(splits["splits"][args.split])
    manifest = resolve_manifest(splits["manifest"])
    m_hash = sha256(manifest)
    if m_hash != splits["manifest_sha256"]:
        sys.exit(f"manifest {manifest} sha256 {m_hash} != splits.json's {splits['manifest_sha256']}")
    views = defaultdict(list)
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                views[f'{r["part"]}/{r["object_id"]}'].append(int(r["scene_idx"]))
    missing = [o for o in objects if o not in views]
    if missing:
        sys.exit(f"{len(missing)} split objects have no ok scenes in the manifest, e.g. {missing[:3]}")
    items = [(o, v) for o in objects for v in sorted(views[o])]
    n_scenes_split = len(items)
    if args.max_scenes and args.max_scenes < len(items):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(items), size=args.max_scenes, replace=False))
        items = [items[i] for i in keep]

    cfg = {"root": str(droot), "size": args.size, "taus": args.taus, "tau_candidates": args.tau_candidates,
           "telea_radius": args.telea_radius,
           "depth_max": args.depth_max, "depth_step": args.depth_step,
           "serr_lim": args.serr_lim, "serr_step": args.serr_step,
           "small_thresholds": args.small_thresholds, "crops": args.crops}
    head, dirty = git_state()
    config = {"script": "scripts/audit_train.py", "args": vars(args), "cfg": cfg,
              "git_commit": head, "git_dirty": dirty,
              "splits_path": str(splits_path), "splits_sha256": sha256(splits_path),
              "manifest": str(manifest), "manifest_sha256": m_hash,
              "n_objects_in_split": len(objects), "n_scenes_in_split": n_scenes_split,
              "n_scenes_audited": len(items),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "cv2": cv2.__version__},
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=_json_default))
    LOG.info("auditing %d / %d scenes of %d %s objects -> %s", len(items), n_scenes_split, len(objects),
             args.split, out)

    # ---- run (ordered imap => deterministic row order and float accumulation order)
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    work = partial(audit_chunk, cfg=cfg)
    hists, rows, budget = make_hists(cfg), [], SkipBudget()
    t0 = time.time()

    def consume(res, ci):
        r, st, h = res
        rows.extend(r)                 # rows and histograms first, so partial outputs stay consistent
        for k in hists:
            hists[k].merge(h[k])
        abort = None
        for s in st:                   # R3: feed the whole chunk in scene order, then re-raise the first abort
            try:
                budget.loaded() if s is None else budget.skipped(s)
            except RuntimeError as e:
                abort = abort or e
        if abort is not None:
            raise abort
        if (ci + 1) % 20 == 0 or ci + 1 == len(chunks):
            el = time.time() - t0
            LOG.info("%d/%d scenes  %.1f min elapsed  %.1f scenes/s  ~%.1f min left  skipped=%d", len(rows),
                     len(items), el / 60, len(rows) / el, el / len(rows) * (len(items) - len(rows)) / 60,
                     len(budget.records))

    complete = False
    try:
        if args.workers == 0:
            _init_worker()
            for ci, ch in enumerate(chunks):
                consume(work(ch), ci)
        else:
            with Pool(args.workers, initializer=_init_worker) as pool:
                for ci, res in enumerate(pool.imap(work, chunks)):
                    consume(res, ci)
        complete = True
    finally:
        try:
            summary = write_outputs(out, rows, hists, budget, cfg, config, len(objects), n_scenes_split,
                                    complete, t0)
            if complete:
                report(summary)
            else:
                LOG.error("run stopped after %d scenes; partial outputs written with complete=false", len(rows))
        except Exception:       # never mask the original error with a failure while saving partial output
            LOG.exception("failed to write outputs")
            if complete:
                raise
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
