#!/usr/bin/env python3
"""Step 3 overlays + hole statistics + stereo shadow side (docs/proj_v3.md sections 2 and 6).

Everything a model sees is drawn THROUGH data.dataset.GraspDataset (model M, eval mode, so holes come
from the corruption seed and are reproducible). Display-only panels (D*, raw D_s, lie_any@12) are read
from the cache by this script and never pass through the dataset.

Outputs in --out:
  scene_XX_slotNNNNN.png  20 scenes (seeded; >= 4 small objects and >= 4 with a grasp opening < 4 px
                          at 224 when available), severity cycling low / mid / high. 16 panels: RGB,
                          D*, raw D_s, model input depth, M_sensor, M_synth, lie_any@12, flag, GT
                          rectangles at 224, Q / angle / width targets, Z-end3 zones of one GT grasp on
                          the flag (drawn directly; data.zones.selected_zone still refuses), and the same
                          sample under the flip (--show-flip-axis) with its rectangles.
  stats.json              (a) per severity on --n-stats seeded train scenes: share within the +-10 %
                          budget, share by shortfall cause, share with no hole, achieved fraction
                          quantiles, covered lie fraction (lie_any@12 inside the hole) and the share of
                          holes covering no lie (for the H1 sensitivity line);
                          time per synth_holes call (mean, p99);
                          (b) stereo shadow side: M_sensor cells in an 8-px ring around the object, split
                          left / right and above / below the object centroid; signed per-scene asymmetry.
                          A baseline puts shadows CONSISTENTLY on one side: per axis the signed median
                          and whether p10 and p90 share a sign. Consistent left/right -> VERTICAL flip
                          (D2); consistent up/down -> horizontal; otherwise "undecided";
                          (c) time per full training sample (holes + Telea + targets), mean and p99.
  config.json             args, git state, versions.
Read-only on the cache and dataset. Refuses an --out inside the dataset root or the cache, and a
non-empty --out without --overwrite (only this script's own files are replaced).

Usage:
  python scripts/overlays.py --out $RUNS/overlays_v1
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.cache_layout import open_cache  # noqa: E402
from data.paths import load_paths  # noqa: E402
from data.dataset import (SEVERITY_ORDER, STREAM_HOLES, GraspDataset, load_data_cfg,  # noqa: E402
                          load_frozen, stream_rng)
from data.geometry import scale_grasps  # noqa: E402
from data.holes import covered_lie_frac, synth_holes  # noqa: E402
from data.jacquard import data_root, grasp_corners  # noqa: E402
from data.provenance import git_fields, refuse_inside  # noqa: E402
from data.zones import zone_polygons  # noqa: E402

SRC, S = (1024, 1024), (224, 224)
SMALL_OPENING_PX_224 = 4.0
RING_PX = 8
MIN_RING_CELLS = 10


def pick_scenes(cache, n, seed, n_special=4, pool=3000):
    idx = cache["index"]
    ok = np.array([i for i, r in enumerate(idx) if r["status"] == "ok"])
    rng = np.random.default_rng([seed, 0])
    cand = rng.choice(ok, size=min(pool, len(ok)), replace=False)
    small = [int(s) for s in cand if idx[s]["small_obj"] == "1"]
    thin = [int(s) for s in cand
            if (cache["grasps"](s)[:, 3] * S[0] / SRC[0] < SMALL_OPENING_PX_224).any()]
    out = []
    for group in (small[:n_special], thin[:n_special]):
        out += [s for s in group if s not in out]
    for s in cand:
        if len(out) >= n:
            break
        if int(s) not in out:
            out.append(int(s))
    return out[:n], {"n_small_candidates": len(small), "n_thin_candidates": len(thin)}


def _rects(ax, g1024, color, lw=0.6):
    g = scale_grasps(g1024, SRC, S)
    for c in grasp_corners(g):
        p = np.vstack([c, c[:1]])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw)


def _img(ax, a, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_axis_off()


def draw_scene(path, cache, d, slot, severity, flip_axis, norm):
    A = cache["arrays"]
    a = d.sample(0, slot, force_flip=False)
    b = d.sample(0, slot, force_flip=True)
    x, xb = a["x"].numpy(), b["x"].numpy()
    m, s = d.rgb_norm
    rgb = np.clip((x[:3] * s + m).transpose(1, 2, 0), 0, 1)
    rgb_b = np.clip((xb[:3] * s + m).transpose(1, 2, 0), 0, 1)
    depth = x[3] * norm[1] + norm[0]
    dstar = np.asarray(A["dstar"][slot])
    ds_raw = np.asarray(A["ds"][slot])
    lo, hi = np.nanpercentile(dstar, [1, 99])
    q, c2, s2, w = (a[k][0].numpy() for k in ("q", "cos2", "sin2", "width"))
    void = a["void"][0].numpy()
    ang = np.where(void, np.nan, 0.5 * np.degrees(np.arctan2(s2, c2)))
    wid = np.where(void, np.nan, w)
    gt = a["gt_grasps"]
    r = cache["index"][slot]
    fig, axs = plt.subplots(4, 4, figsize=(13, 13))
    ax = axs.ravel()
    _img(ax[0], rgb, f"RGB  slot {slot} {r['object_id'][:10]} v{r['view']}")
    _img(ax[1], dstar, "D* (display only)", "viridis", lo, hi)
    _img(ax[2], ds_raw, "raw D_s at 224 (NaN = M_sensor)", "viridis", lo, hi)
    _img(ax[3], depth, "model input depth (Protocol A)", "viridis", lo, hi)
    _img(ax[4], A["m_sensor"][slot], "M_sensor", "gray", 0, 1)
    _img(ax[5], a["m_synth"].numpy(), f"M_synth ({severity}, {a['hole_frac']:.3f} of object)", "gray", 0, 1)
    _img(ax[6], A["lie_any_t12"][slot], "lie_any@12 (display only)", "gray", 0, 1)
    _img(ax[7], x[4], "flag (deployed)", "gray", 0, 1)
    _img(ax[8], rgb, f"GT rectangles at 224 ({len(gt)} rows)")
    _rects(ax[8], gt, "lime")
    _img(ax[9], q, "Q target", "magma", 0, 1)
    _img(ax[10], ang, "angle target (deg, void blank)", "twilight", -90, 90)
    _img(ax[11], wid, "width target (/ W_NORM)", "cividis")
    _img(ax[12], x[4], "Z-end3 zones, first GT grasp (red)", "gray", 0, 1)
    g0 = scale_grasps(gt[:1], SRC, S)[0]
    _rects(ax[12], gt[:1], "cyan", 0.8)
    for poly in zone_polygons(g0, "Z-end3"):
        p = np.asarray(poly.exterior.coords)
        ax[12].fill(p[:, 0], p[:, 1], color="red", alpha=0.5)
    _img(ax[13], rgb_b, f"flipped ({flip_axis}) RGB + flipped GT")
    _rects(ax[13], b["gt_grasps"], "lime")
    _img(ax[14], b["q"][0].numpy(), "flipped Q target", "magma", 0, 1)
    _img(ax[15], xb[4], "flipped flag", "gray", 0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def shadow_side(ms, obj):
    """M_sensor cells in an RING_PX ring around the object, split by the object centroid."""
    ob = obj.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    ring = cv2.dilate(ob, k).astype(bool) & ~ob.astype(bool)
    cells = np.argwhere(ms.astype(bool) & ring)
    if len(cells) < MIN_RING_CELLS:
        return None
    cy, cx = np.argwhere(ob.astype(bool)).mean(0)
    left, right = int((cells[:, 1] < cx).sum()), int((cells[:, 1] > cx).sum())
    up, down = int((cells[:, 0] < cy).sum()), int((cells[:, 0] > cy).sum())
    return {"lr": (right - left) / max(right + left, 1), "ud": (down - up) / max(down + up, 1)}


def _q(v):
    return np.percentile(v, [10, 50, 90]).round(4).tolist() if len(v) else None


def _axis_reading(v):
    """Signed asymmetries of one axis -> median and whether p10 and p90 share a sign (a consistent
    side is what a stereo baseline produces; irregular shapes give asymmetry of either sign)."""
    if not len(v):
        return {"n": 0}
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {"n": int(len(v)), "signed_p10_50_90": [round(float(p), 4) for p in (p10, p50, p90)],
            "consistent_sign": bool(p10 * p90 > 0), "abs_median": round(float(np.median(np.abs(v))), 4)}


def flip_suggestion(lr, ud):
    """Shadows consistently on one side left/right (horizontal baseline) -> vertical flip; consistently
    above/below (vertical baseline) -> horizontal flip; otherwise undecided (look at the overlays)."""
    if lr.get("consistent_sign") and not ud.get("consistent_sign"):
        return "vertical"
    if ud.get("consistent_sign") and not lr.get("consistent_sign"):
        return "horizontal"
    return "undecided"


def _ms(v):
    return {"mean_ms": round(float(np.mean(v)) * 1e3, 2), "p99_ms": round(float(np.percentile(v, 99)) * 1e3, 2)} \
        if len(v) else None


HIGH_FALLBACKS = (0.25, 0.20)       # docs/proj_v3.md section 7, frozen before stats.json existed
HIGH_MIN_SHARE, MID_MIN_SHARE = 0.80, 0.90


def apply_severity_rule(high_shares, mid_share, achieved):
    """Frozen rule (docs/proj_v3.md section 7):
    - mid share_within_tol < 0.90                        -> STOP (primary severity)
    - high = first of 0.30, 0.25, 0.20 with share >= 0.80; none reaches 0.80 -> STOP (no floor below 0.20)
    - separation on ACHIEVED fractions: p10(high at the chosen budget) > median(mid) and
      p10(mid) > median(low); otherwise the labels overlap                -> STOP
    high_shares: {budget: share}; achieved: {"low", "mid", budget: [p10, p50, p90] achieved_frac}."""
    stops = []
    if mid_share < MID_MIN_SHARE:
        stops.append(f"mid within tolerance {mid_share:.3f} < {MID_MIN_SHARE}")
    high = next((b for b in (0.30, *HIGH_FALLBACKS) if high_shares[b] >= HIGH_MIN_SHARE), None)
    if high is None:
        stops.append(f"no high budget reaches {HIGH_MIN_SHARE} within tolerance (0.20 is the floor)")
    else:
        if not achieved[high][0] > achieved["mid"][1]:
            stops.append(f"high {high} p10 achieved {achieved[high][0]} <= mid median {achieved['mid'][1]}")
    if not achieved["mid"][0] > achieved["low"][1]:
        stops.append(f"mid p10 achieved {achieved['mid'][0]} <= low median {achieved['low'][1]}")
    if stops:
        action = "STOP, discuss before training: " + "; ".join(stops)
    elif high != 0.30:
        action = "apply the new high budget by an approved diff to configs/train_data.json"
    else:
        action = "no change"
    return {"high_budget": high, "high_changed": high is not None and high != 0.30,
            "mid_ok": mid_share >= MID_MIN_SHARE, "stop": bool(stops), "action": action}


def stats(cache, slots, hcfg, seed):
    A = cache["arrays"]
    per = {s: [] for s in SEVERITY_ORDER}
    secs = {s: [] for s in SEVERITY_ORDER}
    fb = {b: [] for b in HIGH_FALLBACKS}
    side = []
    for slot in slots:
        ms, obj = np.asarray(A["m_sensor"][slot]), np.asarray(A["obj"][slot])
        lie = np.asarray(A[hcfg["covered_lie_mask"]][slot])          # logging only, never placement
        for i, sev in enumerate(SEVERITY_ORDER):
            rng = stream_rng(seed, STREAM_HOLES, i, slot, "train")
            t0 = time.perf_counter()
            m, rec = synth_holes(rng, ms, obj, float(hcfg["severities"][sev]), hcfg)
            secs[sev].append(time.perf_counter() - t0)
            rec["covered_lie_frac"] = covered_lie_frac(m, lie)
            per[sev].append(rec)
        for b in HIGH_FALLBACKS:                      # same stream as "high", only the budget differs
            rng = stream_rng(seed, STREAM_HOLES, SEVERITY_ORDER.index("high"), slot, "train")
            fb[b].append(synth_holes(rng, ms, obj, b, hcfg)[1])
        sd = shadow_side(ms, obj)
        if sd:
            side.append(sd)
    out = {"n_scenes": len(slots), "per_severity": {}}
    for sev, recs in per.items():
        n = len(recs)
        causes = {}
        for r in recs:
            if r["shortfall_cause"]:
                causes[r["shortfall_cause"]] = causes.get(r["shortfall_cause"], 0) + 1
        holed = [r["covered_lie_frac"] for r in recs if r["achieved"] > 0]
        out["per_severity"][sev] = {
            "budget_frac": float(hcfg["severities"][sev]),
            "share_within_tol": sum(r["within_tol"] for r in recs) / n,
            "share_by_shortfall_cause": {k: v / n for k, v in sorted(causes.items())},
            "share_no_hole": sum(r["achieved"] == 0 for r in recs) / n,
            "covered_lie_frac_p10_50_90": _q(holed),
            "share_holes_covering_no_lie": (sum(v == 0 for v in holed) / len(holed)) if holed else None,
            "achieved_frac_p10_50_90": _q([r["achieved_frac"] for r in recs]),
            "achieved_over_target_p10_50_90": _q([r["achieved"] / r["target"] for r in recs if r["target"] > 0]),
            "synth_holes_time": _ms(secs[sev]),
        }
    ps = out["per_severity"]
    high0 = float(hcfg["severities"]["high"])
    high_shares = {high0: ps["high"]["share_within_tol"],
                   **{b: sum(r["within_tol"] for r in v) / len(v) for b, v in fb.items()}}
    achieved = {"low": ps["low"]["achieved_frac_p10_50_90"], "mid": ps["mid"]["achieved_frac_p10_50_90"],
                high0: ps["high"]["achieved_frac_p10_50_90"],
                **{b: _q([r["achieved_frac"] for r in v]) for b, v in fb.items()}}
    out["high_fallback_share_within_tol"] = {str(k): v for k, v in high_shares.items()}
    out["high_fallback_achieved_frac_p10_50_90"] = {str(k): achieved[k] for k in high_shares}
    out["severity_rule"] = apply_severity_rule(high_shares, ps["mid"]["share_within_tol"], achieved)
    lr = _axis_reading(np.array([s["lr"] for s in side]))
    ud = _axis_reading(np.array([s["ud"] for s in side]))
    out["shadow_side"] = {
        "n_scenes_with_ring_shadow": len(side), "ring_px": RING_PX,
        "left_right": lr, "up_down": ud,
        "sign_convention": "left_right > 0: more shadow right of the centroid; up_down > 0: more below",
        "suggested_flip_axis": flip_suggestion(lr, ud),
        "reading": "a stereo baseline puts shadows CONSISTENTLY on one side (p10 and p90 share a sign); "
                   "consistent left/right -> vertical flip (D2), consistent up/down -> horizontal flip",
    }
    return out


def time_samples(cache_dir, slots, seed):
    """Full training sample (holes + Telea + targets) on slots NOT read by the other passes, in random
    (unsorted) order, timed before the stats pass, so the OS page cache does not flatter it."""
    d = GraspDataset(cache_dir, "train", "M", mode="train", seed=seed, holes="train")
    d._c()                                            # warm-up: the cache reopen is not a sample cost
    t = []
    for slot in slots:
        t0 = time.perf_counter()
        d.sample(0, int(slot))
        t.append(time.perf_counter() - t0)
    return {"n": len(t), "slots": "random, disjoint from the scenes and the stats slots, read first",
            **(_ms(t) or {})}


def main(argv=None):
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=paths["cache_dir"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="scene selection and the stats hole streams")
    ap.add_argument("--n-scenes", type=int, default=20)
    ap.add_argument("--n-stats", type=int, default=500)
    ap.add_argument("--show-flip-axis", default="vertical", choices=["vertical", "horizontal"],
                    help="display only; the training axis is frozen in configs/train_data.json")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.seed < 0 or a.n_scenes < 1 or a.n_stats < 1:
        ap.error("--seed >= 0, --n-scenes >= 1, --n-stats >= 1")
    out = refuse_inside(a.out, Path(data_root()).expanduser())
    croot = Path(a.cache_dir).expanduser().resolve()
    if out == croot or croot in out.parents:
        raise SystemExit(f"refusing to write inside the cache directory {croot}: {out}")
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty; use a fresh --out or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    for p in list(out.glob("scene_*.png")) + [out / "stats.json"]:      # only this script's own files
        if p.exists():
            p.unlink()
    dcfg, frozen = load_data_cfg(), load_frozen()
    config = {"script": "scripts/overlays.py", "args": vars(a), **git_fields(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "cv2": cv2.__version__,
                           "matplotlib": matplotlib.__version__},
              "train_data_cfg": dcfg, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))

    cache = open_cache(a.cache_dir, "train")
    scenes, sel = pick_scenes(cache, a.n_scenes, a.seed)
    norm = (frozen["normalisation"]["ds_comp_mm"]["median"], frozen["normalisation"]["ds_comp_mm"]["robust_sigma"])
    dsets = {sev: GraspDataset(a.cache_dir, "train", "M", mode="eval", holes=sev, flip_axis=a.show_flip_axis)
             for sev in SEVERITY_ORDER}
    for i, slot in enumerate(scenes):
        sev = SEVERITY_ORDER[i % 3]
        d = dsets[sev]
        draw_scene(out / f"scene_{i:02d}_slot{slot:05d}.png", cache, d, slot, sev, a.show_flip_axis, norm)
        print(f"  scene {i + 1}/{len(scenes)} slot {slot} ({sev})", flush=True)
    ok = np.array([j for j, r in enumerate(cache["index"]) if r["status"] == "ok"])
    st_slots = sorted(int(s) for s in np.random.default_rng([a.seed, 1]).choice(ok, min(a.n_stats, len(ok)), replace=False))
    used = set(st_slots) | set(scenes)
    pool = np.array([s for s in ok if int(s) not in used])
    timed = np.random.default_rng([a.seed, 2]).choice(pool, min(200, len(pool)), replace=False)
    print(f"timing {len(timed)} fresh training samples ...", flush=True)
    sample_time = time_samples(a.cache_dir, timed, a.seed)
    print(f"hole statistics and shadow side on {len(st_slots)} scenes ...", flush=True)
    st = stats(cache, st_slots, dcfg["holes"], a.seed)
    st["train_sample_time"] = sample_time
    st.update(scenes=scenes, selection=sel, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({"per_severity": st["per_severity"], "severity_rule": st["severity_rule"],
                      "high_fallback_share_within_tol": st["high_fallback_share_within_tol"],
                      "shadow_side": st["shadow_side"],
                      "train_sample_time": st["train_sample_time"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
