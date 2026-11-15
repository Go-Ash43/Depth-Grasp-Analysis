#!/usr/bin/env python3
"""Freeze candidates from an audit_train.py run (read-only on the audit outputs; no dataset access).

Applies pre-stated rules to summary.json + histograms.npz and prints the values to freeze by hand:
  tau          smallest --tau-candidates value >= --tau-factor x background |D_s - D*| p99
               (background = valid table pixels: the stereo noise floor); reported with its
               0.5 tau / 2 tau rows and the SPEC go/no-go gates (SPEC_trajectory lines 269-270).
  D_s clamp    [min D* - margin, max D* + margin] mm over the AUDITED SAMPLE, rounded outward to
               10 mm, margin > 2 tau (enforced). Sample extrema are not dataset bounds, so the cache
               builder must enforce the property that matters on every scene: no valid pixel with
               raw |D_s - D*| <= 2 tau is changed by the clamp (abort on train; count and log on val/test).
               Applied to the detector's D_s input only, BEFORE resize/Telea and before any
               Protocol-B sentinel. Flags untouched; M_sensor and M_lie are computed from RAW D_s and
               D*, so the clamp cannot change them. (Deviation from the paper's plain pipeline: logged
               in the cache config.)
  normalise    REFERENCE ONLY: (D - median) / robust sigma, robust sigma = (p84.13 - p15.87) / 2, on the
               UNCLAMPED completed 224 input. Clamping before resize/Telea changes the cells an outlier
               averages or inpaints into, even when the result stays inside the clamp range, so these
               numbers are not the frozen constants. The frozen constants are measured by the cache
               builder through the clamped pipeline on every train scene.
  small objs   full-frame size table and the 1024 vs 896 crop rows, for the placement threshold.

Usage:
  python scripts/audit_report.py --audit-dir $RUNS/audit_train_5k
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.jacquard import data_root  # noqa: E402  (reads configs/paths.json only; no dataset access)


def hist(npz, name):
    e, c = npz[f"{name}__edges"], npz[f"{name}__counts"]
    under, over = (int(v) for v in npz[f"{name}__under_over"])
    return e, c, under, over


def quantile(h, q):
    """Linear interpolation inside the bin; None if the quantile lies outside the histogram range."""
    e, c, under, over = h
    n = under + int(c.sum()) + over
    t = q * n
    if n == 0 or t <= under or t > n - over:
        return None
    cum = np.cumsum(c)
    i = int(np.searchsorted(cum, t - under, side="left"))
    prev = int(cum[i - 1]) if i > 0 else 0
    frac = (t - under - prev) / c[i] if c[i] else 0.0
    return float(e[i] + frac * (e[i + 1] - e[i]))


def count_outside(h, lo, hi):
    """(# values < lo, # values >= hi), exact when lo, hi are bin edges (10-mm multiples on a 0.5-mm grid).
    The upper count includes values == hi, which np.clip leaves unchanged: conservative by that bin."""
    e, c, under, over = h
    below = under + int(c[e[1:] <= lo].sum())
    above = over + int(c[e[:-1] >= hi].sum())
    return below, above


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--tau-factor", type=float, default=1.5)
    ap.add_argument("--clamp-margin", type=float, default=100.0,
                    help="mm beyond the observed D* range. Must exceed 2*tau: then a clamped stereo pixel is "
                         "still off by > 2*tau from D*, so M_lie is identical with and without the clamp up "
                         "to the 2*tau sensitivity row. Also absorbs the sample-to-sample spread of min D* "
                         "(~170 mm across 200-scene samples)")
    ap.add_argument("--out", default="", help="default: <audit-dir>/freeze_candidates.json")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    a = ap.parse_args()
    d = Path(a.audit_dir).expanduser()
    path = (Path(a.out).expanduser() if a.out else d / "freeze_candidates.json").resolve()
    droot = Path(data_root()).expanduser().resolve()
    if path == droot or droot in path.parents:
        sys.exit(f"--out {path} is inside the dataset directory {droot}; refusing to write there")
    if path.exists() and not a.overwrite:
        sys.exit(f"{path} exists; pass --overwrite to replace it")
    s = json.load(open(d / "summary.json"))
    npz = np.load(d / "histograms.npz")
    P = s["pooled"]
    if not s.get("complete", False):
        sys.exit("audit run is marked incomplete; rerun it before freezing")
    out = {"audit_dir": str(d), "n_scenes_included": s["n_scenes_included"],
           "sample_note": "complete=true means the requested (sub)sample finished; extrema are sample "
                          "values, not dataset bounds",
           "audit_config": {k: s["config"]["args"][k] for k in ("seed", "max_scenes", "telea_radius")},
           "rules": {"tau_factor": a.tau_factor, "clamp_margin_mm": a.clamp_margin}}

    # ---- tau
    bg_p99 = quantile(hist(npz, "abserr_bg_1024"), 0.99)
    target = a.tau_factor * bg_p99
    cands = [c for c in s["tau_candidates"] if c >= target]
    tau = min(cands) if cands else None
    if tau is None:
        sys.exit(f"no tau candidate >= target {target:.2f} mm (= {a.tau_factor} x background p99 "
                 f"{bg_p99:.2f} mm); extend --tau-candidates in the audit")
    rows = {r["tau_mm"]: r for r in s["tau_table"]}
    keep = ("lie_rate_obj_1024", "lie_eff_rate_obj_224", "lie_frac_rate_obj_224", "frac_scenes_any_obj_lie_eff_224")
    out["tau"] = {"bg_abserr_p99_mm": bg_p99, "target_mm": target, "tau_mm": tau,
                  "rows": {f"{t:g}": {k: rows[t][k] for k in keep} for t in ((tau / 2, tau, 2 * tau) if tau else ())}}
    if tau and not a.clamp_margin > 2 * tau:
        sys.exit(f"--clamp-margin {a.clamp_margin} must exceed 2*tau = {2 * tau} "
                 f"(else truthful pixels near the sample's D* extremes could be clamped)")
    # SPEC_trajectory go/no-go (lines 269-270): True = the natural effect is LARGE ENOUGH TO STUDY
    # (>= ~1%), i.e. the natural-missingness / natural-lies claims are kept. It is not a quality verdict.
    out["gates"] = {
        "m_sensor_rate_obj_1024": s["pooled_hole_rate"]["obj_1024"],          # / all object pixels
        "m_sensor_ge_1pct_keep_natural_missingness_claims": s["pooled_hole_rate"]["obj_1024"] >= 0.01,
        "m_lie_rate_valid_obj_1024_at_tau": rows[tau]["lie_rate_obj_1024"] if tau else None,  # / VALID object px
        "m_lie_ge_1pct_keep_natural_lies_row": bool(tau) and rows[tau]["lie_rate_obj_1024"] >= 0.01}

    # ---- clamp
    dmin, dmax = P["dstar_img_1024"]["min"], P["dstar_img_1024"]["max"]
    lo = math.floor((dmin - a.clamp_margin) / 10) * 10
    hi = math.ceil((dmax + a.clamp_margin) / 10) * 10
    b, ab = count_outside(hist(npz, "ds_valid_img_1024"), lo, hi)
    n_valid = P["ds_valid_img_1024"]["n"]
    out["ds_clamp_mm"] = {"lo": lo, "hi": hi, "dstar_min": dmin, "dstar_max": dmax,
                          "ds_valid_px_below": b, "ds_valid_px_above": ab,
                          "frac_ds_valid_px_clamped": (b + ab) / n_valid}

    # ---- normalisation (completed D_s at 224, the A0/M depth input)
    hc = hist(npz, "ds_completed_img_224")
    q = {p: quantile(hc, p) for p in (0.005, 0.1587, 0.5, 0.8413, 0.995)}
    out["normalisation_reference_unclamped"] = {
        "NOT_FROZEN": "unclamped sample; final constants come from the clamped train cache",
        "center_mm": q[0.5],
        "scale_mm": (q[0.8413] - q[0.1587]) / 2 if None not in (q[0.8413], q[0.1587]) else None,
        "p0.5": q[0.005], "p99.5": q[0.995],
        "mean_std": [P["ds_completed_img_224"]["mean"], P["ds_completed_img_224"]["std"]]}
    hd = hist(npz, "dstar_img_224")                   # A* input (perfect depth): no clamp involved
    out["normalisation_dstar_reference"] = {"center_mm": quantile(hd, 0.5),
                                            "scale_mm": (quantile(hd, 0.8413) - quantile(hd, 0.1587)) / 2}

    # ---- small objects
    out["small_objects"] = {"obj_px_224": s["per_scene"]["obj_px_224"],
                            "thresholds": s["small_object_thresholds"],
                            "crop_rows": [r for r in s["crop_table"] if r["crop_px_at_1024"] in (896, 1024)]}
    out["warnings"] = s["warnings"]
    out["exclusions"] = s["exclusions"]["counts"]

    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "small_objects"}, indent=1))
    so = out["small_objects"]
    print("obj_px_224:", {k: round(v) for k, v in so["obj_px_224"].items() if k in ("p0", "p1", "p5", "p10", "p50")})
    for t in so["thresholds"]:
        print(f"  obj_px_224 < {t['obj_px_224_below']:>4}: {t['n_scenes']:5d} scenes ({100 * t['frac_scenes']:.2f}%), "
              f"{t['n_objects']} objects")
    for r in so["crop_rows"]:
        print(f"  crop {r['crop_px_at_1024']} {r['placement']}: obj&corners {r['frac_object_and_corners_inside']:.4f}, "
              f"objpx p5 {r['objpx224']['p5']:.0f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
