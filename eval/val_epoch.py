"""Per-epoch validation: decode, match, and categorise misses (step 5 proposal rev 2 sections 2 and 9).

Per sample: decode_from_cfg on the model's maps (top-1 cell, output frame 1024), match_both against
every annotation row (primary and strict). Each MISS is categorised separately per metric, in order:

  invalid_decode       decoder valid=False (|(cos, sin)| < angle_eps or opening <= 0)
  void_cell            valid prediction whose top-1 cell has target void == 1 (outside every
                       supervised centre neighbourhood, Q < q_void)
  target_at_cell_miss  the cell is supervised, but the EXACT target cos2 / sin2 / width AT THAT SAME
                       CELL also miss (decode_from_cfg with a one-hot Q at the predicted cell, so the
                       counterfactual uses the same decoder code and the predicted cell, never the
                       targets' own Q maximum)
  regression_miss      the exact targets at that cell would match, the prediction does not

These are operational categories. target_at_cell_miss is NOT a demonstrated jaw-rule failure: non-void
cells reach ~4.9 px (224) from a centre, so the cell's position alone can cause it, and target
deduplication collapses rows with different jaws (no single "owner jaw").

Extra per-scene columns (proj_v4 5.13; reported, never changing a category or top-1 number):
  target_opening_224   exact target opening at the picked cell, px at 224 (empty when the cell is void)
  cfn_match_<metric>   the same exact-regression counterfactual, computed whenever the picked cell is
                       NON-VOID, including invalid predicted decodes (cf_match_* keeps its original, narrower
                       eligibility: valid decode AND non-void; where both exist they are equal)
  tt_match_<metric>    the exact-target baseline: decode of the target maps themselves

run_val: model.eval() + torch.inference_mode(), training mode restored afterwards; every buffer
(BatchNorm running stats, num_batches_tracked) must be unchanged; invalid decodes stay in the
denominator; every expected slot must appear exactly once.
"""
import csv
import io

import numpy as np
import torch

from data.decode import decode_from_cfg
from eval.match import match_both

METRICS = ("primary", "strict")
MISS_CATEGORIES = ("invalid_decode", "void_cell", "target_at_cell_miss", "regression_miss")
CSV_FIELDS = ("condition", "slot", "severity", "hole_frac", "covered_lie_frac", "valid", "x", "y",
              "theta", "opening", "jaw", "q_top", "cell_j", "cell_i", "target_void",
              "match_primary", "match_strict", "cf_match_primary", "cf_match_strict",
              "cat_primary", "cat_strict", "target_opening_224", "cfn_match_primary", "cfn_match_strict",
              "tt_match_primary", "tt_match_strict")


def classify_sample(pred_maps, tgt_maps, gt_rows, labels_cfg):
    """pred_maps: {q, cos2, sin2, width} 2D arrays; tgt_maps: same plus void (2D). -> row dict."""
    for k in ("q", "cos2", "sin2", "width"):
        if np.asarray(pred_maps[k]).ndim != 2 or np.asarray(tgt_maps[k]).ndim != 2:
            raise ValueError(f"{k}: the decoder takes single 2D maps")
    dec = decode_from_cfg(pred_maps, labels_cfg)
    m = match_both(dec["grasp"], dec["valid"], gt_rows, labels_cfg)
    j, i = dec["cell_xy"]
    tvoid = bool(np.asarray(tgt_maps["void"])[i, j])
    cfn = {k: None for k in METRICS}                   # eligibility: non-void picked cell only
    if not tvoid:
        onehot = np.zeros_like(np.asarray(tgt_maps["q"], np.float32))
        onehot[i, j] = 1.0
        cdec = decode_from_cfg({"q": onehot, "cos2": tgt_maps["cos2"], "sin2": tgt_maps["sin2"],
                                "width": tgt_maps["width"]}, labels_cfg)
        if cdec["cell_xy"] != (j, i):
            raise AssertionError("counterfactual decode left the predicted cell")
        cm = match_both(cdec["grasp"], cdec["valid"], gt_rows, labels_cfg)
        cfn = {k: bool(cm[k]["match"]) for k in METRICS}
    cf = cfn if dec["valid"] else {k: None for k in METRICS}   # original narrower eligibility, unchanged
    tdec = decode_from_cfg({k: tgt_maps[k] for k in ("q", "cos2", "sin2", "width")}, labels_cfg)
    tm = match_both(tdec["grasp"], tdec["valid"], gt_rows, labels_cfg)
    w_norm = float(labels_cfg["targets"]["w_norm_px_224"])
    row = {"valid": bool(dec["valid"]), "q_top": dec["q"], "cell_j": int(j), "cell_i": int(i),
           "target_void": tvoid,
           "target_opening_224": None if tvoid else float(np.asarray(tgt_maps["width"])[i, j]) * w_norm,
           **{f"cfn_match_{k}": cfn[k] for k in METRICS},
           **{f"tt_match_{k}": bool(tm[k]["match"]) for k in METRICS}}
    row.update({n: float(v) for n, v in zip(("x", "y", "theta", "opening", "jaw"), dec["grasp"])})
    for k in METRICS:
        hit = bool(m[k]["match"])
        row[f"match_{k}"] = hit
        row[f"cf_match_{k}"] = cf[k]
        if hit:
            cat = ""
        elif not dec["valid"]:
            cat = "invalid_decode"
        elif tvoid:
            cat = "void_cell"
        elif not cf[k]:
            cat = "target_at_cell_miss"
        else:
            cat = "regression_miss"
        row[f"cat_{k}"] = cat
    return row


def buffer_snapshot(model):
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def buffers_equal(model, snap):
    now = dict(model.named_buffers())
    return set(now) == set(snap) and all(torch.equal(now[n], snap[n]) for n in snap)


def run_val(model, loader, expected_slots, labels_cfg, device, condition):
    """-> (rows, summary). loader yields collate() batches of one eval-mode dataset (one condition)."""
    was_training = model.training
    snap = buffer_snapshot(model)
    rows = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                pred = model(batch["x"].to(device, non_blocking=True))
                pm = {k: pred[k].float().cpu().numpy() for k in ("q", "cos2", "sin2", "width")}
                tm = {k: batch[k].numpy() for k in ("q", "cos2", "sin2", "width", "void")}
                for b in range(batch["x"].shape[0]):
                    r = classify_sample({k: pm[k][b, 0] for k in pm}, {k: tm[k][b, 0] for k in tm},
                                        batch["gt_grasps"][b], labels_cfg)
                    r.update({"condition": condition, "slot": int(batch["slot"][b]),
                              "severity": batch["severity"][b],
                              "hole_frac": float(batch["hole_frac"][b]),
                              "covered_lie_frac": float(batch["covered_lie_frac"][b])})
                    rows.append(r)
    finally:
        model.train(was_training)
    if not buffers_equal(model, snap):
        raise AssertionError("validation changed a model buffer (BatchNorm statistics)")
    seen = [r["slot"] for r in rows]
    if len(seen) != len(set(seen)):
        raise AssertionError(f"{condition}: a slot was evaluated more than once")
    want = sorted(int(s) for s in expected_slots)
    if sorted(seen) != want:
        missing = sorted(set(want) - set(seen))[:5]
        extra = sorted(set(seen) - set(want))[:5]
        raise AssertionError(f"{condition}: slots differ from the dataset (missing {missing}, extra {extra})")
    return rows, summarise(rows, condition)


def summarise(rows, condition):
    n = len(rows)
    out = {"condition": condition, "n": n, "invalid_decodes": sum(not r["valid"] for r in rows)}
    for k in METRICS:
        hits = sum(r[f"match_{k}"] for r in rows)
        out[f"top1_{k}"] = hits / n if n else None
        out[f"matches_{k}"] = hits
        out[f"misses_{k}"] = {c: sum(r[f"cat_{k}"] == c for r in rows) for c in MISS_CATEGORIES}
    out["hole_frac_mean"] = float(np.mean([r["hole_frac"] for r in rows])) if n else None
    out["covered_lie_frac_mean"] = float(np.mean([r["covered_lie_frac"] for r in rows])) if n else None
    sev = {}
    for r in rows:
        sev[r["severity"]] = sev.get(r["severity"], 0) + 1
    out["severity_counts"] = sev
    return out


def rows_to_csv_bytes(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in CSV_FIELDS})
    return buf.getvalue().encode()
