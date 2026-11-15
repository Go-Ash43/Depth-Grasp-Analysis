"""Annotation match for a predicted grasp (SPEC_trajectory "Labels, metric and splits";
notes/MAJOR_v1.md section 6; configs/labels.json).

match = any annotation row with  dtheta <= angle_max  AND  IoU > iou_min
  primary: 30 deg, 0.25;  strict: 15 deg, 0.40  (angle inclusive, IoU strict, as in the reference).
- dtheta = |((tp - tg + 90) mod 180) - 90|: grasps are symmetric under a half turn.
- IoU on exact float polygons (shapely) from data.jacquard.grasp_corners, at one resolution (1024),
  never clipped to the frame (corners may leave it). IoU is scale-invariant.
- ALL annotation rows are kept, including jaw-size repeats: jaw size changes the rectangle and its IoU.
- An invalid prediction (decoder valid=False) never matches. A prediction passed as valid=True that is
  non-finite or has opening <= 0 or jaw <= 0 raises (checked before the angle filter). An invalid
  annotation polygon raises.
Deviation from the reference (GR-ConvNet grasp.py iou): it rasterises rectangles; we use exact areas.
"""
import numpy as np
import shapely

from data.jacquard import grasp_corners
from data.labels import validate_labels_cfg


def angle_diff_deg(a, b):
    return np.abs((np.asarray(a, np.float64) - np.asarray(b, np.float64) + 90.0) % 180.0 - 90.0)


def rect_polygons(grasps):
    g = np.asarray(grasps, np.float64).reshape(-1, 5)
    return shapely.polygons(grasp_corners(g))


def iou_one_to_many(pred, gt_rows):
    """IoU of one (5,) grasp against (N, 5) rows; exact polygon areas, no clipping."""
    p = rect_polygons(pred)[0]
    gts = rect_polygons(gt_rows)
    if not shapely.is_valid(p) or p.area <= 0:
        raise ValueError(f"invalid prediction polygon {np.asarray(pred).tolist()}")
    bad = ~shapely.is_valid(gts) | (shapely.area(gts) <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} invalid annotation polygons")
    inter = shapely.area(shapely.intersection(p, gts))
    union = p.area + shapely.area(gts) - inter
    return inter / union


def match(pred, valid, gt_rows, angle_max_deg, iou_min):
    """-> {"match": bool, "best_iou_in_angle": float (0 if none in angle), "n_rows": int}."""
    gt = np.asarray(gt_rows, np.float64).reshape(-1, 5)
    if len(gt) == 0:
        raise ValueError("no annotation rows (R1 excludes such scenes)")
    if not np.isfinite(gt).all() or (gt[:, 3] <= 0).any() or (gt[:, 4] <= 0).any():
        raise ValueError("invalid annotation rows (non-finite, opening <= 0 or jaw <= 0)")
    if not valid:
        return {"match": False, "best_iou_in_angle": 0.0, "n_rows": len(gt)}
    pred = np.asarray(pred, np.float64).reshape(5)
    if not np.isfinite(pred).all() or pred[3] <= 0 or pred[4] <= 0:   # review #10: a broken prediction
        raise ValueError(f"valid=True prediction is malformed (non-finite, opening <= 0 or jaw <= 0): "
                         f"{pred.tolist()}")                          # must never be scored as a miss
    in_angle = angle_diff_deg(pred[2], gt[:, 2]) <= angle_max_deg
    if not in_angle.any():
        return {"match": False, "best_iou_in_angle": 0.0, "n_rows": len(gt)}
    iou = iou_one_to_many(pred, gt[in_angle])
    best = float(iou.max())
    return {"match": bool(best > iou_min), "best_iou_in_angle": best, "n_rows": len(gt)}


def match_both(pred, valid, gt_rows, cfg):
    """Primary and strict match from configs/labels.json["match"]."""
    validate_labels_cfg(cfg, ("match",))
    out = {}
    for k in ("primary", "strict"):
        m = cfg["match"][k]
        out[k] = match(pred, valid, gt_rows, m["angle_max_deg"], m["iou_min_exclusive"])
    return out
