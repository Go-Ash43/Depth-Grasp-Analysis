"""Top-1 grasp from Q / cos 2t / sin 2t / width maps (notes/MAJOR_v1.md section 6; configs/labels.json).

- Any non-finite value in any map raises: NaN from a model is a training failure, not a miss.
- Top-1 cell = np.argmax(Q) (ties -> first in row-major order); its centre is mapped to the output
  resolution with data.geometry.scale_xy (pixel-centre convention), never x * s.
- theta = 0.5 * atan2(sin 2t, cos 2t) in degrees, in (-90, 90], used as-is: never negated.
- opening = width * W_NORM * (out / map) px; jaw = opening / 2 (the reference decoder's rule,
  GR-ConvNet grasp.py:436; not a claim of comparability with published accuracy).
- valid = False (scored as a miss and counted by the caller) if |(cos, sin)| < angle_eps or opening <= 0.
Dtype contract: maps may be float32; all arithmetic here is float64.
"""
import numpy as np

from data.geometry import scale_xy
from data.labels import validate_labels_cfg


def decode_top1(q, cos2, sin2, width, w_norm, out_hw=(1024, 1024), angle_eps=1e-6):
    maps = [np.asarray(m) for m in (q, cos2, sin2, width)]
    shape = maps[0].shape
    if len(shape) != 2 or any(m.shape != shape for m in maps):
        raise ValueError(f"maps must be 2D with one shape, got {[m.shape for m in maps]}")
    for name, m in zip(("q", "cos2", "sin2", "width"), maps):
        if not np.isfinite(m).all():
            raise ValueError(f"non-finite values in the {name} map")
    if w_norm is None or not (np.isfinite(float(w_norm)) and float(w_norm) > 0):
        raise ValueError(f"w_norm must be finite and > 0, got {w_norm}")
    h, w = shape
    sx, sy = out_hw[1] / w, out_hw[0] / h
    if not np.isclose(sx, sy):
        raise ValueError(f"anisotropic decode {shape} -> {out_hw} not supported")
    i, j = divmod(int(np.argmax(maps[0])), w)                 # first maximum in row-major order
    c, s, wd = (float(m[i, j]) for m in maps[1:])
    x, y = scale_xy(np.array([j, i], np.float64), shape, out_hw)
    theta = 0.5 * np.degrees(np.arctan2(s, c))                 # [-90, 90]
    if theta == -90.0:                                         # atan2(-0.0, -1) = -pi: same grasp as +90
        theta = 90.0
    opening = wd * float(w_norm) * sx
    valid = bool(np.hypot(c, s) >= angle_eps and opening > 0)
    return {"grasp": np.array([x, y, theta, opening, opening / 2.0], np.float64), "valid": valid,
            "cell_xy": (j, i), "q": float(maps[0][i, j])}


def decode_from_cfg(maps, cfg, out_hw=(1024, 1024)):
    validate_labels_cfg(cfg, ("decode",))      # smoothing none, tie first row-major, jaw opening/2 (MAJOR_v1 s6)
    want = (int(cfg["size"]), int(cfg["size"]))           # W_NORM is in px at this size: any other
    for k in ("q", "cos2", "sin2", "width"):               # map size would rescale the opening silently
        if np.asarray(maps[k]).shape != want:
            raise ValueError(f"{k} map shape {np.asarray(maps[k]).shape} != configured {want}")
    return decode_top1(maps["q"], maps["cos2"], maps["sin2"], maps["width"],
                       cfg["targets"]["w_norm_px_224"], out_hw, cfg["decode"]["angle_eps"])
