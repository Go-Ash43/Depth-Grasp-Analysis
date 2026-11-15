"""Training targets at 224 px from Jacquard grasps (notes/MAJOR_v1.md section 6; configs/labels.json).

Per pixel p (integer = pixel centre, data/geometry.py), with deduplicated grasps g scaled to 224:
  w_g(p) = exp(-|p - c_g|^2 / (2 sigma^2))
  owner(p) = argmax_g w_g(p)  (nearest centre; ties -> lowest row of the sorted dedup array)
  Q(p) = w_owner(p);  angle = (cos 2t, sin 2t) of the owner;  width = opening_owner / W_NORM
The angle and opening always come from the SAME grasp: no averaging (equal-weight 0 and 60 deg would
average to 30 deg, matching neither). void(p) = 1 where Q(p) < q_void: angle/width are left out of
the loss there (Q is still trained). Q is computed exactly inside a window around each centre and
is 0 outside it, i.e. truncated where w < q_floor (1e-6).

W_NORM is a scale factor, never a cap: any finite opening > 0 is accepted and an opening above
W_NORM gives width > 1, unclipped (the width head must be unbounded above).
Dtype contract: geometry in float64; output maps float32; void uint8 (1 = excluded); owner int32.
Deduplication of jaw-size repeats happens HERE ONLY; the match set (eval/match.py) keeps all rows.
"""
import json
from pathlib import Path

import numpy as np

from data.geometry import in_image, scale_grasps

REPO = Path(__file__).resolve().parents[1]
SRC_HW = (1024, 1024)


# Behaviour the code implements (review finding 11): configs/labels.json must state exactly these, so
# a config can never claim one policy while the code runs another. Changing one means changing the code.
DEDUP_KEY = ["x", "y", "theta_canonical", "opening"]
THETA_CANONICAL = "(-90, 90]"
OWNER_RULE = "largest Gaussian weight (nearest centre); tie -> lowest row of the sorted dedup array"
DECODE_SMOOTHING, DECODE_TIE, JAW_RULE = "none", "first row-major", "opening/2"
MATCH_KEYS = ("primary", "strict")
CFG_SECTIONS = ("targets", "decode", "match")


def _finite_pos(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
        raise ValueError(f"labels cfg: {name} must be a finite number > 0, got {v!r}")


def _expect(v, want, name):
    if v != want:
        raise ValueError(f"labels cfg: {name} = {v!r}, but the code implements {want!r}")


def validate_labels_cfg(cfg, sections=CFG_SECTIONS):
    """Raise ValueError unless the named sections of a labels cfg are well-formed AND describe the
    policies this code implements. Returns cfg. Covers targets, decode and match ONLY: the zones
    section is guarded by data.zones (selected_zone refuses until Stage B), not here."""
    bad = [s for s in sections if s not in CFG_SECTIONS]
    if bad:
        raise ValueError(f"unknown cfg section(s) {bad}")
    for s in sections:
        if not isinstance(cfg.get(s), dict):
            raise ValueError(f"labels cfg: missing section {s!r}")
    if "targets" in sections or "decode" in sections:          # both use the map size
        size = cfg.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"labels cfg: size must be an int > 0, got {size!r}")
    if "targets" in sections:
        t = cfg["targets"]
        _finite_pos(t.get("sigma_px_224"), "targets.sigma_px_224")
        _finite_pos(t.get("q_floor"), "targets.q_floor")
        _finite_pos(t.get("q_void"), "targets.q_void")
        if not t["q_floor"] < t["q_void"] < 1:
            raise ValueError(f"labels cfg: need 0 < q_floor < q_void < 1, got {t['q_floor']}, {t['q_void']}")
        if t.get("w_norm_px_224") is not None:                 # null allowed here; make_targets refuses it
            _finite_pos(t["w_norm_px_224"], "targets.w_norm_px_224")
        _expect(t.get("dedup_key"), DEDUP_KEY, "targets.dedup_key")
        _expect(t.get("theta_canonical"), THETA_CANONICAL, "targets.theta_canonical")
        _expect(t.get("owner"), OWNER_RULE, "targets.owner")
    if "decode" in sections:
        d = cfg["decode"]
        _expect(d.get("smoothing"), DECODE_SMOOTHING, "decode.smoothing")
        _expect(d.get("tie"), DECODE_TIE, "decode.tie")
        _expect(d.get("jaw_rule"), JAW_RULE, "decode.jaw_rule")
        _finite_pos(d.get("angle_eps"), "decode.angle_eps")
    if "match" in sections:
        m = cfg["match"]
        if sorted(map(str, m)) != sorted(MATCH_KEYS):
            raise ValueError(f"labels cfg: match must have exactly {MATCH_KEYS}, got {sorted(m)}")
        for k in MATCH_KEYS:
            if not isinstance(m[k], dict):
                raise ValueError(f"labels cfg: match.{k} must be a mapping, got {m[k]!r}")
            a, iou = m[k].get("angle_max_deg"), m[k].get("iou_min_exclusive")
            _finite_pos(a, f"match.{k}.angle_max_deg")
            if a > 90:
                raise ValueError(f"labels cfg: match.{k}.angle_max_deg must be <= 90, got {a}")
            if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not 0 <= iou < 1:
                raise ValueError(f"labels cfg: match.{k}.iou_min_exclusive must be in [0, 1), got {iou!r}")
    return cfg


def load_labels_cfg(path=None):
    return validate_labels_cfg(json.loads(Path(path or REPO / "configs" / "labels.json").read_text()))


def canonical_theta(theta_deg):
    """Grasp angle mod 180 deg into (-90, 90]: a grasp is symmetric under a half turn."""
    t = (np.asarray(theta_deg, np.float64) + 90.0) % 180.0 - 90.0
    return np.where(t == -90.0, 90.0, t)


def dedup_for_targets(grasps):
    """(N, 5) x, y, theta, opening, jaw -> (K, 4) unique (x, y, theta_canonical, opening), sorted
    lexicographically (np.unique). Jaw-size repeats collapse; theta and theta + 180 collapse."""
    g = np.array(grasps, dtype=np.float64).reshape(-1, 5)
    if not np.isfinite(g).all():
        raise ValueError("non-finite grasp values")
    if (g[:, 3] <= 0).any():
        raise ValueError("grasp opening must be > 0")
    key = g[:, :4].copy()
    key[:, 2] = canonical_theta(key[:, 2])
    return np.unique(key, axis=0)


def _w_norm(w_norm):
    if w_norm is None:
        raise ValueError("w_norm_px_224 is null: set it from the train label range before training")
    w = float(w_norm)
    if not (np.isfinite(w) and w > 0):
        raise ValueError(f"w_norm must be finite and > 0, got {w_norm}")
    return w


def make_targets(grasps_1024, size=224, sigma=2.0, q_void=0.05, w_norm=None, q_floor=1e-6,
                 src_hw=SRC_HW):
    """Grasps at src_hw -> dict of target maps at (size, size); see module docstring."""
    w_norm = _w_norm(w_norm)
    sigma, q_void, q_floor = float(sigma), float(q_void), float(q_floor)
    if not (sigma > 0 and 0 < q_floor < q_void < 1):
        raise ValueError(f"need sigma > 0 and 0 < q_floor < q_void < 1, got {sigma}, {q_floor}, {q_void}")
    hw = (size, size)
    u = dedup_for_targets(grasps_1024)
    if len(u) == 0:
        raise ValueError("no grasps: R1 excludes such scenes before targets are made")
    s = scale_grasps(np.c_[u, np.ones(len(u))], src_hw, hw)      # x, y via pixel-centre scaling
    cx, cy, theta, opening = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    if not in_image(s[:, :2], hw).all():
        raise ValueError("a grasp centre falls outside the target grid")

    best = np.zeros(hw, np.float64)
    owner = np.full(hw, -1, np.int32)
    r = int(np.ceil(sigma * np.sqrt(2.0 * np.log(1.0 / q_floor)))) + 1   # w < q_floor beyond r
    for k in range(len(u)):                                      # sorted order: ties keep the lowest k
        j0, j1 = max(int(np.floor(cx[k])) - r, 0), min(int(np.ceil(cx[k])) + r, size - 1)
        i0, i1 = max(int(np.floor(cy[k])) - r, 0), min(int(np.ceil(cy[k])) + r, size - 1)
        jj = np.arange(j0, j1 + 1, dtype=np.float64)[None, :]
        ii = np.arange(i0, i1 + 1, dtype=np.float64)[:, None]
        w = np.exp(-((jj - cx[k]) ** 2 + (ii - cy[k]) ** 2) / (2.0 * sigma * sigma))
        w[w < q_floor] = 0.0
        win_b, win_o = best[i0:i1 + 1, j0:j1 + 1], owner[i0:i1 + 1, j0:j1 + 1]
        take = w > win_b                                         # strict: an equal weight never steals
        win_b[take] = w[take]
        win_o[take] = k

    has = owner >= 0
    t2 = np.deg2rad(2.0 * theta)
    cos2 = np.zeros(hw, np.float64)
    sin2 = np.zeros(hw, np.float64)
    width = np.zeros(hw, np.float64)
    cos2[has], sin2[has] = np.cos(t2)[owner[has]], np.sin(t2)[owner[has]]
    width[has] = opening[owner[has]] / w_norm                    # may exceed 1: never clipped
    return {"q": best.astype(np.float32), "cos2": cos2.astype(np.float32),
            "sin2": sin2.astype(np.float32), "width": width.astype(np.float32),
            "void": (best < q_void).astype(np.uint8), "owner": owner,
            "grasps_224": np.c_[cx, cy, theta, opening]}


def targets_from_cfg(grasps_1024, cfg):
    validate_labels_cfg(cfg, ("targets",))
    t = cfg["targets"]
    return make_targets(grasps_1024, size=cfg["size"], sigma=t["sigma_px_224"], q_void=t["q_void"],
                        w_norm=t["w_norm_px_224"], q_floor=t["q_floor"])
