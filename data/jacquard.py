"""Jacquard v2 scene loader.

Layout (verified on disk): <data_root>/Jacquard_Dataset_N/<obj_id>/<view>_<obj_id>_<kind>
with kind in {RGB.png, perfect_depth.tiff, stereo_depth.tiff, mask.png, grasps.txt}.
A scene = (object folder, view index). Splits are by object folder (AGENTS.md).

Conventions (AGENTS.md "Conventions"):
- every mask is 1 = invalid, except `obj_mask` (True = object);
- stereo -1 -> m_sensor = 1, checked on the raw metre values BEFORE converting to mm;
- grasp corners are (x, y) pixel coordinates, y pointing down; integer = pixel CENTRE, so a centre is
  in the image iff -0.5 <= x < 1023.5 (data/geometry.py; use its scale_xy / pixel_index, never x * s).

Changes vs. the reference repo (GRConv robotic-grasping):
- corners are kept in (x, y) order; the reference swaps to (row, col) = (y, x);
- angle: jaw-closing axis a = (cos t, sin t) in (x, y-down), t = Jacquard theta used AS-IS (no
  negation): positive theta turns from +x towards +y, i.e. clockwise on screen. The reference builds
  Grasp((y, x), -theta) and its as_gr yields exactly these corners in (row, col) order;
  tests/test_angle.py::test_matches_reference_as_gr ports as_gr line by line and asserts equality.
  "theta is mirrored" in the Jacquard docs is already accounted for by this formula: a renderer or
  decoder must NOT negate theta again, and must not copy the reference's (row, col) code verbatim.
  Real-data sign checks:
  a one-scene overlay check (quick look only: no filter for angles near 0/90 deg,
  and round objects score the same either way) and scripts/check_angle_convention.py (12+ scenes with
  a depth check; this is the deciding one);
- lines are split with .strip() (the reference's l[:-1] eats the last digit when the file has no
  trailing newline);
- stereo depth is loaded and -1 pixels become NaN (never a number that can blend into valid depth);
  the reference only loads perfect depth and normalises it with no invalid handling.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from data.geometry import in_image
from data.paths import load_paths

REPO = Path(__file__).resolve().parents[1]
IMG_SIZE = 1024
STEREO_INVALID = -1.0  # Jacquard's "undetermined" value in stereo_depth.tiff (metres)


def data_root():
    """Dataset root: configs/paths.json, overridden by DEPTH_LIES_PATHS (data.paths.load_paths)."""
    return load_paths()["data_root"]


# ---------------------------------------------------------------- grasps
class SceneDataError(ValueError):
    """A scene file holds unexpected data. Raised deliberately by the strict parsers below, so that
    try_load_scene can skip bad scenes without also swallowing plain ValueErrors from code bugs."""


class EmptyGraspsError(SceneDataError):
    """Scene has no valid grasp annotation; callers (manifest/dataset) should skip it."""


def parse_grasps(text, return_dropped=False):
    """Jacquard grasp lines 'x;y;theta_deg;opening;jaw' -> (N, 5) float64, same column order.

    Rows that are exactly 0;0;0;0;0 are dropped: a full-dataset scan (the dataset audit, docs/proj_v3.md) found 16 such
    rows in 14 of 51,601 scenes, and no other invalid rows. Any other row with opening <= 0 or
    jaw <= 0 is unexpected and raises instead of being silently dropped.
    """
    rows, dropped = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals = [float(v) for v in line.split(";")]
        except ValueError as e:
            raise SceneDataError(f"unparsable grasp line: {line!r}") from e
        if len(vals) != 5:
            raise SceneDataError(f"bad grasp line: {line!r}")
        if all(v == 0.0 for v in vals):        # known dataset artefact: all-zero row
            dropped += 1
            continue
        if not all(np.isfinite(vals)) or vals[3] <= 0 or vals[4] <= 0:
            raise SceneDataError(f"degenerate grasp line (not the known all-zero case): {line!r}")
        rows.append(vals)
    g = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    return (g, dropped) if return_dropped else g


def grasp_corners(grasps):
    """(N, 5) grasps -> (N, 4, 2) rectangle corners in (x, y) pixels, y pointing down.

    axis a = (cos t, sin t) is the jaw-closing direction (length = opening),
    perp p = (-sin t, cos t) spans the jaw size (width = jaw).
    Corner order: c - L/2 a - W/2 p, c + L/2 a - W/2 p, c + L/2 a + W/2 p, c - L/2 a + W/2 p
    (same order as the reference's Grasp.as_gr, but in (x, y)).
    """
    g = np.asarray(grasps, dtype=np.float64).reshape(-1, 5)
    ctr = g[:, None, 0:2]                                   # (N,1,2) x, y
    t = np.deg2rad(g[:, 2])
    a = np.stack([np.cos(t), np.sin(t)], -1)[:, None, :]    # (N,1,2)
    p = np.stack([-np.sin(t), np.cos(t)], -1)[:, None, :]
    hl = (g[:, 3] / 2)[:, None, None]
    hw = (g[:, 4] / 2)[:, None, None]
    sa = np.array([-1, 1, 1, -1], np.float64)[None, :, None]  # sign along axis
    sp = np.array([-1, -1, 1, 1], np.float64)[None, :, None]  # sign along perp
    return ctr + sa * hl * a + sp * hw * p


# ---------------------------------------------------------------- depth
def stereo_to_mm(raw_m):
    """Raw stereo depth (metres, -1 = undetermined) -> (depth_mm float32 with NaN holes, m_sensor uint8).

    -1 is detected on the raw values, before *1000, so it can never be confused with a valid depth.
    Any other value <= 0 or non-finite is unexpected and raises instead of being silently flagged:
    0 mm at a valid pixel would be indistinguishable from the Protocol B missed-hole sentinel.
    No upper bound, by design: valid stereo depth reaches ~275 m (natural stereo lies; 15% of
    sampled train scenes exceed 1680 mm). A raising bound would turn them into skips. The detector
    input is bounded by the frozen clamp instead (RULES R4, data/lies.clamp_ds).
    """
    raw = np.asarray(raw_m, dtype=np.float32)
    m_sensor = (raw == STEREO_INVALID).astype(np.uint8)     # 1 = invalid
    valid = ~m_sensor.astype(bool)
    other_bad = valid & (~np.isfinite(raw) | (raw <= 0))
    if other_bad.any():
        raise SceneDataError(f"stereo depth has {int(other_bad.sum())} non-positive/non-finite pixels that are not -1")
    depth_mm = _m_to_mm(raw, valid, "stereo")
    depth_mm[m_sensor.astype(bool)] = np.nan                # holes stay holes until explicit completion
    return depth_mm, m_sensor


PERFECT_MAX_MM = 3000.0   # loose sanity bound; 5k train audit D* range is 893-1574 mm (traps note)


def perfect_to_mm(raw_m):
    """Perfect depth (metres) -> mm float32. Every pixel must be finite, > 0 and <= PERFECT_MAX_MM.

    Unlike stereo depth, D* is rendered and has no natural outliers, so a value far outside the
    observed range means a broken file, not a phenomenon: raise (the scene becomes a skip)."""
    raw = np.asarray(raw_m, dtype=np.float32)
    bad = ~np.isfinite(raw) | (raw <= 0)
    if bad.any():
        raise SceneDataError(f"perfect depth has {int(bad.sum())} non-positive/non-finite pixels")
    mm = _m_to_mm(raw, np.ones(raw.shape, bool), "perfect")
    far = mm > np.float32(PERFECT_MAX_MM)
    if far.any():
        raise SceneDataError(f"perfect depth has {int(far.sum())} pixels > {PERFECT_MAX_MM:.0f} mm "
                             f"(max {float(mm.max()):.0f} mm): broken file")
    return mm


def _m_to_mm(raw, valid, name):
    """metres -> mm, re-checking finiteness AFTER the multiply (a huge finite float32 can overflow)."""
    with np.errstate(over="ignore"):
        mm = raw * np.float32(1000.0)
    if not np.isfinite(mm[valid]).all():
        raise SceneDataError(f"{name} depth overflows to non-finite when converted to mm")
    return mm


# ---------------------------------------------------------------- scene
def scene_paths(obj_dir, view):
    obj_dir = Path(obj_dir)
    if not obj_dir.is_absolute():
        obj_dir = Path(data_root()) / obj_dir
    obj = obj_dir.name
    kinds = {"rgb": "RGB.png", "depth_perfect": "perfect_depth.tiff", "depth_stereo": "stereo_depth.tiff",
             "obj_mask": "mask.png", "grasps": "grasps.txt"}
    return {k: obj_dir / f"{int(view)}_{obj}_{v}" for k, v in kinds.items()}


def _imread(path, flags):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path), flags)
    if img is None:                                         # file exists, so it is corrupt, not missing
        raise SceneDataError(f"{path}: exists but cv2 could not decode it")
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        raise SceneDataError(f"{path}: expected {IMG_SIZE}x{IMG_SIZE}, got {img.shape}")
    return img


def _read_depth(path):
    d = _imread(path, cv2.IMREAD_UNCHANGED)
    if d.ndim != 2 or d.dtype.kind != "f":
        raise SceneDataError(f"{path}: expected single-channel float depth, got {d.dtype} {d.shape}")
    return d


MASK_BORDER_THRESH = 0.5   # same default as the dataset audit's border threshold


def _object_mask(mask, path):
    """mask.png -> bool (True = object). Enforces what the dataset audit (docs/proj_v3.md) checks: single-channel uint8,
    values {0,1} or {0,255}, both object and background present, and polarity (object = nonzero):
    if more than MASK_BORDER_THRESH of the border pixels are 'object', the mask is likely inverted.
    Re-checked here so a file changed after the audit cannot slip through."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise SceneDataError(f"{path}: object mask must be single-channel uint8, got {mask.dtype} {mask.shape}")
    present = set(np.flatnonzero(np.bincount(mask.ravel(), minlength=256)).tolist())
    if not (present <= {0, 1} or present <= {0, 255}):
        raise SceneDataError(f"{path}: object mask values {sorted(present)[:8]} not in {{0,1}} or {{0,255}}")
    if len(present) < 2:
        raise SceneDataError(f"{path}: object mask is {'empty' if present == {0} else 'full-frame'}")
    fg = mask > 0
    border = np.concatenate([fg[0], fg[-1], fg[1:-1, 0], fg[1:-1, -1]])
    if border.mean() > MASK_BORDER_THRESH:
        raise SceneDataError(f"{path}: {100 * border.mean():.1f}% of border pixels are object "
                             f"(> {100 * MASK_BORDER_THRESH:.0f}%): mask likely inverted (polarity)")
    return fg


def load_scene(obj_dir, view):
    """Load one Jacquard scene. obj_dir: absolute, or relative to data_root (configs/paths.json)."""
    p = scene_paths(obj_dir, view)
    try:
        text = p["grasps"].read_text()                                         # cheapest file first
    except UnicodeDecodeError as e:
        raise SceneDataError(f"{p['grasps']}: not valid text ({e})") from e
    grasps, n_dropped = parse_grasps(text, return_dropped=True)
    if len(grasps) == 0:
        raise EmptyGraspsError(f"{p['grasps']}: 0 valid grasp rows ({n_dropped} all-zero rows dropped)")
    out = ~in_image(grasps[:, 0:2], (IMG_SIZE, IMG_SIZE))                    # pixel-centre convention
    if out.any():
        raise SceneDataError(f"{p['grasps']}: {int(out.sum())} grasp centres outside the "
                             f"{IMG_SIZE}x{IMG_SIZE} image, e.g. {grasps[out][0, :2].tolist()}")
    # validate the stored format BEFORE any conversion (IMREAD_COLOR would silently turn gray/RGBA into 3ch)
    bgr = _imread(p["rgb"], cv2.IMREAD_UNCHANGED)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise SceneDataError(f"{p['rgb']}: expected uint8 3-channel colour, got {bgr.dtype} {bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                                   # uint8 HxWx3, RGB order
    obj_mask = _object_mask(_imread(p["obj_mask"], cv2.IMREAD_UNCHANGED), p["obj_mask"])  # True = object
    depth_stereo, m_sensor = stereo_to_mm(_read_depth(p["depth_stereo"]))
    return {
        "rgb": rgb,
        "depth_perfect": perfect_to_mm(_read_depth(p["depth_perfect"])),        # D*, mm; oracle/A* only
        "depth_stereo": depth_stereo,                                             # D_s, mm, NaN where m_sensor=1
        "obj_mask": obj_mask,
        "m_sensor": m_sensor,
        "grasps": grasps,                                                         # (N,5) x,y,theta_deg,opening,jaw
        "grasp_corners": grasp_corners(grasps),                                   # (N,4,2) x,y
        "n_grasps_dropped": n_dropped,                                            # all-zero rows removed
        "obj_id": Path(p["rgb"]).parent.name,
        "view": int(view),
    }


# Expected data errors: the strict parsers raise SceneDataError (EmptyGraspsError is one; so does a
# file that exists but cannot be decoded), missing files raise FileNotFoundError, cv2 failures cv2.error.
# Other OSErrors (EMFILE "too many open files", EIO, ENOSPC, PermissionError) are resource or system
# faults, and a plain ValueError / TypeError is a code bug: all of these must crash, not become skips.
SCENE_ERRORS = (SceneDataError, FileNotFoundError, cv2.error)


class SkipBudget:
    """RULES R3 hard stop for any loop over scenes: abort if more than `max_total` scenes are skipped
    in total, or `max_run` in a row. Call loaded() after every success and skipped(rec) after every
    skip record from try_load_scene; put summary() into the output's summary JSON."""

    def __init__(self, max_total=50, max_run=10):
        self.max_total, self.max_run = int(max_total), int(max_run)
        self.records, self.run, self.max_run_seen = [], 0, 0

    def loaded(self):
        self.run = 0

    def skipped(self, record):
        self.records.append(record)
        self.run += 1
        self.max_run_seen = max(self.max_run_seen, self.run)
        if len(self.records) > self.max_total:
            raise RuntimeError(f"{len(self.records)} scenes skipped (> {self.max_total}); aborting. "
                               f"Last: {record}")
        if self.run >= self.max_run:
            raise RuntimeError(f"{self.run} scenes skipped in a row (limit {self.max_run}); aborting. "
                               f"Last: {record}")

    def summary(self):
        return {"n_skipped": len(self.records), "max_run_seen": self.max_run_seen,
                "max_total": self.max_total, "max_run": self.max_run, "skips": list(self.records)}


def try_load_scene(obj_dir, view, root=None):
    """load_scene that skips bad scenes: returns (scene, None) or (None, skip_record).

    root: dataset root the scene belongs to (default: data_root() from configs/paths.json). A relative
    obj_dir is resolved against it; an absolute obj_dir must lie inside it (else ValueError: a caller
    mixing datasets is a code bug, and the drive check below would test the wrong root).
    skip_record = {"obj_dir", "view", "error_type", "message"}.
    On every skip, root must still exist; if not (e.g. the USB drive dropped), raise RuntimeError
    instead of turning every remaining scene into a skip.

    Hard stops the cache builder must add on top (expected skips = 0: the full manifest found no
    unreadable scenes, the extras check only the all-zero grasp rows):
      - feed every result into a SkipBudget (aborts at > 50 skipped or 10 in a row);
      - write SkipBudget.summary() to the cache's summary JSON.
    Note: depth_stereo holds NaN at m_sensor=1, so statistics must use the mask or np.nan* functions.
    """
    # resolve(): normalises '..' and symlinks, so the containment check is not merely lexical
    root = Path(root if root is not None else data_root()).expanduser().resolve()
    obj = Path(obj_dir).expanduser()
    obj = (obj if obj.is_absolute() else root / obj).resolve()
    if root not in obj.parents:
        raise ValueError(f"{obj} is not inside the dataset root {root}")
    try:
        return load_scene(obj, view), None
    except SCENE_ERRORS as e:
        if not root.is_dir():
            raise RuntimeError(f"dataset root {root} is missing (drive disconnected?); "
                               f"aborting instead of skipping {obj_dir} view {view}") from e
        return None, {"obj_dir": str(obj_dir), "view": int(view),
                      "error_type": type(e).__name__, "message": str(e)}
