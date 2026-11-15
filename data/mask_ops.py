"""Mask resizing and depth completion. All masks: 1 = invalid.

Per-scene order (SPEC_trajectory "Preprocessing cache at 224 px"):
  load at 1024 -> M_sensor, M_lie at 1024 -> resize_depth (D_s) / resize_mask_any (masks)
  -> cache at 224 -> telea_fill at 224 in data-loader workers.
Depth arrays carry NaN at invalid pixels until telea_fill; any statistic on them must use the
mask or np.nan* functions.
"""
import cv2
import numpy as np


def _as_depth(depth):
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {d.shape}")
    return d


def _as_mask(mask, shape=None, name="mask"):
    """Strict mask check -> bool array. 2D, exactly `shape` if given (no broadcasting), and bool or
    values in {0, 1} only: a 255 / -1 / 0.5 mask is a bug upstream, not something to coerce."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"{name} must be 2D (H, W), got shape {m.shape}")
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"{name} shape {m.shape} != depth shape {tuple(shape)} (refusing to broadcast)")
    if m.dtype != bool:
        if not (np.issubdtype(m.dtype, np.integer) or np.issubdtype(m.dtype, np.floating)):
            raise ValueError(f"{name} has non-numeric dtype {m.dtype}")
        if not ((m == 0) | (m == 1)).all():
            raise ValueError(f"{name} must hold only 0/1 (1 = invalid) or bool; got other values")
    return m.astype(bool)


def _check_downsample(src_shape, size):
    """The any/fraction rules below are defined for DOWNsampling only: when upsampling, INTER_AREA
    interpolates and `> 0` no longer means "covers an invalid source pixel". Raise instead."""
    h, w = size
    if not (1 <= h <= src_shape[0] and 1 <= w <= src_shape[1]):
        raise ValueError(f"only downsampling is supported: {tuple(src_shape[:2])} -> {(h, w)}")


def _coverage(mask, size, name):
    """Fraction of each output cell covered by mask=1 source pixels (INTER_AREA = exact area weights)."""
    m = _as_mask(mask, name=name)
    _check_downsample(m.shape, size)
    h, w = size
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def resize_mask_any(mask, size):
    """Downsample a 1=invalid mask to (h, w); an output pixel is invalid if ANY source pixel it covers is.

    INTER_AREA weights each source pixel by its overlap with the output cell, so a cell touched by
    even a sliver of an invalid pixel gets a value > 0. Works for non-integer factors (1024 -> 224).
    Grows holes by up to one output pixel at their edges, so M_sensor % is higher at 224 than at 1024.
    """
    return (_coverage(mask, size, "mask") > 0).astype(np.uint8)


def resize_mask_fraction(mask, size, min_frac=0.5):
    """Downsample a 1=invalid mask: output = 1 where the covered fraction >= min_frac.

    RULES R2 sensitivity rule for M_lie: under the any-rule a 224 cell counts as lied-about when 1 of
    ~21 source pixels exceeds tau, so the oracle flag partly acts as a false alarm. Reported alongside
    the any-rule (primary), never instead of it. Returns uint8 1 = invalid."""
    if not 0.0 < min_frac <= 1.0:
        raise ValueError(f"min_frac must be in (0, 1], got {min_frac}")
    return (_coverage(mask, size, "mask") >= min_frac - 1e-6).astype(np.uint8)   # float32 slack


def resize_mask_majority(mask, size):
    """Downsample a boolean/0-1 mask to (h, w) by majority: output = 1 where covered fraction > 0.5.
    Used for the object mask (True = object), where the any-rule would grow the object."""
    return _coverage(mask, size, "object mask") > 0.5


def resize_depth(depth_mm, invalid, size):
    """Resize depth to (h, w); returns (float32 depth with NaN at invalid cells, uint8 1=invalid mask).

    Holes are zeroed, the depth is resized with INTER_AREA, then the holes are re-marked with the
    any-rule (resize_mask_any). Both resizes run the same float32 INTER_AREA, so a valid output cell
    gives zero weight to every invalid source pixel and the zero fill cannot contaminate it.
    """
    depth_mm = _as_depth(depth_mm)
    _check_downsample(depth_mm.shape, size)
    hole = _as_mask(invalid, depth_mm.shape, "invalid")
    # Source-side check: a NaN/inf at a pixel marked valid that shares an output cell with a true hole
    # would be masked by the any-rule below and never reach the output-side check.
    if not np.isfinite(depth_mm[~hole]).all():
        raise ValueError("non-finite source depth at pixels marked valid (depth/mask mismatch)")
    d = np.where(hole, 0.0, depth_mm).astype(np.float32)
    out = cv2.resize(d, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    inv = resize_mask_any(hole, size)
    out[inv.astype(bool)] = np.nan
    if not np.isfinite(out[~inv.astype(bool)]).all():   # explicit raise: asserts vanish under -O
        raise ValueError("non-finite depth in a valid output cell (NaN/inf outside the invalid mask?)")
    return out, inv


def telea_fill(depth_mm, invalid, radius=3):
    """Fill pixels with invalid=1 using cv2.inpaint (Telea). float32 is accepted directly (checked on
    cv2 5.0.0), so no dtype conversion. Holes are zeroed first so NaNs cannot leak into the fill.

    `invalid` is the set to complete. Protocol A: the true invalid set M_sensor | M_synth,
    regardless of the flag the detector sees. Protocol B: the reported flag M_rep, which
    includes M_lie only in named oracle conditions. Never derived from D* otherwise.
    Prefer complete_protocol_a / complete_protocol_b, which pick the right set by construction.
    """
    d = _as_depth(depth_mm).astype(np.float32, copy=True)
    hole = _as_mask(invalid, d.shape, "invalid")
    if hole.all():
        # cv2.inpaint has no valid pixel to propagate from and would silently return all zeros
        raise ValueError("invalid mask covers every pixel: no valid depth to complete from")
    if not np.isfinite(d[~hole]).all():
        raise ValueError("non-finite depth outside the invalid mask")
    d[hole] = 0.0
    return cv2.inpaint(d, hole.astype(np.uint8), float(radius), cv2.INPAINT_TELEA)


# Sec. "Protocols" (SPEC_hypotheses): value written into missed holes under Protocol B. 0 mm is the
# conventional "no reading" of real sensors and matches the zeroed-flag input of 6g. Log it in configs.
MISSED_HOLE_SENTINEL_MM = 0.0


def complete_protocol_a(depth_mm, m_true, radius=3):
    """Protocol A (primary): complete over the TRUE invalid set (M_sensor | M_synth) and hold the
    result fixed. Deliberately takes no M_rep: flag-only edits cannot reach the depth, so R and A0
    stay bit-identical under any flag change."""
    return telea_fill(depth_mm, m_true, radius)


def complete_protocol_b(depth_mm, m_true, m_rep, radius=3, sentinel_mm=MISSED_HOLE_SENTINEL_MM):
    """Protocol B (secondary): completion driven by the reported flag M_rep, as in a real pipeline.

    - missed hole (m_true=1, m_rep=0): stays unfilled, marked valid, holds `sentinel_mm`;
    - false alarm (m_true=0, m_rep=1): good depth is overwritten by inpainting;
    - oracle conditions pass an M_rep that includes M_lie, so flagged errors are re-completed.
    Returns (completed float32 depth, missed uint8 mask).

    Intended, not a bug: sentinel pixels are "valid" to Telea, so a flagged region bordering a missed
    one is pulled toward 0 mm, as it would be in a real pipeline that trusts the flag (SPEC_hypotheses
    "Protocols": missed holes are marked valid). Report it as part of the Protocol B effect.
    Raises if no pixel outside M_rep holds real depth (all would-be Telea sources are sentinels):
    the fill would then be all sentinel, silently.
    """
    d = _as_depth(depth_mm).astype(np.float32, copy=True)
    true = _as_mask(m_true, d.shape, "m_true")
    rep = _as_mask(m_rep, d.shape, "m_rep")
    missed = true & ~rep
    if not rep.all() and not (~true & ~rep).any():       # rep.all() is telea_fill's own error
        raise ValueError("Protocol B: every unflagged pixel is a missed-hole sentinel; no real depth "
                         "to complete from")
    d[missed] = np.float32(sentinel_mm)
    return telea_fill(d, rep, radius), missed.astype(np.uint8)
