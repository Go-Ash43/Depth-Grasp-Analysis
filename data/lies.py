"""Natural stereo lies and the detector-input clamp: the ONE implementation used by the cache builder
and the Module 2 harness.

Definitions (SPEC_trajectory "Masks and depth, redefined once"; docs/RULES_preprocessing.md R2, R4):
  M_lie       = 1[M_sensor = 0 and |D_s - D*| > tau]   at native 1024 px, from RAW D_s (strict >)
  deployed    = resize_any(M_sensor) | M_synth_224      (the flag a real pipeline shows the detector)
  oracle      = deployed | resize_rule(M_lie)
  lie_eff_224 = oracle & ~deployed                      (R2: cells where the two flags differ)
  clamp       = valid D_s clipped to [lo, hi] mm, holes untouched; detector input only (R4)

These match the frozen 5k train audit (scripts/audit_train.py, audit_scene) count for count;
tests/test_lies.py asserts that on synthetic scenes, so the cache cannot drift from the numbers that
fixed tau = 12 mm and kept the natural-lies row. All masks are uint8 with 1 = invalid / lie.
"""
import numpy as np

from data.mask_ops import _as_mask, resize_mask_any, resize_mask_fraction


def _depth(depth, name, shape=None):
    d = np.asarray(depth)
    if d.ndim != 2 or d.dtype.kind != "f":
        raise ValueError(f"{name} must be a 2D float array, got {d.dtype} {d.shape}")
    if shape is not None and d.shape != tuple(shape):
        raise ValueError(f"{name} shape {d.shape} != {tuple(shape)} (refusing to broadcast)")
    return d


def _positive(v, name):
    v = float(v)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and > 0, got {v}")
    return v


def m_lie(ds_raw_mm, dstar_mm, m_sensor, tau_mm):
    """M_lie at native resolution (uint8, 1 = lie). ds_raw_mm is the UNCLAMPED stereo depth in mm
    (any value at holes, e.g. NaN from load_scene); dstar_mm the perfect depth in mm.

    Strict '>' (a pixel exactly tau off is not a lie). Hole pixels are never lies. Same float
    arithmetic as audit_scene: |D_s - D*| in the input dtype, compared with tau.
    """
    ds = _depth(ds_raw_mm, "ds_raw_mm")
    dstar = _depth(dstar_mm, "dstar_mm", ds.shape)
    hole = _as_mask(m_sensor, ds.shape, "m_sensor")
    tau = _positive(tau_mm, "tau_mm")
    valid = ~hole
    if not np.isfinite(ds[valid]).all():
        raise ValueError("non-finite stereo depth at a pixel M_sensor marks valid")
    if not np.isfinite(dstar).all():
        raise ValueError("non-finite perfect depth")
    lie = np.zeros(ds.shape, bool)
    lie[valid] = np.abs(ds[valid] - dstar[valid]) > tau
    return lie.astype(np.uint8)


def deployed_224(m_sensor, size, m_synth_224=None):
    """The as-deployed flag at model resolution: resize_any(M_sensor) | M_synth_224 (uint8)."""
    dep = resize_mask_any(_as_mask(m_sensor, name="m_sensor"), size).astype(bool)
    if m_synth_224 is not None:
        dep |= _as_mask(m_synth_224, tuple(size), "m_synth_224")
    return dep.astype(np.uint8)


def lie_eff_224(m_lie_native, m_sensor, size, m_synth_224=None, rule="any", min_frac=0.5):
    """RULES R2: cells where the oracle flag differs from the as-deployed flag (uint8).

    rule="any" (primary): a cell is lied-about if any source pixel is in M_lie.
    rule="fraction" (pre-registered sensitivity): if >= min_frac of the cell is in M_lie.
    Either way, cells already flagged by the deployed flag (sensor hole or synthetic hole) are 0:
    there the model sees the same flag under both settings.
    """
    lie = _as_mask(m_lie_native, name="m_lie")
    hole = _as_mask(m_sensor, lie.shape, "m_sensor")
    if (lie & hole).any():
        raise ValueError("M_lie overlaps M_sensor at native resolution (M_lie must come from valid pixels)")
    if rule == "any":
        lie_s = resize_mask_any(lie, size)
    elif rule == "fraction":
        lie_s = resize_mask_fraction(lie, size, min_frac)
    else:
        raise ValueError(f"rule must be 'any' or 'fraction', got {rule!r}")
    dep = deployed_224(hole, size, m_synth_224).astype(bool)
    return (lie_s.astype(bool) & ~dep).astype(np.uint8)


def lie_eff_from_224(lie_224, deployed):
    """RULES R2 from masks already at model resolution: lie_224 & ~deployed (uint8).

    lie_224 is the CACHED resize of native M_lie (resize_mask_any or resize_mask_fraction, before any
    flag is subtracted); deployed is deployed_224(...) for this sample, including its M_synth. Equals
    lie_eff_224(native M_lie, ...) exactly (tests/test_lies.py); used by data-loader workers, which
    only see the 224 cache.
    """
    lie = _as_mask(lie_224, name="lie_224")
    dep = _as_mask(deployed, lie.shape, "deployed")
    return (lie & ~dep).astype(np.uint8)


def clamp_ds(ds_raw_mm, m_sensor, lo_mm, hi_mm):
    """RULES R4 clamp of the detector's stereo input: valid pixels clipped to [lo, hi] mm.

    Holes are left exactly as they are (NaN, -1, 0, ...), so the clamp can never turn a hole into a
    valid-looking depth or shrink M_sensor. Returns (clamped copy, changed uint8 mask over valid
    pixels). Apply BEFORE resize_depth, Telea and any Protocol-B sentinel; compute M_sensor and M_lie
    from the raw input, never from this output.
    """
    ds = _depth(ds_raw_mm, "ds_raw_mm")
    hole = _as_mask(m_sensor, ds.shape, "m_sensor")
    lo, hi = float(lo_mm), float(hi_mm)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"clamp range must be finite with lo < hi, got [{lo}, {hi}]")
    valid = ~hole
    v = ds[valid]
    if not np.isfinite(v).all():
        raise ValueError("non-finite stereo depth at a pixel M_sensor marks valid")
    out = ds.copy()
    out[valid] = np.clip(v, lo, hi)
    changed = np.zeros(ds.shape, bool)
    changed[valid] = out[valid] != v
    return out, changed.astype(np.uint8)


def clamp_truthful_changed(ds_raw_mm, dstar_mm, m_sensor, lo_mm, hi_mm, tau_mm):
    """RULES R4 hard check: number of valid pixels with raw |D_s - D*| <= 2 tau that the clamp
    changes. Must be 0 on every train scene (else widen the range and rebuild); logged on val/test."""
    ds = _depth(ds_raw_mm, "ds_raw_mm")
    dstar = _depth(dstar_mm, "dstar_mm", ds.shape)
    hole = _as_mask(m_sensor, ds.shape, "m_sensor")
    tau = _positive(tau_mm, "tau_mm")
    if not np.isfinite(dstar).all():        # a NaN D* would compare False and pass the gate silently
        raise ValueError("non-finite perfect depth: the clamp check cannot be evaluated")
    _, changed = clamp_ds(ds, hole, lo_mm, hi_mm)
    valid = ~hole
    truthful = np.zeros(ds.shape, bool)
    truthful[valid] = np.abs(ds[valid] - dstar[valid]) <= 2 * tau
    return int((changed.astype(bool) & truthful).sum())
