"""-1 handling, any-invalid resize and Telea completion on synthetic arrays (no dataset files)."""
import inspect

import cv2
import numpy as np
import pytest

from data.jacquard import SceneDataError, stereo_to_mm
from data.mask_ops import (MISSED_HOLE_SENTINEL_MM, complete_protocol_a, complete_protocol_b,
                           resize_depth, resize_mask_any, resize_mask_fraction, resize_mask_majority,
                           telea_fill)

N, OUT = 1024, 224


def synthetic_stereo_m():
    # smooth plane 0.6..0.9 m with a -1 block hole and a single -1 pixel
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    d = 0.6 + 0.3 * (xx + yy) / (2 * N)
    d[400:420, 500:530] = -1.0
    d[777, 333] = -1.0
    return d


def test_m_sensor_marks_minus_one_before_mm():
    raw = synthetic_stereo_m()
    mm, m = stereo_to_mm(raw)
    assert m.dtype == np.uint8 and set(np.unique(m)) == {0, 1}
    assert m[400:420, 500:530].all() and m[777, 333] == 1
    assert m.sum() == 20 * 30 + 1
    assert np.isnan(mm[m == 1]).all()                          # holes never carry a number
    np.testing.assert_allclose(mm[m == 0], raw[m == 0] * 1000, rtol=1e-6)
    assert mm.dtype == np.float32


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.001, -2.0, 0.0])   # 0 would collide with the sentinel
def test_other_invalid_values_raise(bad):
    raw = synthetic_stereo_m()
    raw[10, 10] = bad
    with pytest.raises(SceneDataError):        # try_load_scene skips only this; a plain ValueError crashes
        stereo_to_mm(raw)


def any_reference(mask, out):
    # independent any-pool: output cell i covers source [i*s, (i+1)*s); include every pixel it touches
    s = mask.shape[0] / out
    edges = [(int(np.floor(i * s)), int(np.ceil((i + 1) * s))) for i in range(out)]
    rows = np.stack([mask[a:b].any(0) for a, b in edges])
    return np.stack([rows[:, a:b].any(1) for a, b in edges], 1).astype(np.uint8)


SIZES = [224, 160]                                             # primary, and the spec's 160-px fallback


@pytest.mark.parametrize("out", SIZES)
def test_resize_keeps_holes(out):
    _, m = stereo_to_mm(synthetic_stereo_m())
    r = resize_mask_any(m, (out, out))
    s = N / out
    assert r.shape == (out, out) and r.dtype == np.uint8
    assert r[int(777 / s), int(333 / s)] == 1                  # single pixel survives
    assert r[int(400 / s):int(419 / s) + 1, int(500 / s):int(529 / s) + 1].all()
    np.testing.assert_array_equal(r, any_reference(m.astype(bool), out))


@pytest.mark.parametrize("out", SIZES)
def test_resize_matches_reference_random_sparse(out):
    rng = np.random.default_rng(0)
    m = (rng.random((N, N)) < 2e-4).astype(np.uint8)          # ~200 isolated pixels, incl. cell borders
    m[0, 0] = m[N - 1, N - 1] = m[31, 32] = m[32, 31] = 1     # 32 = 7*s is an exact cell boundary at 224
    m[63, 64] = m[64, 63] = 1                                  # 64 = 10*s is an exact cell boundary at 160
    np.testing.assert_array_equal(resize_mask_any(m, (out, out)), any_reference(m.astype(bool), out))


@pytest.mark.parametrize("fn", [resize_mask_any, resize_mask_majority, resize_mask_fraction])
def test_mask_upsampling_rejected(fn):
    m = np.zeros((OUT, OUT), np.uint8); m[5, 5] = 1
    with pytest.raises(ValueError, match="downsampling"):
        fn(m, (N, N))
    with pytest.raises(ValueError, match="downsampling"):
        fn(m, (OUT, OUT + 1))                                  # one axis up is still upsampling


def test_depth_upsampling_rejected():
    with pytest.raises(ValueError, match="downsampling"):
        resize_depth(_plane_224(), np.zeros((OUT, OUT), np.uint8), (N, N))


@pytest.mark.parametrize("out", SIZES)
def test_resize_mask_fraction(out):
    # RULES R2 sensitivity rule: a single lied pixel flags a cell under the any-rule but not at >= 50%
    m = np.zeros((N, N), np.uint8)
    m[700, 700] = 1
    s = N / out
    cell = (int(700 / s), int(700 / s))
    assert resize_mask_any(m, (out, out))[cell] == 1
    assert resize_mask_fraction(m, (out, out), 0.5)[cell] == 0
    # 256 px is an exact cell boundary at both 224 (56 cells) and 160 (40 cells): fully covered cells -> 1
    m[:] = 0; m[:256, :256] = 1
    k = int(round(256 / s))
    r = resize_mask_fraction(m, (out, out), 0.5)
    assert r.dtype == np.uint8 and r[:k, :k].all() and r.sum() == k * k
    assert resize_mask_fraction(m, (out, out), 1.0).sum() == k * k
    # fraction rule is never larger than the any-rule
    rng = np.random.default_rng(2)
    m = (rng.random((N, N)) < 0.3).astype(np.uint8)
    assert (resize_mask_fraction(m, (out, out), 0.5) <= resize_mask_any(m, (out, out))).all()
    with pytest.raises(ValueError, match="min_frac"):
        resize_mask_fraction(m, (out, out), 0.0)


def test_telea_fill_float32_finite():
    mm, m = stereo_to_mm(synthetic_stereo_m())
    assert mm.dtype == np.float32
    out = telea_fill(mm, m, radius=3)                          # cv2.inpaint accepts float32 directly
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(out[m == 0], mm[m == 0])     # valid pixels untouched
    valid = mm[m == 0]
    filled = out[m == 1]
    # Telea is not exact even on a constant field (~±0.2% seen on cv2 5.0.0), so allow 1% slack
    assert filled.min() > valid.min() * 0.99 and filled.max() < valid.max() * 1.01


def holey_stereo_m():
    # block hole, single pixel, holes on exact cell boundaries (32 = 7*s), border-touching hole
    d = synthetic_stereo_m()
    d[31, 32] = d[32, 31] = -1.0
    d[0:5, 900:N] = -1.0
    rng = np.random.default_rng(1)
    d[rng.random((N, N)) < 1e-4] = -1.0
    return d


@pytest.mark.parametrize("size", SIZES)
def test_resize_depth_matches_mask_rule(size):
    mm, m = stereo_to_mm(holey_stereo_m())
    out, inv = resize_depth(mm, m, (size, size))
    assert out.shape == (size, size) and out.dtype == np.float32 and inv.dtype == np.uint8
    np.testing.assert_array_equal(inv, resize_mask_any(m, (size, size)))
    np.testing.assert_array_equal(inv, any_reference(m.astype(bool), size))
    np.testing.assert_array_equal(np.isnan(out), inv.astype(bool))    # NaN exactly at invalid cells


@pytest.mark.parametrize("size", SIZES)
def test_resize_depth_valid_cells_ignore_hole_values(size):
    # valid output cells must be bit-identical whatever value sits in the holes -> no contamination
    mm, m = stereo_to_mm(holey_stereo_m())
    hole = m.astype(bool)
    outs = []
    for fill in (0.0, 1e6):
        d = mm.copy(); d[hole] = fill
        r = cv2.resize(d, (size, size), interpolation=cv2.INTER_AREA)
        outs.append(r)
    out, inv = resize_depth(mm, m, (size, size))
    valid = inv == 0
    np.testing.assert_array_equal(outs[0][valid], outs[1][valid])
    np.testing.assert_array_equal(out[valid], outs[0][valid])


def test_resize_depth_rejects_nan_outside_mask():
    mm, m = stereo_to_mm(synthetic_stereo_m())
    mm[100, 100] = np.nan                                       # m[100,100] == 0
    with pytest.raises(ValueError):
        resize_depth(mm, m, (OUT, OUT))


def test_resize_depth_rejects_nan_hidden_next_to_a_hole():
    # NaN at a valid pixel in the SAME 224 cell as a flagged hole: the output cell is masked anyway,
    # so only a source-side check can catch the mismatch
    mm, m = stereo_to_mm(synthetic_stereo_m())
    assert m[777, 333] == 1 and m[777, 332] == 0
    assert int(332 * OUT / N) == int(333 * OUT / N)              # same 224 cell (column 72)
    mm[777, 332] = np.nan
    with pytest.raises(ValueError, match="source depth"):
        resize_depth(mm, m, (OUT, OUT))


def test_resize_depth_then_telea():
    mm, m = stereo_to_mm(holey_stereo_m())                      # 1024: M_sensor computed here
    d224, inv = resize_depth(mm, m, (OUT, OUT))
    out = telea_fill(d224, inv, radius=3)                       # Telea at 224
    assert out.dtype == np.float32 and np.isfinite(out).all()
    valid = inv == 0
    np.testing.assert_array_equal(out[valid], d224[valid])
    v = d224[valid]
    assert out[~valid].min() > v.min() * 0.99 and out[~valid].max() < v.max() * 1.01


def test_resize_depth_and_telea_empty_mask():
    mm, _ = stereo_to_mm(synthetic_stereo_m())
    mm = np.nan_to_num(mm, nan=700.0)
    empty = np.zeros((N, N), np.uint8)
    d224, inv = resize_depth(mm, empty, (OUT, OUT))
    assert inv.sum() == 0 and np.isfinite(d224).all()
    np.testing.assert_array_equal(telea_fill(d224, inv), d224)


def _plane_224():
    yy, xx = np.mgrid[0:OUT, 0:OUT].astype(np.float32)
    return 600.0 + 300.0 * (xx + yy) / (2 * OUT)                # mm, smooth plane


def _check_large_fill(d, hole):
    out = telea_fill(d, hole.astype(np.uint8), radius=3)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(out[~hole], d[~hole])         # valid depth untouched
    v = d[~hole]
    assert out[hole].min() > v.min() * 0.99 and out[hole].max() < v.max() * 1.01


def test_telea_very_large_central_hole():
    # SPEC_trajectory "test discipline": very large holes pass through Telea. Disk covers ~40% of the image.
    d = _plane_224()
    yy, xx = np.mgrid[0:OUT, 0:OUT]
    hole = (xx - OUT / 2) ** 2 + (yy - OUT / 2) ** 2 < 80 ** 2
    assert hole.mean() > 0.35
    d[hole] = np.nan                                           # NaN holes, as stereo_to_mm produces
    _check_large_fill(d, hole)


def test_telea_large_hole_touching_border():
    d = _plane_224()
    hole = np.zeros((OUT, OUT), bool)
    hole[:100, :] = True                                       # top band, touches three borders
    hole[:, :30] = True
    d[hole] = np.nan
    _check_large_fill(d, hole)


def test_telea_almost_all_invalid():
    d = np.full((OUT, OUT), np.nan, np.float32)
    hole = np.ones((OUT, OUT), bool)
    hole[100:104, 100:104] = False
    d[~hole] = 750.0
    out = telea_fill(d, hole.astype(np.uint8))
    assert np.isfinite(out).all()
    # Only one value exists to propagate. Telea's gradient term adds a small numerical ripple far from
    # the source (measured: max 1.6 mm over ~110 px, cv2 5.0.0), so allow 1% like the other Telea tests.
    # A leak of the zeroed hole values would show up as hundreds of mm, far outside this band.
    np.testing.assert_allclose(out, 750.0, rtol=0.01)


def test_telea_all_invalid_raises():
    d = np.full((OUT, OUT), np.nan, np.float32)
    full = np.ones((OUT, OUT), np.uint8)
    with pytest.raises(ValueError, match="every pixel"):
        telea_fill(d, full)
    with pytest.raises(ValueError, match="every pixel"):
        complete_protocol_a(d, full)
    with pytest.raises(ValueError, match="every pixel"):
        complete_protocol_b(d, full, full)


def test_resize_mask_majority():
    m = np.zeros((N, N), bool)
    m[0:N // 2, :] = True                                       # top half object
    r = resize_mask_majority(m, (OUT, OUT))
    assert r.dtype == bool and r[: OUT // 2 - 1].all() and not r[OUT // 2 + 1:].any()
    single = np.zeros((N, N), bool); single[500, 500] = True    # a sliver does not grow the object
    assert not resize_mask_majority(single, (OUT, OUT)).any()


def protocol_setup():
    # 224-px depth with two true holes: one correctly flagged, one missed; plus a false-alarm patch
    mm, m = stereo_to_mm(synthetic_stereo_m())
    d224, true = resize_depth(mm, m, (OUT, OUT))
    t = true.astype(bool)
    flagged_hole = np.zeros_like(t); flagged_hole[87:93, 108:117] = True      # inside the block hole
    missed_hole = t & ~flagged_hole
    assert flagged_hole.any() and missed_hole.any()
    false_alarm = np.zeros_like(t); false_alarm[30:36, 30:36] = True
    assert not (false_alarm & t).any()
    rep = ((t & ~missed_hole) | false_alarm).astype(np.uint8)
    return d224, true, rep, missed_hole, false_alarm


def test_protocol_b_missed_hole_holds_sentinel():
    d224, true, rep, missed_hole, false_alarm = protocol_setup()
    with pytest.raises(ValueError):                             # NaN in a pixel marked valid
        telea_fill(d224, rep)
    out, missed = complete_protocol_b(d224, true, rep)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(missed.astype(bool), missed_hole)
    assert (out[missed_hole] == MISSED_HOLE_SENTINEL_MM).all()  # unfilled, sentinel 0 mm
    untouched = ~true.astype(bool) & ~false_alarm
    np.testing.assert_array_equal(out[untouched], d224[untouched])


def test_protocol_b_all_sources_are_sentinels_raises():
    # every pixel truly invalid, only part of it flagged: the unflagged rest is all sentinel 0 mm,
    # so Telea would silently fill from zeros only
    d = np.full((OUT, OUT), np.nan, np.float32)
    true = np.ones((OUT, OUT), np.uint8)
    rep = np.zeros((OUT, OUT), np.uint8); rep[50:60, 50:60] = 1
    with pytest.raises(ValueError, match="sentinel"):
        complete_protocol_b(d, true, rep)


def test_protocol_b_flagged_hole_next_to_missed_hole_is_pulled_to_sentinel():
    # documented behaviour, not a bug: a flagged region bordered by a missed region fills toward 0 mm
    d = _plane_224()
    true = np.zeros((OUT, OUT), bool); true[100:130, 100:130] = True
    rep = np.zeros((OUT, OUT), bool); rep[100:130, 100:115] = True    # left half flagged, right half missed
    d[true] = np.nan
    out, missed = complete_protocol_b(d, true, rep)
    assert (out[100:130, 115:130] == MISSED_HOLE_SENTINEL_MM).all()
    assert out[100:130, 112:115].mean() < 0.9 * np.nanmean(d[100:130, 95:100])   # contaminated edge


def test_protocol_b_false_alarm_overwrites_good_depth():
    d224, true, rep, _, false_alarm = protocol_setup()
    out, _ = complete_protocol_b(d224, true, rep)
    assert np.isfinite(d224[false_alarm]).all()
    assert not np.array_equal(out[false_alarm], d224[false_alarm])   # inpainting replaced it


def test_protocol_a_ignores_the_reported_flag():
    d224, true, rep, _, _ = protocol_setup()
    # Protocol A has no way to receive M_rep, so a flag-only edit cannot change the depth ...
    assert "m_rep" not in inspect.signature(complete_protocol_a).parameters
    ref = complete_protocol_a(d224, true)
    assert np.isfinite(ref).all()
    np.testing.assert_array_equal(complete_protocol_a(d224, true), ref)          # deterministic
    # ... and it equals Protocol B only when the reported flag is truthful
    np.testing.assert_array_equal(complete_protocol_b(d224, true, true)[0], ref)
    assert not np.array_equal(complete_protocol_b(d224, true, rep)[0], ref)


def _bad_masks():
    m = np.zeros((OUT, OUT), np.uint8); m[50:60, 50:60] = 1
    return {"column": np.zeros((OUT, 1), np.uint8),            # would broadcast across every column
            "row": np.zeros((1, OUT), np.uint8),
            "3d": m[..., None],
            "255": m * 255,
            "minus1": m.astype(np.int8) * -1,
            "half": m * 0.5,
            "nan": np.where(m == 1, np.nan, 0.0)}


@pytest.mark.parametrize("kind", list(_bad_masks()))
def test_bad_masks_rejected_everywhere(kind):
    bad = _bad_masks()[kind]
    d = _plane_224()
    good = np.zeros((OUT, OUT), np.uint8); good[50:60, 50:60] = 1
    with pytest.raises(ValueError):
        resize_depth(d, bad, (56, 56))
    with pytest.raises(ValueError):
        telea_fill(d, bad)
    with pytest.raises(ValueError):
        complete_protocol_a(d, bad)
    with pytest.raises(ValueError):
        complete_protocol_b(d, bad, good)
    with pytest.raises(ValueError):
        complete_protocol_b(d, good, bad)
    if kind not in ("column", "row"):                          # resize alone cannot know the depth shape
        with pytest.raises(ValueError):
            resize_mask_any(bad, (56, 56))


def test_valid_mask_dtypes_accepted():
    d = _plane_224()
    base = np.zeros((OUT, OUT), bool); base[50:60, 50:60] = True
    ref = telea_fill(d, base)
    for m in (base, base.astype(np.uint8), base.astype(np.int64), base.astype(np.float32)):
        np.testing.assert_array_equal(telea_fill(d, m), ref)


def test_depth_must_be_2d():
    with pytest.raises(ValueError, match="2D"):
        telea_fill(np.zeros((OUT, OUT, 1), np.float32), np.zeros((OUT, OUT), np.uint8))


def test_cv2_inpaint_dtypes():
    # documents which dtypes cv2.inpaint accepts on this install
    m = np.zeros((32, 32), np.uint8); m[10:14, 10:14] = 1
    for dt in (np.float32, np.uint16, np.uint8):
        assert cv2.inpaint(np.full((32, 32), 100, dt), m, 3, cv2.INPAINT_TELEA).dtype == dt
