"""data/labels.py + data/decode.py: ownership targets, round trip 1024 -> 224 maps -> decode -> 1024,
counterexamples from the Step 2 review (notes/MAJOR_v1.md section 6). Synthetic only.

Dtype contract: maps float32, geometry float64. Round-trip opening tolerance at 1024:
rtol = 1e-6, atol = 1e-6 px; the decoded cell must equal pixel_index of the scaled centre exactly."""
import json
from pathlib import Path

import numpy as np
import pytest

from data.decode import decode_from_cfg, decode_top1
from data.geometry import pixel_index, scale_grasps, scale_xy
from data.labels import canonical_theta, dedup_for_targets, make_targets

SRC, S = (1024, 1024), (224, 224)
W_NORM = 60.0
KW = dict(size=224, sigma=2.0, q_void=0.05, w_norm=W_NORM)


def _g(x, y, t, op=100.0, jaw=40.0):
    return np.array([[x, y, t, op, jaw]], np.float64)


def _decode(tg):
    return decode_top1(tg["q"], tg["cos2"], tg["sin2"], tg["width"], W_NORM)


def test_config_file_is_well_formed():
    """Valid both before and after the approved W_NORM / zone updates."""
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "labels.json").read_text())
    w = cfg["targets"]["w_norm_px_224"]
    assert w is None or (isinstance(w, (int, float)) and w > 0)
    assert cfg["zones"]["selected"] is None or cfg["zones"]["selected"] in cfg["zones"]["candidates"]
    assert 0 < cfg["targets"]["q_floor"] < cfg["targets"]["q_void"] < 1 and cfg["size"] == 224


def test_real_config_passes_validation():
    from data.labels import load_labels_cfg
    assert load_labels_cfg()["size"] == 224


def _set(cfg, path, value):
    *head, last = path.split(".")
    d = cfg
    for k in head:
        d = d[k]
    if value is _DEL:
        del d[last]
    else:
        d[last] = value


_DEL = object()


@pytest.mark.parametrize("path, value, msg", [
    ("targets.dedup_key", ["x", "y", "theta_canonical"], "dedup_key"),
    ("targets.theta_canonical", "[-90, 90)", "theta_canonical"),
    ("targets.owner", "average of overlapping grasps", "owner"),
    ("targets.sigma_px_224", float("nan"), "sigma"),
    ("targets.sigma_px_224", 0.0, "sigma"),
    ("targets.q_void", 1e-7, "q_floor < q_void"),
    ("targets.w_norm_px_224", -1.0, "w_norm"),
    ("decode.smoothing", "gaussian", "smoothing"),
    ("decode.tie", "last row-major", "tie"),
    ("decode.jaw_rule", "fixed", "jaw_rule"),
    ("decode.angle_eps", 0.0, "angle_eps"),
    ("match.primary.angle_max_deg", 120, "<= 90"),
    ("match.primary.angle_max_deg", float("inf"), "angle_max_deg"),
    ("match.strict.iou_min_exclusive", 1.0, "iou_min_exclusive"),
    ("match.strict.iou_min_exclusive", -0.1, "iou_min_exclusive"),
    ("match.extra", {"angle_max_deg": 10, "iou_min_exclusive": 0.5}, "exactly"),
    ("match.strict", _DEL, "exactly"),
    ("match.primary", None, "mapping"),
    ("size", 224.0, "size"),
    ("decode", _DEL, "missing section"),
])
def test_config_validation_rejects_policy_or_value_drift(path, value, msg):
    """Review #11: the config must describe what the code does; values must be finite and in range."""
    from data.labels import load_labels_cfg, validate_labels_cfg
    cfg = load_labels_cfg()
    _set(cfg, path, value)
    with pytest.raises(ValueError, match=msg):
        validate_labels_cfg(cfg)


def test_cfg_consumers_validate_their_section():
    from data.labels import load_labels_cfg, targets_from_cfg
    from eval.match import match_both
    cfg = load_labels_cfg()
    cfg["decode"]["tie"] = "last row-major"
    with pytest.raises(ValueError, match="tie"):
        decode_from_cfg({k: np.ones(S, np.float32) for k in ("q", "cos2", "sin2", "width")}, cfg)
    cfg = load_labels_cfg()
    cfg["targets"]["owner"] = "x"
    with pytest.raises(ValueError, match="owner"):
        targets_from_cfg(_g(500, 400, 0)[:, :5], cfg)
    cfg = load_labels_cfg()
    cfg["match"]["primary"]["angle_max_deg"] = 0
    with pytest.raises(ValueError, match="angle_max_deg"):
        match_both(_g(500, 400, 0)[0], True, _g(500, 400, 0), cfg)


def test_null_w_norm_is_refused():
    with pytest.raises(ValueError, match="null"):                 # training refuses a null W_NORM
        make_targets(_g(500, 400, 0), w_norm=None)


def test_decode_from_cfg_rejects_wrong_map_size():
    cfg = {"size": 224, "targets": {"w_norm_px_224": W_NORM},
           "decode": {"smoothing": "none", "tie": "first row-major", "jaw_rule": "opening/2", "angle_eps": 1e-6}}
    m = {k: np.ones((112, 112), np.float32) for k in ("q", "cos2", "sin2", "width")}
    with pytest.raises(ValueError, match="configured"):
        decode_from_cfg(m, cfg)
    m = {k: np.ones(S, np.float32) for k in ("q", "cos2", "sin2", "width")}
    assert decode_from_cfg(m, cfg)["valid"]


def test_canonical_theta():
    np.testing.assert_allclose(canonical_theta([90, -90, 89.96, -89.96, 0, 180, 270, -270]),
                               [90, 90, 89.96, -89.96, 0, 0, 90, 90], atol=1e-12)


def test_dedup_collapses_jaw_repeats_and_half_turns():
    g = np.array([[10, 20, 30, 50, 5], [10, 20, 30, 50, 9],      # jaw repeat
                  [10, 20, 90, 50, 5], [10, 20, -90, 50, 5]], float)   # 90 == -90
    u = dedup_for_targets(g)
    assert len(u) == 2 and u.shape[1] == 4


@pytest.mark.parametrize("theta", [30.0, -60.0, 89.5, 0.0])
def test_round_trip_exact_cell_and_values(theta):
    g = _g(401.3, 612.7, theta)                                   # x != y, no half-pixel tie at 224
    tg = make_targets(g, **KW)
    d = _decode(tg)
    c224 = scale_xy(g[0, :2], SRC, S)
    ix, iy = pixel_index(c224, S)
    assert d["cell_xy"] == (int(ix), int(iy))                     # nearest cell, exactly
    np.testing.assert_allclose(d["grasp"][:2], scale_xy(np.array([ix, iy], float), S, SRC), atol=1e-9)
    assert np.all(np.abs(d["grasp"][:2] - g[0, :2]) <= 0.5 * 1024 / 224 + 1e-9)
    assert d["grasp"][2] == pytest.approx(theta, abs=1e-4)        # float32 cos/sin: ~1e-5 deg
    np.testing.assert_allclose(d["grasp"][3], g[0, 3], rtol=1e-6, atol=1e-6)
    assert d["grasp"][4] == pytest.approx(d["grasp"][3] / 2) and d["valid"]


def test_sign_and_transpose_are_caught():
    tg = make_targets(_g(401.3, 612.7, 30.0), **KW)
    assert _decode(tg)["grasp"][2] == pytest.approx(30.0, abs=1e-4)      # not -30
    d_t = decode_top1(tg["q"].T, tg["cos2"].T, tg["sin2"].T, tg["width"].T, W_NORM)
    assert not np.allclose(d_t["grasp"][:2], _decode(tg)["grasp"][:2], atol=5)   # x/y swap detected


def test_q_at_fractional_centre_is_below_one_but_bounded():
    tg = make_targets(_g(401.3, 612.7, 0.0), **KW)
    assert np.exp(-0.5 / 8) - 1e-6 <= tg["q"].max() < 1.0


def test_same_centre_0_and_60_one_owner_no_average():
    g = np.array([[500.0, 400.0, 60.0, 120.0, 30.0], [500.0, 400.0, 0.0, 80.0, 30.0]])
    tg = make_targets(g, **KW)
    owned = tg["owner"] >= 0
    assert set(np.unique(tg["owner"][owned])) == {0}              # sorted: theta 0 row first
    ang = 0.5 * np.degrees(np.arctan2(tg["sin2"][owned], tg["cos2"][owned]))
    np.testing.assert_allclose(ang, 0.0, atol=1e-5)               # never 30
    np.testing.assert_allclose(tg["width"][owned], 80 * 224 / 1024 / W_NORM, rtol=1e-6)   # same owner's opening
    d = _decode(tg)
    assert tg["void"][d["cell_xy"][1], d["cell_xy"][0]] == 0


def test_0_and_60_three_px_apart_each_centre_keeps_its_own_grasp():
    dx = 3 * 1024 / 224                                           # 3 cells at 224
    g = np.array([[400.0, 500.0, 0.0, 80.0, 30.0], [400.0 + dx, 500.0, 60.0, 120.0, 30.0]])
    tg = make_targets(g, **KW)
    for row in g:
        ix, iy = pixel_index(scale_xy(row[:2], SRC, S), S)
        ang = 0.5 * np.degrees(np.arctan2(tg["sin2"][iy, ix], tg["cos2"][iy, ix]))
        assert ang == pytest.approx(row[2], abs=1e-4)
        assert tg["width"][iy, ix] == pytest.approx(row[3] * 224 / 1024 / W_NORM, rel=1e-6)
        assert tg["void"][iy, ix] == 0


def test_angle_and_width_from_the_same_owner_everywhere():
    rng = np.random.default_rng(0)
    g = np.c_[rng.uniform(300, 700, 40), rng.uniform(300, 700, 40), rng.uniform(-90, 90, 40),
              rng.uniform(20, 150, 40), rng.uniform(5, 40, 40)]
    tg = make_targets(g, **KW)
    s = tg["grasps_224"]
    o = tg["owner"]
    m = o >= 0
    np.testing.assert_allclose(tg["cos2"][m], np.cos(np.deg2rad(2 * s[o[m], 2])), atol=1e-6)
    np.testing.assert_allclose(tg["width"][m], s[o[m], 3] / W_NORM, rtol=1e-6)


def test_width_above_w_norm_is_not_clipped():
    tg = make_targets(_g(500, 400, 0.0, op=400.0), **KW)          # 87.5 px at 224 > 60
    assert tg["width"].max() == pytest.approx(400 * 224 / 1024 / W_NORM, rel=1e-6) and tg["width"].max() > 1
    d = _decode(tg)
    np.testing.assert_allclose(d["grasp"][3], 400.0, rtol=1e-6, atol=1e-6)


def test_void_exactly_where_q_below_threshold():
    tg = make_targets(np.array([[300.0, 300, 10, 60, 20], [700, 650, -40, 90, 20]]), **KW)
    np.testing.assert_array_equal(tg["void"], (tg["q"] < 0.05).astype(np.uint8))
    assert tg["void"].dtype == np.uint8 and tg["q"].dtype == np.float32


def test_jaw_repeats_do_not_change_targets():
    a = make_targets(np.array([[500.0, 400, 20, 90, 10]]), **KW)
    b = make_targets(np.array([[500.0, 400, 20, 90, 10], [500.0, 400, 20, 90, 25]]), **KW)
    for k in ("q", "cos2", "sin2", "width", "void"):
        np.testing.assert_array_equal(a[k], b[k])


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        make_targets(_g(500, 400, 0, op=0.0), **KW)
    with pytest.raises(ValueError):
        make_targets(np.zeros((0, 5)), **KW)
    with pytest.raises(ValueError):
        make_targets(_g(np.nan, 400, 0), **KW)


# ------------------------------------------------------------------ decoder edge cases
def _maps():
    q = np.zeros(S, np.float32)
    q[100, 50] = 1.0
    return q, np.ones(S, np.float32), np.zeros(S, np.float32), np.full(S, 0.5, np.float32)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_decoder_raises_on_non_finite(bad, which):
    m = list(_maps())
    m[which][7, 9] = bad
    with pytest.raises(ValueError, match="non-finite"):
        decode_top1(*m, W_NORM)


def test_decoder_invalid_cases():
    q, c, s, w = _maps()
    c[100, 50] = 0.0
    assert not decode_top1(q, c, s, w, W_NORM)["valid"]            # zero angle vector
    q, c, s, w = _maps()
    w[100, 50] = 0.0
    assert not decode_top1(q, c, s, w, W_NORM)["valid"]            # opening 0
    w[100, 50] = -0.2
    assert not decode_top1(q, c, s, w, W_NORM)["valid"]            # opening < 0


def test_decoder_tie_first_row_major_and_theta_range():
    q, c, s, w = _maps()
    q[100, 60] = 1.0
    q[90, 200] = 1.0                                               # earlier row wins
    assert decode_top1(q, c, s, w, W_NORM)["cell_xy"] == (200, 90)
    c[90, 200], s[90, 200] = -1.0, -0.0                            # atan2(-0, -1) = -pi
    assert decode_top1(q, c, s, w, W_NORM)["grasp"][2] == 90.0


def test_decode_from_cfg_requires_w_norm():
    cfg = {"size": 224, "targets": {"w_norm_px_224": None},
           "decode": {"smoothing": "none", "tie": "first row-major", "jaw_rule": "opening/2", "angle_eps": 1e-6}}
    q, c, s, w = _maps()
    with pytest.raises(ValueError):
        decode_from_cfg({"q": q, "cos2": c, "sin2": s, "width": w}, cfg)
