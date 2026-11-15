"""eval/match.py: angle wrap, exact boundaries, exact float IoU without clipping, all rows kept."""
import numpy as np
import pytest

from eval.match import angle_diff_deg, iou_one_to_many, match, match_both

CFG = {"match": {"primary": {"angle_max_deg": 30, "iou_min_exclusive": 0.25},
                 "strict": {"angle_max_deg": 15, "iou_min_exclusive": 0.40}}}


def G(x, y, t, op=40.0, jaw=20.0):
    return np.array([x, y, t, op, jaw], np.float64)


@pytest.mark.parametrize("a, b, d", [(89, -89, 2), (0, 180, 0), (30, 0, 30), (-80, 80, 20), (90, -90, 0),
                                     (45, -45, 90)])
def test_angle_diff(a, b, d):
    assert angle_diff_deg(a, b) == pytest.approx(d, abs=1e-12)


def test_identical_and_half_turn():
    g = G(100, 100, 25)
    assert iou_one_to_many(g, g[None])[0] == pytest.approx(1.0)
    assert match(G(100, 100, 205), True, g[None], 30, 0.25)["match"]


def test_iou_boundary_is_strict():
    # axis-aligned 40 x 20 rectangles shifted by d along x: overlap (40 - d) * 20, IoU 0.25 at d = 24
    gt = G(100, 100, 0)[None]
    assert iou_one_to_many(G(124, 100, 0), gt)[0] == 0.25
    assert not match(G(124, 100, 0), True, gt, 30, 0.25)["match"]
    assert match(G(123.9, 100, 0), True, gt, 30, 0.25)["match"]


def test_angle_boundary_is_inclusive():
    gt = G(100, 100, 0)[None]
    m30 = match(G(100, 100, 30.0), True, gt, 30, 0.25)
    assert m30["best_iou_in_angle"] > 0.25 and m30["match"]
    assert not match(G(100, 100, 30.0 + 1e-9), True, gt, 30, 0.25)["match"]
    m15 = match(G(100, 100, 15.0), True, gt, 15, 0.40)
    assert m15["best_iou_in_angle"] > 0.40 and m15["match"]
    assert not match(G(100, 100, 15.0 + 1e-9), True, gt, 15, 0.40)["match"]


def test_out_of_frame_corners_are_not_clipped():
    # both rectangles extend to x < 0; unclipped overlap 30 x 20, union 1000 -> 0.6
    assert iou_one_to_many(G(5, 100, 0), G(-5, 100, 0)[None])[0] == pytest.approx(0.6, abs=1e-12)


def test_jaw_repeats_are_kept_and_differ():
    gt = np.array([[100, 100, 0, 40, 20], [100, 100, 0, 40, 6]], float)
    iou = iou_one_to_many(G(100, 100, 0), gt)
    assert iou[0] == pytest.approx(1.0) and iou[1] == pytest.approx(6 / 20)
    assert match(G(100, 100, 0), True, gt, 30, 0.25)["n_rows"] == 2


def test_invalid_prediction_never_matches_and_bad_gt_raises():
    gt = G(100, 100, 0)[None]
    assert not match(G(100, 100, 0), False, gt, 30, 0.25)["match"]
    with pytest.raises(ValueError):
        match(G(100, 100, 0), True, np.array([[100, 100, 0, 0, 20]], float), 30, 0.25)
    with pytest.raises(ValueError):
        match(G(100, 100, 0), True, np.zeros((0, 5)), 30, 0.25)


def test_match_both():
    r = match_both(G(100, 100, 20), True, G(100, 100, 0)[None], CFG)
    assert r["primary"]["match"] and not r["strict"]["match"]       # 20 deg: in 30, out of 15


@pytest.mark.parametrize("pred", [G(100, 100, np.nan), G(np.nan, 100, 0), G(100, 100, 0, op=-40.0),
                                  G(100, 100, 0, op=0.0), G(100, 100, 0, jaw=0.0),
                                  G(100, 100, 0, jaw=-20.0), G(np.inf, 100, 0)])
def test_malformed_valid_prediction_raises_and_invalid_is_a_miss(pred):
    """Review #10: checked BEFORE the angle filter, so a NaN angle cannot pass as an ordinary miss and a
    negative opening (same polygon as a positive one) cannot match."""
    gt = G(100, 100, 0)[None]
    with pytest.raises(ValueError, match="malformed"):
        match(pred, True, gt, 30, 0.25)
    assert match(pred, False, gt, 30, 0.25) == {"match": False, "best_iou_in_angle": 0.0, "n_rows": 1}
