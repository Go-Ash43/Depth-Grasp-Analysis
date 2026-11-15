"""data/augment.py: one flip, exact, commuting with targets and decoding; angle rule t -> -t
(docs/proj_v3.md D2). Asymmetric grasps (t = 30, x != y) so a sign flip or x/y swap cannot pass."""
import numpy as np
import pytest

from data.augment import flip_array, flip_grasps, flip_targets
from data.decode import decode_top1
from data.jacquard import grasp_corners
from data.labels import make_targets
from eval.match import angle_diff_deg

S, SRC = (224, 224), (1024, 1024)
KW = dict(size=224, sigma=2.0, q_void=0.05, w_norm=100.0)
G = np.array([[300.0, 520.0, 30.0, 120.0, 40.0],
              [710.0, 260.0, -55.0, 60.0, 25.0]])          # far apart: no ownership ties


@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_flip_twice_is_identity(axis):
    a = np.random.default_rng(0).random((224, 224, 3))
    np.testing.assert_array_equal(flip_array(flip_array(a, axis), axis), a)
    np.testing.assert_array_equal(flip_grasps(flip_grasps(G, axis, SRC), axis, SRC), G)


@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_flipped_rectangle_is_the_mirrored_rectangle(axis):
    """Geometry, not formula: the corners of the flipped grasp are the mirrored corners."""
    c = grasp_corners(G)
    m = c.copy()
    if axis == "vertical":
        m[..., 1] = 1023 - m[..., 1]
    else:
        m[..., 0] = 1023 - m[..., 0]
    f = grasp_corners(flip_grasps(G, axis, SRC))
    for i in range(len(G)):
        a = sorted(map(tuple, np.round(m[i], 9)))
        b = sorted(map(tuple, np.round(f[i], 9)))
        np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_flip_commutes_with_targets(axis):
    t = make_targets(G, **KW)
    ft = flip_targets(t, axis)
    tf = make_targets(flip_grasps(G, axis, SRC), **KW)
    for k in ("q", "cos2", "sin2", "width"):
        np.testing.assert_allclose(ft[k], tf[k], atol=1e-6, err_msg=k)
    away = np.abs(tf["q"] - 0.05) > 1e-5                      # void may differ only at the threshold
    np.testing.assert_array_equal(ft["void"][away], tf["void"][away])


@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_decode_of_flipped_targets_is_the_flipped_grasp(axis):
    t = make_targets(G[:1], **KW)
    d0 = decode_top1(t["q"], t["cos2"], t["sin2"], t["width"], 100.0)["grasp"]
    ft = flip_targets(t, axis)
    d1 = decode_top1(ft["q"], ft["cos2"], ft["sin2"], ft["width"], 100.0)["grasp"]
    want = flip_grasps(d0[None], axis, SRC)[0]
    np.testing.assert_allclose(d1[[0, 1, 3, 4]], want[[0, 1, 3, 4]], atol=1e-6)
    assert angle_diff_deg(d1[2], want[2]) < 1e-9             # exact: (cos, -sin) -> -atan2
    assert angle_diff_deg(d1[2], -30.0) < 1e-4               # t -> -t (float32 maps)


def test_sin2_negated_cos2_kept_and_no_negative_zero():
    t = make_targets(G, **KW)
    ft = flip_targets(t, "vertical")
    np.testing.assert_array_equal(np.flip(ft["cos2"], 0), t["cos2"])
    np.testing.assert_array_equal(np.flip(ft["sin2"], 0), -t["sin2"] + 0.0)
    assert not np.signbit(ft["sin2"][ft["sin2"] == 0]).any()


def test_flip_targets_drops_unflipped_entries():
    t = make_targets(G, **KW)
    assert {"owner", "grasps_224"} <= set(t)
    assert set(flip_targets(t, "vertical")) == {"q", "cos2", "sin2", "width", "void"}


def test_unknown_axis_raises():
    with pytest.raises(ValueError):
        flip_array(np.zeros((4, 4)), "diagonal")
