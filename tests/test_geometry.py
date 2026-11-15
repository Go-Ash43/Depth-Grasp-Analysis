"""Pixel-centre convention (data/geometry.py) agrees with cv2.resize, which is what aligns labels with images."""
import cv2
import numpy as np
import pytest

from data.geometry import in_image, pixel_index, scale_grasps, scale_xy

N = 1024


@pytest.mark.parametrize("out", [224, 160])
def test_scale_xy_matches_cv2_resize(out):
    # ramp image whose value at each pixel centre IS its x (resp. y) coordinate. INTER_AREA averages
    # per-pixel-constant values, so output pixel j holds its centre's source coordinate up to a
    # partial-pixel term of at most 0.125 / S^2 dst px (0.006 at 224, 0.003 at 160)
    ramp = np.tile(np.arange(N, dtype=np.float32), (N, 1))
    rx = cv2.resize(ramp, (out, out), interpolation=cv2.INTER_AREA)[0]       # x along columns
    ry = cv2.resize(ramp.T.copy(), (out, out), interpolation=cv2.INTER_AREA)[:, 0]
    j = np.arange(out, dtype=np.float64)
    src = np.stack([rx, ry], -1).astype(np.float64)
    np.testing.assert_allclose(scale_xy(src, (N, N), (out, out)), np.stack([j, j], -1), atol=0.01)


def test_plain_multiply_is_off_by_039px_at_224():
    s = 224 / N
    x = np.array([[500.0, 500.0]])
    assert np.allclose(scale_xy(x, (N, N), (224, 224)) - x * s, 0.5 * s - 0.5)
    assert abs(0.5 * s - 0.5 + 0.39) < 0.005


def test_scale_xy_round_trip():
    pts = np.array([[0.0, 0.0], [1023.4, 17.2], [-0.5, 512.0]])
    back = scale_xy(scale_xy(pts, (N, N), (224, 224)), (224, 224), (N, N))
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_in_image_edges():
    pts = np.array([[-0.5, 0], [N - 0.51, 0], [N - 0.5, 0], [-0.51, 0], [0, N - 0.5]])
    np.testing.assert_array_equal(in_image(pts, (N, N)), [True, True, False, False, False])


def test_pixel_index_and_rejects_outside():
    ix, iy = pixel_index(np.array([[-0.5, 0.49], [N - 0.51, 0.5]]), (N, N))
    np.testing.assert_array_equal(ix, [0, N - 1])
    np.testing.assert_array_equal(iy, [0, 1])
    with pytest.raises(ValueError, match="outside"):
        pixel_index(np.array([[N - 0.5, 0.0]]), (N, N))


def test_scale_grasps():
    g = np.array([[511.5, 100.0, 30.0, 80.0, 20.0]])
    s = 224 / N
    out = scale_grasps(g, (N, N), (224, 224))
    np.testing.assert_allclose(out[0, :2], scale_xy(g[:, :2], (N, N), (224, 224))[0])
    np.testing.assert_allclose(out[0, 2:], [30.0, 80.0 * s, 20.0 * s])
    np.testing.assert_array_equal(g, [[511.5, 100.0, 30.0, 80.0, 20.0]])    # input untouched
    with pytest.raises(ValueError, match="anisotropic"):
        scale_grasps(g, (N, N), (224, 160))
