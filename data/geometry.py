"""Pixel-coordinate convention: the single source of truth for every loader, renderer, decoder and metric.

Integer coordinates are pixel CENTRES: pixel (row i, col j) covers x in [j - 0.5, j + 0.5) and
y in [i - 0.5, i + 0.5). This is the convention of cv2.resize, so grasp labels scaled with scale_xy
stay aligned with images resized by data/mask_ops. Consequences:
  - a point is in an H x W image iff -0.5 <= x < W - 0.5 and -0.5 <= y < H - 0.5;
  - its pixel index is floor(x + 0.5);
  - resizing maps x -> (x + 0.5) * W_dst / W_src - 0.5. Plain x * W_dst / W_src is off by
    0.5 * (1 - s) pixels (0.39 px at 1024 -> 224), a few % of a ~21-px object.
"""
import numpy as np


def _xy(xy):
    a = np.asarray(xy, dtype=np.float64)
    if a.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) x, y coordinates, got shape {a.shape}")
    return a


def in_image(xy, hw):
    """Bool (...,) : point inside an image of shape hw = (H, W) under the pixel-centre convention."""
    a = _xy(xy)
    h, w = hw[:2]
    x, y = a[..., 0], a[..., 1]
    return (x >= -0.5) & (x < w - 0.5) & (y >= -0.5) & (y < h - 0.5)


def pixel_index(xy, hw):
    """(..., 2) x, y -> integer (ix, iy) of the containing pixel. Raises if any point is outside."""
    a = _xy(xy)
    if not in_image(a, hw).all():
        raise ValueError(f"{int((~in_image(a, hw)).sum())} points outside the {hw[0]}x{hw[1]} image")
    ix = np.floor(a[..., 0] + 0.5).astype(np.int64)
    iy = np.floor(a[..., 1] + 0.5).astype(np.int64)
    return ix, iy


def scale_xy(xy, src_hw, dst_hw):
    """Map (..., 2) x, y from an image of shape src_hw to one of shape dst_hw (pixel-centre aligned,
    matching cv2.resize)."""
    a = _xy(xy)
    sx = dst_hw[1] / src_hw[1]
    sy = dst_hw[0] / src_hw[0]
    out = np.empty_like(a)
    out[..., 0] = (a[..., 0] + 0.5) * sx - 0.5
    out[..., 1] = (a[..., 1] + 0.5) * sy - 0.5
    return out


def scale_grasps(grasps, src_hw, dst_hw):
    """(N, 5) x, y, theta_deg, opening, jaw -> same at dst resolution. Centres via scale_xy, lengths
    by the scale factor, theta unchanged. Isotropic scaling only: an anisotropic resize would change
    the rectangle's angle and side lengths, which this does not model."""
    g = np.array(grasps, dtype=np.float64).reshape(-1, 5)
    sx = dst_hw[1] / src_hw[1]
    sy = dst_hw[0] / src_hw[0]
    if not np.isclose(sx, sy):
        raise ValueError(f"anisotropic resize {src_hw} -> {dst_hw} not supported for grasps")
    g[:, 0:2] = scale_xy(g[:, 0:2], src_hw, dst_hw)
    g[:, 3:5] *= sx
    return g
