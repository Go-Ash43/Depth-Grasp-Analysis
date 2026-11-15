"""Training augmentation: identity or ONE flip along the stereo baseline (docs/proj_v3.md D2).

Why only one flip: stereo holes and lies come from occlusion along the camera baseline, so they sit on
one side of the object. A flip across the baseline or a 90-degree rotation would create hole patterns
the test set never contains (M_sensor and M_lie are what we measure). Deviation from GR-ConvNet
(random rotation and zoom), recorded in docs/proj_v3.md.

Conventions (AGENTS.md): x right, y down, integer = pixel centre, jaw-closing axis (cos t, sin t),
t as written. Flips are exact index permutations (no resampling):
  vertical   (axis "vertical",   rows reversed):    y -> (H - 1) - y,  t -> -t
  horizontal (axis "horizontal", columns reversed): x -> (W - 1) - x,  t -> -t (180 - t == -t mod 180)
Angle maps: t -> -t keeps cos 2t and negates sin 2t, for either flip. Flipped grasp angles are put
back into the canonical (-90, 90] (data.labels.canonical_theta).
"""
import numpy as np

from data.labels import canonical_theta

AXES = ("vertical", "horizontal")


def _axis(axis):
    if axis not in AXES:
        raise ValueError(f"flip axis must be one of {AXES}, got {axis!r}")
    return 0 if axis == "vertical" else 1


def flip_array(a, axis):
    """(H, W, ...) array -> flipped copy (contiguous)."""
    return np.ascontiguousarray(np.flip(np.asarray(a), _axis(axis)))


TARGET_KEYS = ("q", "cos2", "sin2", "width", "void")


def flip_targets(t, axis):
    """Target maps dict -> flipped copy holding ONLY q, cos2, sin2, width, void. Other entries of
    make_targets (owner, grasps_224) are dropped, never passed on unflipped."""
    out = {k: flip_array(t[k], axis) for k in TARGET_KEYS}
    out["sin2"] = (-out["sin2"]).astype(t["sin2"].dtype, copy=False)   # t -> -t
    out["sin2"][out["sin2"] == 0] = 0.0                                  # no -0.0
    return out


def flip_grasps(g, axis, hw):
    """(N, 5) x, y, t, opening, jaw in an image of shape hw -> flipped copy (same resolution)."""
    g = np.array(g, np.float64).reshape(-1, 5)
    if _axis(axis) == 0:
        g[:, 1] = (hw[0] - 1) - g[:, 1]
    else:
        g[:, 0] = (hw[1] - 1) - g[:, 0]
    g[:, 2] = canonical_theta(-g[:, 2])            # keep (-90, 90]: 90 stays 90, not -90
    g[:, 2][g[:, 2] == 0] = 0.0
    return g
