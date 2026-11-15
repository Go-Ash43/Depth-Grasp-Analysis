"""perfect_to_mm upper bound (notes/traps_for_unwritten_code.md, "Depth upper bound")."""
import numpy as np
import pytest

from data.jacquard import PERFECT_MAX_MM, SceneDataError, perfect_to_mm, stereo_to_mm


def test_typical_range_passes():
    raw = np.full((4, 4), 1.5, np.float32)
    raw[0, 0], raw[1, 1] = 0.893, 1.574
    np.testing.assert_allclose(perfect_to_mm(raw), raw * 1000, rtol=1e-6)


def test_bound_is_inclusive():
    raw = np.full((2, 2), PERFECT_MAX_MM / 1000, np.float32)
    assert perfect_to_mm(raw).max() <= PERFECT_MAX_MM


def test_above_bound_raises():
    raw = np.full((4, 4), 1.5, np.float32)
    raw[2, 3] = 3.5
    with pytest.raises(SceneDataError, match="> 3000 mm"):
        perfect_to_mm(raw)


def test_stereo_still_unbounded():
    """Stereo outliers (up to ~275 m) are natural lies, never errors."""
    raw = np.full((4, 4), 1.5, np.float32)
    raw[0, 0] = 274.813
    raw[1, 1] = -1.0
    d, m = stereo_to_mm(raw)
    assert d[0, 0] == pytest.approx(274813, rel=1e-6)
    assert m[1, 1] == 1 and np.isnan(d[1, 1])
