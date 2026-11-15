"""Half-pixel regression for the diagnostic scripts: point lookups must follow the pixel-centre
convention of data/geometry.py (index = floor(x + 0.5)), not np.rint / np.round, which round
half to even (rint(2.5) = 2, but the point x = 2.5 lies in pixel 3)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cac = _load("check_angle_convention")


@pytest.mark.parametrize("x, col", [(2.5, 3), (3.5, 4), (-0.5, 0), (9.4999, 9)])
def test_angle_check_sample_uses_pixel_centre(x, col):
    img = np.arange(10)[None, :].repeat(3, 0)            # value = column index
    assert cac.sample(img, np.array([[x, 1.0]]))[0] == col


@pytest.mark.parametrize("mod", [cac], ids=["check_angle_convention"])
def test_in_frame_edges(mod):
    pts = np.array([[-0.5, 0.0], [-0.5001, 0.0], [9.4999, 0.0], [9.5, 0.0], [0.0, 2.4999], [0.0, 2.5]])
    np.testing.assert_array_equal(mod.in_frame(pts, (3, 10)), [True, False, True, False, True, False])


def test_angle_check_sample_rejects_out_of_frame():
    with pytest.raises(ValueError):
        cac.sample(np.zeros((3, 10)), np.array([[9.5, 0.0]]))
