"""Angle round-trip: parse -> corners vs. corners computed independently with plain trigonometry.

Convention under test: image frame, x right, y DOWN; Jacquard theta rotates the jaw-closing axis
from +x towards +y (clockwise on screen). Corner order: (-axis,-perp), (+axis,-perp), (+axis,+perp), (-axis,+perp).
"""
import math

import numpy as np
import pytest

from data.jacquard import grasp_corners, parse_grasps

CX, CY, L, W = 300.0, 200.0, 80.0, 20.0


def expected_corners(theta_deg):
    # independent: rotate the axis-aligned box offsets with the 2x2 rotation matrix (y-down frame)
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    out = []
    for u, v in [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]:
        out.append((CX + c * u - s * v, CY + s * u + c * v))
    return np.array(out)


def corners_from_line(theta_deg, newline=True):
    line = f"{CX};{CY};{theta_deg};{L};{W}" + ("\n" if newline else "")
    return grasp_corners(parse_grasps(line))[0]


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0, -30.0, 135.0, -90.0, 179.5])
def test_corners_match_trig(theta):
    np.testing.assert_allclose(corners_from_line(theta), expected_corners(theta), atol=1e-9)


def test_literal_0_and_90():
    # hand-written numbers, no trig at all
    np.testing.assert_allclose(corners_from_line(0), [[260, 190], [340, 190], [340, 210], [260, 210]], atol=1e-9)
    # 90 deg: jaw-closing axis points straight DOWN the image (y increases)
    np.testing.assert_allclose(corners_from_line(90), [[310, 160], [310, 240], [290, 240], [290, 160]], atol=1e-9)


def test_positive_theta_points_down_right():
    # +45 deg in a y-down frame: the +axis end is to the right AND lower (larger y)
    k = corners_from_line(45.0)
    end_plus = (k[1] + k[2]) / 2
    assert end_plus[0] > CX and end_plus[1] > CY


def test_rejects_known_wrong_conventions():
    good = expected_corners(30.0)
    neg = expected_corners(-30.0)                              # negated theta (mirrored)
    got = corners_from_line(30.0)
    assert not np.allclose(got, neg)
    swapped = grasp_corners(parse_grasps(f"{CY};{CX};30;{L};{W}"))[0]   # x/y swapped
    assert not np.allclose(swapped, good)


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0, -30.0, 135.0])
def test_theta_recovered_mod_180(theta):
    k = corners_from_line(theta)
    d = k[1] - k[0]                                            # along the jaw-closing axis
    rec = math.degrees(math.atan2(d[1], d[0]))
    assert abs((rec - theta + 90) % 180 - 90) < 1e-9
    assert np.isclose(np.linalg.norm(d), L) and np.isclose(np.linalg.norm(k[2] - k[1]), W)


def _reference_as_gr(center, angle, length, width):
    """Verbatim port of GRConv robotic-grasping utils/dataset_processing/grasp.py, Grasp.as_gr
    (L363-382). center = (row, col); returns (4, 2) points in (row, col)."""
    xo = np.cos(angle)
    yo = np.sin(angle)
    y1 = center[0] + length / 2 * yo
    x1 = center[1] - length / 2 * xo
    y2 = center[0] - length / 2 * yo
    x2 = center[1] + length / 2 * xo
    return np.array([
        [y1 - width / 2 * xo, x1 - width / 2 * yo],
        [y2 - width / 2 * xo, x2 - width / 2 * yo],
        [y2 + width / 2 * xo, x2 + width / 2 * yo],
        [y1 + width / 2 * xo, x1 + width / 2 * yo],
    ])


def _reference_load_line(x, y, theta, w, h):
    """Verbatim port of GraspRectangles.load_from_jacquard_file (L102-104), one line, scale 1."""
    return _reference_as_gr(np.array([y, x]), -theta / 180.0 * np.pi, w, h)


@pytest.mark.parametrize("theta", [0.0, 20.33, 45.0, 90.0, -30.0, 135.0, -90.0, 179.5])
def test_matches_reference_as_gr(theta):
    # the reference code itself, not this repo's formula: same corners, same order, after (row, col) -> (x, y)
    ref_xy = _reference_load_line(CX, CY, theta, L, W)[:, ::-1]
    np.testing.assert_allclose(corners_from_line(theta), ref_xy, atol=1e-9)


def test_last_line_without_newline_keeps_digits():
    g = parse_grasps("1;2;3;4;5\n10;20;30;40;55")
    np.testing.assert_array_equal(g, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 55]])
