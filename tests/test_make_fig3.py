"""scripts/make_fig3.py helpers: rectangles at 224 follow the scoring geometry (angle convention, pixel-centre
scaling), crops contain everything drawn, outcomes and invalid decodes are read as written. Synthetic only."""
import numpy as np

from scripts import make_fig3 as mf


def test_corners_224_axis_and_scale():
    # theta = 0: opening along +x, jaw along y; centre 511.5 (1024) -> 111.5 (224)
    c = mf.corners_224(np.array([[511.5, 511.5, 0.0, 102.4, 51.2]]))[0]
    assert np.allclose(c[:, 0].max() - c[:, 0].min(), 102.4 * 224 / 1024)
    assert np.allclose(c[:, 1].max() - c[:, 1].min(), 51.2 * 224 / 1024)
    assert np.allclose(c.mean(0), [111.5, 111.5])
    # theta = 90 (y down): the opening now spans y
    c90 = mf.corners_224(np.array([[511.5, 511.5, 90.0, 102.4, 51.2]]))[0]
    assert np.allclose(c90[:, 1].max() - c90[:, 1].min(), 102.4 * 224 / 1024)


def test_crop_box_contains_object_and_points_and_is_clipped():
    obj = np.zeros((224, 224), bool)
    obj[100:110, 50:60] = True
    pts = np.array([[40.0, 105.0], [70.0, 98.0]])
    x0, x1, y0, y1 = mf.crop_box(obj, pts, margin=4, hw=(224, 224))
    assert x0 <= 40 - 0.5 and x1 >= 70 + 0.5 and y0 <= 98 - 0.5 and y1 >= 109 + 0.5
    assert abs((x1 - x0) - (y1 - y0)) < 1e-9                       # square when not clipped
    edge = np.zeros((224, 224), bool)
    edge[0:5, 0:5] = True
    bx0, _, by0, _ = mf.crop_box(edge, np.empty((0, 2)), margin=10, hw=(224, 224))
    assert bx0 == -0.5 and by0 == -0.5


def test_pred_grasp_and_outcome():
    row = {"valid": "True", "x": "1", "y": "2", "theta": "3", "opening": "4", "jaw": "5",
           "match_primary": "True", "match_strict": "False"}
    assert mf.pred_grasp(row).tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert mf.outcome(row) == "hit/miss"
    assert mf.pred_grasp({**row, "valid": "False"}) is None


def test_filled_fraction_counts_cell_centres_inside():
    mask = np.zeros((20, 20), bool)
    mask[:, :10] = True
    rect = np.array([[4.5, 4.5], [14.5, 4.5], [14.5, 14.5], [4.5, 14.5]])   # centres x 5..14
    assert np.isclose(mf.filled_fraction(rect, mask), 0.5)
    assert mf.filled_fraction(np.array([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]), mask) is None
