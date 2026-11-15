"""scripts/audit_train.py: histogram binning on edges (normalisation quantiles) and crop clipping
(small-object decision)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from data.jacquard import grasp_corners

_spec = importlib.util.spec_from_file_location(
    "audit_train", Path(__file__).resolve().parents[1] / "scripts" / "audit_train.py")
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)


def _boundary_values(edges):
    e = np.asarray(edges, np.float64)
    vals = [e, np.nextafter(e, -np.inf), np.nextafter(e, np.inf), (e[:-1] + e[1:]) / 2]
    return np.concatenate(vals + [np.random.default_rng(0).uniform(e[0] - 5, e[-1] + 5, 20000)])


@pytest.mark.parametrize("edges", [
    np.linspace(0.0, 5000.0, 10001),        # depth
    np.linspace(-1000.0, 1000.0, 20001),    # signed error (0.1 mm: inexact in float)
    np.linspace(-1.0, 1.0, 31),
])
def test_fast_path_matches_searchsorted(edges):
    h = at.Hist(edges)
    assert h.step is not None, "uniform edges must take the fast path"
    x = _boundary_values(edges)
    np.testing.assert_array_equal(h.index(x), np.searchsorted(edges, x, side="right") - 1)


def test_last_edge_is_overflow_and_first_edge_is_bin0():
    h = at.Hist(np.linspace(-1000.0, 1000.0, 20001))
    h.add(np.array([-1000.0, 1000.0, 0.0]))
    assert h.under == 0 and h.over == 1 and h.counts[0] == 1 and h.counts.sum() == 2


def test_out_of_range_quantile_is_none():
    h = at.Hist(np.linspace(0.0, 10.0, 11))
    h.add(np.array([-5.0] * 6 + [5.0] * 4))
    assert h.quantile(0.5) is None and h.quantile(0.0) == -5.0 and h.quantile(1.0) == 5.0
    assert h.quantile(0.9) == 6.0                    # upper edge of the bin holding 5.0


def test_tau_grid_pairs_every_candidate():
    cands = [1, 2.5, 16, 25]
    g = at.tau_grid(cands)
    assert all(c / 2 in g and c in g and 2 * c in g for c in cands)


def _scene(x0, x1, y0, y1, grasp):
    obj = np.zeros((1024, 1024), bool)
    obj[y0:y1 + 1, x0:x1 + 1] = True
    g = np.array([grasp], np.float64)
    return obj, (x0, x1, y0, y1), g[:, :2], grasp_corners(g)


def test_crop_clips_object_outside_centred_crop():
    # centred 512 crop covers indices 256..767; object x 200..299 is cut at x = 256 (44 of 100 columns left)
    obj, bbox, ctr, cor = _scene(200, 299, 500, 599, [250.0, 550.0, 0.0, 40.0, 10.0])
    r = at.crop_rows(obj, int(obj.sum()), bbox, ctr, cor, {"size": 224, "crops": [512, 1024]})
    full = r["crop512_obj_objpx224"]                 # object-centred crop keeps the whole object
    assert r["crop512_ctr_obj_in"] == 0 and r["crop512_obj_obj_in"] == 1
    assert 0 < r["crop512_ctr_objpx224"] < 0.5 * full       # ~44% of the area survives, not all of it
    assert r["crop512_ctr_ctr_in"] == 0 and r["crop512_obj_ctr_in"] == 1
    assert r["crop1024_ctr_obj_in"] == 1 and r["crop1024_obj_obj_in"] == 1
    assert abs(full - 100 * 100 * (224 / 512) ** 2) < 0.1 * full


def test_corners_can_leave_crop_while_centre_stays():
    # centre 2 px inside the crop's left edge, jaw axis along x with opening 40: corners poke out
    obj, bbox, ctr, cor = _scene(258, 300, 500, 540, [258.0, 520.0, 0.0, 40.0, 10.0])
    r = at.crop_rows(obj, int(obj.sum()), bbox, ctr, cor, {"size": 224, "crops": [512]})
    assert r["crop512_ctr_ctr_in"] == 1 and r["crop512_ctr_cor_in"] == 0
