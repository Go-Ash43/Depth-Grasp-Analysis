"""data/holes.py: whole ellipses, placement independent of M_lie (no lie input at all), object-level
budget and cap, records (docs/proj_v3.md D1; notes/holes_decision_v1.md). Synthetic masks only."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from data.holes import covered_lie_frac, ellipse_mask, synth_holes

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "train_data.json").read_text())["holes"]
HW = (96, 96)


def _obj(r0=20, r1=76, c0=20, c1=76):
    o = np.zeros(HW, np.uint8)
    o[r0:r1, c0:c1] = 1
    return o


def _z():
    return np.zeros(HW, np.uint8)


def _run(seed, ms, obj, frac, cfg=CFG):
    return synth_holes(np.random.default_rng(seed), ms, obj, frac, cfg)


def _union(rec):
    u = np.zeros(HW, bool)
    for e in rec["ellipses"]:
        u |= ellipse_mask(HW, *e)
    return u


def test_no_lie_input_by_construction():
    """Condition 1: neither centres nor shapes can depend on M_lie, because the function takes none."""
    params = list(inspect.signature(synth_holes).parameters)
    assert params == ["rng", "m_sensor", "obj", "budget_frac", "cfg"]


@pytest.mark.parametrize("seed", range(6))
def test_synth_is_whole_ellipses_minus_sensor(seed):
    rng = np.random.default_rng(100 + seed)
    obj = _obj()
    ms = (rng.random(HW) < 0.05).astype(np.uint8)
    m, rec = _run(seed, ms, obj, 0.15)
    np.testing.assert_array_equal(m.astype(bool), _union(rec) & ~ms.astype(bool))   # exact, no clipping
    assert rec["n_ellipses"] == len(rec["ellipses"]) >= 1


@pytest.mark.parametrize("frac", [0.05, 0.15, 0.30])
def test_budget_is_measured_on_the_object_and_met(frac):
    obj = _obj()
    m, rec = _run(1, _z(), obj, frac)
    on_obj = int((m.astype(bool) & obj.astype(bool)).sum())
    assert rec["achieved"] == on_obj and rec["n_obj"] == int(obj.sum())
    assert rec["within_tol"] and abs(on_obj - frac * obj.sum()) <= CFG["budget_tol"] * frac * obj.sum()
    assert rec["shortfall_cause"] is None


@pytest.mark.parametrize("seed", range(20))
def test_budget_met_whenever_the_cap_does_not_bind(seed):
    """No rejection any more: with a free object every severity lands within tolerance."""
    obj = _obj()
    for frac in (0.05, 0.15, 0.30):
        assert _run(seed, _z(), obj, frac)[1]["within_tol"]


def test_cells_off_the_object_do_not_count():
    obj = _obj(40, 48, 40, 48)                                  # 64-cell object, ellipses spill off it
    m, rec = _run(3, _z(), obj, 0.30)
    assert rec["achieved"] == int((m.astype(bool) & obj.astype(bool)).sum())
    assert int(m.sum()) >= rec["achieved"]


def test_sensor_cells_are_not_synth_and_do_not_count():
    obj = _obj()
    ms = _z()
    ms[30:40, 30:60] = 1
    m, rec = _run(4, ms, obj, 0.15)
    assert not (m.astype(bool) & ms.astype(bool)).any()
    assert rec["achieved"] == int((m.astype(bool) & obj.astype(bool)).sum())


def test_object_cap_keeps_half_the_object_valid_and_records_the_shortfall():
    obj = _obj()
    n = int(obj.sum())
    ms = _z()
    rows = np.argwhere(obj.astype(bool))[: int(0.4 * n)]       # 40 % of the object already invalid
    ms[rows[:, 0], rows[:, 1]] = 1
    m, rec = _run(5, ms, obj, 0.30)
    invalid = (ms.astype(bool) | m.astype(bool)) & obj.astype(bool)
    assert invalid.sum() <= 0.5 * n
    assert rec["achieved"] <= rec["cap"] and not rec["within_tol"] and rec["shortfall_cause"] == "cap"


def test_tiny_object_and_zero_budget():
    obj = _obj(50, 52, 50, 52)                                  # 4 cells: 5 % is 0.2 cell
    m, rec = _run(7, _z(), obj, 0.05)
    assert m.sum() == 0 and rec["shortfall_cause"] == "under_tolerance"
    m, rec = _run(7, _z(), _obj(), 0.0)
    assert m.sum() == 0 and rec["within_tol"] and rec["shortfall_cause"] is None


def test_no_eligible_centre():
    obj = _obj()
    m, rec = _run(8, obj, obj, 0.15)                            # the whole object is a sensor hole
    assert m.sum() == 0 and rec["shortfall_cause"] in ("cap", "no_eligible_centre")


def test_deterministic_given_the_generator():
    obj = _obj()
    a = _run(9, _z(), obj, 0.15)
    b = _run(9, _z(), obj, 0.15)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_covered_lie_frac():
    h = _z()
    h[10:20, 10:20] = 1                                         # 100 hole cells
    lie = _z()
    lie[10:15, 10:20] = 1                                       # 50 of them are lies
    lie[50, 50] = 1                                             # a lie outside the hole does not count
    assert covered_lie_frac(h, lie) == 0.5
    assert covered_lie_frac(_z(), lie) == 0.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        _run(0, _z(), _obj(), 1.0)
    with pytest.raises(ValueError):
        _run(0, _z(), _z(), 0.15)                               # no object (R1 excludes these)
    with pytest.raises(ValueError):
        _run(0, _z(), _obj(), 0.15, {**CFG, "n_ellipses": [0, 3]})
