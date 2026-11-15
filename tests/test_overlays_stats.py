"""scripts/overlays.py helpers: the flip direction is read from a CONSISTENT sign, not from absolute
asymmetry (second review of Phase 1, fix 2). Synthetic masks only."""
import numpy as np

from scripts import overlays as ov


def _scene(shadow_dx):
    """Square object with a sensor shadow strip on one side (dx < 0: left, > 0: right)."""
    obj = np.zeros((96, 96), np.uint8)
    obj[30:60, 30:60] = 1
    ms = np.zeros_like(obj)
    if shadow_dx < 0:
        ms[30:60, 24:30] = 1
    else:
        ms[30:60, 60:66] = 1
    return ms, obj


def test_shadow_side_sign():
    assert ov.shadow_side(*_scene(+1))["lr"] == 1.0
    assert ov.shadow_side(*_scene(-1))["lr"] == -1.0
    assert abs(ov.shadow_side(*_scene(+1))["ud"]) < 1e-9


def test_consistent_left_right_suggests_vertical_flip():
    lr = ov._axis_reading(np.full(50, 0.8) + np.linspace(-0.1, 0.1, 50))
    ud = ov._axis_reading(np.linspace(-0.6, 0.6, 50))                 # large but mixed-sign asymmetry
    assert lr["consistent_sign"] and not ud["consistent_sign"]
    assert ov.flip_suggestion(lr, ud) == "vertical"
    assert ov.flip_suggestion(ud, lr) == "horizontal"


ACH = {"low": [0.045, 0.05, 0.052], "mid": [0.14, 0.15, 0.155], 0.30: [0.27, 0.30, 0.31],
       0.25: [0.23, 0.25, 0.26], 0.20: [0.18, 0.20, 0.21]}


def test_severity_rule_as_frozen():
    r = ov.apply_severity_rule({0.30: 0.85, 0.25: 0.9, 0.20: 0.95}, 0.95, ACH)
    assert r["high_budget"] == 0.30 and not r["high_changed"] and r["mid_ok"] and not r["stop"]
    assert r["action"] == "no change"
    r = ov.apply_severity_rule({0.30: 0.70, 0.25: 0.82, 0.20: 0.95}, 0.95, ACH)
    assert r["high_budget"] == 0.25 and r["high_changed"] and not r["stop"]
    r = ov.apply_severity_rule({0.30: 0.70, 0.25: 0.75, 0.20: 0.81}, 0.95, ACH)
    assert r["high_budget"] == 0.20 and not r["stop"]


def test_severity_rule_stops():
    shares = {0.30: 0.85, 0.25: 0.9, 0.20: 0.95}
    assert ov.apply_severity_rule(shares, 0.89, ACH)["stop"]                     # mid below 0.90
    floor = ov.apply_severity_rule({0.30: 0.70, 0.25: 0.75, 0.20: 0.79}, 0.95, ACH)
    assert floor["stop"] and floor["high_budget"] is None and "floor" in floor["action"]   # no 0.20 fallthrough
    overlap = {**ACH, 0.30: [0.14, 0.29, 0.31]}                                  # high p10 <= mid median
    assert ov.apply_severity_rule(shares, 0.95, overlap)["stop"]
    low_mid = {**ACH, "mid": [0.04, 0.15, 0.16]}                                 # mid p10 <= low median
    assert ov.apply_severity_rule(shares, 0.95, low_mid)["stop"]


def test_absolute_asymmetry_alone_is_undecided():
    """Irregular shapes: big |asymmetry| on both axes but no consistent side -> undecided."""
    mixed = ov._axis_reading(np.r_[np.full(25, -0.7), np.full(25, 0.7)])
    assert mixed["abs_median"] > 0.5 and not mixed["consistent_sign"]
    assert ov.flip_suggestion(mixed, mixed) == "undecided"
    assert ov.flip_suggestion({"n": 0}, {"n": 0}) == "undecided"
