"""scripts/audit_train.py audit_scene + summarise on a hand-built scene: the per-scene counts and
pooled rates behind the Phase-0 gates (M_sensor and M_lie rates), checked against hand-computed values.
The audit script is imported, not edited (it produced the frozen values)."""
import numpy as np
import pytest

from data.jacquard import SkipBudget
from tests.synthetic import AUDIT_CFG, N, hand_scene, load_audit_module, small_scene

at = load_audit_module()

# Hand-computed (tests/synthetic.py: every feature aligned to 32 source px = 7 cells)
OBJ_1024, OBJ_224 = 320 * 320, 70 * 70
HOLE_1024 = 32 * 32 + 1                 # hole block + shared-cell hole pixel
HOLE_224 = 7 * 7 + 1                    # its cells (the single pixel fills one cell by the any-rule)
LIE_1024 = {6: 1024 + 3, 12: 1024 + 2, 24: 1024 + 2}    # exact-tau pixel counts only at tau = 6
LIE_EFF_224 = {6: 49 + 2, 12: 49 + 1, 24: 49 + 1}       # shared cell never counts
LIE_FRAC_224 = {6: 49, 12: 49, 24: 49}                  # single-pixel lies cover < 50% of a cell
S_OBJ_1024, S_OBJ_224 = 64 * 64, 14 * 14                # small_scene(): 4096 px, 196 cells
S_HOLE_1024, S_HOLE_224 = 32 * 32, 7 * 7                # 25% of the object is a hole


def _row(scene, view=0):
    hists = at.make_hists(AUDIT_CFG)
    row = at.audit_scene(scene, hists, AUDIT_CFG)
    return {"part": "P", "object_id": f"obj{view}", "view": view, **row}, hists


def test_per_scene_counts():
    r, _ = _row(hand_scene())
    assert r["status"] == "ok"
    assert r["obj_px_1024"] == OBJ_1024 and r["obj_px_224"] == OBJ_224
    assert r["n_valid_obj_1024"] == OBJ_1024 - HOLE_1024
    assert r["n_valid_obj_224"] == OBJ_224 - HOLE_224
    assert r["n_valid_1024"] == N * N - HOLE_1024 and r["n_valid_224"] == 224 * 224 - HOLE_224
    assert r["sensor_obj_1024"] == pytest.approx(HOLE_1024 / OBJ_1024, rel=1e-12)
    assert r["sensor_obj_224"] == pytest.approx(HOLE_224 / OBJ_224, rel=1e-12)
    for t in AUDIT_CFG["taus"]:
        assert r[f"lie_obj_1024@{t:g}"] == LIE_1024[t] == r[f"lie_img_1024@{t:g}"]   # no table lies
        assert r[f"lie_eff_obj_224@{t:g}"] == LIE_EFF_224[t] == r[f"lie_eff_img_224@{t:g}"]
        assert r[f"lie_frac_obj_224@{t:g}"] == LIE_FRAC_224[t]


def test_small_scene_counts():
    r, _ = _row(small_scene())
    assert (r["obj_px_1024"], r["obj_px_224"]) == (S_OBJ_1024, S_OBJ_224)
    assert (r["n_valid_obj_1024"], r["n_valid_obj_224"]) == (S_OBJ_1024 - S_HOLE_1024, S_OBJ_224 - S_HOLE_224)
    assert all(r[f"lie_obj_1024@{t:g}"] == 0 for t in AUDIT_CFG["taus"])


def test_summarise_pools_counts_and_applies_r1():
    """Two included scenes of very different object size and hole rate, plus one R1-excluded scene.
    Pooled rate = sum of counts / sum of pixels; the test also asserts that this differs from the
    unweighted mean of per-scene rates, so averaging rates could not pass it."""
    a, hists = _row(hand_scene(), view=0)
    b, hb_ok = _row(small_scene(), view=1)
    dead = hand_scene()                                    # every pixel a hole -> R1 no_valid_depth_224
    dead["m_sensor"][:] = 1
    dead["depth_stereo"][:] = np.nan
    bad, hb = _row(dead, view=2)
    assert bad["status"] == "no_valid_depth_224"
    for h in (hb_ok, hb):                                  # hb is empty for the excluded scene
        for k in hists:
            hists[k].merge(h[k])
    s = at.summarise([a, b, bad], hists, SkipBudget(), AUDIT_CFG, n_objects_split=3, n_scenes_split=3)

    assert s["n_scenes_included"] == 2 and s["exclusions"]["counts"] == {"no_valid_depth_224": 1}
    assert s["exclusions"]["objects_dropped_entirely"] == ["P/obj2"]

    def pooled_and_mean(num_a, den_a, num_b, den_b):
        pooled, mean = (num_a + num_b) / (den_a + den_b), (num_a / den_a + num_b / den_b) / 2
        assert abs(pooled - mean) > 0.01 * pooled          # the two scenes really discriminate
        return pooled

    ph = s["pooled_hole_rate"]                             # gate inputs: included scenes only
    assert ph["obj_1024"] == pytest.approx(pooled_and_mean(HOLE_1024, OBJ_1024, S_HOLE_1024, S_OBJ_1024), rel=1e-12)
    assert ph["obj_224"] == pytest.approx(pooled_and_mean(HOLE_224, OBJ_224, S_HOLE_224, S_OBJ_224), rel=1e-12)
    assert ph["img_1024"] == pytest.approx((HOLE_1024 + S_HOLE_1024) / (2 * N ** 2), rel=1e-12)
    assert ph["img_224"] == pytest.approx((HOLE_224 + S_HOLE_224) / (2 * 224 ** 2), rel=1e-12)
    rows = {r["tau_mm"]: r for r in s["tau_table"]}
    v1024 = (OBJ_1024 - HOLE_1024, S_OBJ_1024 - S_HOLE_1024)
    v224 = (OBJ_224 - HOLE_224, S_OBJ_224 - S_HOLE_224)
    for t in AUDIT_CFG["taus"]:                            # small scene has no lies: num_b = 0
        assert rows[t]["lie_rate_obj_1024"] == pytest.approx(
            pooled_and_mean(LIE_1024[t], v1024[0], 0, v1024[1]), rel=1e-12)
        assert rows[t]["lie_eff_rate_obj_224"] == pytest.approx(
            pooled_and_mean(LIE_EFF_224[t], v224[0], 0, v224[1]), rel=1e-12)
        assert rows[t]["lie_frac_rate_obj_224"] == pytest.approx(
            pooled_and_mean(LIE_FRAC_224[t], v224[0], 0, v224[1]), rel=1e-12)
        assert rows[t]["is_candidate"] == (t == 12)
    assert s["object_fully_invalid"] == {"n_scenes_1024": 0, "n_scenes_224": 0}
