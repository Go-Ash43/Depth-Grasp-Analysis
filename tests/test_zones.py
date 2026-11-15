"""data/zones.py and scripts/zone_diagnostics.py (Stage A helpers): analytic coverage, union G_theta,
G0 / G90, empty and out-of-frame handling, small openings, supersampling convergence. Synthetic only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import shapely

from data.zones import (CANDIDATES, coverage, g0_polygon, g90_polygons, region_stats, with_predicted_jaw,
                        zone_features, zone_polygons)

HW = (224, 224)
_spec = importlib.util.spec_from_file_location(
    "zone_diagnostics", Path(__file__).resolve().parents[1] / "scripts" / "zone_diagnostics.py")
zd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zd)


def G(x=50.0, y=50.0, t=0.0, L=16.0, W=4.0):
    return np.array([x, y, t, L, W], np.float64)


def test_axis_aligned_end4_coverage_is_analytic():
    # + zone: x in [54, 58], y in [48, 52]; - zone: x in [42, 46]. Cells are [j - .5, j + .5].
    cov = coverage(shapely.union_all(zone_polygons(G(), "Z-end4")), HW)
    edge = np.array([0.5, 1, 1, 1, 0.5])
    exp = np.zeros(HW)
    exp[48:53, 54:59] = np.outer(edge, edge)
    exp[48:53, 42:47] = np.outer(edge, edge)
    np.testing.assert_allclose(cov, exp, atol=1e-12)
    assert cov.sum() == pytest.approx(32.0)


def test_union_g_theta_is_area_weighted():
    f = np.zeros(HW, np.uint8)
    f[48:53, 54] = 1                                              # half-covered column of the + zone
    r = region_stats(zone_polygons(G(), "Z-end4"), f)
    assert r["g"] == pytest.approx((0.25 + 0.5 * 3 + 0.25) / 32.0)
    assert region_stats(zone_polygons(G(), "Z-end4"), np.ones(HW, np.uint8))["g"] == pytest.approx(1.0)
    assert region_stats(zone_polygons(G(), "Z-end4"), np.zeros(HW, np.uint8))["g"] == 0.0


@pytest.mark.parametrize("cand", CANDIDATES)
def test_half_turn_and_g90(cand):
    g = G(80.3, 61.7, 37.0, 22.0, 11.0)
    a = coverage(shapely.union_all(zone_polygons(g, cand)), HW)
    b = coverage(shapely.union_all(zone_polygons(g + [0, 0, 180, 0, 0], cand)), HW)
    np.testing.assert_allclose(a, b, atol=1e-9)
    c = coverage(shapely.union_all(g90_polygons(g, cand)), HW)
    d = coverage(shapely.union_all(zone_polygons(g + [0, 0, 90, 0, 0], cand)), HW)
    np.testing.assert_allclose(c, d, atol=1e-12)


@pytest.mark.parametrize("cand", CANDIDATES)
@pytest.mark.parametrize("L", [1.0, 2.0, 3.0, 4.0, 22.0])
def test_zones_never_overlap(cand, L):
    z = zone_polygons(G(L=L, W=L / 2, t=23.0), cand)
    assert z[0].intersection(z[1]).area == pytest.approx(0.0, abs=1e-12)


def test_band_small_opening_meets_at_centre():
    z = zone_polygons(G(L=3.0, W=2.0), "Z-band")                  # band [max(1.5 - 2, 0), 1.5] = [0, 1.5]
    assert sum(p.area for p in z) == pytest.approx(2 * 1.5 * 2.0)
    z4 = zone_polygons(G(L=10.0, W=2.0), "Z-band")                # [3, 5]
    assert sum(p.area for p in z4) == pytest.approx(2 * 2.0 * 2.0)


@pytest.mark.parametrize("cand", CANDIDATES)
def test_g0_continuous_area_equals_union_area(cand):
    g = G(100.2, 90.6, -33.0, 20.0, 10.0)
    area = shapely.union_all(zone_polygons(g, cand)).area
    assert g0_polygon(g, area).area == pytest.approx(area, rel=1e-9)
    f = zone_features(g, np.zeros(HW, np.uint8), cand)
    assert abs(f["g0_cells_binary"] - f["cells_binary"]) <= 0.5 * f["cells_binary"]   # reported, loosely sane


def test_out_of_frame_and_empty():
    g = G(x=1.0, y=100.0, L=200.0, W=10.0)                         # - zone at x in [-99, -49]: outside
    f = zone_features(g, np.zeros(HW, np.uint8), "Z-end4")
    assert f["jaw_empty"] == [False, True] and not f["empty"]
    assert f["frac_out"] == pytest.approx(0.5, abs=1e-12)
    far = zone_features(G(x=-300.0, y=100.0), np.zeros(HW, np.uint8), "Z-end4")
    assert far["empty"] and np.isnan(far["g_theta"]) and far["frac_out"] == pytest.approx(1.0)


def test_supersampling_converges_to_exact():
    g = G(70.37, 81.91, 37.0, 21.0, 9.0)
    union = shapely.union_all(zone_polygons(g, "Z-end4"))
    exact = coverage(union, HW)
    ii, jj = np.nonzero(exact > 0)
    errs = {}
    for n in (8, 16, 32):
        o = (np.arange(n) + 0.5) / n - 0.5
        ox, oy = np.meshgrid(o, o)
        est = np.array([shapely.contains_xy(union, j + ox.ravel(), i + oy.ravel()).mean()
                        for i, j in zip(ii, jj)])
        e = np.abs(est - exact[ii, jj])
        assert e.max() <= 4.0 / n                                  # bound shrinks with resolution
        errs[n] = e.mean()
    assert errs[32] < errs[8]                                      # aggregate convergence


def test_per_jaw_areas_half_out_of_frame():
    f = zone_features(G(x=1.0, y=100.0, L=200.0, W=10.0), np.zeros(HW, np.uint8), "Z-end4")
    plus, minus = f["jaws"]                                       # + zone x in [51, 101], - zone outside
    assert plus["area_cont"] == pytest.approx(500.0) and plus["area_in"] == pytest.approx(500.0)
    assert minus["area_cont"] == pytest.approx(500.0) and minus["area_in"] == 0.0
    assert minus["empty"] and not plus["empty"] and minus["cells_binary"] == 0
    # x in [51, 101]: 49 full + 2 half columns; y in [95, 105]: 9 full + 2 half rows. Cells >= 0.5:
    # full rows x (49 + 2 halves) = 9 * 51, half rows x full columns (0.5 each) = 2 * 49 -> 557
    assert plus["cells_binary"] == 9 * 51 + 2 * 49


def test_empty_zone_still_counts_flag_contact():
    # reviewer case: Z-end4, opening 1, jaw 0.5 -> union 0.25 cell: empty, G_theta NaN, but touches flags
    g = G(x=50.3, y=60.2, L=1.0, W=0.5)
    allflag = np.ones(HW, np.uint8)
    f = zone_features(g, allflag, "Z-end4")
    assert f["empty"] and np.isnan(f["g_theta"]) and f["flag_area"] == pytest.approx(0.25)
    rec = zd.scene_stats(g, allflag, allflag, candidates=("Z-end4",))
    assert all(p["touched"] for p in rec["Z-end4"]["pert"])
    assert rec["Z-end4"]["area_on_object"] == pytest.approx(1.0)   # small zone kept in geometry stats
    s = zd.summarise([rec], ("Z-end4",))["Z-end4"]
    assert s["area_on_object"]["n"] == 1 and s["area_on_object"]["median"] == pytest.approx(1.0)
    assert s["share_scenes_flag_in_zone"] == 1.0 and s["jaw_empty_rate"] == 1.0
    assert s["grasp_any_jaw_empty_rate"] == 1.0


def test_w_norm_check_states():
    r = {"w_norm_proposed_px_224": 60.0}
    assert zd.w_norm_check(None, r)["status"] == "unset"
    assert zd.w_norm_check(60, r)["status"] == "match"
    assert zd.w_norm_check(50.0, r)["status"] == "mismatch"


def test_output_guard_protects_production_cache(tmp_path):
    prod, other = tmp_path / "cache", tmp_path / "cache_other"
    with pytest.raises(SystemExit):
        zd.safe_out(prod / "diag", prod, other)                   # other input cache selected
    with pytest.raises(SystemExit):
        zd.safe_out(other / "diag", prod, other)
    assert zd.safe_out(tmp_path / "runs" / "diag", prod, other) == (tmp_path / "runs" / "diag").resolve()


def test_overwrite_handling(tmp_path):
    out = tmp_path / "diag"
    out.mkdir()
    (out / "config.json").write_text("{}")                        # only config exists: still refused
    with pytest.raises(SystemExit):
        zd.check_outputs(out, overwrite=False)
    od = out / "overlays"
    od.mkdir()
    (od / "slot1_a_0.png").write_bytes(b"x")
    (od / "manifest.json").write_text("{}")
    (od / "keep_me.txt").write_text("not ours")
    zd.check_outputs(out, overwrite=True)
    assert not (out / "config.json").exists() and not (od / "slot1_a_0.png").exists()
    assert not (od / "manifest.json").exists() and (od / "keep_me.txt").exists()


def _summary_with_groups(groups, cands=("Z-end4", "Z-band")):
    return {"subgroups": {g: {"n": len(v), "candidates": zd.summarise(v, cands)} for g, v in groups.items()}}


def test_console_report_empty_opening_lt4_subgroup():
    rec = zd.scene_stats(G(L=16.0, W=8.0), np.zeros(HW, np.uint8), np.ones(HW, np.uint8),
                         candidates=("Z-end4", "Z-band"))                  # opening 16 px: not < 4
    rep = zd.console_report(_summary_with_groups({"all": [rec], "obj_ge200": [rec], "opening_lt4": []}))
    assert rep["opening_lt4"] == {"n": 0, "note": "no scenes in this subgroup"}
    assert rep["all"]["n"] == 1 and set(rep["all"]["Z-end4"]) == set(zd.REPORT_KEYS)


def test_console_report_empty_obj_ge200_subgroup():
    rec = zd.scene_stats(G(L=3.0, W=1.5), np.zeros(HW, np.uint8), np.ones(HW, np.uint8),
                         candidates=("Z-end4", "Z-band"))
    rep = zd.console_report(_summary_with_groups({"all": [rec], "obj_ge200": [], "opening_lt4": [rec]}))
    assert rep["obj_ge200"]["n"] == 0
    assert rep["opening_lt4"]["n"] == 1 and "Z-band" in rep["opening_lt4"]


# ------------------------------------------------------------------ Stage A sampling / overlays (review 11)
def _fake_cache(n=30, seed=1):
    """Synthetic stand-in for open_cache's train split: grasps at 1024, m_sensor / obj at 224."""
    rng = np.random.default_rng(seed)
    grasps = {i: np.c_[rng.uniform(300, 700, 4), rng.uniform(300, 700, 4), rng.uniform(-90, 90, 4),
                       rng.uniform(5, 150, 4), rng.uniform(5, 40, 4)] for i in range(n)}
    ms = (rng.random((n,) + HW) < 0.1).astype(np.uint8)
    obj = np.zeros((n,) + HW, np.uint8)
    obj[:, 60:170, 60:170] = 1
    index = [{"part": "P", "object_id": f"o{i}", "view": "0", "obj_px_224": "500",
              "small_obj": "0" if i % 3 else "1", "status": "ok"} for i in range(n)]
    return grasps, ms, obj, index


def _run(seed, n_scenes, n_overlays, cache):
    grasps, ms, obj, index = cache
    r_s, r_o = zd.make_rngs(seed)
    st = zd.sample_standins(list(range(len(index))), n_scenes, lambda s: grasps[s], 224, r_s)
    rows, recs, groups = zd.compute_stats(st, ms, obj, index, ("Z-end4",), 2.0, 0.5, 64)
    picks = zd.select_overlays([r["opening224"] for r in rows], n_overlays, r_o)
    summ = {g: zd.summarise(v, ("Z-end4",)) for g, v in groups.items()}
    return st, rows, summ, picks


def test_overlay_count_never_changes_standins_or_statistics():
    cache = _fake_cache()
    st0, rows0, s0, p0 = _run(0, 12, 0, cache)
    st12, rows12, s12, p12 = _run(0, 12, 12, cache)
    assert [s for s, _ in st0] == [s for s, _ in st12]
    for (_, a), (_, b) in zip(st0, st12):
        np.testing.assert_array_equal(a, b)
    assert rows0 == rows12 and s0 == s12
    assert p0 == [] and len(p12) == 12 and len(set(p12)) == 12


def test_records_hold_no_coverage_maps():
    cache = _fake_cache(5)
    r_s, _ = zd.make_rngs(0)
    st = zd.sample_standins(list(range(5)), 5, lambda s: cache[0][s], 224, r_s)
    _, recs, _ = zd.compute_stats(st, cache[1], cache[2], cache[3], ("Z-end4", "Z-band"), 2.0, 0.5, 64)
    assert all("cov" not in rec[c] for rec in recs for c in rec)


@pytest.mark.parametrize("n, k", [(5, 12), (6, 4), (30, 12), (1, 1), (10, 0), (10, 1)])
def test_overlay_selection_is_exact_and_backfilled(n, k):
    openings = np.r_[np.full(n // 2, 1.0), np.arange(n - n // 2) + 10.0]     # ties among the smallest
    picks = zd.select_overlays(openings, k, np.random.default_rng(3))
    assert len(picks) == min(k, n) and len(set(picks)) == len(picks)
    smallest = list(np.argsort(openings, kind="stable")[:min(k, n) // 2])
    assert picks[:len(smallest)] == [int(i) for i in smallest]


def test_overlay_and_scene_count_validation():
    with pytest.raises(ValueError):
        zd.select_overlays([1.0, 2.0], -1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        zd.sample_standins([0, 1], 0, lambda s: None, 224, np.random.default_rng(0))
    with pytest.raises(ValueError):
        zd.sample_standins([], 5, lambda s: None, 224, np.random.default_rng(0))


def test_per_scene_rows_have_per_jaw_columns():
    cache = _fake_cache(3)
    _, rows, _, _ = _run(0, 3, 0, cache)
    for side in ("plus", "minus"):
        for k in zd.JAW_KEYS:
            assert f"Z-end4:jaw_{side}_{k}" in rows[0]


def test_selected_zone_refuses_provisional():
    from data.labels import load_labels_cfg
    from data.zones import selected_zone
    cfg = load_labels_cfg()
    if cfg["zones"]["selected"] is None:                           # until Stage B passes
        with pytest.raises(RuntimeError, match="provisional"):
            selected_zone(cfg)
    ok = {"zones": {"selected": "Z-end3", "candidates": list(CANDIDATES)}}
    assert selected_zone(ok) == "Z-end3"
    with pytest.raises(ValueError):
        selected_zone({"zones": {"selected": "Z-foo", "candidates": list(CANDIDATES)}})


def test_predicted_jaw_rule():
    assert with_predicted_jaw([1, 2, 3, 30, 99])[4] == 15.0


# ------------------------------------------------------------------ Stage A helpers
def test_label_ranges():
    g = np.array([[0, 0, 0, 1024 * 10 / 224, 1024 * 4 / 224], [0, 0, 0, 1024 * 3 / 224, 1024 * 6 / 224]])
    r = zd.label_ranges(g)
    assert r["opening_224"]["max"] == pytest.approx(10.0) and r["opening_224"]["share_lt_4px"] == 0.5
    assert r["w_norm_proposed_px_224"] == 10.0
    assert r["jaw_over_opening_per_grasp"]["median"] == pytest.approx((0.4 + 2.0) / 2)


def test_scene_stats_geometric_change_hand_values():
    # Z-end4 of L = 16, W = 4: two 4 x 4 squares (union 32). A 1-px axis shift: xor = 2 * 8 = 16 -> 0.5;
    # a diagonal shift: per square xor = 2 * (16 - 9) = 14 -> 28 / 32.
    ms = np.zeros(HW, np.uint8)
    rec = zd.scene_stats(G(), ms, np.ones(HW, np.uint8), candidates=("Z-end4",))["Z-end4"]
    shifts = {(p["kind"], i): p["geo"] for i, p in enumerate(rec["pert"])}
    geo = [p["geo"] for p in rec["pert"] if p["kind"] == "shift"]
    assert sorted(set(np.round(geo, 12))) == [0.5, 0.875]
    assert all(p["dg"] == 0.0 for p in rec["pert"])               # no flags: dG = 0 (finite, not NaN)
    assert not any(p["touched"] for p in rec["pert"]) and len(shifts) == 10


def test_scene_stats_flag_touching_and_summary():
    ms = np.zeros(HW, np.uint8)
    ms[48:53, 58] = 1                                              # edge column of the + zone
    obj = np.zeros(HW, np.uint8)
    obj[40:60, 40:60] = 1
    rec = zd.scene_stats(G(), ms, obj, candidates=("Z-end4",))
    s = zd.summarise([rec], ("Z-end4",))["Z-end4"]
    assert s["n_scenes"] == 1 and s["share_moves_flag_touching"] > 0
    assert s["abs_dg_flag_touching_moves"]["max"] > 0 and s["empty_rate"] == 0.0
    assert rec["Z-end4"]["area_on_object"] == pytest.approx(1.0)
