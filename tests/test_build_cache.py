"""scripts/build_cache.py + data/cache_layout.py on synthetic scenes (no dataset access).

Covers: count-for-count agreement with the frozen audit (lies at every tau, completed-depth histogram),
R4 order and hard check, R1 statuses, cached-array invariants, and a full write/read round trip that
is byte-identical across two builds."""
import argparse
import csv
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest

from data.cache_layout import array_specs, lie_name, open_cache
from data.lies import m_lie
from data.mask_ops import resize_depth, resize_mask_any
from tests.synthetic import (ABOVE_CLAMP, AUDIT_CFG, N, hand_scene, load_audit_module, noisy_scene,
                             small_scene)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_cache", REPO / "scripts" / "build_cache.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)
at = load_audit_module()

CFG = bc.frozen_cfg(REPO / "configs" / "frozen_day1.json")      # real frozen values, no data access
S = CFG["size"]


def full(sc, seed=0):
    """Add the load_scene keys the synthetic builder omits."""
    sc = dict(sc)
    sc["rgb"] = np.random.default_rng(seed).integers(0, 256, (N, N, 3), dtype=np.uint8)
    sc["n_grasps_dropped"] = 0
    return sc


def _cell(p):
    return tuple(int(v * 7 // 32) for v in p)


def test_frozen_cfg_values():
    assert CFG["taus"] == [6.0, 12.0, 24.0] and CFG["tau"] == 12.0
    assert CFG["clamp"] == [790.0, 1680.0] and S == 224 and CFG["min_obj_px"] == 200


# ------------------------------------------------------------------ same numbers as the frozen audit
@pytest.mark.parametrize("make", [hand_scene, small_scene, lambda: noisy_scene(0), lambda: noisy_scene(1)],
                         ids=["hand", "small", "noisy0", "noisy1"])
def test_counts_match_audit(make):
    sc = full(make())
    row_a = at.audit_scene(sc, at.make_hists(AUDIT_CFG), AUDIT_CFG)
    row, arrays, g, _ = bc.process_scene(sc, CFG, abort_on_clamp=False)
    assert row["status"] == row_a["status"] == "ok"
    assert row["obj_px_224"] == row_a["obj_px_224"]
    for t in CFG["taus"]:
        k = f"{t:g}"
        for f in (f"lie_obj_1024@{k}", f"lie_eff_obj_224@{k}", f"lie_frac_obj_224@{k}"):
            assert row[f] == row_a[f], f
    np.testing.assert_array_equal(g, sc["grasps"])


@pytest.mark.parametrize("make", [hand_scene, lambda: noisy_scene(0)], ids=["hand", "noisy0"])
def test_completed_depth_matches_audit_without_clamp(make):
    """With a clamp wide enough to change nothing, ds_comp is exactly the audit's completed input, so
    the normalisation histogram is the audit's histogram."""
    sc = full(make())
    cfg = dict(CFG, clamp=[1e-3, 1e7])
    ha = at.make_hists(AUDIT_CFG)
    at.audit_scene(sc, ha, AUDIT_CFG)
    row, arrays, _, stats = bc.process_scene(sc, cfg, abort_on_clamp=True)
    assert row["n_clamped"] == 0
    hb = at.make_hists(AUDIT_CFG)
    for k in bc.STAT_HISTS:
        hb[k].add(stats[k])
        np.testing.assert_array_equal(hb[k].counts, ha[k].counts, err_msg=k)
        assert (hb[k].under, hb[k].over, hb[k].n) == (ha[k].under, ha[k].over, ha[k].n)


# ------------------------------------------------------------------ R4
def test_lie_uses_raw_depth_and_input_is_clamped():
    sc = full(hand_scene())
    row, arrays, _, _ = bc.process_scene(sc, CFG, abort_on_clamp=True)
    c = _cell(ABOVE_CLAMP)
    for t in CFG["taus"]:
        assert arrays[lie_name("any", t)][c] == 1             # raw 2000 mm vs D* 1450: a lie
    assert np.nanmax(arrays["ds"]) <= CFG["clamp"][1]
    assert row["n_clamped"] == 1 and row["n_clamped_truthful"] == 0


def _truthful_far_scene():
    sc = full(hand_scene())
    sc["depth_perfect"] = sc["depth_perfect"].copy()
    sc["depth_stereo"] = sc["depth_stereo"].copy()
    sc["depth_perfect"][10:20, 10:20] = 1700.0                # D* above the clamp range...
    sc["depth_stereo"][10:20, 10:20] = 1700.0                 # ...and the stereo agrees: truthful
    return sc


def test_r4_aborts_on_train_and_counts_elsewhere():
    sc = _truthful_far_scene()
    with pytest.raises(RuntimeError, match="R4"):
        bc.process_scene(sc, CFG, abort_on_clamp=True)
    row, arrays, _, _ = bc.process_scene(sc, CFG, abort_on_clamp=False)
    assert row["n_clamped_truthful"] == 100
    d_ref, _ = resize_depth(sc["depth_perfect"], np.zeros((N, N), np.uint8), (S, S))
    np.testing.assert_array_equal(arrays["dstar"], d_ref)     # D* is never clamped
    assert arrays["dstar"].max() == pytest.approx(1700.0)


# ------------------------------------------------------------------ R1
def test_r1_statuses():
    sc = full(hand_scene())
    sc["obj_mask"] = np.zeros((N, N), bool)
    sc["obj_mask"][100, 100] = True                            # vanishes under the majority rule
    row, arrays, g, stats = bc.process_scene(sc, CFG, abort_on_clamp=True)
    assert row["status"] == "no_object_224" and arrays is None and g is None and stats == {}

    sc = full(hand_scene())
    ms = np.zeros((N, N), np.uint8)
    ms[::4, ::4] = 1                                           # every 224 cell touches a hole
    sc["m_sensor"] = ms
    sc["depth_stereo"] = np.where(ms.astype(bool), np.nan, sc["depth_perfect"]).astype(np.float32)
    row, arrays, _, _ = bc.process_scene(sc, CFG, abort_on_clamp=True)
    assert row["status"] == "no_valid_depth_224" and arrays is None


# ------------------------------------------------------------------ cached-array invariants
@pytest.mark.parametrize("seed", [0, 1])
def test_array_invariants(seed):
    sc = full(noisy_scene(seed), seed)
    row, a, _, _ = bc.process_scene(sc, CFG, abort_on_clamp=False)
    specs = array_specs(S, CFG["taus"])
    assert set(a) == set(specs)
    for n, (dt, sh) in specs.items():
        assert a[n].dtype == dt and a[n].shape == sh, n
    ms = a["m_sensor"].astype(bool)
    np.testing.assert_array_equal(ms, resize_mask_any(sc["m_sensor"], (S, S)).astype(bool))
    np.testing.assert_array_equal(np.isnan(a["ds"]), ms)
    assert np.isfinite(a["ds_comp"]).all()
    np.testing.assert_array_equal(a["ds_comp"][~ms], a["ds"][~ms])
    for n in specs:
        if n.startswith("lie_") or n in ("m_sensor", "obj"):
            assert set(np.unique(a[n])) <= {0, 1}, n
    for t in CFG["taus"]:                                      # any-rule is a superset of the fraction rule
        assert (a[lie_name("fraction", t)] <= a[lie_name("any", t)]).all()
        lie = m_lie(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], t)
        np.testing.assert_array_equal(a[lie_name("any", t)], resize_mask_any(lie, (S, S)))


# ------------------------------------------------------------------ driver: write / read round trip
SCENES = {("P/objA", 0): lambda: full(hand_scene()),
          ("P/objA", 1): lambda: full(noisy_scene(0), 1),
          ("P/objB", 0): None,                                   # load skip
          ("P/objC", 0): lambda: full(small_scene(), 2)}


def _build(tmp_path, monkeypatch, name, cfg=CFG, scenes=None, overwrite=False, workers=0, chunk=2):
    scenes = SCENES if scenes is None else scenes
    if workers:                                    # Pool pickles partial(build_chunk): module must be importable
        monkeypatch.setitem(sys.modules, "build_cache", bc)

    def fake_try_load(obj_dir, view, root=None):
        make = scenes[(obj_dir, view)]
        if make is None:
            return None, {"obj_dir": obj_dir, "view": view, "error_type": "SceneDataError", "message": "bad"}
        return make(), None

    monkeypatch.setattr(bc, "try_load_scene", fake_try_load)
    man = tmp_path / f"manifest_{name}.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "object_id", "scene_idx", "status"])
        w.writeheader()
        for (o, v) in scenes:
            p, oid = o.split("/")
            w.writerow({"part": p, "object_id": oid, "scene_idx": v, "status": "ok"})
    splits = {"splits": {"train": sorted({o for o, _ in scenes}, reverse=True)}}   # builder must sort
    args = argparse.Namespace(max_scenes=0, seed=0, overwrite=overwrite, workers=workers, chunk=chunk)
    out = tmp_path / name
    c = dict(cfg, root=str(tmp_path))
    summary = bc.build_split("train", args, c, splits, man, out, {"git_commit": None})
    return out, summary


def _summary(out):
    return json.loads((out / "train" / "summary.json").read_text())


def _splits_for(scenes=None):
    """The splits dict a synthetic train cache was built from (open_cache checks membership against it)."""
    scenes = SCENES if scenes is None else scenes
    return {"splits": {"train": sorted({o for o, _ in scenes})}}


def _open(out, scenes=None, **kw):
    return open_cache(out, "train", splits=_splits_for(scenes), **kw)


def test_round_trip(tmp_path, monkeypatch):
    out, summary = _build(tmp_path, monkeypatch, "c1")
    cache = _open(out)
    idx = cache["index"]
    order = [(r["part"] + "/" + r["object_id"], int(r["view"])) for r in idx]
    assert order == [("P/objA", 0), ("P/objA", 1), ("P/objB", 0), ("P/objC", 0)]   # sorted, fixed slots
    assert [r["status"] for r in idx] == ["ok", "ok", "load_skip", "ok"]
    assert summary["n_ok"] == 3 and summary["excluded_by_reason"] == {"load_skip": 1}
    assert summary["skips_by_error_type"] == {"SceneDataError": 1}
    assert summary["clamp"]["n_truthful_clamped_px"] == 0
    assert summary["train_stats"]["normalisation"]["ds_comp"] is not None
    A = cache["arrays"]
    for slot, key in ((0, ("P/objA", 0)), (1, ("P/objA", 1)), (3, ("P/objC", 0))):
        sc = SCENES[key]()
        row, ref, g, _ = bc.process_scene(sc, CFG, abort_on_clamp=True)
        for n in ref:
            np.testing.assert_array_equal(A[n][slot], ref[n], err_msg=n)
        np.testing.assert_array_equal(cache["grasps"](slot), sc["grasps"])
        assert int(idx[slot]["obj_px_224"]) == row["obj_px_224"]
    for n in A:                                                   # excluded slot: all zero
        assert not np.any(A[n][2]), n
    assert len(cache["grasps"](2)) == 0


def test_two_builds_byte_identical(tmp_path, monkeypatch):
    o1, _ = _build(tmp_path, monkeypatch, "c1")
    o2, _ = _build(tmp_path, monkeypatch, "c2")
    files = sorted(p.name for p in (o1 / "train").iterdir() if p.suffix in (".npy", ".csv"))
    assert "index.csv" in files and "ds_comp.npy" in files
    for f in files:
        assert (o1 / "train" / f).read_bytes() == (o2 / "train" / f).read_bytes(), f


R4_SCENES = {**SCENES, ("P/objC", 0): _truthful_far_scene}     # last slot fails R4


def test_failed_build_keeps_records_but_is_not_complete(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="R4"):
        _build(tmp_path, monkeypatch, "bad", scenes=R4_SCENES)
    d = tmp_path / "bad" / "train"
    assert not (d / "COMPLETE").exists()
    with pytest.raises(RuntimeError, match="COMPLETE"):
        _open(tmp_path / "bad", R4_SCENES)
    s = _summary(tmp_path / "bad")
    assert s["complete"] is False and "R4" in s["error"] and "train_stats" not in s
    assert s["n_processed"] == 3                                  # slots 0-2 done before slot 3 aborted
    assert s["skips_by_error_type"] == {"SceneDataError": 1}
    with open(d / "index.csv", newline="") as fh:
        assert [r["status"] for r in csv.DictReader(fh)] == ["ok", "ok", "load_skip"]


def test_skip_budget_abort_keeps_skip_records(tmp_path, monkeypatch):
    scenes = {(f"P/o{i:02d}", 0): None for i in range(12)}         # 10 in a row aborts (R3)
    with pytest.raises(RuntimeError, match="in a row"):
        _build(tmp_path, monkeypatch, "skips", scenes=scenes)
    s = _summary(tmp_path / "skips")
    assert s["complete"] is False and "in a row" in s["error"]
    assert len(s["skip_budget"]["skips"]) >= 10 and s["skip_budget"]["max_run_seen"] >= 10
    assert not (tmp_path / "skips" / "train" / "COMPLETE").exists()


def test_failed_overwrite_leaves_no_stale_summary(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "c1")
    assert _summary(out)["complete"] is True
    with pytest.raises(RuntimeError, match="R4"):
        _build(tmp_path, monkeypatch, "c1", scenes=R4_SCENES, overwrite=True)
    s = _summary(out)
    assert s["complete"] is False and "R4" in s["error"]
    assert not (out / "train" / "COMPLETE").exists()


def _drive_gone():
    raise RuntimeError("dataset root /x is missing (drive disconnected?)")


def test_loader_exception_keeps_earlier_records_in_chunk(tmp_path, monkeypatch):
    scenes = {("P/objA", 0): SCENES[("P/objA", 0)], ("P/objA", 1): _drive_gone}   # same chunk (size 2)
    with pytest.raises(RuntimeError, match="drive disconnected"):
        _build(tmp_path, monkeypatch, "gone", scenes=scenes)
    s = _summary(tmp_path / "gone")
    assert s["complete"] is False and "drive disconnected" in s["error"] and "P/objA view 1" in s["error"]
    assert s["n_processed"] == 1 and s["n_ok"] == 1 and "train_stats" not in s
    assert s["skip_budget"]["n_skipped"] == 0                      # the failing scene is not a skip
    with open(tmp_path / "gone" / "train" / "index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["object_id"], r["view"], r["status"]) for r in rows] == [("objA", "0", "ok")]
    assert not (tmp_path / "gone" / "train" / "COMPLETE").exists()


def test_failed_grasp_save_is_not_success(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        _build(tmp_path, monkeypatch, "nosave")
    s = _summary(tmp_path / "nosave")
    assert s["complete"] is False and "OSError" in s["error"] and "train_stats" not in s
    assert s["n_processed"] == 4
    assert not (tmp_path / "nosave" / "train" / "COMPLETE").exists()


# ------------------------------------------------------------------ estimator = the one that froze tau
def test_q_interp_matches_audit_report():
    from scripts.audit_report import quantile as ar_quantile
    h = bc.stat_hists(CFG)["abserr_bg_1024"]
    h.add(np.random.default_rng(0).lognormal(0.8, 0.6, 5000))
    for q in (0.01, 0.1587, 0.5, 0.8413, 0.99):
        assert bc.q_interp(h, q) == ar_quantile((h.edges, h.counts, h.under, h.over), q)


def test_noise_floor_pick_uses_interpolation():
    """A case where the upper bin edge would cross 8.0 mm (-> tau 15) but the interpolated p99, the
    estimator that froze tau, does not (-> tau 12): the confirmation must agree with the freeze."""
    hists = bc.stat_hists(CFG)
    hists["abserr_bg_1024"].add(np.r_[np.full(986, 1.0), np.full(14, 7.95)])
    h = hists["abserr_bg_1024"]
    assert bc.TAU_FACTOR * h.quantile(0.99) > 12                   # upper edge: would pick 15
    st = bc.train_stats(hists, CFG, 12.0)["noise_floor"]
    assert bc.TAU_FACTOR * st["bg_abserr_p99_mm"] <= 12
    assert st["tau_rule_pick_mm"] == 12 and st["agrees"]


def test_refuses_nonempty_without_overwrite(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, "c1")
    with pytest.raises(SystemExit):
        _build(tmp_path, monkeypatch, "c1")


def test_open_cache_refuses_smoke(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "c1")
    p = out / "train" / "summary.json"
    s = json.loads(p.read_text())
    s["smoke"] = True
    p.write_text(json.dumps(s))
    with pytest.raises(RuntimeError, match="smoke"):
        _open(out)
    assert _open(out, allow_smoke=True)["summary"]["smoke"]


# ------------------------------------------------------------------ review 10/11: builder hardening
def _no_object_scene():
    sc = full(hand_scene())
    sc["obj_mask"] = np.zeros((N, N), bool)
    sc["obj_mask"][100, 100] = True                            # vanishes at 224: no_object_224
    return sc


def _no_valid_depth_scene():
    sc = full(hand_scene())
    ms = np.zeros((N, N), np.uint8)
    ms[::4, ::4] = 1                                           # every 224 cell touches a hole
    sc["m_sensor"] = ms
    sc["depth_stereo"] = np.where(ms.astype(bool), np.nan, sc["depth_perfect"]).astype(np.float32)
    return sc


EXCL_SCENES = {**SCENES, ("P/objD", 0): _no_object_scene, ("P/objE", 0): _no_valid_depth_scene}


@pytest.mark.parametrize("bad", [["--chunk", "0"], ["--chunk", "-1"], ["--workers", "-1"],
                                 ["--max-scenes", "-1"]])
def test_builder_rejects_bad_args(monkeypatch, bad):
    monkeypatch.setattr(sys, "argv", ["build_cache.py", *bad])
    with pytest.raises(SystemExit):
        bc.parse_args()


def test_audit_rejects_bad_args(monkeypatch):
    for bad in (["--chunk", "-1"], ["--workers", "-1"], ["--max-scenes", "-1"]):
        monkeypatch.setattr(sys, "argv", ["audit_train.py", *bad])
        with pytest.raises(SystemExit):
            at.parse_args()


def test_missing_slot_record_is_not_complete(tmp_path, monkeypatch):
    real = bc.build_chunk

    def drop_last(items, **kw):                                  # simulate a lost record
        r, a, g, sk, h, err = real(items, **kw)
        if any(row["slot"] == 3 for row in r):
            k = [row["slot"] for row in r].index(3)
            r, a, g = r[:k] + r[k + 1:], a[:k] + a[k + 1:], g[:k] + g[k + 1:]
        return r, a, g, sk, h, err

    monkeypatch.setattr(bc, "build_chunk", drop_last)
    with pytest.raises(RuntimeError, match="records for 4 slots"):
        _build(tmp_path, monkeypatch, "lost")
    s = _summary(tmp_path / "lost")
    assert s["complete"] is False and "missing" in s["error"] and "train_stats" not in s
    assert not (tmp_path / "lost" / "train" / "COMPLETE").exists()


def test_round_trip_with_both_exclusions_and_a_load_skip(tmp_path, monkeypatch):
    out, s = _build(tmp_path, monkeypatch, "excl", scenes=EXCL_SCENES)
    cache = _open(out, EXCL_SCENES)                              # status-aware reader accepts it
    st = {r["object_id"]: r for r in cache["index"]}
    assert st["objD"]["status"] == "no_object_224" and int(st["objD"]["n_grasps"]) > 0
    assert st["objE"]["status"] == "no_valid_depth_224" and int(st["objE"]["n_grasps"]) > 0
    assert st["objB"]["status"] == "load_skip" and st["objB"]["n_grasps"] == ""
    for r in cache["index"]:
        n = len(cache["grasps"](int(r["slot"])))
        assert n == (int(r["n_grasps"]) if r["status"] == "ok" else 0)
    assert s["excluded_by_reason"] == {"load_skip": 1, "no_object_224": 1, "no_valid_depth_224": 1}


def _tamper(out, fn):
    d = out / "train"
    fn(d)
    return d


@pytest.mark.parametrize("name, fn, msg", [
    ("shifted_offsets", lambda d: np.save(d / "grasp_offsets.npy",
                                          np.load(d / "grasp_offsets.npy") + np.r_[0, 1, 1, 1, 1, 1, 0]),
     "stored grasps|never decrease"),
    ("decreasing", lambda d: np.save(d / "grasp_offsets.npy",
                                     np.load(d / "grasp_offsets.npy")[[0, 2, 1, 3, 4, 5, 6]]), "never decrease"),
    ("float_offsets", lambda d: np.save(d / "grasp_offsets.npy",
                                        np.load(d / "grasp_offsets.npy").astype(np.float64)), "1-D integers"),
    ("grasps_f32", lambda d: np.save(d / "grasps_1024.npy",
                                     np.load(d / "grasps_1024.npy").astype(np.float32)), "float64"),
])
def test_reader_rejects_bad_offsets_and_grasps(tmp_path, monkeypatch, name, fn, msg):
    out, _ = _build(tmp_path, monkeypatch, name, scenes=EXCL_SCENES)
    _tamper(out, fn)
    with pytest.raises(RuntimeError, match=msg):
        _open(out, EXCL_SCENES)


def _edit_index(d, fn):
    p = d / "index.csv"
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0])
    fn(rows)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _edit_summary(d, **kw):
    p = d / "summary.json"
    s = json.loads(p.read_text())
    s.update(kw)
    p.write_text(json.dumps(s))


@pytest.mark.parametrize("fn, msg", [
    (lambda d: _edit_index(d, lambda rows: rows[0].update(slot="1")), "has slot"),
    (lambda d: _edit_index(d, lambda rows: rows[0].update(n_grasps="0")), "stored grasps"),
    (lambda d: _edit_index(d, lambda rows: rows[0].update(n_grasps="")), "n_grasps"),
    (lambda d: _edit_summary(d, complete=False), "complete"),
    (lambda d: _edit_summary(d, error="boom"), "error"),
    (lambda d: _edit_summary(d, n_ok=99), "n_ok"),
    (lambda d: _edit_summary(d, n_slots=99), "n_slots"),
])
def test_reader_rejects_bad_index_or_summary(tmp_path, monkeypatch, fn, msg):
    _reader_case(tmp_path, monkeypatch, fn, msg)


def _drop_summary_field(d, key):
    p = d / "summary.json"
    s = json.loads(p.read_text())
    del s[key]
    p.write_text(json.dumps(s))


@pytest.mark.parametrize("key", ["n_slots", "n_processed", "n_ok"])
def test_reader_requires_count_fields(tmp_path, monkeypatch, key):
    _reader_case(tmp_path, monkeypatch, lambda d: _drop_summary_field(d, key), f"lacks required field '{key}'")


def _reader_case(tmp_path, monkeypatch, fn, msg):
    out, _ = _build(tmp_path, monkeypatch, "t", scenes=EXCL_SCENES)
    fn(out / "train")
    with pytest.raises(RuntimeError, match=msg):
        _open(out, EXCL_SCENES)


def _slot_of(d, obj_id):
    with open(d / "index.csv", newline="") as fh:
        return [r["object_id"] for r in csv.DictReader(fh)].index(obj_id)


def test_reader_rejects_grasps_on_excluded_slot(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "x", scenes=EXCL_SCENES)
    d = out / "train"
    k = _slot_of(d, "objD")                                       # no_object_224: empty slice expected
    off = np.load(d / "grasp_offsets.npy")
    g = np.load(d / "grasps_1024.npy")
    assert off[k + 1] == off[k]
    g = np.insert(g, off[k], [500.0, 500.0, 0.0, 50.0, 10.0], axis=0)   # one extra row inside slot k
    off[k + 1:] += 1                                              # offsets stay monotone and end at G
    np.save(d / "grasps_1024.npy", g)
    np.save(d / "grasp_offsets.npy", off)
    with pytest.raises(RuntimeError, match=r"excluded slot \d+ \(no_object_224\) stores 1 grasps"):
        _open(out, EXCL_SCENES)


def test_reader_rejects_ok_row_with_empty_slice(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "y", scenes=EXCL_SCENES)
    d = out / "train"
    k = _slot_of(d, "objD")
    _edit_index(d, lambda rs: rs[k].update(status="ok"))          # claims ok, keeps n_grasps > 0, 0 stored
    with pytest.raises(RuntimeError, match="stored grasps"):
        _open(out, EXCL_SCENES)


def _boom_flush(self):
    raise OSError("flush: no space left")


def test_flush_error_alone_is_primary_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(np.memmap, "flush", _boom_flush)
    with pytest.raises(OSError, match="no space left"):
        _build(tmp_path, monkeypatch, "fl")
    s = _summary(tmp_path / "fl")
    assert s["complete"] is False and "flush failed" in s["error"] and "train_stats" not in s
    assert not (tmp_path / "fl" / "train" / "COMPLETE").exists()


def test_flush_error_after_processing_error_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setattr(np.memmap, "flush", _boom_flush)
    with pytest.raises(RuntimeError, match="R4"):                # the original error, not the flush one
        _build(tmp_path, monkeypatch, "fl2", scenes=R4_SCENES)
    s = _summary(tmp_path / "fl2")
    assert s["complete"] is False and "R4" in s["error"] and "train_stats" not in s
    assert not (tmp_path / "fl2" / "train" / "COMPLETE").exists()


@pytest.mark.skipif(multiprocessing.get_start_method() != "fork",
                    reason="workers inherit the monkeypatched loader only under fork")
def test_workers_parity_byte_identical(tmp_path, monkeypatch):
    o0, _ = _build(tmp_path, monkeypatch, "w0", scenes=EXCL_SCENES, workers=0, chunk=2)
    o2, _ = _build(tmp_path, monkeypatch, "w2", scenes=EXCL_SCENES, workers=2, chunk=2)
    files = sorted(p.name for p in (o0 / "train").iterdir() if p.suffix in (".npy", ".csv"))
    assert "ds_comp.npy" in files and "index.csv" in files
    for f in files:
        assert (o0 / "train" / f).read_bytes() == (o2 / "train" / f).read_bytes(), f
    s0, s2 = _summary(o0), _summary(o2)
    assert s0["train_stats"]["normalisation"] == s2["train_stats"]["normalisation"]


# ------------------------------------------------------------------ review finding 3: split identity
def _swap_views(rows):
    rows[0]["view"], rows[1]["view"] = rows[1]["view"], rows[0]["view"]


@pytest.mark.parametrize("fn, splits, msg", [
    (lambda d: _edit_summary(d, split="val"), None, "belongs to split 'val'"),
    (lambda d: _edit_index(d, lambda rs: rs[1].update(view=rs[0]["view"])), None, "duplicate"),
    (lambda d: _edit_index(d, _swap_views), None, "sorted order"),
    (lambda d: None, {"splits": {"train": ["P/objA", "P/objB"]}}, "not in split"),
    (lambda d: None, {"splits": {"train": ["P/objA", "P/objB", "P/objC", "P/objZ"]}}, "missing from the cache"),
    (lambda d: None, {"splits": {"val": ["P/objA"]}}, "not defined"),
    (lambda d: _edit_summary(d, n_scenes_in_split=99), None, "n_scenes_in_split"),
])
def test_reader_rejects_wrong_split_identity(tmp_path, monkeypatch, fn, splits, msg):
    out, _ = _build(tmp_path, monkeypatch, "id")
    fn(out / "train")
    with pytest.raises(RuntimeError, match=msg):
        open_cache(out, "train", splits=splits or _splits_for())


def test_reader_smoke_cache_may_hold_a_subset(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "sub")
    _edit_summary(out / "train", smoke=True)
    sp = {"splits": {"train": ["P/objA", "P/objB", "P/objC", "P/objZ"]}}     # objZ not built: fine for smoke
    assert _open(out, allow_smoke=True)["summary"]["smoke"]
    assert open_cache(out, "train", allow_smoke=True, splits=sp)["summary"]["smoke"]


def test_reader_production_default_checks_splits_hash(tmp_path, monkeypatch):
    import data.cache_layout as cl
    out, _ = _build(tmp_path, monkeypatch, "h")
    f = tmp_path / "splits.json"
    f.write_text(json.dumps(_splits_for()))
    monkeypatch.setattr(cl, "SPLITS_FILE", f)
    with pytest.raises(RuntimeError, match="lacks splits_sha256"):            # synthetic: no provenance
        open_cache(out, "train")
    _edit_summary(out / "train", splits_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="splits changed"):
        open_cache(out, "train")
    _edit_summary(out / "train", splits_sha256=cl._sha256(f))
    assert open_cache(out, "train")["summary"]["n_ok"] == 3
