"""scripts/accept_cache.py on a tiny synthetic cache: a clean build passes every check; each tampering
is caught by the check meant for it (review finding 4). Synthetic only, no dataset access."""
import csv
import json

import numpy as np
import pytest

from data.cache_layout import lie_name, open_cache
from scripts import accept_cache as ac
from tests.test_build_cache import CFG, EXCL_SCENES, _build, _edit_index, _splits_for

SPLITS = _splits_for(EXCL_SCENES)
SLOT_OK, SLOT_OK2, SLOT_EXCL = 0, 1, 2          # objA v0, objA v1 ok; objB load_skip (sorted order)


def _cache(tmp_path, monkeypatch, name="acc"):
    out, _ = _build(tmp_path, monkeypatch, name, scenes=EXCL_SCENES)
    return out


def _open(out):
    return open_cache(out, "train", splits=SPLITS)


def _payload(out):
    prob = ac.Problems()
    ac.check_payload(_open(out), CFG, prob, chunk=2)
    return prob.counts


def _poke(out, name, slot, fn):
    """Edit one slot of one array in place (np.save keeps the .npy header)."""
    p = out / "train" / f"{name}.npy"
    a = np.load(p)
    fn(a[slot])
    np.save(p, a)


def _first(mask, want):
    """(row, col) of the first cell of mask == want."""
    ij = np.argwhere(mask == want)
    assert len(ij), "synthetic scene lacks the needed cell"
    return tuple(ij[0])


def test_clean_cache_passes_payload_and_rederive(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)
    assert _payload(out) == {}
    _fake_loader(monkeypatch)
    cache = _open(out)
    prob = ac.Problems()
    ac.check_rederive(cache, dict(CFG, root=str(tmp_path)), ac.pick_slots(cache["index"], 10, 0, "train"), prob)
    assert prob.counts == {}


def _fake_loader(monkeypatch):
    def fake(obj_dir, view, root=None):
        return EXCL_SCENES[(obj_dir, view)](), None
    monkeypatch.setattr(ac.bc, "try_load_scene", fake)


def _nonbinary(a):
    a[0, 0] = 2


def _nan_comp_on_valid(out):
    ms = np.load(out / "train" / "m_sensor.npy")[SLOT_OK]
    ij = _first(ms, 0)
    _poke(out, "ds_comp", SLOT_OK, lambda a: a.__setitem__(ij, np.nan))


def _comp_ne_ds(out):
    ms = np.load(out / "train" / "m_sensor.npy")[SLOT_OK]
    ij = _first(ms, 0)
    _poke(out, "ds_comp", SLOT_OK, lambda a: a.__setitem__(ij, a[ij] + 1.0))


def _break_nesting(out):
    t_lo, t_hi = sorted(CFG["taus"])[0], sorted(CFG["taus"])[-1]
    lo = np.load(out / "train" / f"{lie_name('any', t_lo)}.npy")[SLOT_OK]
    ij = _first(lo, 0)
    _poke(out, lie_name("any", t_hi), SLOT_OK, lambda a: a.__setitem__(ij, 1))


def _swap_rows(out):
    for p in (out / "train").glob("*.npy"):
        if p.name in ("grasps_1024.npy", "grasp_offsets.npy"):
            continue
        a = np.load(p)
        a[[SLOT_OK, SLOT_OK2]] = a[[SLOT_OK2, SLOT_OK]]
        np.save(p, a)


@pytest.mark.parametrize("tamper, check", [
    (lambda out: _poke(out, "m_sensor", SLOT_OK, _nonbinary), "mask_not_binary"),
    (_nan_comp_on_valid, "ds_comp_nonfinite"),
    (_comp_ne_ds, "ds_comp_ne_ds_outside_holes"),
    (_break_nesting, "lie_not_nested"),
    (lambda out: _edit_index(out / "train", lambda rs: rs[SLOT_OK].update(obj_px_224="1")), "count_obj_px_224"),
    (lambda out: _poke(out, "rgb", SLOT_EXCL, lambda a: a.__setitem__((0, 0, 0), 1)), "excluded_slot_nonzero"),
])
def test_payload_tampering_is_caught(tmp_path, monkeypatch, tamper, check):
    out = _cache(tmp_path, monkeypatch)
    tamper(out)
    assert _payload(out).get(check, 0) >= 1


def test_swapped_rows_are_caught_by_payload_or_rederive(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)
    _swap_rows(out)
    _fake_loader(monkeypatch)
    prob = ac.Problems()
    ac.check_rederive(_open(out), dict(CFG, root=str(tmp_path)), [SLOT_OK, SLOT_OK2], prob)
    assert prob.counts.get("rederive_array", 0) >= 1


def test_rederive_catches_a_value_payload_cannot_see(tmp_path, monkeypatch):
    """A changed completion value INSIDE a hole is finite and off the valid set: only E sees it."""
    out = _cache(tmp_path, monkeypatch)
    ms = np.load(out / "train" / "m_sensor.npy")[SLOT_OK]
    ij = _first(ms, 1)
    _poke(out, "ds_comp", SLOT_OK, lambda a: a.__setitem__(ij, a[ij] + 0.5))
    assert _payload(out) == {}
    _fake_loader(monkeypatch)
    prob = ac.Problems()
    ac.check_rederive(_open(out), dict(CFG, root=str(tmp_path)), [SLOT_OK], prob)
    assert prob.examples["rederive_array"] == [f"slot {SLOT_OK} P/objA v0: ds_comp"]


def _rederive(out, tmp_path, slots=(SLOT_OK, SLOT_OK2)):
    prob = ac.Problems()
    ac.check_rederive(_open(out), dict(CFG, root=str(tmp_path)), list(slots), prob)
    return prob


def test_annotation_count_change_is_its_own_check(tmp_path, monkeypatch):
    """n_grasps_dropped comes from parsing the grasp file, not numpy / cv2: never VERSION-MISMATCH."""
    out = _cache(tmp_path, monkeypatch)
    _edit_index(out / "train", lambda rs: rs[SLOT_OK].update(n_grasps_dropped="7"))
    _fake_loader(monkeypatch)
    prob = _rederive(out, tmp_path)
    assert prob.counts == {"rederive_annotation_field": 1}
    assert ac.verdict({"train": prob.as_dict()}, {"counts": {}}, {"train": False}) == "FAIL"


def test_rebuilt_array_set_must_equal_the_cache_set(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)
    _fake_loader(monkeypatch)
    real = ac.bc.process_scene

    def drop_obj(*a, **k):
        row, arrs, g, st = real(*a, **k)
        return row, {n: v for n, v in arrs.items() if n != "obj"}, g, st
    monkeypatch.setattr(ac.bc, "process_scene", drop_obj)
    assert _rederive(out, tmp_path, [SLOT_OK]).counts == {"rederive_array_set": 1}


def test_one_failing_rebuild_is_logged_and_the_rest_still_run(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)

    def loader(obj_dir, view, root=None):
        if view == 0:
            raise RuntimeError("non-finite value after Telea")
        return EXCL_SCENES[(obj_dir, view)](), None
    monkeypatch.setattr(ac.bc, "try_load_scene", loader)
    prob = _rederive(out, tmp_path)                          # slot 0 (view 0) raises, slot 1 (view 1) ok
    assert prob.counts == {"rederive_exception": 1}
    assert "Telea" in prob.examples["rederive_exception"][0]


def test_coverage_matches_builder_order_and_catches_a_missing_scene(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)
    man = tmp_path / "manifest_acc.csv"
    prob = ac.Problems()
    ac.check_coverage("train", _open(out)["index"], SPLITS, man, prob)
    assert prob.counts == {}
    with open(man, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if not (r["object_id"] == "objA" and r["scene_idx"] == "1")]
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ac.check_coverage("train", _open(out)["index"], SPLITS, man, prob)
    assert prob.counts["coverage"] == 1


def test_coverage_catches_object_in_another_split(tmp_path, monkeypatch):
    out = _cache(tmp_path, monkeypatch)
    sp = {"splits": {**SPLITS["splits"], "val": ["P/objA"]}}
    prob = ac.Problems()
    ac.check_coverage("train", _open(out)["index"], sp, tmp_path / "manifest_acc.csv", prob)
    assert prob.counts["object_in_other_split"] == 1


CFG_NOW = {"size": 224, "taus": [6.0, 12.0], "x": 1}
SPL = {"manifest_sha256": "m", "splits": {"train": ["o1"], "val": ["o2"]}}


def _summ(**kw):
    base = {"cfg": CFG_NOW, "size": 224, "taus": [6.0, 12.0], "lie_rules": ["any", "fraction"],
            "splits_sha256": "s", "manifest_sha256": "m", "git_dirty": False, "git_commit": "c",
            "versions": {"numpy": "1", "cv2": "2"}}
    return {**base, **kw}


@pytest.mark.parametrize("summary, check", [
    (_summ(cfg={**CFG_NOW, "x": 2}), "cfg_differs_from_frozen"),
    (_summ(taus=[6.0]), "summary_layout_differs"),              # top-level taus drive the payload checks
    (_summ(size=160), "summary_layout_differs"),
    (_summ(lie_rules=["any"]), "summary_layout_differs"),
    (_summ(splits_sha256="x"), "splits_hash"),
    (_summ(manifest_sha256="x"), "manifest_hash"),
    (_summ(git_dirty=True), "unclean_build"),
    (_summ(git_commit=None), "unclean_build"),
    (_summ(versions=None), "build_versions_missing"),
    (_summ(versions={"numpy": "1"}), "build_versions_missing"),
    (_summ(versions={"numpy": "1", "cv2": 5}), "build_versions_missing"),
])
def test_metadata_checks(summary, check):
    prob = ac.Problems()
    assert ac.check_metadata("train", summary, CFG_NOW, SPL, "s", prob) is False
    assert set(prob.counts) == {check}


def test_clean_metadata_passes_and_commit_may_differ_between_splits():
    """A separately scheduled build (e.g. test_sealed) has another commit: compatibility is content."""
    for s in (_summ(), _summ(git_commit="later", versions={"numpy": "9", "cv2": "9"})):
        prob = ac.Problems()
        assert ac.check_metadata("train", s, CFG_NOW, SPL, "s", prob) and prob.counts == {}


def test_splits_overlap_is_global():
    prob = ac.Problems()
    ac.check_splits_disjoint({"splits": {"train": ["o1"], "test_sealed": ["o1"]}}, prob)
    assert prob.counts == {"splits_overlap": 1}


def test_verdict_is_per_split_and_only_numeric_rebuild_differences_are_excused():
    clean = {"counts": {}}
    num = {"counts": {"rederive_array": 3, "rederive_index_field": 1}}
    assert ac.verdict({"train": clean}, clean, {"train": True}) == "PASS"
    assert ac.verdict({"train": num}, clean, {"train": False}) == "VERSION-MISMATCH"
    assert ac.verdict({"train": num}, clean, {"train": True}) == "FAIL"
    # a mismatch on val must not excuse a numeric failure on train, whose versions match
    assert ac.verdict({"train": num, "val": num}, clean, {"train": True, "val": False}) == "FAIL"
    for structural in ("rederive_load_skip", "rederive_status", "rederive_grasps", "mask_not_binary"):
        assert ac.verdict({"train": {"counts": {structural: 1}}}, clean, {"train": False}) == "FAIL"
    assert ac.verdict({"train": clean}, {"counts": {"splits_overlap": 1}}, {"train": True}) == "FAIL"


# ------------------------------------------------------------------ main(): end to end on a synthetic cache
def _prod(tmp_path, monkeypatch, versions=None):
    """Synthetic cache made to look like a production build: splits file + hash, manifest hash, clean
    provenance, current (or given) versions. Loader faked; data root is a tmp dir."""
    import data.cache_layout as cl
    out = _cache(tmp_path, monkeypatch)
    man = tmp_path / "manifest_acc.csv"
    sf = tmp_path / "splits.json"
    sf.write_text(json.dumps({"manifest": str(man), "manifest_sha256": cl._sha256(man), **SPLITS}))
    monkeypatch.setattr(cl, "SPLITS_FILE", sf)
    monkeypatch.setattr(ac, "SPLITS_FILE", sf)
    monkeypatch.setattr(ac, "data_root", lambda: str(tmp_path / "dataset"))
    now = {"python": "x", "numpy": np.__version__, "cv2": ac.cv2.__version__}
    _edit_summary(out / "train", splits_sha256=cl._sha256(sf), manifest_sha256=cl._sha256(man),
                  git_commit="abc", git_dirty=False, versions=versions or now)
    _fake_loader(monkeypatch)
    return out


def _edit_summary(d, **kw):
    p = d / "summary.json"
    s = json.loads(p.read_text())
    s.update(kw)
    p.write_text(json.dumps(s))


def _main(out, rep, *extra):
    code = ac.main(["--cache-dir", str(out), "--splits", "train", "--out", str(rep), "--n-rederive", "2",
                    *map(str, extra)])
    return code, json.loads((rep / "report.json").read_text())


def test_main_passes_a_clean_cache(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch)
    code, r = _main(out, tmp_path / "rep")
    assert code == 0 and r["verdict"] == "PASS", r
    assert r["splits"]["train"]["heavy_checks_run"] and len(r["splits"]["train"]["rederive_slots"]) == 2
    assert json.loads((tmp_path / "rep" / "config.json").read_text())["args"]["n_rederive"] == 2


def test_main_error_replaces_a_stale_pass(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch)
    rep = tmp_path / "rep"
    assert _main(out, rep)[1]["verdict"] == "PASS"

    def boom(*a, **k):
        raise OSError("drive gone")
    monkeypatch.setattr(ac, "check_payload", boom)
    code, r = _main(out, rep, "--overwrite")
    assert code == 3 and r["verdict"] == "ERROR" and "drive gone" in r["error"]


def test_main_config_write_failure_after_a_pass_leaves_no_stale_pass(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch)
    rep = tmp_path / "rep"
    assert _main(out, rep)[1]["verdict"] == "PASS"

    def boom():
        raise OSError("cannot read git state")
    monkeypatch.setattr(ac, "git_fields", boom)              # fails while building the new config
    code, r = _main(out, rep, "--overwrite")
    assert code == 3 and r["verdict"] == "ERROR" and "git state" in r["error"]


def test_main_split_without_an_ok_scene_fails(tmp_path, monkeypatch):
    """Stand-in for a split whose every scene is excluded: pick_slots finds nothing to rebuild."""
    out = _prod(tmp_path, monkeypatch)
    monkeypatch.setattr(ac, "pick_slots", lambda *a, **k: [])
    code, r = _main(out, tmp_path / "rep")
    assert code == 1 and r["per_split"]["train"]["counts"] == {"rederive_no_eligible_scene": 1}


def test_unknown_build_versions_never_excuse_a_rebuild_difference():
    assert ac.versions_match({"versions": None}, {"numpy": "1", "cv2": "2"}) is True
    assert ac.versions_match({"versions": {"numpy": "1", "cv2": "3"}}, {"numpy": "1", "cv2": "2"}) is False


def test_main_metadata_failure_skips_heavy_checks(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch)
    s = json.loads((out / "train" / "summary.json").read_text())
    _edit_summary(out / "train", cfg={**s["cfg"], "telea_radius": 5.0})

    def must_not_run(*a, **k):
        raise AssertionError("payload ran despite a metadata failure")
    monkeypatch.setattr(ac, "check_payload", must_not_run)
    code, r = _main(out, tmp_path / "rep")
    assert code == 1 and r["per_split"]["train"]["counts"] == {"cfg_differs_from_frozen": 1}
    assert r["splits"]["train"]["heavy_checks_run"] is False


def test_main_numeric_difference_under_other_versions_is_version_mismatch(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch, versions={"python": "x", "numpy": "0.0", "cv2": "0.0"})
    ms = np.load(out / "train" / "m_sensor.npy")
    poked = 0
    for slot in ac.pick_slots(_open(out)["index"], 2, 0, "train"):          # the slots E will rebuild
        if (ms[slot] == 1).any():                 # a hole value: invisible to D, numeric for E
            ij = _first(ms[slot], 1)
            _poke(out, "ds_comp", slot, lambda a: a.__setitem__(ij, a[ij] + 0.5))
            poked += 1
    assert poked, "no rebuilt slot has a hole"
    code, r = _main(out, tmp_path / "rep")
    assert code == 2 and r["verdict"] == "VERSION-MISMATCH", r["per_split"]


def test_main_load_skip_under_other_versions_is_still_fail(tmp_path, monkeypatch):
    out = _prod(tmp_path, monkeypatch, versions={"python": "x", "numpy": "0.0", "cv2": "0.0"})
    monkeypatch.setattr(ac.bc, "try_load_scene", lambda obj_dir, view, root=None: (
        None, {"obj_dir": obj_dir, "view": view, "error_type": "SceneDataError", "message": "gone"}))
    code, r = _main(out, tmp_path / "rep")
    assert code == 1 and "rederive_load_skip" in r["per_split"]["train"]["counts"]


def test_cli_refuses_out_inside_cache_and_existing_report(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with pytest.raises(SystemExit, match="cache directory"):
        ac.main(["--cache-dir", str(cache_dir), "--out", str(cache_dir / "acc")])
    out = tmp_path / "acc"
    out.mkdir()
    (out / "report.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="already exist"):
        ac.main(["--cache-dir", str(cache_dir), "--out", str(out)])


@pytest.mark.parametrize("bad", [["--splits", "train,train"], ["--splits", "foo"], ["--chunk", "0"],
                                 ["--n-rederive", "-1"], ["--n-rederive", "0"], ["--seed", "-1"]])
def test_cli_rejects_bad_args(bad):
    with pytest.raises(SystemExit):
        ac.parse_args(["--out", "x", *bad])
