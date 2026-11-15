"""scripts/check_cache_totals.py: R3 total-skip acceptance check on synthetic split summaries."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_cache_totals", Path(__file__).resolve().parents[1] / "scripts" / "check_cache_totals.py")
cct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cct)

SPLITS = ("train", "val", "test_module")


def _split(root, name, n_skipped=0, complete=True, error=None, smoke=False, marker=True):
    d = root / name
    d.mkdir(parents=True)
    recs = [{"obj_dir": f"P/o{i}", "view": 0, "error_type": "SceneDataError", "message": "x"}
            for i in range(n_skipped)]
    (d / "summary.json").write_text(json.dumps({"split": name, "complete": complete, "error": error,
                                                "smoke": smoke, "n_ok": 10,
                                                "skip_budget": {"n_skipped": n_skipped, "skips": recs}}))
    if marker:
        (d / "COMPLETE").write_text("x\n")


def _edit(root, split, fn):
    p = root / split / "summary.json"
    s = json.loads(p.read_text())
    fn(s)
    p.write_text(json.dumps(s))


def _all(root, skips=(0, 0, 0), **kw):
    for s, n in zip(SPLITS, skips):
        _split(root, s, n, **kw)


def test_pass_and_total_counts_loader_skips(tmp_path):
    _all(tmp_path, (20, 20, 10))
    r = cct.check_totals(tmp_path, SPLITS)
    assert r["verdict"] == "PASS" and r["total_skips"] == 50


def test_total_above_50_fails_even_if_each_split_is_fine(tmp_path):
    _all(tmp_path, (30, 15, 6))                                   # each <= 50, total 51
    r = cct.check_totals(tmp_path, SPLITS)
    assert r["verdict"] == "FAIL" and any("total loader skips 51" in p for p in r["problems"])


@pytest.mark.parametrize("kw, msg", [({"complete": False}, "incomplete"), ({"marker": False}, "incomplete"),
                                     ({"error": "boom"}, "error"), ({"smoke": True}, "smoke")])
def test_bad_split_fails(tmp_path, kw, msg):
    _all(tmp_path, **kw)
    r = cct.check_totals(tmp_path, SPLITS)
    assert r["verdict"] == "FAIL" and any(msg in p for p in r["problems"])


def test_missing_split_fails(tmp_path):
    _split(tmp_path, "train")
    assert cct.check_totals(tmp_path, SPLITS)["verdict"] == "FAIL"


def test_sealed_not_read_unless_requested(tmp_path):
    _all(tmp_path)
    (tmp_path / "test_sealed").mkdir()
    (tmp_path / "test_sealed" / "summary.json").write_text("not json")   # would crash if opened
    assert cct.check_totals(tmp_path, SPLITS)["verdict"] == "PASS"


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.update(skip_budget={}), "missing or invalid"),
    (lambda s: s["skip_budget"].update(n_skipped=False), "missing or invalid"),     # bool is not a count
    (lambda s: s["skip_budget"].update(n_skipped=-1), "missing or invalid"),
    (lambda s: s["skip_budget"].update(n_skipped=2), "stored skip records"),         # 0 records stored
    (lambda s: s["skip_budget"].pop("skips"), "stored skip records"),
    (lambda s: s.pop("smoke"), "production build"),                                  # missing smoke field
    (lambda s: s.update(smoke=None), "production build"),
])
def test_invalid_metadata_fails(tmp_path, mutate, msg):
    _all(tmp_path)
    _edit(tmp_path, "val", mutate)
    r = cct.check_totals(tmp_path, SPLITS)
    assert r["verdict"] == "FAIL" and any(msg in p for p in r["problems"])


def test_copied_summary_under_wrong_split_fails(tmp_path):
    _all(tmp_path)
    (tmp_path / "val" / "summary.json").write_text((tmp_path / "train" / "summary.json").read_text())
    r = cct.check_totals(tmp_path, SPLITS)
    assert r["verdict"] == "FAIL" and any("belongs to split 'train'" in p for p in r["problems"])


def test_unknown_split_name_rejected(tmp_path):
    _all(tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        cct.check_totals(tmp_path, ("train", "validation"))
