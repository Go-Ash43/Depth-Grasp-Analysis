"""End-to-end main() of scripts/zone_diagnostics.py on a tiny synthetic cache in a temp dir (review 12):
argument validation and the no-eligible-scene check happen BEFORE any cleanup; a valid --overwrite run
removes only the script's own files and writes provenance, an exact overlay manifest and per-jaw columns.
No dataset or cache directory on the drive is read (data_root is redirected to the temp dir)."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import data.cache_layout as cache_layout
from data.cache_layout import INDEX_FIELDS, LAYOUT_VERSION, array_specs

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("zone_diagnostics_main", REPO / "scripts" / "zone_diagnostics.py")
zd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zd)

N, SIZE, TAUS = 4, 224, [12.0]
# zone_diagnostics opens the cache the production way (splits=None): the synthetic cache records the
# hash of this splits file and the env fixture points data.cache_layout.SPLITS_FILE at it.
SPLITS_TEXT = json.dumps({"splits": {"train": [f"P/o{i}" for i in range(N)]}})


def write_cache(root, statuses=("ok",) * N):
    """Minimal train split in the data/cache_layout format: ok slots get grasps, excluded ones none."""
    d = root / "train"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, (dt, sh) in array_specs(SIZE, TAUS).items():
        a = np.zeros((N,) + sh, dt)
        if name == "m_sensor":
            a[:] = (rng.random(a.shape) < 0.1)
        if name == "obj":
            a[:, 60:170, 60:170] = 1
        np.save(d / f"{name}.npy", a)
    gs, off, rows = [], [0], []
    for i, st in enumerate(statuses):
        g = (np.c_[rng.uniform(300, 700, 3), rng.uniform(300, 700, 3), rng.uniform(-90, 90, 3),
                   rng.uniform(5, 150, 3), rng.uniform(5, 40, 3)] if st == "ok" else np.zeros((0, 5)))
        gs.append(g)
        off.append(off[-1] + len(g))
        rows.append({"slot": i, "part": "P", "object_id": f"o{i}", "view": 0, "status": st,
                     "obj_px_224": 12100, "small_obj": 0, "n_grasps": 3})
    np.save(d / "grasps_1024.npy", np.concatenate(gs).astype(np.float64))
    np.save(d / "grasp_offsets.npy", np.array(off, np.int64))
    with open(d / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    (d / "summary.json").write_text(json.dumps({
        "layout_version": LAYOUT_VERSION, "split": "train", "size": SIZE, "taus": TAUS, "smoke": False,
        "complete": True, "error": None, "git_commit": "abc123", "git_dirty": False,
        "n_ok": sum(s == "ok" for s in statuses), "n_slots": N, "n_processed": N, "n_scenes_in_split": N,
        "splits_sha256": hashlib.sha256(SPLITS_TEXT.encode()).hexdigest()}))
    (d / "COMPLETE").write_text("x\n")
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(zd, "data_root", lambda: str(tmp_path / "dataset"))   # never the real drive path
    sf = tmp_path / "splits.json"
    sf.write_text(SPLITS_TEXT)
    monkeypatch.setattr(cache_layout, "SPLITS_FILE", sf)
    out = tmp_path / "diag"
    (out / "overlays").mkdir(parents=True)
    old = {out / "summary.json": "OLD", out / "config.json": "OLD", out / "per_scene.csv": "OLD",
           out / "overlays" / "slot9_old_0.png": "OLD", out / "overlays" / "manifest.json": "OLD",
           out / "overlays" / "keep_me.txt": "not ours"}
    for p, t in old.items():
        p.write_text(t)
    return tmp_path, out, old


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zone_diagnostics.py", *map(str, argv)])
    zd.main()


def untouched(old):
    return all(p.exists() and p.read_text() == t for p, t in old.items())


@pytest.mark.parametrize("bad", [["--seed", "-1"], ["--n-scenes", "0"], ["--n-overlays", "-1"]])
def test_invalid_args_leave_outputs_intact(env, monkeypatch, bad):
    tmp, out, old = env
    cache = write_cache(tmp / "cache")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--cache-dir", cache, "--out", out, "--overwrite", *bad)
    assert untouched(old)


def test_no_eligible_scene_leaves_outputs_intact(env, monkeypatch):
    tmp, out, old = env
    cache = write_cache(tmp / "cache", statuses=("no_object_224", "no_valid_depth_224", "no_grasps", "load_skip"))
    with pytest.raises(SystemExit, match="no eligible"):
        run_main(monkeypatch, "--cache-dir", cache, "--out", out, "--overwrite")
    assert untouched(old)


def test_refuses_existing_outputs_without_overwrite(env, monkeypatch):
    tmp, out, old = env
    cache = write_cache(tmp / "cache")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--cache-dir", cache, "--out", out)
    assert untouched(old)


def test_valid_overwrite_run(env, monkeypatch):
    tmp, out, old = env
    cache = write_cache(tmp / "cache", statuses=("ok", "no_object_224", "ok", "ok"))
    run_main(monkeypatch, "--cache-dir", cache, "--out", out, "--overwrite", "--n-scenes", 3, "--n-overlays", 2)
    od = out / "overlays"
    assert not (od / "slot9_old_0.png").exists() and (od / "keep_me.txt").read_text() == "not ours"
    man = json.loads((od / "manifest.json").read_text())
    assert sorted(o["file"] for o in man["overlays"]) == sorted(p.name for p in od.glob("slot*.png"))
    assert len(man["overlays"]) == 2 and man["n_scenes"] == 3
    cfg = json.loads((out / "config.json").read_text())
    src = cfg["source_cache"]
    assert src["git_commit"] == "abc123" and src["smoke"] is False and src["layout_version"] == LAYOUT_VERSION
    assert len(src["summary_sha256"]) == 64 and "git_commit" in cfg
    s = json.loads((out / "summary.json").read_text())
    assert s["n_scenes"] == 3 and s["w_norm_check"]["status"] in ("unset", "match", "mismatch")
    with open(out / "per_scene.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and {int(r["slot"]) for r in rows} == {0, 2, 3}     # excluded slot 1 never sampled
    assert "Z-end4:jaw_plus_area_in" in rows[0] and "Z-band:jaw_minus_empty" in rows[0]
