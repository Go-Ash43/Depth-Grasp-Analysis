"""data/paths.py override, scripts/cache_manifest.py write / verify, training/env.py lock parsing.
Synthetic files only; no dataset, no GPU."""
import json

import pytest

from data.paths import ENV_VAR, load_paths
from scripts import cache_manifest as cm
from training import env as pc


# ------------------------------------------------------------------ paths override
def test_paths_default_and_override(tmp_path):
    base = load_paths(env={})
    o = tmp_path / "paths_pod.json"
    o.write_text(json.dumps({"cache_dir": "/root/cache"}))
    got = load_paths(env={ENV_VAR: str(o)})
    assert got["cache_dir"] == "/root/cache"
    assert {k: v for k, v in got.items() if k != "cache_dir"} == {k: v for k, v in base.items() if k != "cache_dir"}


def test_paths_unknown_key_refused(tmp_path):
    o = tmp_path / "paths_pod.json"
    o.write_text(json.dumps({"cache_dirr": "/root/cache"}))                 # a typo must not fall back
    with pytest.raises(ValueError, match="unknown path keys"):
        load_paths(env={ENV_VAR: str(o)})


# ------------------------------------------------------------------ cache manifest
def _fake_cache(root):
    for sp in ("train", "val"):
        d = root / sp
        d.mkdir(parents=True)
        (d / "COMPLETE").write_text("ok")
        (d / "rgb.npy").write_bytes(bytes(range(256)) * 50)
        (d / "index.csv").write_text(f"slot\n0\n{sp}\n")
    return root


def _run(*args):
    return cm.main([*map(str, args)])


def test_manifest_roundtrip_and_tampering(tmp_path):
    cache = _fake_cache(tmp_path / "cache")
    out = tmp_path / "man"
    assert _run("write", "--cache-dir", cache, "--splits", "train,val", "--out", out) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_files"] == 6 and [f["path"] for f in man["files"]] == sorted(f["path"] for f in man["files"])
    mf = out / "manifest.json"
    assert _run("verify", "--cache-dir", cache, "--manifest", mf, "--out", tmp_path / "v0") == 0

    b = bytearray((cache / "train" / "rgb.npy").read_bytes())
    b[100] ^= 1                                                              # same size, one bit flipped
    (cache / "train" / "rgb.npy").write_bytes(bytes(b))
    (cache / "val" / "extra.npy").write_bytes(b"x")
    (cache / "val" / "index.csv").unlink()
    assert _run("verify", "--cache-dir", cache, "--manifest", mf, "--out", tmp_path / "v1", "--workers", "3") == 1
    rep = json.loads((tmp_path / "v1" / "verify.json").read_text())
    assert rep["changed"] == ["train/rgb.npy"] and rep["extra"] == ["val/extra.npy"]
    assert rep["missing"] == ["val/index.csv"] and rep["verdict"] == "FAIL"


def test_manifest_refuses_incomplete_split_and_out_inside_cache(tmp_path):
    cache = _fake_cache(tmp_path / "cache")
    (cache / "val" / "COMPLETE").unlink()
    with pytest.raises(SystemExit, match="COMPLETE"):
        _run("write", "--cache-dir", cache, "--splits", "val", "--out", tmp_path / "m")
    with pytest.raises(SystemExit):
        _run("write", "--cache-dir", cache, "--splits", "train", "--out", cache / "inside")


# ------------------------------------------------------------------ lock parsing (training/env.py)
def test_parse_lock(tmp_path):
    p = tmp_path / "lock.txt"
    p.write_text("# comment\nnumpy==2.4.6\n\n--extra-index-url https://x\ntorch==2.14.1+cu132 ; python_version>'3'\n")
    assert pc.parse_lock(p) == {"numpy": "2.4.6", "torch": "2.14.1+cu132"}
    p.write_text("numpy>=2\n")
    with pytest.raises(ValueError, match="exact pin"):
        pc.parse_lock(p)


def test_real_lock_parses():
    pins = pc.parse_lock()
    assert {"numpy", "opencv-python-headless", "torch"} <= set(pins)


# ------------------------------------------------------------------ report lifecycle
def test_manifest_verify_error_replaces_an_old_pass(tmp_path, monkeypatch):
    cache = _fake_cache(tmp_path / "cache")
    assert _run("write", "--cache-dir", cache, "--splits", "train", "--out", tmp_path / "m") == 0
    mf, v = tmp_path / "m" / "manifest.json", tmp_path / "v"
    assert _run("verify", "--cache-dir", cache, "--manifest", mf, "--out", v) == 0
    assert json.loads((v / "verify.json").read_text())["verdict"] == "PASS"

    def boom(*a, **k):
        raise OSError("disk vanished")
    monkeypatch.setattr(cm, "hash_all", boom)
    assert _run("verify", "--cache-dir", cache, "--manifest", mf, "--out", v, "--overwrite") == 3
    rep = json.loads((v / "verify.json").read_text())
    assert rep["verdict"] == "ERROR" and "disk vanished" in rep["error"] and rep["traceback"]


def test_manifest_write_crash_leaves_no_manifest(tmp_path, monkeypatch):
    cache = _fake_cache(tmp_path / "cache")
    out = tmp_path / "m"
    assert _run("write", "--cache-dir", cache, "--splits", "train", "--out", out) == 0

    def boom(*a, **k):
        raise OSError("disk vanished")
    monkeypatch.setattr(cm, "hash_all", boom)
    with pytest.raises(OSError):
        _run("write", "--cache-dir", cache, "--splits", "train", "--out", out, "--overwrite")
    assert not (out / "manifest.json").exists()                      # never the old one beside a new config
